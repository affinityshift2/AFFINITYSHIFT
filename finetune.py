import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

class MemberCaptionDataset(Dataset):
    def __init__(self, directory: Path, tokenizer: CLIPTokenizer, resolution: int):
        self.directory = directory
        self.tokenizer = tokenizer
        self.samples = []
        for image_path in sorted(directory.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                caption_path = image_path.with_suffix('.txt')
                if not caption_path.is_file():
                    raise FileNotFoundError(f'Missing caption for {image_path}: {caption_path}')
                caption = caption_path.read_text(encoding='utf-8').strip()
                if not caption:
                    raise ValueError(f'Empty caption: {caption_path}')
                self.samples.append((image_path, caption))
        if not self.samples:
            raise ValueError(f'No image-caption pairs found in {directory}')
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, caption = self.samples[index]
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            pixels = self.transform(rgb)
        tokens = self.tokenizer(
            caption, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        ).input_ids[0]
        return {'pixel_values': pixels, 'input_ids': tokens}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-model', required=True, help='Local SD v1.4 Diffusers directory')
    parser.add_argument('--member-dir', required=True, help='Directory of member images and matching .txt captions')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset', required=True, choices=['mscoco', 'flickr30k'])
    parser.add_argument('--max-steps', type=int, default=None, help='Default: 150000 COCO; 60000 Flickr30k')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--resume-from', default=None, help='Path to accelerate checkpoint directory')
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 150_000 if args.dataset == 'mscoco' else 60_000
    if args.batch_size != 1 or args.resolution != 512 or args.learning_rate != 1e-4:
        print('WARNING: supplied batch size / resolution / learning rate differ from the manuscript.')
    if args.gradient_accumulation_steps != 1:
        print('WARNING: gradient accumulation changes the effective batch size from the manuscript.')
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    return args
