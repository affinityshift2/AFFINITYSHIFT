#!/usr/bin/env python3
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


def main():
    args = parse_args()
    pretrained = Path(args.pretrained_model).expanduser().resolve()
    member_dir = Path(args.member_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not pretrained.is_dir() or not member_dir.is_dir():
        raise FileNotFoundError('The pre-trained model directory and member directory must exist.')
    output.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder='unet')
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder='scheduler')

    # Full UNet fine-tuning; VAE and caption encoder remain frozen.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    vae.eval()
    text_encoder.eval()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    dataset = MemberCaptionDataset(member_dir, tokenizer, args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    weight_dtype = (torch.float16 if accelerator.mixed_precision == 'fp16'
                    else torch.bfloat16 if accelerator.mixed_precision == 'bf16'
                    else torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    config = vars(args).copy()
    config['num_member_images'] = len(dataset)
    if accelerator.is_main_process:
        (output / 'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    accelerator.print(f'Training on {len(dataset)} MEMBER images; {args.max_steps} optimizer updates')

    global_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        state_path = Path(args.resume_from) / 'step.json'
        if not state_path.is_file():
            raise FileNotFoundError(f'Missing step counter: {state_path}')
        global_step = int(json.loads(state_path.read_text(encoding='utf-8'))['global_step'])
        accelerator.print(f'Resumed at optimizer step {global_step}')

    progress = tqdm(total=args.max_steps, initial=global_step,
                    disable=not accelerator.is_local_main_process, desc='SD v1.4 fine-tuning')
    unet.train()
    # Resume uses a new shuffled loader traversal. For exact interruption/restart
    # equivalence, a stateful sampler would additionally be required.
    while global_step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pixels = batch['pixel_values'].to(dtype=weight_dtype)
                with torch.no_grad():
                    # Draw latent samples from the frozen VAE, as in standard SD training.
                    latents = vae.encode(pixels).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unsupported prediction type: {noise_scheduler.config.prediction_type}')
                loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f'{loss.detach().item():.4f}')
                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt = output / f'checkpoint-{global_step}'
                    accelerator.wait_for_everyone()
                    accelerator.save_state(str(ckpt))
                    if accelerator.is_main_process:
                        (ckpt / 'step.json').write_text(
                            json.dumps({'global_step': global_step}), encoding='utf-8'
                        )
                if global_step >= args.max_steps:
                    break
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Standalone full SD pipeline for the downstream inversion/reconstruction step.
        # Keep the unfine-tuned pre-trained model separately for DDIM inversion.
        trained_unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained, unet=trained_unet, torch_dtype=torch.float32
        )
        pipeline.save_pretrained(output / 'pipeline')
        (output / 'completed_steps.json').write_text(
            json.dumps({'global_step': global_step, 'member_images': len(dataset)}, indent=2),
            encoding='utf-8'
        )
        accelerator.print(f'Final fine-tuned pipeline: {output / "pipeline"}')

if __name__ == '__main__':
    main()
