import argparse
import csv
import gc
import math
import os
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from skimage.metrics import structural_similarity

WIDTH = HEIGHT = 512
INVERSION_STEPS = 100
INVERSION_GUIDANCE = 1.0
RECONSTRUCTION_STEPS = 50
RECONSTRUCTION_GUIDANCE = 3.5
ENSEMBLE_K = 5
ENSEMBLE_SEEDS = (42, 123, 456, 789, 1024)
PERTURBATION_SCALE = 0.05
CSV_COLUMNS = ("Image_Name", "Prompt", "SSIM", "LPIPS", "Ensemble_Variance")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-model", required=True,
                        help="Local SD v1.4 model directory BEFORE fine-tuning")
    parser.add_argument("--finetuned-model", default=None,
                        help="Complete fine-tuned SD v1.4 diffusers checkpoint")
    parser.add_argument("--lora-root", default=None,
                        help="Alternative to --finetuned-model: PEFT LoRA root with unet/ and optional text_encoder/")
    parser.add_argument("--input-dir", required=True,
                        help="Images and same-stem .txt captions")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--reconstruction-dir", default=None)
    parser.add_argument("--temp-dir", default=None,
                        help="Temporary inverted latents (automatically cleared after success)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if bool(args.finetuned_model) == bool(args.lora_root):
        parser.error("Supply exactly one of --finetuned-model or --lora-root")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def model_pipeline(model_path, device):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe.unet.config, "in_channels", None) != 4:
        raise ValueError("Expected SD v1.x latent UNet with four input channels")
    if getattr(pipe.scheduler.config, "prediction_type", "epsilon") != "epsilon":
        raise ValueError("Expected epsilon prediction for this SD v1.4 implementation")
    return pipe


def merge_legacy_peft_lora(pipe, root):
    """Merge same PEFT adapter layout as the user-provided old script."""
    from peft import PeftModel
    unet = Path(root) / "unet"
    if not unet.is_dir():
        raise FileNotFoundError(f"Missing PEFT adapter directory: {unet}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(unet)).merge_and_unload()
    text_encoder = Path(root) / "text_encoder"
    if text_encoder.is_dir():
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, str(text_encoder)
        ).merge_and_unload()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    return pipe


def make_scheduler(pipe, inverse, steps, device):
    base = dict(pipe.scheduler.config)
    cls = DDIMInverseScheduler if inverse else DDIMScheduler
    # Use the SAME beta schedule, timestep spacing and alpha convention
    # for inversion and reconstruction, except for their step counts.
    scheduler = cls.from_config(base, clip_sample=False, set_alpha_to_one=False,
                                timestep_spacing="leading", steps_offset=0)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


def preprocess_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.resize((WIDTH, HEIGHT), resample=Image.Resampling.LANCZOS)


def prompt_embeddings(pipe, caption, device, guidance):
    tokens = pipe.tokenizer(
        caption, padding="max_length", truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        conditional = pipe.text_encoder(tokens.input_ids.to(device))[0]
        if guidance <= 1.0:
            return conditional, None
        uncond_tokens = pipe.tokenizer(
            "", padding="max_length", truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        unconditional = pipe.text_encoder(uncond_tokens.input_ids.to(device))[0]
    return conditional, unconditional


def predict_noise(pipe, latent, timestep, conditional, unconditional, guidance):
    if unconditional is None:
        return pipe.unet(latent, timestep, encoder_hidden_states=conditional).sample
    # Keep batch size 1 to avoid doubling activations on 4090.
    uncond = pipe.unet(latent, timestep, encoder_hidden_states=unconditional).sample
    cond = pipe.unet(latent, timestep, encoder_hidden_states=conditional).sample
    return uncond + guidance * (cond - uncond)


def image_to_latent(pipe, image, device):
    image_np = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    pixel = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    pixel = pixel.to(device=device, dtype=pipe.vae.dtype)
    with torch.inference_mode():
        # Deterministic encoding: avoid unreported stochastic VAE sampling.
        latent = pipe.vae.encode(pixel).latent_dist.mean
    return latent * pipe.vae.config.scaling_factor


def latent_to_image(pipe, latent):
    with torch.inference_mode():
        decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
    arr = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).round()
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


@torch.inference_mode()
def invert_image(pipe, image, caption, device):
    scheduler = make_scheduler(pipe, True, INVERSION_STEPS, device)
    latent = image_to_latent(pipe, image, device)
    conditional, _ = prompt_embeddings(pipe, caption, device, INVERSION_GUIDANCE)
    for t in scheduler.timesteps:  # ascending timesteps
        predicted = predict_noise(pipe, latent, t, conditional, None, INVERSION_GUIDANCE)
        latent = scheduler.step(predicted, t, latent).prev_sample
    return latent


@torch.inference_mode()
def reconstruct(pipe, inverted_latent, caption, device):
    scheduler = make_scheduler(pipe, False, RECONSTRUCTION_STEPS, device)
    latent = inverted_latent.clone()
    conditional, unconditional = prompt_embeddings(
        pipe, caption, device, RECONSTRUCTION_GUIDANCE
    )
    for t in scheduler.timesteps:  # descending timesteps
        predicted = predict_noise(
            pipe, latent, t, conditional, unconditional, RECONSTRUCTION_GUIDANCE
        )
        latent = scheduler.step(predicted, t, latent, eta=0.0).prev_sample
    return latent_to_image(pipe, latent)


def rgb_for_lpips(image, device):
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    return (tensor.permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0)


def compute_ssim(original, reconstructed):
    a = np.asarray(original.convert("L"), dtype=np.uint8)
    b = np.asarray(reconstructed.convert("L"), dtype=np.uint8)
    return float(structural_similarity(a, b, data_range=255))


def compute_lpips(original, reconstructed, lpips_model, device):
    with torch.inference_mode():
        result = lpips_model(
            rgb_for_lpips(original, device), rgb_for_lpips(reconstructed, device)
        )
    return float(result.item())


def compute_ensemble_variance(pipe, inverted_latent, caption, device):
    recon_gray = []
    latent_std = inverted_latent.float().std(unbiased=False)
    for seed in ENSEMBLE_SEEDS:
        generator = torch.Generator(device=device).manual_seed(seed)
        epsilon = torch.randn(
            inverted_latent.shape, generator=generator, device=device,
            dtype=inverted_latent.dtype,
        )
        perturbed = inverted_latent + (PERTURBATION_SCALE * latent_std * epsilon).to(
            inverted_latent.dtype
        )
        reconstruction = reconstruct(pipe, perturbed, caption, device)
        gray = np.asarray(reconstruction.convert("L"), dtype=np.float32) / 255.0
        recon_gray.append(gray)
    stack = np.stack(recon_gray, axis=0)
    return float(np.var(stack, axis=0, ddof=0).mean())


def dataset_items(directory):
    files = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    items = []
    for image in files:
        caption_file = image.with_suffix(".txt")
        if not caption_file.is_file():
            raise FileNotFoundError(f"Missing required caption: {caption_file}")
        caption = caption_file.read_text(encoding="utf-8").strip()
        if not caption:
            raise ValueError(f"Empty caption: {caption_file}")
        items.append((image, caption))
    return items


def cleanup_pipeline(pipe, device):
    del pipe
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run():
    args = parse_args()
    items = dataset_items(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = Path(args.reconstruction_dir) if args.reconstruction_dir else None
    if reconstruction_dir:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

    import tempfile
    with tempfile.TemporaryDirectory(dir=args.temp_dir) as work_dir:
        work = Path(work_dir)

        # Pass 1: invert EVERY image using the ORIGINAL, pre-fine-tuning model.
        print("[1/2] DDIM inversion using pre-fine-tuning SD v1.4")
        pre = model_pipeline(args.pretrained_model, args.device)
        for index, (path, caption) in enumerate(items):
            latent = invert_image(pre, preprocess_image(path), caption, args.device)
            torch.save(latent.detach().cpu(), work / f"{index:08d}.pt")
            print(f"  inverted {index + 1}/{len(items)}: {path.name}")
        cleanup_pipeline(pre, args.device)

        # Pass 2: load target checkpoint (or the original model + merged LoRA).
        print("[2/2] Reconstructing with FINE-TUNED SD v1.4")
        target_path = args.finetuned_model or args.pretrained_model
        target = model_pipeline(target_path, args.device)
        if args.lora_root:
            target = merge_legacy_peft_lora(target, args.lora_root)
        lpips_model = lpips.LPIPS(net="vgg").to(args.device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)

        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for index, (path, caption) in enumerate(items):
                original = preprocess_image(path)
                latent = torch.load(
                    work / f"{index:08d}.pt", map_location=args.device,
                    weights_only=True,
                ).to(dtype=target.unet.dtype)
                reconstruction = reconstruct(target, latent, caption, args.device)
                if reconstruction_dir:
                    reconstruction.save(reconstruction_dir / f"{path.stem}_reconstruction.png")
                s = compute_ssim(original, reconstruction)
                l = compute_lpips(original, reconstruction, lpips_model, args.device)
                v = compute_ensemble_variance(target, latent, caption, args.device)
                if not all(map(math.isfinite, (s, l, v))):
                    raise ValueError(f"Non-finite metric(s) for {path.name}")
                writer.writerow({
                    "Image_Name": path.name, "Prompt": caption,
                    "SSIM": f"{s:.8f}", "LPIPS": f"{l:.8f}",
                    "Ensemble_Variance": f"{v:.10f}",
                })
                file.flush()
                print(f"  {index+1}/{len(items)} {path.name}: SSIM={s:.5f}, "
                      f"LPIPS={l:.5f}, EV={v:.8f}")
        cleanup_pipeline(target, args.device)
    print(f"Finished {len(items)} images. CSV: {output_csv}")


if __name__ == "__main__":
    run()
