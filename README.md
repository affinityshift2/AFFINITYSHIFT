
# AffinityShift

**Membership Inference for Diffusion Models via Intrinsic-Property-Conditioned Model–Image Affinity**

This repository contains the implementation of **AffinityShift**, a weakly supervised, one-class membership inference framework for diffusion models.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Hugging Face Diffusers
- Transformers
- Accelerate
- PEFT
- LPIPS
- scikit-image
- NumPy
- Pillow
- torchvision

Install dependencies:

```bash
pip install torch torchvision diffusers transformers accelerate peft lpips scikit-image numpy pillow tqdm
```

## Repository Structure

```text
AffinityShift/
├── README.md
├── finetune_sd14_affinityshift.py
└── affinity_metrics_sd14.py
```

Additional feature-extraction and detection scripts will be added as they are validated.

## Dataset Preparation

Each image must have a corresponding `.txt` file containing its text prompt.

```text
dataset/
├── member/
│   ├── image001.jpg
│   └── image001.txt
└── non_member/
    ├── image002.jpg
    └── image002.txt
```

Only member images are used to fine-tune the target diffusion model.

## Fine-Tuning

Fine-tune Stable Diffusion v1.4 using the member dataset.

**MS-COCO:**

```bash
python finetune_sd14_affinityshift.py \
  --pretrained-model /path/to/sd-v1-4 \
  --member-dir /path/to/mscoco/member \
  --output-dir /path/to/output/mscoco \
  --dataset mscoco
```

**Flickr30k:**

```bash
python finetune_sd14_affinityshift.py \
  --pretrained-model /path/to/sd-v1-4 \
  --member-dir /path/to/flickr30k/member \
  --output-dir /path/to/output/flickr30k \
  --dataset flickr30k
```

## Model–Image Affinity Extraction

Extract SSIM, LPIPS, and Ensemble Variance using the pretrained and fine-tuned SD v1.4 models.

```bash
python affinity_metrics_sd14.py \
  --pretrained-model /path/to/sd-v1-4 \
  --finetuned-model /path/to/output/mscoco/pipeline \
  --input-dir /path/to/dataset/member \
  --output-csv /path/to/results/affinity.csv \
  --reconstruction-dir /path/to/results/reconstructions
```

Repeat for non-member images using the same target-model checkpoint.

## Reproducibility

- Use image resolution 512 × 512.
- Fine-tune using only member images.
- Use disjoint non-member reference-fitting, calibration, and evaluation subsets.
- Estimate reference functions using only non-member samples.
- Calibrate the decision threshold on held-out non-members.
- Use consistent model checkpoints, captions, and preprocessing.

## Implementation Status

Stable Diffusion v1.4 fine-tuning and affinity extraction scripts are available. Intrinsic-property extraction, reference-based detection, and Stable Diffusion v3.5 Large scripts will be added separately.

## Citation

If you use this code, please cite:

*AffinityShift: Membership Inference for Diffusion Models via Intrinsic-Property-Conditioned Model–Image Affinity.*

Citation details will be provided upon publication.
