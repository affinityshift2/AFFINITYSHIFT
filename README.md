
# AffinityShift

**Membership Inference for Diffusion Models via Intrinsic-Property-Conditioned Model–Image Affinity**

This repository contains the implementation of **AffinityShift**, a weakly supervised, one-class membership inference framework for diffusion models.

## 🔍 Framework Overview

<p align="left">
  <img src="overview.PNG" alt="Framework Overview" width="650">
</p>

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

**Dataset**

```bash
python finetune_sd14_affinityshift.py \
  --pretrained-model /path/to/model \
  --member-dir /path/to/dataset \
  --output-dir /path/to/output/folder \
  --dataset name
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



## 2️⃣ Intrinsic Image Properties

This pathway extracts two target-model-independent features: **Image Predictability** and **Complexity**.

### 🔹 Image Predictability

Estimate image predictability using the pretrained [SReC model](https://github.com/caoscott/SReC).

```bash
python image_predictibility.py \
  --file images.txt \
  --load /path/to/srec_checkpoint.pth \
  --srec-repo /path/to/SReC \
  --output-csv results/predictability.csv \
  --mode official-bpsp
```

**Note:** `official-bpsp` computes an SReC-based predictability proxy. Exact local/global predictability, as defined in the paper, requires `--mode paper-dual-context` with a verified probability extractor.

### 🔹 Image Complexity

Compute complexity from six descriptors: edge density, gradient magnitude, Laplacian variance, Shannon entropy, colorfulness, and JPEG PSNR.

First, fit normalization statistics using reference-fitting non-members:

```bash
python complexity.py fit \
  --images nonmember_reference.txt \
  --stats results/complexity_stats.json \
  --output-csv results/reference_complexity.csv
```

Then apply the saved statistics to other images:

```bash
python complexity.py transform \
  --images evaluation_images.txt \
  --stats results/complexity_stats.json \
  --output-csv results/evaluation_complexity.csv
```

The same normalization statistics are used for members, calibration non-members, and evaluation non-members.


## Implementation Status

Stable Diffusion v1.4 fine-tuning and affinity extraction scripts are available. Intrinsic-property extraction, reference-based detection, and Stable Diffusion v3.5 Large scripts will be added separately.

# Classification: 
After all features are in added in excel sheet, need to execute this command to classify

```bash
python RidgeRegression.py \
  --input /path/to/combined_features.xlsx \
  --output results/affinityshift_predictions.xlsx \
  --summary results/affinityshift_summary.json
```
