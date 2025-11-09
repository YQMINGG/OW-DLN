# OW-DLN: Open World Defect Learning Network

<div align="center">

**An Industrial Defect Detection Framework for Open World Scenarios**

[English](README_EN.md) | 简体中文

</div>

---

## 📋 Table of Contents

- [Project Introduction](#project-introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Complete Training Pipeline](#complete-training-pipeline)
- [Model Inference](#model-inference)
- [Performance Evaluation](#performance-evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Project Introduction

OW-DLN (Open World Defect Learning Network) is an innovative industrial defect detection framework specifically designed to handle **open world scenarios**, i.e., capable of identifying novel defect types not seen during training.

### Core Challenge

Traditional defect detection models assume that all defect types in the test set have appeared in the training set, which is unrealistic in actual industrial scenarios:
- ✗ New defect types constantly emerge
- ✗ Labeling all defect types is too costly
- ✗ Some rare defects are difficult to collect sufficient samples

### Solution

OW-DLN implements open world defect detection through the following innovative pipeline:

```
Defect-free Data Preparation → MFI Repair and Reconstruction → DVPS Training → Pseudo-label Generation → OW-DLN Two-stage Training → Open World Detection
```

---

## Core Features

### 🎯 Open World Detection Capability
- Capable of detecting and localizing novel defect types not seen during training
- Distinguishing between known defects and unknown defects
- Avoiding misclassification of unknown defects as known categories

### 🔧 Mask-Free Inpainting and Reconstruction (MFI)
- No need for manual annotation of defect masks
- Automatically learns defect regions and performs repair
- Generates high-quality defect-free reference images

### 🎨 Defect Variation Prediction Segmentation (DVPS)
- Contrastive learning-based defect localization
- Input: Original image + MFI repaired image
- Output: Precise defect segmentation mask

### 🏷️ Automatic Pseudo-label Generation
- Combines DVPS and preliminary OW-DLN results
- Intelligent label merging and filtering
- Generates high-quality pseudo-labels for unknown defects

### 📊 Professional Evaluation Metrics
- U-Recall: Unknown defect recall rate
- WI (Wilderness Impact): Wilderness impact
- A-OSE: Absolute Open Set Error

---

### Key Components

1. **SD-Inpainting** (Optional)
   - Location: `MFI/SD-Inpainting/`
   - Function: Preliminary defect repair, generating defect-free reference images
   - Applicable scenarios: Lack of clean defect-free images

2. **MFI (Mask-Free Inpainting)**
   - Location: `MFI/`
   - Function: Mask-free defect repair and reconstruction
   - Model: VQ-VAE + PixelSNAIL
   - Output: High-quality defect-free images

3. **DVPS (Defect Variation Prediction Segmentation)**
   - Location: `DVPS/` (formerly `unet/`)
   - Function: Contrast-based defect segmentation
   - Architecture: U-Net
   - Input: Original image + MFI repaired image
   - Output: Defect segmentation mask

4. **OW-DLN (Open World Detection Network)**
   - Location: `ultralytics/`
   - Function: Open world defect detection
   - Based on: YOLO11 + custom modules (DAAM, ECA, etc.)
   - Config: `ultralytics/cfg/models/11/OW-DLN.yaml`

---

## Environment Setup

```bash
# Clone repository
git clone 
cd OW-DLN

# Create conda environment
conda create -n owdln python=3.10
conda activate owdln

# Install PyTorch (choose according to CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install this project (editable mode)
pip install -e .
```

### requirements.txt

```txt
# Core dependencies
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0

# Data processing
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
albumentations>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0

# Other tools
pyyaml>=6.0
tensorboard>=2.10.0
omegaconf>=2.1.0
```

---

## Quick Start

### 1. Data Preparation

Project data organized by module:

```bash
# MFI data structure (for repair and reconstruction training)
MFI/dataset/
├── defect_images/          # Original defective images
├── repaired_images/        # MFI-repaired images
└── unlabeled/              # Unlabeled data (for pseudo-label generation)
    ├── defect_images/
    └── repaired_images/

# DVPS data structure (for defect segmentation training)
DVPS/dataset/
├── train/
│   ├── defect_images/      # Original defective images
│   ├── inpainted_images/   # MFI-repaired images
│   └── masks/              # Defect ground truth masks
└── val/
    ├── defect_images/
    ├── inpainted_images/
    └── masks/

# YOLO data structure (for OW-DLN training)
yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. Quick Test (Using Pre-trained Model)

```bash
# Download pre-trained model

# Run inference
python predict.py \
    --model weights/owdln_best.pt \
    --source data/test/images \
    --conf 0.25 \
    --save-dir runs/detect/test
```

---

## Complete Training Pipeline

### Step 1: MFI Repair and Reconstruction Training

#### 1.1 (Optional) SD-Inpainting Preprocessing

If defect-free images are not available, use SD-Inpainting first:

```bash
cd MFI/SD-Inpainting

# Configure Stable Diffusion model
# Edit config file to specify model path and input data

# Run inpainting
python inpaint_images.py \
    --input_dir ../../data/images/train \
    --mask_dir ../../data/masks/train \
    --output_dir ../../data/inpainted/train \
    --model_path models/stable-diffusion-inpainting
```

#### 1.2 MFI Training

```bash
cd MFI

# 1. Train VQ-VAE
python train_MFI.py \
    --data_dir ./data/images/train \
    --epochs 100 \
    --batch_size 16 \
    --lr 3e-4


# 2. Perform defect repair and reconstruction
python rebuild.py \
    --vqvae_path checkpoint/vqvae_best.pt \
    --pixelsnail_path checkpoint/pixelsnail_best.pt \
    --input_dir ../data/images/train \
    --output_dir ../data/mfi_repaired/train
```

**Expected Output**:
- `vqvae_best.pt`: Trained VQ-VAE model

- `mfi_repaired/`: MFI-repaired defect-free images

### Step 2: DVPS Training

DVPS uses original defective images, MFI-repaired images, and ground truth masks for training:

```bash
cd DVPS

# Prepare DVPS training data
python split_dataset.py \
    --defect_images ../data/images/train \
    --inpainted_images ../data/mfi_repaired/train \
    --masks ../data/masks/train \
    --output_dir dataset \
    --train_ratio 0.8

# Train DVPS
python train_dvps.py \
    --data_dir dataset \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir checkpoints
```

**Training Monitoring**:

```bash
# View training process with TensorBoard
tensorboard --logdir DVPS/runs
```

**Expected Output**:
- `checkpoints/dvps_best.pt`: Trained DVPS model
- Training logs and visualization results

### Step 3: Pseudo-label Generation

#### 3.1 DVPS Inference to Generate Defect Masks

```bash
cd DVPS

# Use DVPS to generate defect masks
python predict.py \
    --model checkpoints/dvps_best.pt \
    --defect_images ../data/images/unlabeled \
    --inpainted_images ../data/mfi_repaired/unlabeled \
    --output_dir predictions/masks
```

#### 3.2 Region Merging to Generate Detection Boxes

```bash
# Generate detection box labels from masks
python create_labels_from_masks.py \
    --masks_dir DVPS/predictions/masks \
    --output_dir data/pseudo_labels/dvps \
    --min_area 100 \
    --format yolo
```

#### 3.3 OW-DLN Stage 1 Training (Known Classes Only)

```bash
# Train OW-DLN using known class labels
python train.py \
    --model ultralytics/cfg/models/11/OW-DLN.yaml \
    --data configs/data_known_only.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 512 \
    --name owdln_stage1
```

#### 3.4 Label Merging and Filtering

```bash
# Inference on unlabeled data using OW-DLN Stage1
python predict.py \
    --model runs/detect/owdln_stage1/weights/best.pt \
    --source data/images/unlabeled \
    --save-txt \
    --conf 0.25

# Merge DVPS and OW-DLN results to generate final pseudo-labels
python merge_pseudo_labels.py \
    --dvps_labels data/pseudo_labels/dvps \
    --owdln_labels runs/detect/predict/labels \
    --output_dir data/pseudo_labels/merged \
    --confidence_threshold 0.5 \
    --iou_threshold 0.5
```

**Label Filtering Strategy**:
- Keep regions detected by both DVPS and OW-DLN (high confidence)
- Regions detected by DVPS but not OW-DLN are marked as unknown class
- Filter out boxes with small area or high overlap

### Step 4: OW-DLN Two-stage Training

#### 4.1 Prepare Final Training Data

```bash
# Merge known class labels and pseudo-labels
python prepare_final_dataset.py \
    --known_labels data/labels/train \
    --pseudo_labels data/pseudo_labels/merged \
    --output_dir data/labels_final \
    --unknown_class_id 5  # Unknown class ID
```

#### 4.2 OW-DLN Stage 2 Training

```bash
# Train using complete data (known classes + pseudo-labels)
python train.py \
    --model ultralytics/cfg/models/11/OW-DLN.yaml \
    --data configs/data_with_unknown.yaml \
    --epochs 150 \
    --batch 16 \
    --imgsz 512 \
    --name owdln_stage2 \
    --pretrained runs/detect/owdln_stage1/weights/best.pt
```

**Training Config (data_with_unknown.yaml)**:

```yaml
# Dataset configuration
path: data
train: images/train
val: images/val

**Expected Output**:
- `runs/detect/owdln_stage2/weights/best.pt`: Final model
- Training curves and evaluation metrics

---

## Model Inference

### Single Image Inference

```bash
# Basic inference
python predict.py \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --source test_image.jpg \
    --conf 0.25 \
    --save

# Advanced options
python predict.py \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --source test_image.jpg \
    --conf 0.25 \
    --iou 0.5 \
    --save \
    --save-txt \
    --save-conf \
    --line-thickness 2
```

### Batch Inference

```bash
# Inference on entire folder
python predict.py \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --source data/test/images \
    --conf 0.25 \
    --save-dir runs/detect/test_results

# Inference on video
python predict.py \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --source video.mp4 \
    --conf 0.25 \
    --save
```


## Performance Evaluation

### Standard YOLO Metrics

```bash
# Evaluate OW-DLN model
python train.py \
    --mode val \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --data configs/data_with_unknown.yaml \
    --batch 16 \
    --imgsz 512
```

**Output Metrics**:
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Detailed metrics for each class
- Confusion matrix

### Open Set Detection Specific Metrics

Using our implemented open set evaluation tool:

```bash
# Evaluate open world performance
python evaluate_openset.py \
    --model runs/detect/owdln_stage2/weights/best.pt \
    --data configs/data_with_unknown.yaml \
    --known-classes 0 1 2 3 4 \
    --unknown-id 5 \
    --conf 0.001 \
    --iou 0.6 \
    --save openset_evaluation.txt
```

**Open Set Metrics**:

| Metric | Meaning | Target Value |
|--------|---------|--------------|
| **U-Recall** | Unknown class recall: proportion of real unknown objects correctly identified as unknown | Higher is better (0-1) |
| **WI** | Wilderness Impact: proportion of known classes misclassified as unknown | Lower is better (0-1) |
| **A-OSE** | Absolute Open Set Error: comprehensive error rate | Lower is better (0-∞) |

**Ideal Model Characteristics**:
- High U-Recall (>0.8): Can effectively discover unknown defects
- Low WI (<0.1): Does not overly affect known class detection
- Low A-OSE (<0.2): Excellent overall open set performance


For detailed usage, please refer to: [OPENSET_EVALUATION_GUIDE.md](OPENSET_EVALUATION_GUIDE.md)

---

## Project Structure

```
OW-DLN/
├── README.md                          # This document
├── README_EN.md                       # English documentation
├── OPENSET_EVALUATION_GUIDE.md        # Open set evaluation guide
├── requirements.txt                   # Dependency list
├── setup.py                           # Installation script
│
├── MFI/                               # Mask-Free Inpainting
│   ├── SD-Inpainting/                 # Stable Diffusion Inpainting (optional)
│   ├── train_vqvae.py                 # VQ-VAE training script
│   ├── train_pixelsnail.py            # PixelSNAIL training script
│   ├── rebuild.py                     # Defect repair and reconstruction
│   ├── vqvae.py                       # VQ-VAE model definition
│   └── pixelsnail.py                  # PixelSNAIL model definition
│
├── DVPS/                              # Defect Variation Prediction Segmentation
│   ├── dvps.py                        # DVPS U-Net model definition
│   ├── train_dvps.py                  # DVPS training script
│   ├── predict.py                     # DVPS inference script
│   ├── dataset.py                     # Dataset loader
│   ├── split_dataset.py               # Dataset splitting
│   └── checkpoints/                   # Model weights
│
├── ultralytics/                       # OW-DLN detection framework
│   ├── cfg/
│   │   └── models/11/
│   │       └── OW-DLN.yaml            # OW-DLN model configuration
│   ├── nn/
│   │   ├── DAAM.py                    # Dual Attention Aggregation Module
│   │   ├── ECA.py                     # Efficient Channel Attention
│   │   └── modules.py                 # Other custom modules
│   ├── models/
│   │   └── yolo/detect/
│   │       └── train.py               # Training logic
│   └── utils/
│       ├── metrics.py                 # Evaluation metrics
│       └── openset_metrics.py         # Open set specific metrics
│
├── configs/                           # Configuration files
│   ├── data_known_only.yaml           # Known classes only data config
│   └── data_with_unknown.yaml         # Config with unknown classes
│
├── scripts/                           # Utility scripts
│   ├── merge_pseudo_labels.py         # Pseudo-label merging
│   ├── create_labels_from_masks.py    # Mask to label conversion
│   └── prepare_final_dataset.py       # Dataset preparation
│
├── train.py                           # OW-DLN training entry
├── predict.py                         # Inference entry
├── evaluate_openset.py                # Open set evaluation entry
└── test_openset_metrics.py            # Metrics unit test
```

---

## Dataset Format

### Image Data

```
images/
├── train/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val/
    └── ...
```

### YOLO Format Labels

```
labels/
├── train/
│   ├── img_001.txt
│   │   # Format: class x_center y_center width height
│   │   # 0 0.5 0.5 0.1 0.1  (normalized coordinates 0-1)
│   │   # class_id x_center y_center width height
│   └── ...
└── val/
    └── ...
```

### Mask Data (for DVPS)

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   │   # YOLO format: class x_center y_center width height
    │   │   # Normalized coordinates (0-1)
    │   │   # Example: 0 0.5 0.5 0.1 0.1
    │   └── ...
    └── val/
        └── ...
```

---


## Changelog

### v1.0.0 (2025-01-08)
- ✨ Initial version release
- ✅ Implemented complete MFI-DVPS-OW-DLN pipeline
- ✅ Added open set evaluation metrics (U-Recall, WI, A-OSE)
- ✅ Provided pre-trained models and complete documentation

---


---

## License

This project is open-sourced under the AGPL-3.0 license. See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project is based on the following excellent works:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [VQ-VAE-2](https://arxiv.org/abs/1906.00446)
- [PixelSNAIL](https://arxiv.org/abs/1712.09763)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

---


<div align="center">

**If this project helps you, please give us a ⭐️!**

Made with ❤️ for Open World Detection

</div>
