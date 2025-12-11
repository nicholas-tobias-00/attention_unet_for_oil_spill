# Data Structure and Preparation Guide

This document provides detailed information about the expected data structure for the Oil Spill Detection system.

## Overview

The system expects SAR (Synthetic Aperture Radar) satellite images organized in a specific directory structure with corresponding segmentation masks. All images should be in **TIF format**.

Complete dataset for oil spills can be downloaded from the following directories (total 127 GB):
1. https://zenodo.org/records/8346860
2. https://zenodo.org/records/8253899
3. https://zenodo.org/records/8346860


## Directory Structure

### Training/Validation Data

Downloaded dataset should be extracted with the following directory:

```
dataset/
├── 01_Train_Val_Oil_Spill_images/
│   └── Oil/
│       ├── 00000.tif
│       ├── 00001.tif
│       ├── 00002.tif
│       └── ...
├── 01_Train_Val_Oil_Spill_mask/
│   └── Mask_oil/
│       ├── 00000.tif
│       ├── 00001.tif
│       ├── 00002.tif
│       └── ...
├── 01_Train_Val_Lookalike_images/
│   └── Lookalike/
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_Lookalike_mask/
│   └── Mask_lookalike/
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_No_Oil_images/
│   └── No_oil/
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_No_Oil_mask/
│   └── Mask_no_oil/
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
└── 02_Test_images_and_ground_truth/
    ├── Images/
    │   ├── Lookalike/
    │   │   └── *.tif
    │   ├── No oil/
    │   │   └── *.tif
    │   └── Oil/
    │       └── *.tif
    └── Mask/
        ├── Lookalike/
        │   └── *.tif
        ├── No oil/
        │   └── *.tif
        └── Oil/
            └── *.tif
```

## Image Categories

### 1. Oil Spill
- **Description**: SAR images containing actual oil spills
- **Characteristics**: Dark patches on the ocean surface due to dampening of surface waves
- **Image folder**: `01_Train_Val_Oil_Spill_images/Oil/`
- **Mask folder**: `01_Train_Val_Oil_Spill_mask/Mask_oil/`
- **Mask**: Binary mask highlighting the oil spill regions (white=255, background=0)

### 2. Lookalike
- **Description**: SAR images with features that resemble oil spills but are not
- **Examples**: 
  - Algae blooms
  - Low wind areas
  - Rain cells
  - Biogenic slicks
- **Image folder**: `01_Train_Val_Lookalike_images/Lookalike/`
- **Mask folder**: `01_Train_Val_Lookalike_mask/Mask_lookalike/`
- **Mask**: Binary mask highlighting the lookalike regions

### 3. No Oil
- **Description**: Clean ocean SAR images without oil spills or lookalikes
- **Image folder**: `01_Train_Val_No_Oil_images/No_oil/`
- **Mask folder**: `01_Train_Val_No_Oil_mask/Mask_no_oil/`
- **Mask**: Typically all-black masks (all zeros) or empty regions

## Image Specifications

### Input Images

- **Format**: TIF (GeoTIFF)
- **Channels**: 
  - Grayscale (1 channel) - SAR backscatter intensity
- **Size**: Any size (will be resized to 256x256 by default)
- **Bit Depth**: 8-bit or 16-bit
- **Naming**: Zero-padded numeric format (e.g., `00000.tif`, `00001.tif`, `00002.tif`)

### Mask Images

- **Format**: TIF
- **Channels**: Grayscale (1 channel)
- **Pixel Values**: 
  - 0 (black) = Background/No oil
  - 255 (white) = Oil spill/Lookalike region
- **Size**: Should match input image size (or will be resized automatically)
- **Naming**: Should match corresponding input image name exactly (e.g., `00000.tif`)

## Data Preparation Steps

### Step 1: Download Dataset

Download the oil spill SAR dataset and extract it into the `dataset/` folder.

### Step 2: Verify Directory Structure

Ensure your dataset follows this structure:
```bash
dataset/
├── 01_Train_Val_Oil_Spill_images/Oil/
├── 01_Train_Val_Oil_Spill_mask/Mask_oil/
├── 01_Train_Val_Lookalike_images/Lookalike/
├── 01_Train_Val_Lookalike_mask/Mask_lookalike/
├── 01_Train_Val_No_Oil_images/No_oil/
├── 01_Train_Val_No_Oil_mask/Mask_no_oil/
└── 02_Test_images_and_ground_truth/
    ├── Images/{Lookalike,No oil,Oil}/
    └── Mask/{Lookalike,No oil,Oil}/
```

### Step 3: Verify Data Integrity

Use the following script to verify your data:

```python
import os
from PIL import Image

def verify_dataset(data_root):
    categories = [
        ('Oil', '01_Train_Val_Oil_Spill_images/Oil', '01_Train_Val_Oil_Spill_mask/Mask_oil'),
        ('Lookalike', '01_Train_Val_Lookalike_images/Lookalike', '01_Train_Val_Lookalike_mask/Mask_lookalike'),
        ('No Oil', '01_Train_Val_No_Oil_images/No_oil', '01_Train_Val_No_Oil_mask/Mask_no_oil'),
    ]
    
    for name, img_rel, mask_rel in categories:
        img_dir = os.path.join(data_root, img_rel)
        mask_dir = os.path.join(data_root, mask_rel)
        
        if not os.path.exists(img_dir):
            print(f"  {name}: Image directory NOT FOUND")
            continue
            
        img_files = set([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        mask_files = set([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
        print(f"\n{name.upper()}:")
        print(f"  Images: {len(img_files)}")
        print(f"  Masks: {len(mask_files)}")
        
        # Check for missing masks
        missing = img_files - mask_files
        if missing:
            print(f"  ⚠ Missing masks for: {list(missing)[:5]}...")
        
        # Check first few image-mask pairs
        for img_file in sorted(list(img_files))[:3]:
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            img = Image.open(img_path)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                print(f"  {img_file}: Image {img.size}, Mask {mask.size}")
            else:
                print(f"  {img_file}: Image {img.size}, Mask MISSING")

# Run verification
verify_dataset('./dataset')
```

## Data Augmentation (Optional)

The dataset loader supports standard PyTorch transformations. You can add augmentation:

```python
import torchvision.transforms as transforms

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

## SAR Image Characteristics

### Understanding SAR Imagery

- **Single-band**: Most SAR images are single-channel (grayscale)
- **Backscatter**: Intensity represents radar backscatter from ocean surface
- **Oil Signature**: Appears as dark patches (low backscatter)
- **Noise**: Speckle noise is common in SAR imagery

### Preprocessing Recommendations

1. **Speckle Filtering**: Apply Lee or Frost filter for noise reduction
2. **Normalization**: Normalize pixel values to [0, 1] or standardize
3. **Contrast Enhancement**: Histogram equalization for better visualization
4. **Despeckling**: Consider multi-temporal averaging if available

## Example Dataset Structure

A minimal working example:

```
dataset/
├── 01_Train_Val_Oil_Spill_images/
│   └── Oil/                      # TIF images with oil spills
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_Oil_Spill_mask/
│   └── Mask_oil/                 # Corresponding binary masks
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_Lookalike_images/
│   └── Lookalike/                # TIF images with lookalikes
├── 01_Train_Val_Lookalike_mask/
│   └── Mask_lookalike/           # Corresponding binary masks
├── 01_Train_Val_No_Oil_images/
│   └── No_oil/                   # Clean TIF images
├── 01_Train_Val_No_Oil_mask/
│   └── Mask_no_oil/              # Corresponding masks (all black)
└── 02_Test_images_and_ground_truth/
    ├── Images/
    │   ├── Lookalike/
    │   ├── No oil/
    │   └── Oil/
    └── Mask/
        ├── Lookalike/
        ├── No oil/
        └── Oil/
```

## Common Issues and Solutions

### Issue 1: Missing Masks
- **Solution**: Ensure each image has a corresponding mask with the same filename (e.g., `00000.tif` image → `00000.tif` mask)

### Issue 2: Size Mismatch
- **Solution**: The system automatically resizes images, but verify source data isn't corrupted

### Issue 3: Incorrect Mask Values
- **Solution**: Masks should be binary (0 and 255), not normalized (0 and 1)

### Issue 4: Wrong File Format
- **Solution**: Ensure all files are in TIF format (`.tif` extension)

### Issue 5: Incorrect Folder Structure
- **Solution**: Follow the exact folder naming convention (e.g., `01_Train_Val_Oil_Spill_images/Oil/`)

## Dataset Statistics

For best results, aim for:

- **Minimum**: 100 images per category
- **Recommended**: 500+ images per category
- **Ideal**: 1000+ images per category
- **Balance**: Similar number of samples across categories

## Creating Synthetic Data

If you don't have enough real data, consider:

1. **Data Augmentation**: Rotate, flip, scale existing images
2. **Synthetic SAR Generation**: Use simulators for SAR image generation
3. **Transfer Learning**: Pre-train on similar datasets

## Validation

Before training, verify:

- [ ] All images are in TIF format (`.tif`)
- [ ] All images load correctly
- [ ] Masks are binary (0 and 255)
- [ ] Image-mask pairs match by filename
- [ ] Correct folder structure is followed
- [ ] No corrupted files
- [ ] Zero-padded numeric naming (00000.tif, 00001.tif, etc.)

## References

- ESA Sentinel-1 SAR Data: https://sentinel.esa.int/
- Oil Spill Detection Tutorial: Various online resources
- SAR Image Processing: IEEE GRSS tutorials
