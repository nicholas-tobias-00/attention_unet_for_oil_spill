# Data Structure and Preparation Guide

This document provides detailed information about the expected data structure for the Oil Spill Detection system.

## Overview

The system expects SAR (Synthetic Aperture Radar) satellite images organized in a specific directory structure with corresponding segmentation masks.

## Directory Structure

```
data/
├── images/
│   ├── oil/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   ├── lookalike/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── no_oil/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── masks/
    ├── oil/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    ├── lookalike/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── no_oil/
        ├── image_001.png
        ├── image_002.png
        └── ...
```

## Image Categories

### 1. Oil
- **Description**: SAR images containing actual oil spills
- **Characteristics**: Dark patches on the ocean surface due to dampening of surface waves
- **Mask**: Binary mask highlighting the oil spill regions (white=255, background=0)

### 2. Lookalike
- **Description**: SAR images with features that resemble oil spills but are not
- **Examples**: 
  - Algae blooms
  - Low wind areas
  - Rain cells
  - Biogenic slicks
- **Mask**: Binary mask highlighting the lookalike regions

### 3. No Oil
- **Description**: Clean ocean SAR images without oil spills or lookalikes
- **Mask**: Typically all-black masks (all zeros) or empty regions

## Image Specifications

### Input Images

- **Format**: PNG, JPG, JPEG, TIF, or TIFF
- **Channels**: 
  - RGB (3 channels) - default
  - Grayscale (1 channel) - for SAR mode
- **Size**: Any size (will be resized to 256x256 by default)
- **Bit Depth**: 8-bit or 16-bit
- **Naming**: Consistent naming convention (e.g., `image_001.png`, `sar_20240101_001.tif`)

### Mask Images

- **Format**: PNG, JPG, JPEG, TIF, or TIFF (PNG recommended)
- **Channels**: Grayscale (1 channel)
- **Pixel Values**: 
  - 0 (black) = Background/No oil
  - 255 (white) = Oil spill/Lookalike region
- **Size**: Should match input image size (or will be resized automatically)
- **Naming**: Should match corresponding input image name exactly

## Data Preparation Steps

### Step 1: Organize Raw Data

1. Create the main data directory structure:
```bash
mkdir -p data/images/oil data/images/lookalike data/images/no_oil
mkdir -p data/masks/oil data/masks/lookalike data/masks/no_oil
```

2. Sort your SAR images into the appropriate category folders

### Step 2: Create Segmentation Masks

For each image, create a corresponding binary segmentation mask:

```python
import numpy as np
from PIL import Image

# Example: Create a mask
mask = np.zeros((height, width), dtype=np.uint8)
# Mark oil regions as 255 (white)
mask[y1:y2, x1:x2] = 255
Image.fromarray(mask).save('masks/oil/image_001.png')
```

### Step 3: Verify Data Integrity

Use the following script to verify your data:

```python
import os
from PIL import Image

def verify_dataset(data_root):
    categories = ['oil', 'lookalike', 'no_oil']
    
    for category in categories:
        img_dir = os.path.join(data_root, 'images', category)
        mask_dir = os.path.join(data_root, 'masks', category)
        
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        
        print(f"\n{category.upper()}:")
        print(f"  Images: {len(img_files)}")
        print(f"  Masks: {len(mask_files)}")
        
        # Check for missing masks
        missing = img_files - mask_files
        if missing:
            print(f"  Missing masks for: {missing}")
        
        # Check image-mask pairs
        for img_file in list(img_files)[:3]:  # Check first 3
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            print(f"  {img_file}: Image {img.size}, Mask {mask.size}")

# Run verification
verify_dataset('./data')
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

A minimal working example with 10 images per category:

```
data/
├── images/
│   ├── oil/              # 10 images with oil spills
│   ├── lookalike/        # 10 images with lookalikes
│   └── no_oil/           # 10 clean images
└── masks/
    ├── oil/              # 10 corresponding masks
    ├── lookalike/        # 10 corresponding masks
    └── no_oil/           # 10 corresponding masks (can be all black)
```

## Common Issues and Solutions

### Issue 1: Missing Masks
- **Solution**: Ensure each image has a corresponding mask with the same filename

### Issue 2: Size Mismatch
- **Solution**: The system automatically resizes images, but verify source data isn't corrupted

### Issue 3: Incorrect Mask Values
- **Solution**: Masks should be binary (0 and 255), not normalized (0 and 1)

### Issue 4: Empty Directories
- **Solution**: Ensure each category directory contains at least a few samples

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

- [ ] All images load correctly
- [ ] Masks are binary (0 and 255)
- [ ] Image-mask pairs match
- [ ] At least 50+ samples per category
- [ ] No corrupted files
- [ ] Consistent naming convention

## References

- ESA Sentinel-1 SAR Data: https://sentinel.esa.int/
- Oil Spill Detection Tutorial: Various online resources
- SAR Image Processing: IEEE GRSS tutorials
