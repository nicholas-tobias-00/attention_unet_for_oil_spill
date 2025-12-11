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
  - Other natural phenomena causing dark patches
- **Image folder**: `01_Train_Val_Lookalike_images/Lookalike/`
- **Mask folder**: `01_Train_Val_Lookalike_mask/Mask_lookalike/`
- **Mask**: Binary mask highlighting the lookalike regions

### 3. No Oil
- **Description**: Clean ocean SAR images without oil spills or lookalikes
- **Image folder**: `01_Train_Val_No_Oil_images/No_oil/`
- **Mask folder**: `01_Train_Val_No_Oil_mask/Mask_no_oil/`
- **Mask**: Typically all-black masks (all zeros) or empty regions

## Data Preparation Steps (For Additional Sentinel-1 GRD Demonstration / Testing Dataset)

This section outlines the steps to download and preprocess additional Sentinel-1 GRD data for oil spill detection. Note that this **outputs GeoTIFF files without labels; manual annotation is required for training data.**

### Step 1: Download Dataset

1. Register on [Copernicus Open Access Hub](https://browser.dataspace.copernicus.eu/)
2. Pin or draw desired area for detection
3. Perform custom search with the following metrics:
    - Data Source   : Sentinel-1 (C-SAR and Level-1 GRD)
    - Additional Filters : 
        - Operational Mode : IW
        - Polarisation Channels : VV + VH
4. Configure Date
5. Search and Download Dataset (~1.5 GB for each block)

### Step 2: Additional Transformation with SeNtinel Application Platform (SNAP Toolbox)

1. Download SNAP on [European Space Agency](https://step.esa.int/main/download/snap-download/)
2. Open SNAP and load the downloaded Sentinel-1 GRD data
3. Apply the following preprocessing steps:
    - Apply Orbit File
    - Thermal Noise Removal
    - Radiometric Calibration
    - Speckle Filtering (Lee 5×5)
    - Range-Doppler Terrain Correction
    - Dual Polarization Sigma0 dB
    - Subset/Resample to 256×256 pixel tiles
4. Export the processed tiles as GeoTIFF format

**Ethical Considerations:**

| Aspect | Consideration |
|--------|---------------|
| **Data Licensing** | Copernicus data free and open under CC BY 4.0 |
| **Attribution** | "Contains modified Copernicus Sentinel data [2024]" |
| **Privacy** | No personal data in satellite imagery |


## References

- ESA Sentinel-1 SAR Data: https://browser.dataspace.copernicus.eu/
- Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C. & Monterrubio-Martínez, E. (2024) ‘Marine oil spill detection and segmentation in SAR data with two steps Deep Learning framework’, Marine Pollution Bulletin, 204, p. 116549
