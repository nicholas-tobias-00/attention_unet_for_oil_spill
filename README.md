# Oil Spill Detection using Attention U-Net

A deep learning system for detecting oil spills in SAR (Synthetic Aperture Radar) satellite images using Attention U-Net architecture. The system supports semantic segmentation of images into three categories: Oil, Lookalike, and No oil.

**Recommended Model:** `attention_unet_oil_spill_improved.pth` (Weighted BCE) - Achieves better F1-Score (65.6%) and IoU (48.8%) compared to standard BCE.

## Features

- **Attention U-Net Architecture**: Advanced segmentation model with attention gates for improved feature selection
- **Multi-Category Support**: Handles Oil, Lookalike, and No oil image categories
- **Comprehensive Metrics**: IoU, F1 score, Precision, and Recall evaluation
- **WandB Integration**: Experiment tracking and visualization with Weights & Biases
- **PyTorch Implementation**: Efficient training and inference using PyTorch
- **Jupyter Notebooks**: Interactive notebooks for training and evaluation

## Project Structure

```
attention-u-net_for_oil-spill/
├── models/
│   ├── attention_unet_oil_spill.pth        # Pre-trained model (BCE loss)
│   └── attention_unet_oil_spill_improved.pth  # Pre-trained model (Weighted BCE - Recommended)
├── notebooks/
│   ├── oil_spill_detection.ipynb           # Oil spill detection training & evaluation
│   └── deforestation_reproduction.ipynb    # Deforestation paper reproduction
├── dataset/                                # Oil spill SAR dataset
│   ├── 01_Train_Val_Oil_Spill_images/
│   │   └── Oil/                            # Oil spill TIF images
│   ├── 01_Train_Val_Oil_Spill_mask/
│   │   └── Mask_oil/                       # Binary masks for oil
│   ├── 01_Train_Val_Lookalike_images/
│   │   └── Lookalike/                      # Lookalike TIF images
│   ├── 01_Train_Val_Lookalike_mask/
│   │   └── Mask_lookalike/                 # Binary masks for lookalike
│   ├── 01_Train_Val_No_Oil_images/
│   │   └── No_oil/                         # Clean ocean TIF images
│   ├── 01_Train_Val_No_Oil_mask/
│   │   └── Mask_no_oil/                    # Binary masks for no oil
│   └── 02_Test_images_and_ground_truth/
│       ├── Images/                         # Test images
│       │   ├── Oil/
│       │   ├── Lookalike/
│       │   └── No oil/
│       └── Mask/                           # Test masks
│           ├── Oil/
│           ├── Lookalike/
│           └── No oil/
├── deforestation-adaptation/               # Deforestation project
│   ├── models/
│   │   └── attention_unet_best.pth         # Deforestation model
│   └── dataset/
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── DATA_GUIDE.md                           # Data preprocessing with SNAP
```

## Installation

### Prerequisites

- **Python 3.11+**
- **CUDA-capable GPU** (recommended for training)
  - Minimum 8GB VRAM recommended
  - CPU training is possible but significantly slower
- **Weights & Biases account** (optional, for experiment tracking)
  - Sign up at [wandb.ai](https://wandb.ai)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/nicholas-tobias-00/attention-u-net_for_oil-spill.git
cd attention-u-net_for_oil-spill
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure Weights & Biases (optional):**
```bash
wandb login
# Enter your API key when prompted
# Or set environment variable: export WANDB_API_KEY=your_api_key
```

## Data Preparation

### Dataset Source

This project uses **Sentinel-1 SAR** satellite imagery for oil spill detection. The dataset should contain preprocessed SAR images in GeoTIFF format.

**📖 For complete preprocessing instructions** (downloading Sentinel-1 data and processing with SNAP), **refer to [DATA_GUIDE.md](DATA_GUIDE.md)**

### Directory Structure

Organize your SAR satellite images in the following structure:

```
dataset/
├── 01_Train_Val_Oil_Spill_images/
│   └── Oil/                            # TIF images with oil spills
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_Oil_Spill_mask/
│   └── Mask_oil/                       # Binary masks for oil
│       ├── 00000.tif
│       ├── 00001.tif
│       └── ...
├── 01_Train_Val_Lookalike_images/
│   └── Lookalike/                      # Lookalike images
│       └── *.tif
├── 01_Train_Val_Lookalike_mask/
│   └── Mask_lookalike/                 # Lookalike masks
│       └── *.tif
├── 01_Train_Val_No_Oil_images/
│   └── No_oil/                         # Clean ocean images
│       └── *.tif
├── 01_Train_Val_No_Oil_mask/
│   └── Mask_no_oil/                    # No oil masks
│       └── *.tif
└── 02_Test_images_and_ground_truth/
    ├── Images/{Oil, Lookalike, No oil}/
    └── Mask/{Oil, Lookalike, No oil}/
```

### Image Format

- **Input Images**: Sentinel-1 SAR images in **GeoTIFF (.tif)** format
- **Masks**: Binary segmentation masks in **GeoTIFF (.tif)** format
  - White (255) = Oil spill/lookalike region
  - Black (0) = Background/Ocean
- **Naming**: Zero-padded numeric format (e.g., `00000.tif`, `00001.tif`)
- **Image Size**: Automatically resized to 256×256 during training

### Supported Categories

1. **Oil**: SAR images containing actual oil spills (dark patches)
2. **Lookalike**: SAR images with features resembling oil spills:
   - Algae blooms
   - Low wind areas
   - Rain cells
   - Biogenic slicks
3. **No Oil**: Clean ocean SAR images without oil or lookalikes

## Model Architecture

The Attention U-Net architecture consists of:

- **Encoder**: 4 levels with convolutional blocks and max pooling
- **Bottleneck**: Deep feature extraction layer
- **Decoder**: 4 levels with attention gates and upsampling
- **Attention Gates**: Focus on relevant spatial regions during upsampling
- **Output**: Single-channel binary segmentation mask

### Key Components

- **ConvBlock**: Double convolution with batch normalization and ReLU
- **AttentionGate**: Attention mechanism for skip connections
- **UpConv**: Upsampling with convolution

## Usage

### 1. Oil Spill Detection

Open and run `notebooks/oil_spill_detection.ipynb` for:
- Training the Attention U-Net model on oil spill data
- Evaluating model performance with multiple metrics
- Visualizing predictions and segmentation results
- Comparing different loss functions (BCE vs Weighted BCE)

### 2. Deforestation Detection (Paper Reproduction)

Open and run `notebooks/deforestation_reproduction.ipynb` for:
- Reproducing results from the deforestation detection paper
- Training on 4-band satellite imagery
- Comparing with baseline results from the paper
- Automatic model loading if pre-trained weights exist

### Metrics

The system computes the following segmentation metrics:

- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives

## Pre-Trained Models

### Download from HuggingFace

Pre-trained models are available at: **[Bobsicle/attention_u_net_oil_spill](https://huggingface.co/Bobsicle/attention_u_net_oil_spill)**

**Two models available:**
1. `attention_unet_oil_spill.pth` - Trained with standard BCE loss
2. `attention_unet_oil_spill_improved.pth` - **Trained with Weighted BCE (RECOMMENDED)**

### Download Instructions

**Option 1: Manual Download**
1. Visit [Bobsicle/attention_u_net_oil_spill](https://huggingface.co/Bobsicle/attention_u_net_oil_spill)
2. Download `attention_unet_oil_spill_improved.pth`
3. Place it in the `models/` directory:
   ```bash
   mkdir -p models
   # Move downloaded file to models/attention_unet_oil_spill_improved.pth
   ```

**Option 2: Using Python/HuggingFace CLI**
```python
from huggingface_hub import hf_hub_download

# Download the recommended model
model_path = hf_hub_download(
    repo_id="Bobsicle/attention_u_net_oil_spill",
    filename="attention_unet_oil_spill_improved.pth",
    local_dir="./models"
)
print(f"Model downloaded to: {model_path}")
```

### Model Comparison

| Model | Loss Function | F1-Score | IoU | Recall | Best For |
|-------|--------------|----------|-----|--------|----------|
| `attention_unet_oil_spill.pth` | BCE | 43.7% | 28.0% | 29.1% | High precision |
| `attention_unet_oil_spill_improved.pth` | **Weighted BCE** | **65.6%** | **48.8%** | **58.8%** | **Overall performance** |

**Recommendation:** Use `attention_unet_oil_spill_improved.pth` for better F1-Score and IoU, especially when detecting small oil spill regions.

### Local Model Paths

- **Oil spill models** save to: `./models/attention_unet_oil_spill_improved.pth`
- **Deforestation models** save to: `./deforestation-adaptation/models/attention_unet_best.pth`

## Performance

### Expected Results

#### 1. Deforestation Detection Attention U-Net

| Baseline vs Reproduced | Accuracy | Precision | Recall | F1 |
|------------------------|----------|-----------|--------|-----|
| Baseline (Paper) | 97.5% | 97.6% | 97.5% | 97.5% |
| Reproduced | 97% | 98.1% | 95.9% | 97% |

#### 2. Oil Spill Detection Attention U-Net

| Metrics | BCE | Weighted BCE (Recommended) |
|---------|-----|---------------------------|
| Accuracy | 97.9% | **98.3%** |
| Precision | 87.6% | 74.1% |
| Recall | 29.1% | **58.8%** |
| F1-Score | 43.7% | **65.6%** |
| IoU | 28.0% | **48.8%** |

**Key Insights:**
- **Weighted BCE model** (`attention_unet_oil_spill_improved.pth`) achieves:
  - 50% improvement in F1-Score (43.7% → 65.6%)
  - 74% improvement in IoU (28.0% → 48.8%)
  - Better recall (58.8% vs 29.1%) for detecting oil spills
- Trade-off: Slightly lower precision (74.1% vs 87.6%)
- **Recommended** for operational use due to superior overall performance

## References

1. John, D. & Zhang, C. (2022) ‘An attention-based U-Net for detecting deforestation within satellite sensor imagery’, International Journal of Applied Earth Observations and Geoinformation, 107, p. 102685.

2. Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C. & Monterrubio-Martínez, E. (2024) ‘Marine oil spill detection and segmentation in SAR data with two steps Deep Learning framework’, Marine Pollution Bulletin, 204, p. 116549.

3. HarisIqbal88. (n.d.). PlotNeuralNet [Computer software]. GitHub. https://github.com/HarisIqbal88/PlotNeuralNet/tree/master

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Weights & Biases for experiment tracking tools
- SAR satellite data providers