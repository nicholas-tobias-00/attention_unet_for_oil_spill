# Oil Spill using Attention U-Net

A deep learning system for detecting oil spills in SAR (Synthetic Aperture Radar) satellite images using Attention U-Net architecture. The system supports classification of images into three categories: Oil, Lookalike, and No oil.

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
│   └── attention_unet.pth                  # Pre-Trained PyTorch models
├── notebooks/
│   ├── train.ipynb                         # Training notebook
│   └── evaluation.ipynb                    # Evaluation notebook
├── dataset/                                # Data directory (create this)
│   ├── images/
│   │   ├── oil/                            # Oil spill images
│   │   ├── lookalike/                      # Lookalike images
│   │   └── no_oil/                         # No oil images
│   └── masks/
│       ├── oil/                            # Segmentation masks for oil
│       ├── lookalike/                      # Segmentation masks for lookalike
│       └── no_oil/                         # Segmentation masks for no oil
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── DATA_GUIDE.md                           # Data Processing Guide with SNAP
```

## Installation

### Prerequisites

- Python 3.11
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nicholas-tobias-00/attention-u-net_for_oil-spill.git
cd attention-u-net_for_oil-spill
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Directory Structure

Organize your SAR satellite images in the following structure:

```
dataset/
├── 01_Train_Val_Lookalike_images/
│   └── lookalike/                      # Images with oil spill lookalikes
|       ├── 000000.tif
|       └── ...
├── 01_Train_Val_Lookalike_masks/
│   └── Mask_lookalike/                 # Images with oil spill lookalikes
|       ├── 000000.tif
|       └── ...
├── 01_Train_Val_No_Oil_images
|   └── ...
├── 01_Train_Val_No_Oil_mask
|   └── ...
└── ...

### Image Format

- **Input Images**: Processed SAR satellite images in TIF format
- **Masks**: Binary segmentation masks where:
  - White (255) = Oil spill/lookalike region
  - Black (0) = Background
- **Image Size**: Images will be automatically resized to 256x256 (configurable)

### Supported Categories

1. **Oil**: Images containing actual oil spills
2. **Lookalike**: Images with features that resemble oil spills (e.g., algae blooms, low wind areas)
3. **No oil**: Clean ocean images without oil or lookalikes

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

## Training

### Using Jupyter Notebook

1. Configure training parameters in `configs/config.py`
2. Open and run `notebooks/train.ipynb`
3. Monitor training with WandB (if enabled)
```

## Evaluation

### Using Jupyter Notebook

1. Ensure you have a trained model checkpoint in `checkpoints/best_model.pth`
2. Open and run `notebooks/evaluation.ipynb`
3. View metrics and visualizations

### Metrics

The system computes the following segmentation metrics:

- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives

## Pre-Trained Models

- Reproduced deforestation models are configured to save automatically to ./deforestation-adaptation/models/attention_unet_best.pth.pth
- Oil spill models are configured to save automatically to ./models/attention_unet_oil_spill_improved.pth

## Performance

### Expected Results

1. Deforestation Detection Attention U-Net

| Baseline vs Reproduced | Accuracy | Precision | Recall | F1 |
|------------------------|----------|-----------|--------|-----|
| Baseline | 97.5% | 97.6% | 97.5% | 97.5% |
| Reproduced | 97% | 98.1% | 95.9% | 97% |

2. Oil Spill Attention U-Net

| Metrics | BCE | Weighted BCE |
|---------|-----|--------------|
| Accuracy | 97.9% | 98.3% |
| Precision | 87.6% | 74.1% |
| Recall | 29.1% | 58.8% |
| F1-Score | 43.7% | 65.6% |
| IoU | 28.0% | 48.8% |

## References

1. John, D. & Zhang, C. (2022) ‘An attention-based U-Net for detecting deforestation within satellite sensor imagery’, International Journal of Applied Earth Observations and Geoinformation, 107, p. 102685.

2. Trujillo-Acatitla, R., Tuxpan-Vargas, J., Ovando-Vázquez, C. & Monterrubio-Martínez, E. (2024) ‘Marine oil spill detection and segmentation in SAR data with two steps Deep Learning framework’, Marine Pollution Bulletin, 204, p. 116549.

3. HarisIqbal88. (n.d.). PlotNeuralNet [Computer software]. GitHub. https://github.com/HarisIqbal88/PlotNeuralNet/tree/master

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Weights & Biases for experiment tracking tools
- SAR satellite data providers