# Oil Spill Detection using Attention U-Net

A deep learning system for detecting oil spills in SAR (Synthetic Aperture Radar) satellite images using Attention U-Net architecture. The system supports classification of images into three categories: Oil, Lookalike, and No oil.

## Features

- **Attention U-Net Architecture**: Advanced segmentation model with attention gates for improved feature selection
- **Multi-Category Support**: Handles Oil, Lookalike, and No oil image categories
- **Comprehensive Metrics**: IoU, F1 score, Precision, and Recall evaluation
- **WandB Integration**: Experiment tracking and visualization with Weights & Biases
- **PyTorch Implementation**: Efficient training and inference using PyTorch
- **Jupyter Notebooks**: Interactive notebooks for training and evaluation
- **Data Preprocessing Pipeline**: Automated data loading and preprocessing for SAR images

## Project Structure

```
attention-u-net_for_oil-spill-detection/
├── models/
│   └── attention_unet.py          # Attention U-Net model architecture
├── utils/
│   ├── dataset.py                 # Dataset loaders and data preprocessing
│   ├── metrics.py                 # Evaluation metrics (IoU, F1, etc.)
│   └── utils.py                   # Utility functions
├── configs/
│   └── config.py                  # Configuration parameters
├── notebooks/
│   ├── train.ipynb                # Training notebook
│   └── evaluation.ipynb           # Evaluation notebook
├── data/                          # Data directory (create this)
│   ├── images/
│   │   ├── oil/                   # Oil spill images
│   │   ├── lookalike/             # Lookalike images
│   │   └── no_oil/                # No oil images
│   └── masks/
│       ├── oil/                   # Segmentation masks for oil
│       ├── lookalike/             # Segmentation masks for lookalike
│       └── no_oil/                # Segmentation masks for no oil
├── checkpoints/                   # Model checkpoints (created during training)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nicholas-tobias-00/attention-u-net_for_oil-spill-detection.git
cd attention-u-net_for_oil-spill-detection
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
data/
├── images/
│   ├── oil/          # Images containing oil spills
│   ├── lookalike/    # Images with oil spill lookalikes
│   └── no_oil/       # Images without oil spills
└── masks/
    ├── oil/          # Binary masks for oil regions
    ├── lookalike/    # Binary masks for lookalike regions
    └── no_oil/       # Binary masks (typically all zeros)
```

### Image Format

- **Input Images**: SAR satellite images in PNG, JPG, JPEG, TIF, or TIFF format
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

### Configuration

Edit `configs/config.py` to customize:

- Model architecture (channels, base filters)
- Training hyperparameters (batch size, learning rate, epochs)
- Data settings (image size, categories, train/val split)
- WandB integration settings
- Checkpoint and logging options

### WandB Integration

To enable WandB experiment tracking:

1. Sign up at [wandb.ai](https://wandb.ai)
2. Login: `wandb login`
3. Configure in `configs/config.py`:
```python
WANDB_CONFIG = {
    'enabled': True,
    'project': 'oil-spill-detection',
    'entity': 'your-username',
}
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
- **Dice Coefficient**: Similar to F1, commonly used in medical imaging

## Model Checkpoints

### Saving Checkpoints

- **Best Model**: Automatically saved when validation metric improves
- **Periodic Checkpoints**: Saved every N epochs (configurable)
- **Location**: `checkpoints/` directory

### Loading Checkpoints

```python
from models.attention_unet import get_model
from utils.utils import load_checkpoint

model = get_model(in_channels=3, out_channels=1)
load_checkpoint('checkpoints/best_model.pth', model)
```

## Usage Examples

### Training from Scratch

```python
import torch
from models.attention_unet import get_model
from utils.dataset import get_data_loaders
from configs.config import *

# Create model
model = get_model(
    in_channels=MODEL_CONFIG['in_channels'],
    out_channels=MODEL_CONFIG['out_channels'],
    base_channels=MODEL_CONFIG['base_channels']
)

# Get data loaders
train_loader, val_loader = get_data_loaders(
    data_root=DATA_CONFIG['data_root'],
    batch_size=TRAIN_CONFIG['batch_size'],
    categories=DATA_CONFIG['categories']
)

# Train model (see notebooks for full training loop)
```

### Inference

```python
import torch
from PIL import Image
from models.attention_unet import get_model
from utils.utils import load_checkpoint

# Load model
model = get_model(in_channels=3, out_channels=1)
load_checkpoint('checkpoints/best_model.pth', model)
model.eval()

# Prepare image
image = Image.open('path/to/image.png')
# ... apply transformations ...

# Predict
with torch.no_grad():
    output = model(image)
    prediction = torch.sigmoid(output) > 0.5
```

## Performance

### Expected Results

With proper training on a good dataset, you should achieve:

- **IoU**: > 0.70
- **F1 Score**: > 0.80
- **Precision**: > 0.75
- **Recall**: > 0.75

*Note: Results depend heavily on dataset quality and size.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@misc{oil-spill-attention-unet,
  author = {Nicholas Tobias},
  title = {Oil Spill Detection using Attention U-Net},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/nicholas-tobias-00/attention-u-net_for_oil-spill-detection}
}
```

## References

- Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas"
- Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Weights & Biases for experiment tracking tools
- SAR satellite data providers