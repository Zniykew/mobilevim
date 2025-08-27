# MobileVim: Lightweight Multi-scale Feature Fusion for Intestinal Depth Estimation

## Overview
MobileVim is a lightweight deep learning model designed for accurate depth estimation in intestinal images. It integrates wavelet transform techniques with state-space models to achieve efficient multi-scale feature fusion while maintaining a compact architecture suitable for mobile and embedded applications.

## Features
- **Lightweight Architecture**: Three model variants (xx_small, x_small, small) with different computational complexities
- **Multi-scale Feature Fusion**: Utilizes pyramid depthwise convolution and wavelet transform for effective feature extraction
- **State-space Model Integration**: Incorporates Mamba-based modules for efficient sequence modeling
- **Attention Mechanisms**: Includes implicit surface decoding and deformable attention for precise depth prediction
- **Medical Image Adaptation**: Optimized for intestinal depth estimation tasks

## Requirements
- Python 3.10
- PyTorch 2.1.2
- torchvision 0.16.2
- torchaudio 2.1.2
- CUDA 11.8 compatible environment

## Environment Setup
```bash
conda create -n MobileVim python=3.10
conda activate MobileVim
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Model Architecture
MobileVim consists of the following key components:

1. **Patch Embedding**: Converts input images into feature patches
2. **Encoder Stages**: Multiple stages of MobileViM blocks with pyramid depthwise convolution and feed-forward networks
3. **Attention Modules**: MobileViM modules with global and local feature processing
4. **Decoder**: Pixel decoder with attention feature processing and implicit surface decoding
5. **Wavelet Transform Integration**: MBWTConv2d for efficient multi-scale feature representation

The model supports three configurations:
- **xx_small**: Ultra lightweight with 2 encoder stages
- **x_small**: Balanced lightweight with 3 encoder stages
- **small**: Higher capacity with 4 encoder stages

## Usage

### Training
To train the model on medical image datasets:
```bash
python medical_predict.py --config Config.py
```

### Pretraining
For ImageNet pretraining:
```bash
python imagenet_pretrain.py
```

### Evaluation
To evaluate the model performance:
```bash
python medical_predict.py --config Config.py --eval
```

## Dataset Preparation
The model supports various medical image datasets including:
- Blender synthetic data
- Colonoscopy images
- Small medical image datasets

Please refer to `med_dataloader/dataloader_total.py` for data loading and preprocessing details.

## Model Weights
Pretrained weights are stored in the following directories:
- `weights/`: ImageNet pretrained weights
- `medical_weights/`: Medical task-specific weights
- `results/`: Depth estimation results and trained models

## Core Technologies

### Multi-Branch Wavelet Transform Convolution
The MBWTConv2d module integrates wavelet transform into convolutional neural networks, enabling efficient multi-scale feature decomposition and reconstruction.

### MobileViM Module
This module separates global and local feature processing paths, leveraging state-space models for efficient global dependency modeling while maintaining local spatial details.

### Implicit Surface Decoding
The ISD (Implicit Surface Decoder) module uses cross-attention mechanisms to decode depth information from high-level features.

### Pyramid Depthwise Convolution
Efficient multi-scale feature extraction using depthwise separable convolutions with different kernel sizes.

## License
[Specify license information here]

## Acknowledgements
This project builds upon research in lightweight neural networks, wavelet transforms, and medical image analysis.

## Contact
For questions or issues, please contact [your contact information here].