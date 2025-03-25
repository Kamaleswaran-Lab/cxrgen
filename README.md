# CXRGen: Chest X-Ray Generation from EHR Data

A deep learning framework for generating chest X-ray images from electronic health record (EHR) data using transformer-based architectures and CLIP models.

## Overview

This project implements multiple deep learning models for predicting chest X-ray images from patient EHR data using CXR-TFT : a transformer-based framework that fuses EHR and image data. 

## Project Structure

```
├── src/
│   ├── models/         # Model implementations
│   │   ├── transformer.py    # Transformer model
│   │   ├── transformernn.py  # Transformer with neural network components
│   │   ├── clip.py          # CLIP model implementation
│   │   └── mlp.py           # MLP baseline model
│   ├── data/          # Data processing
│   ├── training/      # Training logic
│   ├── configs/       # Configuration files
│   │   ├── config_tft.py    # Transformer configuration
│   │   └── config_clip.py   # CLIP configuration
│   └── utils.py       # Utility functions
├── slurmjobs/         # HPC job scripts
├── tests/             # Test files
├── docs/              # Documentation
└── requirements.txt   # Dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MehakArora/cxrgen.git
cd cxrgen
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- `DATA_DIR`: Directory containing chest X-ray images
- `CHECKPOINT_DIR`: Directory for saving model checkpoints
- `WANDB_API_KEY`: Your Weights & Biases API key
- `WANDB_LOCAL_SAVE`: Local directory for W&B files
- `MIMIC_CLASSIFIER`: Path to MIMIC classifier model
- `INTERMEDIATE_DIR`: Directory for intermediate files

## Usage

### Training

To train a model, use the following command:

```bash
python src/train.py
```

The training script supports different model types:
- MLP: `model_type='mlp'`
- Transformer (EHR and CXR embeddings are added at the input): `model_type='transformer'`
- Transformer with concatenation (EHR CXR embeddings are concatenated at the input): `model_type='transformer_concat'`

### Configuration

Model configurations can be modified in the respective config files:
- `src/configs/config_tft.py` for transformer models
- `src/configs/config_clip.py` for CLIP models

Key configuration parameters include:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Logging and monitoring settings

## Data Format

The project expects the following data structure:
- Chest X-ray images in the specified `DATA_DIR`
- EHR matrices in a sub-folder called `longitudinal_data/ehr_matrices`
- Image embeddings in a sub-folder called `longitudinal_data/image_embeddings`

## Training Process

1. Data is split into train/validation/test sets
2. Models are trained with configurable parameters
3. Training progress is tracked using Weights & Biases
4. Model checkpoints are saved periodically
5. Best model is selected based on validation performance

## Dependencies

- PyTorch
- NumPy
- Pillow
- Weights & Biases
- Transformers

See `requirements.txt` for specific versions.
