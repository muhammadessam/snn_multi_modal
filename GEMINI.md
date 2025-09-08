
# Gemini Code Assistant

This document provides an overview of the multi-modal classification project, designed to be used as a context for the Gemini AI assistant.

## Project Overview

The goal of this project is to develop a multi-modal spiking neural network transformer, using QKFormer as its core architecture. The model is designed for general classification tasks and will incorporate a fusion method to combine information from different modalities. The model will be tested on the CREMA-D, AVE, and UrbanSound8K-AV datasets.

Currently, the project is in the phase of adapting the model for uni-modal processing. The model results over the CREMA-D dataset are good, but the model is facing challenges in converging when trained on the AVE dataset.

## References

The QKFormer model architecture is based on the following paper:

-   **Local Path:** `./pdf/+2024 - NeurIPS - QKFormer.pdf`
-   **arXiv Link:** [https://arxiv.org/pdf/2403.16552](https://arxiv.org/pdf/2403.16552)

## Complete File Structure

```
multi-claude/
├── LICENSE                                        # Project license
├── pyproject.toml                                # Python dependencies and project config
├── uv.lock                                       # UV package manager lock file
├── train_crema_video.py                          # MAIN ENTRY POINT: CREMA-D video training script
├── train_crema_audio.py                          # MAIN ENTRY POINT: CREMA-D audio training script
├── configs/
│   └── neuron_factory.py                        # Spiking neuron factory and configuration
├── datasets/
│   ├── crema/
│   │   ├── crema_video.py                       # CREMA-D video dataset loader
│   │   └── crema_audio.py                       # CREMA-D audio dataset loader
│   └── ave/
│       └── video_dataset.py                     # AVE dataset loader (placeholder)
├── models/
│   └── video_model.py                           # QKFormer hierarchical spiking transformer
├── jobs/
│   ├── CLAUDE.md                                 # Jobs folder documentation
│   └── kisski/
│       └── video/
│           └── crema/
│               └── crema_video_lr_batch_size_exp.sh  # SLURM experiment script
├── out/
│   └── crema/
│       ├── video/                               # Video training outputs and checkpoints
│       └── audio/                               # Audio training outputs and checkpoints
└── pdfs/
    └── QKFormer.pdf                             # Original QKFormer research paper
```

## Architecture Overview

### Core Components

1. **HierarchicalSpikingTransformer** (`models/video_model.py`): Main model architecture
   - 3-stage hierarchical structure with different embedding dimensions
   - Uses TokenQKAttention for efficient attention computation
   - Supports both video and image inputs via temporal dimension handling

1b. **HierarchicalAudioSpikingTransformer** (`models/audio_model.py`): Audio-specific model architecture
   - 3-stage hierarchical structure optimized for audio spectrograms
   - Uses ChannelQKAttention in stages 1&2 for feature channel selection
   - Uses full self-attention in stage 3 for global temporal-frequency relationships
   - Designed for single-channel spectrograms with channel-wise feature attention

2. **Neuron Factory** (`configs/neuron_factory.py`): Centralized neuron creation
   - Supports multiple neuron types: LIF, PLIF, IF, QIF
   - Configurable surrogate functions: sigmoid, atan, soft_sign, etc.
   - Default parameters for each neuron type

3. **CREMA-D Dataset Handlers**: 
   - **Video** (`datasets/crema/crema_video.py`): Loads pre-extracted video frames
   - **Audio** (`datasets/crema/crema_audio.py`): Processes raw audio files into temporal spectrograms
   - Handles 6-class emotion classification: ANG, DIS, FEA, HAP, NEU, SAD
   - Supports data augmentation with torchvision v2 transforms

## Building and Running

The project is based on Python and uses `uv` for dependency management. The main dependencies are listed in the `pyproject.toml` file.

### Environment Setup

Before running the training scripts, you need to activate the virtual environment:

```bash
source .venv/bin/activate
```

### Training

The project includes separate training scripts for the audio and video models:

-   **Audio Model Training (for CREMA-D):**
    ```bash
    python train_crema_audio.py [arguments]
    ```

-   **Video Model Training (for CREMA-D):**
    The video modality for the CREMA-D dataset is handled by `vision_model.py` and trained with `train_video.py`.
    ```bash
    python train_crema_video.py [arguments]
    ```

Both training scripts accept a variety of command-line arguments to configure the model architecture, training parameters, and data paths. For a full list of options, run the scripts with the `--help` flag.

Example of running the video training:
```bash
python train_video.py --data-path /user/mohamed.saleh01/u18697/projects/datasets/crema/Crema_frams_80_10_10 --model qkf_snn --epochs 100 --batch-size 32 --lr 0.0005 --patience-epochs 40 --embed-dims 192 --num-heads 2 4 6 --depths 2 4 8 --time-step 8 --output-dir ./out/crema/video
```

### Job Scripts

The `jobs` directory contains shell scripts that provide examples of how to run the training scripts, including parameter sweeps and submissions to a Slurm cluster. These scripts can be used as a reference for configuring and running training experiments.

## Data Requirements

### Video Data
The video training expects pre-processed CREMA-D frames in this structure:
```
data_path/
├── train/
│   ├── {video_id}_{emotion}_XX/
│   │   ├── frame_0000.jpg
│   │   └── ...
└── val/
│   │── {video_id}_{emotion}_XX/
│   │    ├── frame_0000.jpg
│   │    └── ...
└── test/
    └── {video_id}_{emotion}_XX/
        ├── frame_0000.jpg
        └── ...
```

### Audio Data  
The audio training expects raw CREMA-D audio files:
```
AudioWAV/
├── {actor_id}_{sentence}_{emotion}_{intensity}.wav
├── 1001_DFA_ANG_XX.wav
├── 1001_DFA_DIS_XX.wav
├── 1001_IEO_FEA_HI.wav
└── ...
```