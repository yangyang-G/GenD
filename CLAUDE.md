# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **GenD** (Generalized Deepfake Detection), a research project for deepfake detection that generalizes across benchmarks. It uses parameter-efficient fine-tuning of vision foundation models (CLIP, DINOv3, Perception Encoder) by only training Layer Normalization parameters (0.03% of total parameters) with L2 normalization and metric learning.

## Environment Setup

```bash
conda create --name GenD python=3.12 uv -y
conda activate GenD
uv pip install -r requirements.txt
```

## Common Development Commands

### Training

Run a training experiment defined in `src/exp/`:

```bash
python run_exp.py <experiment-name>
```

Example with minimal local data:
```bash
python run_exp.py example-training
```

Run with debug mode (limited batches/epochs):
```bash
python run_exp.py <experiment-name> --debug
```

Override config values via CLI:
```bash
python run_exp.py <experiment-name> --lr=1e-4 --max_epochs=10
```

### Testing/Evaluation

Test using a trained experiment:
```bash
python run_exp.py <test-exp-name> --from_exp <trained-exp-name> --test
```

Test with pre-trained Hugging Face models:
```bash
python run_exp.py GenD_CLIP--CDFv2-example --test
python run_exp.py GenD_PE--CDFv2-example --test
python run_exp.py GenD_DINO--CDFv2-example --test
```

### Code Formatting

Format code with ruff:
```bash
ruff format .
ruff check . --fix
```

Configuration is in `pyproject.toml` (line-length=120, ignores certain rules).

### Gradio Web UI

Launch the interactive demo:
```bash
python app/run.py
```

### Dataset Preprocessing

Extract faces from videos using RetinaFace detector:
```bash
python detector.py -i <input_videos> --mask_folder <mask_videos> -m at_least -n 32 -o <output_dir> --det_thres 0.1 -s 1.3 --target_size none
```

Create dataset path files:
```bash
find datasets/FF/DF/* -type f | sort > config/datasets/FF/DF.txt
```

## Architecture Overview

### Entry Points

- **`run.py`**: Main training/testing script using PyTorch Lightning
- **`run_exp.py`**: Experiment runner that applies config modifiers from `src/exp/`
- **`detector.py`**: Face detection and extraction from videos using RetinaFace
- **`app/run.py`**: Gradio web UI for inference

### Core Model (`src/model/`)

- **`base.py`**: `BaseDeepakeDetectionModel` - PyTorch Lightning module with metric logging, video aggregation, AUROC/AP calculation
- **`GenD.py`**: Main model class implementing the paper's approach
  - Initializes feature extractor (CLIP/DINO/Perception Encoder)
  - Initializes classification head
  - Freezes parameters except specified LayerNorm layers
  - Optional PEFT/LoRA support

### Encoders (`src/encoders/`)

- **`clip_encoder.py`**: CLIP vision encoder wrapper
- **`dino_encoder.py`**: DINOv3 encoder wrapper
- **`perception_encoder.py`**: Perception Encoder wrapper

### Configuration System (`src/config.py`)

Pydantic-based configuration with:
- `Backbone`: CLIP variants, Perception Encoder, DINOv3
- `Head`: Linear, LinearNorm (with L2 normalization)
- `Loss`: Cross-entropy, uniformity, alignment losses
- `Augmentations`: Training augmentations (flip, affine, blur, JPEG, etc.)

### Experiments (`src/exp/`)

Experiments are defined as dictionaries mapping names to lists of config modifiers:
- **`examples.py`**: Minimal training/testing examples
- **`wacv_rebuttal.py`**: Paper experiments
- **`third_party.py`**: Third-party model configs (Effort, ForAda, FS-VFM)

Example experiment definition:
```python
"my-exp": [
    Config(backbone=C.Backbone.CLIP_L_14, lr=3e-4),
    lambda c: setattr(c, "run_name", "custom-name")
]
```

### Dataset System (`src/dataset/`)

- **`data_module.py`**: PyTorch Lightning DataModule
- **`dataset.py`**: DeepfakeDataset with frame sampling, augmentations
- **`augmentations.py`**: Image augmentations (torchvision-based)

Dataset paths managed via `src/utils/files.py` using `Files` class that extends list with `map()`, `unique()`, `cat()` methods.

### Loss Functions (`src/loss.py`, `src/losses/`)

- **`unifalign.py`**: Alignment and uniformity losses for hyperspherical feature learning
- **`loss.py`**: Main loss combining cross-entropy, alignment, uniformity

### Key Design Patterns

1. **Experiment Configuration**: All experiments defined in `src/exp/` files, loaded into dictionary in `src/exp/__init__.py`

2. **Feature Extraction**: Backbones return `features` tensor; heads return `HeadOutput` with `logits_labels` and `l2_embeddings`

3. **Video Aggregation**: Frame predictions aggregated to video-level using mean/median in `compute_across_videos()`

4. **Metrics**: Frame-level and video-level AUROC, AP, accuracy logged via `log_all_metrics()`

5. **Checkpoint Loading**: Models can load from local checkpoints or Hugging Face (`yermandy/GenD_*`)

### Hugging Face Integration (`src/hf/`)

- **`modeling_gend.py`**: Standalone HF-compatible model for inference without Lightning dependencies

### Utilities (`src/utils/`)

- **`files.py`**: Dataset path management, run directory finding
- **`logger.py`**: Rich-based colored logging
- **`checks.py`**: Pre-training validation checks
- **`decorators.py`**: `@TryExcept` for graceful error handling
