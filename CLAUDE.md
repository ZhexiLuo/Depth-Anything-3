# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Depth Anything 3 (DA3) is a depth estimation model that predicts spatially consistent geometry from arbitrary visual inputs. It supports monocular/multi-view depth estimation, camera pose estimation, and 3D Gaussian Splatting.

## Installation

```bash
pip install xformers torch>=2 torchvision
pip install -e .  # Basic installation
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70  # Gaussian head
pip install -e ".[app]"  # Gradio app (requires python>=3.10)
pip install -e ".[all]"  # All dependencies
```

## Common Commands

### CLI Usage
```bash
# Start backend server (caches model to GPU)
da3 backend --model-dir depth-anything/DA3NESTED-GIANT-LARGE --gallery-dir workspace/gallery

# Auto-detect input type and process
da3 auto <input_path> --export-format glb --export-dir <output_dir> --use-backend

# Process video
da3 video <video_path> --fps 15 --export-format glb --use-backend

# Launch Gradio web UI
da3 gradio --model-dir depth-anything/DA3NESTED-GIANT-LARGE

# Launch gallery server
da3 gallery --gallery-dir workspace/gallery
```

### Benchmark Evaluation
```bash
# Download benchmark datasets
hf download depth-anything/DA3-BENCH --local-dir workspace/benchmark_dataset --repo-type dataset

# Run full evaluation
python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-GIANT

# Evaluate specific dataset/mode
python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom] eval.modes=[pose]

# Print saved metrics only
python -m depth_anything_3.bench.evaluator eval.print_only=true
```

### DA3-Streaming (Long Video Processing)
```bash
cd da3_streaming
pip install -r requirements.txt
bash ./scripts/download_weights.sh
python da3_streaming.py --image_dir ./path_of_images --config ./configs/base_config.yaml
```

## Architecture

### Core Components
- `src/depth_anything_3/api.py`: Main API class `DepthAnything3` - handles model loading, inference, and export
- `src/depth_anything_3/model/da3.py`: Core network `DepthAnything3Net` - backbone + head + optional camera/GS decoders
- `src/depth_anything_3/cli.py`: CLI entry point using Typer

### Model Architecture (DepthAnything3Net)
- **Backbone**: DinoV2 feature extractor (`model/dinov2/`)
- **Head**: DPT or DualDPT for depth prediction (`model/dpt.py`, `model/dualdpt.py`)
- **Camera Decoder**: Optional pose estimation (`model/cam_dec.py`, `model/cam_enc.py`)
- **GS Head**: Optional 3D Gaussian Splatting (`model/gsdpt.py`, `model/gs_adapter.py`)

### Configuration System
- Model configs: `src/depth_anything_3/configs/*.yaml`
- Registry: `src/depth_anything_3/registry.py` maps model names to config paths
- Config loading: `depth_anything_3.cfg.create_object()` instantiates objects from YAML

### Key Directories
- `src/depth_anything_3/services/`: Backend server and inference service
- `src/depth_anything_3/app/`: Gradio web application
- `src/depth_anything_3/bench/`: Benchmark evaluation pipeline
- `src/depth_anything_3/utils/export/`: Export formats (GLB, NPZ, COLMAP, GS)
- `da3_streaming/`: Streaming inference for long videos

## Model Variants

| Model | Use Case |
|-------|----------|
| DA3-GIANT/LARGE/BASE/SMALL | Multi-view depth + pose estimation |
| DA3METRIC-LARGE | Metric depth (real-world scale) |
| DA3MONO-LARGE | Monocular relative depth |
| DA3NESTED-GIANT-LARGE | Combined any-view + metric scaling |

Models with `-1.1` suffix are retrained versions with improved street scene performance.

## Code Style

- Formatter: Black (line-length 99)
- Import sorting: isort with black profile
- Type hints used throughout
- Config-driven architecture using OmegaConf
