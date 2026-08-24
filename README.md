# CCM-MIL

CCM-MIL (Coordinate-Cascade Mamba) for whole-slide image (WSI) survival analysis.

This repository contains the reference implementation of the paper "CCM-MIL: Coordinate-Cascade Mamba with Spatial Topology Modeling for Whole-Slide Image Survival Analysis" (submitted to the Journal of Imaging).

## Overview

CCM-MIL is a two-stage cascade architecture for multiple instance learning (MIL) on histopathological whole-slide images:

- Stage 1: patch features are scattered onto a coordinate-derived 2D grid and encoded by multi-directional Mamba2 scans (4 or 8 directions); an importance head produces a per-patch importance map.
- Stage 2: the importance map guides a center-out reordering of the patch sequence, which is re-encoded by Mamba2 layers with a prepended global token.
- LSMR: a logits-space mean residual combines the attention-pooled prediction with a MeanMIL-style patch-logits mean through a learned gate.

## Installation

```bash
# 1. Create conda environment
conda create -n ccmmil python=3.10 -y
conda activate ccmmil

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Install mamba dependencies
pip install packaging causal-conv1d==1.1.1

# 4. Install other dependencies
pip install scikit-survival==0.22.2 pandas==2.2.1 tensorboardx h5py wandb tensorboard lifelines

# 5. Install mamba-ssm
# Follow https://github.com/state-spaces/mamba for installation,
# or use the pre-built wheels if available.
```

## Data Preparation

This codebase operates on pre-extracted WSI patch features (not raw images). Features can be extracted with tools such as [CLAM](https://github.com/mahmoodlab/CLAM) using backbones such as ResNet50 or PLIP.

### Expected directory structure

```
DATA_ROOT_DIR/
  pt_files/
    resnet50/
      slide_1.pt
      slide_2.pt
  h5_files/          # optional, for ss-path mode
    slide_1.h5
    slide_2.h5
```

Each `.pt` file contains a tensor of shape `[N, feature_dim]`, where N is the number of patches of the slide (typically no more than ~20,000 at 1024x1024 tiling) and feature_dim is 1024 for ResNet50.

Each `.h5` file contains `features` (shape `[N, D]`) and `coords` (shape `[N, 2]`).

### Provided data splits

This repository includes pre-defined 5-fold cross-validation splits for 6 TCGA cohorts:

- BLCA (Bladder Urothelial Carcinoma)
- COADREAD (Colon/Rectum Adenocarcinoma)
- KIRC (Kidney Renal Clear Cell Carcinoma)
- KIRP (Kidney Renal Papillary Cell Carcinoma)
- LUAD (Lung Adenocarcinoma)
- STAD (Stomach Adenocarcinoma)

Splits are in `splits/TCGA_*_survival_kfold/` and case labels are in `dataset_csv/`.

## Quick Start

### Single experiment (BLCA, 1 fold for a quick test)

```bash
python main_survival.py \
  --drop_out 0.4 --early_stopping --lr 0.0001 \
  --k 1 --k_start 0 --k_end 1 \
  --label_frac 1.0 \
  --batch_size 1 --weighted_sample --bag_loss nll_surv \
  --task "TCGA_BLCA_survival" --backbone resnet50 \
  --in_dim 1024 --k_fold True \
  --ccm_stage1_dir 4 \
  --ccm_stage2_mode center_out \
  --ccm_selection_mode soft \
  --ccm_soft_topk_ratio 0.3 \
  --ccm_stage2_layers 1 \
  --ccm_v3_grid_mode square_norm \
  --reg 0.001 \
  --max_epochs 100 \
  --seed 1 \
  --use_h5 True \
  --mode ss-path \
  --exp_code "ccm_mil_v3_2/BLCA_1024" \
  --results_dir "./experiments/BLCA" \
  --data_root_dir "/path/to/BLCA" \
  --patch_dim "BLCA_patch_1024/h5_files" \
  --split_dir "./splits/TCGA_BLCA_survival_kfold"
```

### Full 5-fold cross-validation

Remove `--k 1 --k_start 0 --k_end 1` to run all 5 folds.

### Ablation experiments

Run all ablation experiments reported in the paper:

```bash
bash run_ablation_ccm_v3_2_paper_all.sh
```

Options:

```bash
# Run selected groups only
bash run_ablation_ccm_v3_2_paper_all.sh --tables 2,3a

# Quick test (1 fold)
bash run_ablation_ccm_v3_2_paper_all.sh --quick-test
```

Set the `DATA_ROOT` environment variable (or edit `DATA_ROOT` in the script) to point to your feature directory. The D2 cross-model comparison additionally requires the AMALA variant model (`ccm_mil_v3_2a`).

## Model Architecture

```
Input: dict {'patch_features': (N, D), 'patch_coords': (N, 2)}
  |
Embedding: Linear(D -> 512) + GELU + Dropout
  |
Stage 1: multi-directional Mamba2 (4 directions: 0, 90, 180, 270 degrees)
  - scatter patches onto a 2D grid
  - scan along each direction
  - importance head for per-patch scoring
  - differentiable soft selection (Gumbel relaxation during training)
  |
Stage 2: importance-guided reordering + Mamba2
  - reorder patches (center_out / risk_gradient / structure_guided)
  - global token + reordered sequence -> Mamba2 layers
  |
Output: LSMR fusion
  - CCM path: attention-pooled feature fusion -> classifier
  - mean path: patch_logits.mean()
  - gate: (1 - w) * logits_ccm + w * logits_mean
  |
Survival: hazards = sigmoid(logits), S = cumprod(1 - hazards)
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `ccm_stage1_dir` | 4 | Number of scan directions (1/2/4/8) |
| `ccm_stage2_mode` | center_out | Reordering mode (center_out / risk_gradient / structure_guided / none) |
| `ccm_soft_topk_ratio` | 0.3 | Selection ratio; only active in the hard and straight-through variants (the default soft mode retains all tokens) |
| `ccm_stage2_layers` | 1 | Number of Mamba2 layers in Stage 2 |
| `ccm_drop_path_rate` | 0.0 | Drop-path rate for Stage 1 |
| `ablation_mode` | none | Ablation switch (no_stage1 / no_stage2 / no_lsmr / ... ) |
| `drop_out` | 0.4 | Dropout rate |
| `lr` | 1e-4 | Learning rate |
| `reg` | 0.001 | Weight decay |
| `max_epochs` | 100 | Maximum training epochs (early stopping patience 15) |
| `seed` | 1 | Random seed (fixed per fold) |

## Ablation Modes

| Mode | Description |
|---|---|
| `none` | Full CCM-MIL (default) |
| `no_stage1` | Skip Stage 1, keep Stage 2 |
| `no_stage2` | Skip Stage 2, keep the Stage 1 global token |
| `no_lsmr` | Disable LSMR logits fusion |
| `grid_only` | Keep the grid embedding but skip directional Mamba scanning |
| `single_direction` | Use only one scan direction in Stage 1 |
| `no_selection` | Skip Stage 2 patch selection |
| `random_mask` | Random importance scores |

## Citation

If you use this code, please cite:

```bibtex
@article{wang2026ccmmil,
  title={CCM-MIL: Coordinate-Cascade Mamba with Spatial Topology Modeling for Whole-Slide Image Survival Analysis},
  author={Wang, Lixiang and Zhang, Yuluan and Que, Tengcheng and Hu, Yanling},
  journal={Journal of Imaging},
  note={under review},
  year={2026}
}
```

## License

This project is released under the MIT License.
