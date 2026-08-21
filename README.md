# CCM-MIL v3.2

Clinical Cascade Mamba (CCM-MIL) for Whole Slide Image (WSI) Survival Analysis.

This repository contains the official implementation of the CCM-MIL v3.2 model with **Logits-Space Mean Residual (LSMR)** as the default Stage-2 enhancement for survival prediction on computational pathology.

## Overview

CCM-MIL is a two-stage cascade architecture for Multiple Instance Learning (MIL) on histopathological Whole Slide Images:

- **Stage 1**: Coarse multi-directional Mamba2 scanning (4 or 8 directions) with grid-based spatial scattering.
- **Stage 2**: Fine semantic reordering via adaptive Soft Top-K selection and local Mamba2 refinement.
- **LSMR**: Logits-Space Mean Residual fusion that combines CCM feature-space attention with MeanMIL-style logits-space averaging.

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

# 5. Install mamba-ssm (requires custom fork or official package)
# Please follow https://github.com/state-spaces/mamba for installation
# or use the pre-built wheels if available.
```

## Data Preparation

This codebase operates on **pre-extracted WSI patch features** (not raw images). Features should be extracted using tools like [CLAM](https://github.com/mahmoodlab/CLAM) with backbones such as ResNet50 or PLIP.

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

Each `.pt` file contains a tensor of shape `[N_patches, feature_dim]` (e.g., `[~65000, 1024]`).

Each `.h5` file contains `features` (shape `[N, D]`) and `coords` (shape `[N, 2]`).

### Provided data splits

This repository includes pre-defined 5-fold cross-validation splits for 6 TCGA cancer types:
- BLCA (Bladder Urothelial Carcinoma)
- COADREAD (Colon/Rectum Adenocarcinoma)
- KIRC (Kidney Renal Clear Cell Carcinoma)
- KIRP (Kidney Renal Papillary Cell Carcinoma)
- LUAD (Lung Adenocarcinoma)
- STAD (Stomach Adenocarcinoma)

Splits are in `splits/TCGA_*_survival_kfold/` and case labels are in `dataset_csv/`.

## Quick Start

### Single experiment (BLCA, 1 fold for quick test)

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
  --ccm_soft_topk_ratio 0.45 \
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

Run all ablation experiments described in the paper:

```bash
bash run_ablation_ccm_v3_2_paper_all.sh
```

Options:
```bash
# Run specific tables only
bash run_ablation_ccm_v3_2_paper_all.sh --tables 2,3a

# Quick test (1 fold)
bash run_ablation_ccm_v3_2_paper_all.sh --quick-test
```

## Model Architecture

```
Input: dict {'patch_features': (N, D), 'patch_coords': (N, 2)}
  ↓
Embedding: Linear(D → 512) + GELU + Dropout
  ↓
Stage 1: Multi-directional Mamba2 (4 dirs: 0°, 90°, 180°, 270°)
  - Scatter patches to 2D grid
  - Scan along each direction
  - Directional attention head for importance scoring
  - Soft Top-K mask with Gumbel-Softmax
  ↓
Stage 2: Semantic Reordering + Mamba2
  - Reorder patches by importance (center_out / risk_gradient / structure_guided)
  - Global token + reordered sequence → Mamba2 layers
  ↓
Output: LSMR fusion
  - CCM path: attention-weighted feature fusion → classifier
  - MeanMIL path: patch_logits.mean()
  - Adaptive gate: (1-w) * logits_ccm + w * logits_mean
  ↓
Survival: hazards = sigmoid(logits), S = cumprod(1 - hazards)
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `ccm_stage1_dir` | 4 | Number of scan directions (1/2/4/8) |
| `ccm_stage2_mode` | center_out | Reordering mode (center_out / risk_gradient / structure_guided / none) |
| `ccm_soft_topk_ratio` | 0.45 | Soft Top-K selection ratio |
| `ccm_stage2_layers` | 1 | Number of Mamba2 layers in Stage 2 |
| `ccm_drop_path_rate` | 0.0 | Drop path rate for Stage 1 |
| `ablation_mode` | none | Ablation switch (no_stage1 / no_stage2 / no_lsmr / ... ) |
| `drop_out` | 0.4 | Dropout rate |
| `lr` | 1e-4 | Learning rate |
| `reg` | 0.001 | Weight decay |

## Ablation Modes

| Mode | Description |
|---|---|
| `none` | Full CCM-MIL v3.2 (default) |
| `no_stage1` | Skip Stage 1, only Stage 2 |
| `no_stage2` | Skip Stage 2, only Stage 1 global token |
| `no_lsmr` | Disable LSMR logits fusion |
| `grid_only` | Keep grid embedding but skip directional Mamba |
| `single_direction` | Use only 1 direction in Stage 1 |
| `no_selection` | Skip Stage-2 patch selection |
| `random_mask` | Random importance scores |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ccmmil2025,
  title={CCM-MIL: Clinical Cascade Mamba for WSI Survival Analysis},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is released under the MIT License.
