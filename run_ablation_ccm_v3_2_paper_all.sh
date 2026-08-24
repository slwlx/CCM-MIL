#!/bin/bash
# CCM-MIL v3.2 ablation suite
# Runs all or selected ablation groups under a single output directory.
#
# Usage:
#   bash run_ablation_ccm_v3_2_paper_all.sh                     # run all groups
#   bash run_ablation_ccm_v3_2_paper_all.sh --tables 2,3a,3b    # selected groups
#   bash run_ablation_ccm_v3_2_paper_all.sh --quick-test        # single-fold smoke test
#   bash run_ablation_ccm_v3_2_paper_all.sh --output-dir ./experiments/my_ablation
#
# Groups:
#   2   two-stage cascade ablation (OnlyS1, OnlyS2, NoLSMR)
#   3a  Stage-1 direction ablation (Dir1, Dir2, Dir4)
#   3b  Stage-1 grid construction ablation (SquareNorm, Aspect)
#   3c  Stage-2 reordering ablation (CenterOut, RiskGradient, StructureGuided, None)
#   3d  LSMR mechanism analysis (WithLSMR, NoLSMR, LSMR vs AMALA)

set -e

# Defaults
TABLES="all"
QUICK_TEST=false
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tables)
      TABLES="$2"
      shift 2
      ;;
    --quick-test)
      QUICK_TEST=true
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "CCM-MIL v3.2 ablation suite"
      echo ""
      echo "Usage: bash run_ablation_ccm_v3_2_paper_all.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --tables TABLE_LIST   comma-separated groups to run (default: all)"
      echo "                        choices: 2, 3a, 3b, 3c, 3d"
      echo "  --quick-test          single-fold smoke test (--k 1 --k_start 0 --k_end 1)"
      echo "  --output-dir DIR      output directory (default: ./experiments/ccm_v3_2_paper_ablation_YYYYMMDD)"
      echo "  -h, --help            show this help"
      echo ""
      echo "Examples:"
      echo "  bash run_ablation_ccm_v3_2_paper_all.sh                           # run everything"
      echo "  bash run_ablation_ccm_v3_2_paper_all.sh --tables 2,3a             # groups 2 and 3a only"
      echo "  bash run_ablation_ccm_v3_2_paper_all.sh --tables 3c --quick-test  # smoke test for 3c"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage."
      exit 1
      ;;
  esac
done

# Output directory
if [[ -z "$OUTPUT_DIR" ]]; then
  BASE_DIR="./experiments/ccm_v3_2_paper_ablation_$(date +%Y%m%d)"
else
  BASE_DIR="$OUTPUT_DIR"
fi

mkdir -p "$BASE_DIR/logs"

echo "----------------------------------------"
echo "CCM-MIL v3.2 ablation suite"
echo "----------------------------------------"
echo "Base dir:    $BASE_DIR"
echo "Groups:      $TABLES"
echo "Quick test:  $QUICK_TEST"
echo "Start time:  $(date)"
echo "----------------------------------------"

# Shared hyperparameters
BASE_CMD="python main_survival.py \
  --drop_out 0.4 --early_stopping --lr 0.0001 \
  --label_frac 1.0 \
  --batch_size 1 --weighted_sample --bag_loss nll_surv \
  --backbone resnet50 \
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
  --mode ss-path"

# Quick test: single fold only
if $QUICK_TEST; then
  BASE_CMD="$BASE_CMD --k 1 --k_start 0 --k_end 1"
  echo "[QUICK TEST] Running a single fold only."
else
  BASE_CMD="$BASE_CMD --k 5 --k_start 0 --k_end 5"
fi

# Dataset root; override by exporting DATA_ROOT before running.
DATA_ROOT="${DATA_ROOT:-/path/to/tcga_features}"

# Per-cohort configuration
declare -A TASKS DATA_ROOTS PATCH_DIMS SPLIT_DIRS

TASKS[BLCA]="TCGA_BLCA_survival"
DATA_ROOTS[BLCA]="${DATA_ROOT}/BLCA"
PATCH_DIMS[BLCA]="BLCA_patch_1024/h5_files"
SPLIT_DIRS[BLCA]="./splits/TCGA_BLCA_survival_kfold"

TASKS[COADREAD]="TCGA_COADREAD_survival"
DATA_ROOTS[COADREAD]="${DATA_ROOT}/COAD"
PATCH_DIMS[COADREAD]="COAD_patch_1024/h5_files"
SPLIT_DIRS[COADREAD]="./splits/TCGA_COADREAD_survival_kfold"

TASKS[KIRC]="TCGA_KIRC_survival"
DATA_ROOTS[KIRC]="${DATA_ROOT}/KIRC"
PATCH_DIMS[KIRC]="KIRC_patch_1024/h5_files"
SPLIT_DIRS[KIRC]="./splits/TCGA_KIRC_survival_kfold"

TASKS[KIRP]="TCGA_KIRP_survival"
DATA_ROOTS[KIRP]="${DATA_ROOT}/KIRP"
PATCH_DIMS[KIRP]="KIRP_patch_1024/h5_files"
SPLIT_DIRS[KIRP]="./splits/TCGA_KIRP_survival_kfold"

TASKS[STAD]="TCGA_STAD_survival"
DATA_ROOTS[STAD]="${DATA_ROOT}/STAD"
PATCH_DIMS[STAD]="STAD_patch_1024/h5_files"
SPLIT_DIRS[STAD]="./splits/TCGA_STAD_survival_kfold"

TASKS[LUAD]="TCGA_LUAD_survival"
DATA_ROOTS[LUAD]="${DATA_ROOT}/LUAD"
PATCH_DIMS[LUAD]="LUAD_patch_1024/h5_files"
SPLIT_DIRS[LUAD]="./splits/TCGA_LUAD_survival_kfold"

CANCERS=(BLCA COADREAD KIRC KIRP STAD LUAD)
MODEL="ccm_mil_v3_2"

# Run one group of experiments.
# EXP_LIST format: "Label1|args1;Label2|args2;..."
run_experiments() {
  local SUBDIR="$1"
  local EXP_LIST="$2"
  local TOTAL=0

  for CANCER in "${CANCERS[@]}"; do
    TASK="${TASKS[$CANCER]}"
    DATA_ROOT="${DATA_ROOTS[$CANCER]}"
    PATCH_DIM="${PATCH_DIMS[$CANCER]}"
    SPLIT_DIR="${SPLIT_DIRS[$CANCER]}"

    IFS=';' read -ra EXP_ARRAY <<< "$EXP_LIST"

    for EXP in "${EXP_ARRAY[@]}"; do
      # Split LABEL|ARGS
      if [[ "$EXP" == *"|"* ]]; then
        LABEL="${EXP%%|*}"
        EXTRA_ARGS="${EXP#*|}"
      else
        LABEL="$EXP"
        EXTRA_ARGS=""
      fi

      # Cross-model variant (Label|model_type|args)
      if [[ "$EXTRA_ARGS" == ccm_mil_v3* ]]; then
        MODEL_NAME="$EXTRA_ARGS"
        EXTRA_ARGS=""
      elif [[ "$LABEL" == *"crossmodel"* ]] && [[ "$EXTRA_ARGS" == *"ccm_mil_v3"* ]]; then
        MODEL_NAME="$EXTRA_ARGS"
        EXTRA_ARGS=""
      else
        MODEL_NAME="$MODEL"
      fi

      EXP_CODE="${MODEL_NAME}_${CANCER}_1024_${LABEL}"
      RESULTS_DIR="${BASE_DIR}/${SUBDIR}/${CANCER}/${LABEL}"
      LOG_FILE="${RESULTS_DIR}/${CANCER}_${LABEL}.log"

      echo ""
      echo "----------------------------------------"
      echo "[$((++TOTAL))] ${CANCER} | ${LABEL}"
      echo "Model: ${MODEL_NAME}"
      if [[ -n "$EXTRA_ARGS" ]]; then
        echo "Args: ${EXTRA_ARGS}"
      fi
      echo "Results: ${RESULTS_DIR}"
      echo "----------------------------------------"

      mkdir -p "$RESULTS_DIR"

      $BASE_CMD \
        --task "$TASK" \
        --data_root_dir "$DATA_ROOT" \
        --patch_dim "$PATCH_DIM" \
        --split_dir "$SPLIT_DIR" \
        --model_type "$MODEL_NAME" \
        --exp_code "$EXP_CODE" \
        --results_dir "$RESULTS_DIR" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"

      echo "Completed: ${CANCER} ${LABEL} at $(date)"
    done
  done

  echo ""
  echo "${SUBDIR} completed: ${TOTAL} experiments"
}

# Check whether a group is selected
should_run() {
  local t="$1"
  if [[ "$TABLES" == "all" ]]; then
    return 0
  fi
  if [[ ",${TABLES}," == *",${t},"* ]]; then
    return 0
  fi
  return 1
}

OVERALL_TOTAL=0

# Group 2: two-stage cascade ablation
if should_run "2"; then
  echo ""
  echo "== Group 2: two-stage cascade ablation =="
  run_experiments "table2_cascade" \
    "OnlyS1|--ablation_mode no_stage2;OnlyS2|--ablation_mode no_stage1;NoLSMR|--ablation_mode no_lsmr"
fi

# Group 3a: Stage-1 direction ablation
if should_run "3a"; then
  echo ""
  echo "== Group 3a: direction ablation =="
  run_experiments "table3a_direction" \
    "Dir1|--ccm_stage1_dir 1;Dir2|--ccm_stage1_dir 2;Dir4|--ccm_stage1_dir 4"
fi

# Group 3b: Stage-1 grid construction ablation
if should_run "3b"; then
  echo ""
  echo "== Group 3b: grid construction ablation =="
  run_experiments "table3b_grid" \
    "SquareNorm|--ccm_v3_grid_mode square_norm;Aspect|--ccm_v3_grid_mode aspect"
fi

# Group 3c: Stage-2 reordering ablation
if should_run "3c"; then
  echo ""
  echo "== Group 3c: reordering ablation =="
  run_experiments "table3c_reorder" \
    "CenterOut|--ccm_stage2_mode center_out;RiskGradient|--ccm_stage2_mode risk_gradient;StructureGuided|--ccm_stage2_mode structure_guided;None|--ccm_stage2_mode none"
fi

# Group 3d: LSMR mechanism analysis
if should_run "3d"; then
  echo ""
  echo "== Group 3d: LSMR mechanism analysis =="

  # D1: LSMR on/off
  echo "-- Part D1: LSMR on/off --"
  run_experiments "table3d_lsmr/d1_switch" \
    "WithLSMR|;NoLSMR|--ablation_mode no_lsmr"

  # D2: cross-model comparison (LSMR vs AMALA); requires the ccm_mil_v3_2a model variant
  echo "-- Part D2: LSMR vs AMALA --"
  run_experiments "table3d_lsmr/d2_crossmodel" \
    "LSMR_crossmodel|ccm_mil_v3_2;AMALA_crossmodel|ccm_mil_v3_2a"
fi

echo ""
echo "----------------------------------------"
echo "All requested ablations completed."
echo "Results dir: $BASE_DIR"
echo "End time: $(date)"
echo "----------------------------------------"
