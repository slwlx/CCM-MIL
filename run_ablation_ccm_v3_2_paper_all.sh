#!/bin/bash
# ============================================================================
# CCM-MIL v3.2 Paper Ablation — Master Controller
# ============================================================================
# 一键运行全部或部分消融实验，统一实验主目录。
#
# 用法:
#   # 运行全部消融实验
#   bash run_all_ablations.sh
#
#   # 仅运行指定表的实验
#   bash run_all_ablations.sh --tables 2,3a,3b
#
#   # 快速单fold验证模式
#   bash run_all_ablations.sh --quick-test
#
#   # 指定统一输出目录
#   bash run_all_ablations.sh --output-dir ./experiments/my_ablation
#
# 支持的表:
#   2   — Table 2:  两阶段架构消融 (OnlyS1, OnlyS2, NoLSMR)
#   3a  — Table 3A: Stage-1方向消融 (Dir1, Dir2, Dir4)
#   3b  — Table 3B: Stage-1网格消融 (SquareNorm, Aspect)
#   3c  — Table 3C: Stage-2重排序消融 (CenterOut, RiskGradient, StructureGuided, None)
#   3d  — Table 3D: LSMR机制剖析 (WithLSMR, NoLSMR, LSMRvsAMALA)
# ============================================================================

set -e

# ── 默认配置 ──
TABLES="all"
QUICK_TEST=false
OUTPUT_DIR=""

# ── 解析参数 ──
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
      echo "CCM-MIL v3.2 Paper Ablation — Master Controller"
      echo ""
      echo "Usage: bash run_all_ablations.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --tables TABLE_LIST   指定要运行的表，逗号分隔 (默认: all)"
      echo "                        可选: 2, 3a, 3b, 3c, 3d"
      echo "  --quick-test          快速单fold验证模式 (--k 1 --k_start 0 --k_end 1)"
      echo "  --output-dir DIR      指定统一输出目录 (默认: ./experiments/ccm_v3_2_paper_ablation_YYYYMMDD)"
      echo "  -h, --help            显示帮助"
      echo ""
      echo "Examples:"
      echo "  bash run_all_ablations.sh                          # 运行全部"
      echo "  bash run_all_ablations.sh --tables 2,3a            # 仅Table 2和3A"
      echo "  bash run_all_ablations.sh --tables 3c --quick-test # 快速测试3C"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage."
      exit 1
      ;;
  esac
done

# ── 统一输出目录 ──
if [[ -z "$OUTPUT_DIR" ]]; then
  BASE_DIR="./experiments/ccm_v3_2_paper_ablation_$(date +%Y%m%d)"
else
  BASE_DIR="$OUTPUT_DIR"
fi

mkdir -p "$BASE_DIR/logs"

echo "========================================"
echo "CCM-MIL v3.2 — Paper Ablation Suite"
echo "========================================"
echo "Unified base dir: $BASE_DIR"
echo "Tables to run:    $TABLES"
echo "Quick test mode:  $QUICK_TEST"
echo "Start time:       $(date)"
echo "========================================"

# ── 公共超参 ──
BASE_CMD="python main_survival.py \
  --drop_out 0.4 --early_stopping --lr 0.0001 \
  --label_frac 1.0 \
  --batch_size 1 --weighted_sample --bag_loss nll_surv \
  --backbone resnet50 \
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
  --mode ss-path"

# 快速测试模式：只跑1个fold
if $QUICK_TEST; then
  BASE_CMD="$BASE_CMD --k 1 --k_start 0 --k_end 1"
  echo "[QUICK TEST MODE] Running single fold only!"
else
  BASE_CMD="$BASE_CMD --k 5 --k_start 0 --k_end 5"
fi

# ── 癌种配置 ──
declare -A TASKS DATA_ROOTS PATCH_DIMS SPLIT_DIRS

TASKS[BLCA]="TCGA_BLCA_survival"
DATA_ROOTS[BLCA]="/home/wlx/github/MambaMIL-main/dataset/BLCA"
PATCH_DIMS[BLCA]="BLCA_patch_1024/h5_files"
SPLIT_DIRS[BLCA]="./splits/TCGA_BLCA_survival_kfold"

TASKS[COADREAD]="TCGA_COADREAD_survival"
DATA_ROOTS[COADREAD]="/home/wlx/github/MambaMIL-main/dataset/COAD"
PATCH_DIMS[COADREAD]="COAD_patch_1024/h5_files"
SPLIT_DIRS[COADREAD]="./splits/TCGA_COADREAD_survival_kfold"

TASKS[KIRC]="TCGA_KIRC_survival"
DATA_ROOTS[KIRC]="/home/wlx/github/MambaMIL-main/dataset/KIRC"
PATCH_DIMS[KIRC]="KIRC_patch_1024/h5_files"
SPLIT_DIRS[KIRC]="./splits/TCGA_KIRC_survival_kfold"

TASKS[KIRP]="TCGA_KIRP_survival"
DATA_ROOTS[KIRP]="/home/wlx/github/MambaMIL-main/dataset/KIRP"
PATCH_DIMS[KIRP]="KIRP_patch_1024/h5_files"
SPLIT_DIRS[KIRP]="./splits/TCGA_KIRP_survival_kfold"

TASKS[STAD]="TCGA_STAD_survival"
DATA_ROOTS[STAD]="/home/wlx/github/MambaMIL-main/dataset/STAD"
PATCH_DIMS[STAD]="STAD_patch_1024/h5_files"
SPLIT_DIRS[STAD]="./splits/TCGA_STAD_survival_kfold"

TASKS[LUAD]="TCGA_LUAD_survival"
DATA_ROOTS[LUAD]="/home/wlx/github/MambaMIL-main/dataset/LUAD"
PATCH_DIMS[LUAD]="LUAD_patch_1024/h5_files"
SPLIT_DIRS[LUAD]="./splits/TCGA_LUAD_survival_kfold"

CANCERS=(BLCA COADREAD KIRC KIRP STAD LUAD)
MODEL="ccm_mil_v3_2"

# ── 辅助函数: 运行一组实验 ──
run_experiments() {
  local SUBDIR="$1"
  local EXP_LIST="$2"
  local TOTAL=0

  for CANCER in "${CANCERS[@]}"; do
    TASK="${TASKS[$CANCER]}"
    DATA_ROOT="${DATA_ROOTS[$CANCER]}"
    PATCH_DIM="${PATCH_DIMS[$CANCER]}"
    SPLIT_DIR="${SPLIT_DIRS[$CANCER]}"

    # 解析实验列表: "Label1|args1;Label2|args2;..."
    IFS=';' read -ra EXP_ARRAY <<< "$EXP_LIST"

    for EXP in "${EXP_ARRAY[@]}"; do
      # 安全地分割 LABEL|ARGS
      if [[ "$EXP" == *"|"* ]]; then
        LABEL="${EXP%%|*}"
        EXTRA_ARGS="${EXP#*|}"
      else
        LABEL="$EXP"
        EXTRA_ARGS=""
      fi

      # 跨模型变体处理 (Label|model_type|args)
      if [[ "$EXTRA_ARGS" == ccm_mil_v3* ]]; then
        MODEL_NAME="$EXTRA_ARGS"
        EXTRA_ARGS=""
      elif [[ "$LABEL" == *"crossmodel"* ]] && [[ "$EXTRA_ARGS" == *"ccm_mil_v3"* ]]; then
        # D2特殊处理: Label_crossmodel|model_type
        MODEL_NAME="$EXTRA_ARGS"
        EXTRA_ARGS=""
      else
        MODEL_NAME="$MODEL"
      fi

      EXP_CODE="${MODEL_NAME}_${CANCER}_1024_${LABEL}"
      RESULTS_DIR="${BASE_DIR}/${SUBDIR}/${CANCER}/${LABEL}"
      LOG_FILE="${RESULTS_DIR}/${CANCER}_${LABEL}.log"

      echo ""
      echo "========================================"
      echo "[$((++TOTAL))] ${CANCER} | ${LABEL}"
      echo "Model: ${MODEL_NAME}"
      if [[ -n "$EXTRA_ARGS" ]]; then
        echo "Args: ${EXTRA_ARGS}"
      fi
      echo "Results: ${RESULTS_DIR}"
      echo "========================================"

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
  echo "----------------------------------------"
  echo "${SUBDIR} completed! Total: ${TOTAL} experiments"
  echo "----------------------------------------"
}

# ── 判断是否需要运行某个表 ──
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

# ============================================================================
# Table 2: Two-Stage Cascade Ablation
# ============================================================================
if should_run "2"; then
  echo ""
  echo "########################################"
  echo "# Table 2: Two-Stage Cascade Ablation  #"
  echo "########################################"
  run_experiments "table2_cascade" \
    "OnlyS1|--ablation_mode no_stage2;OnlyS2|--ablation_mode no_stage1;NoLSMR|--ablation_mode no_lsmr"
fi

# ============================================================================
# Table 3A: Stage-1 Direction Ablation
# ============================================================================
if should_run "3a"; then
  echo ""
  echo "########################################"
  echo "# Table 3A: Direction Ablation         #"
  echo "########################################"
  run_experiments "table3a_direction" \
    "Dir1|--ccm_stage1_dir 1;Dir2|--ccm_stage1_dir 2;Dir4|--ccm_stage1_dir 4"
fi

# ============================================================================
# Table 3B: Stage-1 Grid Construction Ablation
# ============================================================================
if should_run "3b"; then
  echo ""
  echo "########################################"
  echo "# Table 3B: Grid Construction Ablation #"
  echo "########################################"
  run_experiments "table3b_grid" \
    "SquareNorm|--ccm_v3_grid_mode square_norm;Aspect|--ccm_v3_grid_mode aspect"
fi

# ============================================================================
# Table 3C: Stage-2 Reordering Ablation
# ============================================================================
if should_run "3c"; then
  echo ""
  echo "########################################"
  echo "# Table 3C: Reordering Ablation        #"
  echo "########################################"
  run_experiments "table3c_reorder" \
    "CenterOut|--ccm_stage2_mode center_out;RiskGradient|--ccm_stage2_mode risk_gradient;StructureGuided|--ccm_stage2_mode structure_guided;None|--ccm_stage2_mode none"
fi

# ============================================================================
# Table 3D: LSMR Mechanism Analysis
# ============================================================================
if should_run "3d"; then
  echo ""
  echo "########################################"
  echo "# Table 3D: LSMR Mechanism Analysis    #"
  echo "########################################"

  # D1: LSMR On/Off
  echo ""
  echo "--- Part D1: LSMR On/Off ---"
  run_experiments "table3d_lsmr/d1_switch" \
    "WithLSMR|;NoLSMR|--ablation_mode no_lsmr"

  # D2: Cross-model comparison (LSMR vs AMALA)
  echo ""
  echo "--- Part D2: LSMR vs AMALA ---"
  run_experiments "table3d_lsmr/d2_crossmodel" \
    "LSMR_crossmodel|ccm_mil_v3_2;AMALA_crossmodel|ccm_mil_v3_2a"
fi

echo ""
echo "========================================"
echo "ALL REQUESTED ABLATIONS COMPLETED!"
echo "Unified results dir: $BASE_DIR"
echo "End time: $(date)"
echo "========================================"
