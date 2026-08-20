#!/usr/bin/env bash
# Overnight Track-A reproduction diagnostics (read-only / no retrain).
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${CUDA_PROBE:=0}"
: "${CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${AUDIT_DIR:=/mnt/hdd/dexora/audit/cursor}"

cd "$DEXORA_ROOT"
export PYTHONPATH="$DEXORA_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$AUDIT_DIR" /mnt/hdd/dexora/logs

echo "==> [1/3] GT temporal alignment (CPU)"
/home/wangrenpeng/miniconda3/envs/dexora/bin/python scripts/audit_temporal_alignment.py \
  --out "$AUDIT_DIR/gt_temporal_alignment.json" \
  2>&1 | tee /mnt/hdd/dexora/logs/overnight_gt_alignment.log

echo "==> [2/3] 50k phase-bucket open-loop probe (GPU $CUDA_PROBE)"
CUDA_VISIBLE_DEVICES="$CUDA_PROBE" \
  /home/wangrenpeng/miniconda3/envs/dexora/bin/python scripts/audit_openloop_phase_buckets.py \
  --ckpt "$CKPT" \
  --per-bucket 20 \
  --out "$AUDIT_DIR/openloop_phase_buckets_50k.json" \
  2>&1 | tee /mnt/hdd/dexora/logs/overnight_phase_probe.log

echo "==> [3/3] faithful sim eval (GPU $CUDA_PROBE, skip if output exists)"
FAITH_OUT="/mnt/hdd/dexjoco/outputs/dexora/overnight_faithful_chunk32_seed0_ep1"
if [[ -f "$FAITH_OUT/summary.json" ]]; then
  echo "skip faithful eval: $FAITH_OUT/summary.json exists"
else
  CUDA_VISIBLE_DEVICES="$CUDA_PROBE" bash scripts/s_dexjoco_faithful_eval.sh \
    TAG=overnight_faithful OUTPUT_ROOT=/mnt/hdd/dexjoco/outputs/dexora \
    2>&1 | tee -a /mnt/hdd/dexora/logs/overnight_faithful_eval.log
fi

echo "==> DONE $(date -Is)" | tee "$AUDIT_DIR/overnight_DONE.log"
