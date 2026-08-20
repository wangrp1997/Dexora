#!/usr/bin/env bash
# Shared first-plan probe: Dexora (untrimmed + trim1) and π0.5 handfresh.
# Read-only; does not touch training. Prefer a GPU with ~40GB+ free.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

OUT_DIR=/mnt/hdd/dexora/audit/cursor
LOG=/mnt/hdd/dexora/logs/shared_first_plan_probe.log
mkdir -p "$OUT_DIR" /mnt/hdd/dexora/logs
UNTRIM=/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264
UNTRIM_STATS=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json
TRIM1=/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_trim1_h264
TRIM1_STATS=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_trim1_relative_rot/dataset_statistics.json
CKPT50=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k
CKPT_TRIM=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-trim1-baseline-5k
PI05_ROOT=/mnt/ssd/checkpoints/openpi_dexjoco/pi05_dexjoco_lora/pi05_dexjoco_lora_state44_handfresh
N_EP="${N_EPISODES:-20}"

echo "==> shared first-plan probe start $(date -Is) GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG"

run_dexora() {
  local tag="$1" ckpt="$2" repo="$3" stats="$4" out="$5"
  if [[ -f "$out" ]]; then
    echo "skip dexora $tag: $out exists" | tee -a "$LOG"
    return 0
  fi
  echo "==> dexora $tag -> $out" | tee -a "$LOG"
  python scripts/probe_dexora_first_plan.py \
    --checkpoint "$ckpt" \
    --repo-dir "$repo" \
    --stats-file "$stats" \
    --n-episodes "$N_EP" \
    --tag "$tag" \
    --out "$out" 2>&1 | tee -a "$LOG"
}

run_pi05() {
  local step="$1"
  local out="$OUT_DIR/pi05_first_plan_probe_${step}.json"
  local ckpt="$PI05_ROOT/$step"
  if [[ -f "$out" ]]; then
    echo "skip pi05 $step: $out exists" | tee -a "$LOG"
    return 0
  fi
  if [[ ! -d "$ckpt" ]]; then
    echo "skip pi05 $step: missing $ckpt" | tee -a "$LOG"
    return 0
  fi
  echo "==> pi05 $step -> $out" | tee -a "$LOG"
  # openpi env for policy load
  PATH="/home/wangrenpeng/miniconda3/envs/openpi/bin:$PATH" \
  python scripts/probe_pi05_first_plan.py \
    --checkpoint "$ckpt" \
    --repo-dir "$UNTRIM" \
    --n-episodes "$N_EP" \
    --out "$out" 2>&1 | tee -a "$LOG"
}

# Priority: trim1 5k first, then untrimmed Dexora ladder, then π0.5 gaps.
run_dexora "trim1_5k" "$CKPT_TRIM/checkpoint-5000" "$TRIM1" "$TRIM1_STATS" \
  "$OUT_DIR/dexora_first_plan_trim1_5k.json"

run_dexora "untrim_1k" "$CKPT50/checkpoint-1000" "$UNTRIM" "$UNTRIM_STATS" \
  "$OUT_DIR/dexora_first_plan_untrim_1k.json"
run_dexora "untrim_10k" "$CKPT50/checkpoint-10000" "$UNTRIM" "$UNTRIM_STATS" \
  "$OUT_DIR/dexora_first_plan_untrim_10k.json"
run_dexora "untrim_50k" "$CKPT50/checkpoint-50000" "$UNTRIM" "$UNTRIM_STATS" \
  "$OUT_DIR/dexora_first_plan_untrim_50k.json"

# Pretrain as Dexora "0" (IO-expanded weights at assemble root).
if [[ -f "$CKPT50/../dexora-400m-pretrain-assemble/pytorch_model.bin" ]] || \
   [[ -d /mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble ]]; then
  run_dexora "pretrain0" "/mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble" \
    "$UNTRIM" "$UNTRIM_STATS" "$OUT_DIR/dexora_first_plan_pretrain0.json" || true
fi

for step in 10000 20000 40000; do
  run_pi05 "$step"
done

python - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/mnt/hdd/dexora/audit/cursor")
rows = []
for p in sorted(root.glob("*first_plan*.json")):
    if p.name.endswith("_summary.json"):
        continue
    d = json.loads(p.read_text())
    rows.append({
        "file": p.name,
        "tag": d.get("tag") or d.get("checkpoint", "").split("/")[-1],
        "n": d.get("n"),
        "xyz_m": round(float(d.get("mean_first_plan_xyz_jump_m", float("nan"))), 4),
        "rot": round(float(d.get("mean_first_plan_rot_jump_rad", float("nan"))), 3),
        "hand": round(float(d.get("mean_first_plan_hand_mean", float("nan"))), 3),
        "gt_hand": round(float(d.get("mean_gt_action_hand_mean", float("nan"))), 3),
    })
out = root / "shared_first_plan_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
print(json.dumps({"rows": rows}, indent=2))
print(f"wrote {out}")
PY

echo "==> shared first-plan DONE $(date -Is)" | tee -a "$LOG" | tee "$OUT_DIR/shared_first_plan_DONE.log"
