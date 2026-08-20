#!/usr/bin/env bash
# Strict 5k comparison: direct untrim vs trim1 on the same trim1 frame0 observations.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

OBS_REPO="${OBS_REPO:-/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_trim1_h264}"
UNTRIM_STATS="${UNTRIM_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
TRIM1_STATS="${TRIM1_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_trim1_relative_rot/dataset_statistics.json}"
UNTRIM_CKPT="${UNTRIM_CKPT:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-gate500/checkpoint-5000}"
TRIM1_CKPT="${TRIM1_CKPT:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-trim1-baseline-5k/checkpoint-5000}"
N_EPISODES="${N_EPISODES:-20}"
NOISE_SEEDS="${NOISE_SEEDS:-0,1,2,3,4}"
OUT_DIR="${OUT_DIR:-/mnt/hdd/dexora/audit/matched_untrim_trim1_5k}"
LOG="${LOG:-/mnt/hdd/dexora/logs/matched_untrim_trim1_5k_probe.log}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

run_probe() {
  local tag="$1" checkpoint="$2" stats_file="$3" out="$4"
  if [[ "$FORCE" != 1 && -f "$out" ]]; then
    echo "skip $tag: $out exists" | tee -a "$LOG"
    return 0
  fi
  echo "==> $tag checkpoint=$checkpoint" | tee -a "$LOG"
  python scripts/probe_dexora_first_plan.py \
    --checkpoint "$checkpoint" \
    --repo-dir "$OBS_REPO" \
    --stats-file "$stats_file" \
    --n-episodes "$N_EPISODES" \
    --noise-seeds "$NOISE_SEEDS" \
    --tag "$tag" \
    --out "$out" 2>&1 | tee -a "$LOG"
}

UNTRIM_OUT="$OUT_DIR/direct_untrim_5k_on_trim1_frame0.json"
TRIM1_OUT="$OUT_DIR/trim1_5k_on_trim1_frame0.json"
SUMMARY_OUT="$OUT_DIR/matched_5k_summary.json"

: > "$LOG"
echo "==> matched 5k probe start $(date -Is) GPU=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "    observations=$OBS_REPO episodes=$N_EPISODES noise_seeds=$NOISE_SEEDS" | tee -a "$LOG"

run_probe direct_untrim_5k_on_trim1_frame0 "$UNTRIM_CKPT" "$UNTRIM_STATS" "$UNTRIM_OUT"
run_probe trim1_5k_on_trim1_frame0 "$TRIM1_CKPT" "$TRIM1_STATS" "$TRIM1_OUT"

python - "$UNTRIM_OUT" "$TRIM1_OUT" "$SUMMARY_OUT" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

untrim_path, trim1_path, summary_path = map(Path, sys.argv[1:])
untrim = json.loads(untrim_path.read_text())
trim1 = json.loads(trim1_path.read_text())

if untrim["repo_dir"] != trim1["repo_dir"]:
    raise RuntimeError("matched comparison requires the same observation repo")
if untrim["noise_seeds"] != trim1["noise_seeds"]:
    raise RuntimeError("matched comparison requires identical noise seeds")

untrim_xyz = untrim["metrics"]["first_plan_xyz_jump_m"]
trim1_xyz = trim1["metrics"]["first_plan_xyz_jump_m"]
summary = {
    "comparison": "direct untrim 5k vs trim1 5k",
    "observation_repo": untrim["repo_dir"],
    "observation_frame": "trim1 frame0 (equivalent to untrim old frame1)",
    "noise_seeds": untrim["noise_seeds"],
    "n_episodes": untrim["n_episodes"],
    "direct_untrim_5k": untrim_xyz,
    "trim1_5k": trim1_xyz,
    "trim1_minus_untrim_median_m": trim1_xyz["median"] - untrim_xyz["median"],
    "trim1_median_reduction_fraction": 1.0 - trim1_xyz["median"] / untrim_xyz["median"],
    "inputs": {
        "direct_untrim_5k": str(untrim_path),
        "trim1_5k": str(trim1_path),
    },
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

print("\nmodel          mean_cm  median_cm  p90_cm  max_cm")
for name, values in (("direct-untrim", untrim_xyz), ("trim1", trim1_xyz)):
    print(
        f"{name:14s} {values['mean'] * 100:7.2f} "
        f"{values['median'] * 100:9.2f} {values['p90'] * 100:7.2f} "
        f"{values['max'] * 100:7.2f}"
    )
print(f"wrote {summary_path}")
PY

echo "==> matched 5k probe done $(date -Is)" | tee -a "$LOG"
