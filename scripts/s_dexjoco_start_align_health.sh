#!/usr/bin/env bash
# Health gate for start-align ablation checkpoints (seed0, step 2000).
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${CKPT_ROOT:=/mnt/hdd/dexora/checkpoints}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${HEALTH_DIR:=/mnt/hdd/dexora/health/start_align_ab}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${STEP:=2000}"

mkdir -p "$HEALTH_DIR"

for tag in A-oversample B-horizon C-combined; do
  root="$CKPT_ROOT/dexora-dexjoco-start-align-${tag}-2k"
  out="$HEALTH_DIR/${tag}_seed0_step${STEP}.json"
  log="$HEALTH_DIR/${tag}_seed0_step${STEP}.log"
  echo "==> health $tag"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python -m eval_sim.sample_health \
      --ckpt-root "$root" \
      --steps "$STEP" \
      --noise-seed 0 \
      --stats-file "$DEXORA_STATS" \
      --out "$out" \
      2>&1 | tee "$log"
done

python - <<PY
import json
from pathlib import Path

health_dir = Path("${HEALTH_DIR}")
step = "${STEP}"
for tag in ("A-oversample", "B-horizon", "C-combined"):
    p = health_dir / f"{tag}_seed0_step{step}.json"
    if not p.is_file():
        print(tag, "MISSING")
        continue
    d = json.loads(p.read_text())
    v = d.get("verdict", d)
    es = d.get("per_step", {}).get(str(step), {}).get("episode_start", {})
    xyz = es.get("first_action_xyz_jump_m", float("nan"))
    hand = es.get("first_action_hand_mean", float("nan"))
    print(
        f"{tag}: pass={v.get('pass')} start_align={v.get('start_align')} "
        f"xyz={xyz:.3f}m hand={hand:.3f}"
    )
PY
