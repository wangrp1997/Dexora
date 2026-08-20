#!/usr/bin/env bash
# Matched 5k ablation: relative wrist rotvec representation + epsilon target.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export CONFIG_PATH="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly_epsilon.yaml"
export DEXORA_STATS="${DEXORA_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-epsilon-5k}"
export MAX_TRAIN_STEPS=5000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=500,1000,2500,5000
export SAVE_INITIAL_CHECKPOINT=1
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"

LOG_DIR="${LOG_DIR:-/mnt/hdd/dexora/logs}"
RUN_LOG="$LOG_DIR/dexjoco_relative_rot_epsilon_5k_train.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "==> matched ablation: relative rotvec + epsilon, 5k (GPU ${CUDA_VISIBLE_DEVICES})"
bash scripts/s_dexjoco_finetune.sh 2>&1 | tee "$RUN_LOG"

for seed in 0 1 2; do
  echo "==> sample health seed=${seed}"
  set +e
  python -m eval_sim.sample_health \
    --ckpt-root "$OUTPUT_DIR" \
    --steps 0,500,1000,2500,5000 \
    --noise-seed "$seed" \
    --out "$LOG_DIR/dexjoco_relative_rot_epsilon_5k_health_seed${seed}.json" \
    > "$LOG_DIR/dexjoco_relative_rot_epsilon_5k_health_seed${seed}.log" 2>&1
  status=$?
  set -e
  echo "    health seed=${seed} exit=${status}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/hdd/dexora/logs")
files = [root / f"dexjoco_relative_rot_epsilon_5k_health_seed{seed}.json" for seed in range(3)]
summary = {}
for path in files:
    if not path.is_file():
        summary[path.stem] = {"missing": True}
        continue
    payload = json.loads(path.read_text())
    summary[path.stem] = {
        "verdict": payload["verdict"],
        "step_5000": payload["per_step"].get("5000"),
    }
(root / "dexjoco_relative_rot_epsilon_5k_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

echo "==> ablation complete: $OUTPUT_DIR"
