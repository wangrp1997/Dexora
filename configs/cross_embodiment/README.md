# Cross-Embodiment Fine-tuning Configs

These configs reproduce the three embodiment-projection experiments in
Dexora §IV-C (Tab. II, Fig. 9). The Dexora policy is trained natively in a
36-D dual-arm dual-hand action space, and lower-DoF embodiments are obtained
by **dimension reduction** — we keep the tensor shape fixed at 36-D and (a)
mask the unused action / state dimensions and (b) mask the absent camera.

```
36-D Dexora state vector  =  [ right_arm(6) | right_hand(12) | left_arm(6) | left_hand(12) ]
```

We map each embodiment into that canonical layout and set `state_elem_mask`
(per-sample) accordingly. The mask is consumed by `RDTRunner` exactly the
way it is in stage-1/stage-3 training, so no architectural change is needed.

| Config | Embodiment | DoF | Active Dexora indices |
|---|---|---|---|
| `ec1_franka.yaml`     | Franka Panda + 1-DoF gripper                       | 6 + 1   | right_arm[0:6], right_hand[0]       |
| `ec2_aloha.yaml`      | Cobot Magic ALOHA — 2 × (6-DoF arm + 1-DoF gripper) | 14      | right_arm[0:6], right_hand[0], left_arm[0:6], left_hand[0] |
| `ec3_g1_inspire.yaml` | Unitree G1 7-DoF arm + Inspire Hand 6-DoF          | 13      | right_arm[0:7]\*, right_hand[0:6]   |
| `ec4_dexjoco_bimanual_assembly.yaml` | DexJoCo dual Panda + dual Allegro (insert) | 44 | native 16-DoF hands; Stage-1 I/O is expanded from 36D to 44D |

\* For G1's 7-DoF arm we use one extra dim borrowed from the right-hand block;
this convention can be changed in the YAML if your retargeting differs.

## Cameras

* `ec1_franka.yaml`: a single wrist camera. We set
  `camera_active_mask = [0, 0, 1, 0]` (head, left-wrist, right-wrist, third-view)
  so that the dataset always returns a "background mean colour" image for
  the missing views (the existing `cond_mask_prob` / `cam_ext_mask_prob`
  mechanism in `train/dataset.py`).
* `ec2_aloha.yaml`: head + both wrist cameras → `[1, 1, 1, 0]`.
* `ec3_g1_inspire.yaml`: head + ego (third-view) → `[1, 0, 0, 1]`.
* `ec4_dexjoco_bimanual_assembly.yaml`: ego + both wrists (no front) → `[1, 1, 1, 0]`.
  Unlike the paper's lower-DoF EC examples, this adaptation preserves all Allegro joints:
`state46` is converted to native policy44. Fine-tuning expands Stage-1 I/O from 36-D
to 44-D by inheriting same-shape layers and scale-init of `state_adaptor.0` + `fc2` only.

## Usage

```bash
# Stage-1 / stage-3 / post-train scripts all accept --config_path.
# Cross-embodiment is purely a fine-tuning recipe: load the stage-3 Dexora
# checkpoint, then continue training under the embodiment-specific config.

bash post_train.sh \
    OUTPUT_DIR=checkpoints/dexora-ec1-franka \
    STAGE1_CKPT=checkpoints/dexora-400m-posttrain \
    # ↓ key: use the embodiment YAML
    CONFIG_PATH=configs/cross_embodiment/ec1_franka.yaml \
    --no_quality_weights     # cross-embodiment fine-tune is small (100 demos),
                             # the paper does not use the discriminator here.
```

DexJoCo insert (EC-4) after Stage-1 on Dexora `airbot_assemble`:

```bash
bash scripts/download_dexora_pretrain.sh          # FAMILY=airbot_assemble
bash scripts/s_dexora_pretrain_assemble.sh
PRETRAINED=checkpoints/dexora-400m-pretrain-assemble \
    bash scripts/s_dexjoco_finetune.sh
```

The released DexJoCo videos are AV1. On the current training host, OpenCV cannot
decode them and previously returned black frames. Run
`bash scripts/transcode_dexjoco_videos.sh` first; the fine-tuning launcher now
defaults to the validated H.264 mirror and aborts if any sampled camera is black.
