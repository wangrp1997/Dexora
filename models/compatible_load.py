"""Load Stage-1 RDT weights into a possibly wider I/O model.

36-D AIRBOT joint layout and 44-D DexJoCo TCP+Allegro are **not** the same
skill space. On expand we copy every shape-matched tensor (backbone, lang/img
adaptors, later state-adaptor MLP layers, ``final_layer.norm_final`` / ``fc1``)
and only rebuild:

* ``state_adaptor.0``  (input 72 -> 88)
* ``model.final_layer.ffn_final.fc2``  (output 36 -> 44)

Rebuilt weights are ``N(0, src.std)`` with zero bias — a scale-matched cold
start, **not** a sampling-stability guarantee. Gate the raw diffusion output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Only the two layers whose *shapes* change on 36 -> 44. Trailing dots avoid
# matching ``state_adaptor.10`` etc.
DEFAULT_REINIT_PREFIXES: Tuple[str, ...] = (
    "state_adaptor.0.",
    "model.final_layer.ffn_final.fc2.",
)


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Prefer a directory that actually contains weights."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"pretrained path is not a directory: {path}")
    for name in ("pytorch_model.bin", "model.safetensors"):
        if (root / name).is_file():
            return root
    ckpts = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for cand in reversed(ckpts):
        for name in ("pytorch_model.bin", "model.safetensors"):
            if (cand / name).is_file():
                return cand
    raise FileNotFoundError(f"no pytorch_model.bin / model.safetensors under {path}")


def peek_action_dim(pretrained_path: str | Path) -> Optional[int]:
    """Read ``action_dim`` from a checkpoint ``config.json`` if present."""
    root = Path(pretrained_path)
    candidates = [root / "config.json"]
    if root.is_dir():
        candidates.extend(sorted(root.glob("checkpoint-*/config.json")))
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        if "action_dim" in cfg:
            return int(cfg["action_dim"])
    return None


def _load_raw_state_dict(ckpt_dir: Path, map_location: str = "cpu") -> dict:
    bin_path = ckpt_dir / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location=map_location)
    st_path = ckpt_dir / "model.safetensors"
    if st_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(st_path), device=map_location)
    raise FileNotFoundError(f"no weights in {ckpt_dir}")


def _tensor_std(t: torch.Tensor) -> float:
    return float(t.detach().float().std().clamp_min(1e-8).item())


def _scale_init_like_src(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Fill ``dst`` with N(0, src.std) for weights; zeros for bias."""
    if dst.ndim >= 2:
        std = _tensor_std(src)
        try:
            nn.init.normal_(dst, mean=0.0, std=std, generator=generator)
        except TypeError:
            noise = torch.randn(dst.shape, generator=generator, dtype=torch.float32)
            dst.copy_((noise * std).to(dst.dtype))
        return std
    nn.init.zeros_(dst)
    return 0.0


def _assign_param(model: nn.Module, key: str, src: torch.Tensor, generator: torch.Generator) -> float:
    parts = key.split(".")
    mod: nn.Module | torch.Tensor = model
    for p in parts[:-1]:
        if p.isdigit():
            mod = mod[int(p)]  # type: ignore[index]
        else:
            mod = getattr(mod, p)
    leaf = parts[-1]
    dst = getattr(mod, leaf)
    if not isinstance(dst, torch.Tensor):
        raise AttributeError(f"{key} is not a tensor")
    with torch.no_grad():
        return _scale_init_like_src(dst, src, generator=generator)


def _load_partial_copy_map(path: str | Path) -> dict[str, dict[int, int]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"partial-copy map must be a JSON object: {path}")
    result: dict[str, dict[int, int]] = {}
    for name in ("state", "action"):
        entries = raw.get(name, {})
        if not isinstance(entries, dict):
            raise ValueError(f"partial-copy map '{name}' must be an object")
        parsed: dict[int, int] = {}
        for target, source in entries.items():
            target_index, source_index = int(target), int(source)
            if target_index < 0 or source_index < 0:
                raise ValueError("partial-copy indices must be non-negative")
            parsed[target_index] = source_index
        result[name] = parsed
    return result


def _copy_partial_io(
    model: nn.Module,
    src: dict,
    dst: dict,
    mapping: dict[str, dict[int, int]],
) -> set[str]:
    """Copy only explicitly mapped state/action dimensions into rebuilt I/O."""
    copied: set[str] = set()
    state_map = mapping["state"]
    action_map = mapping["action"]

    state_key = "state_adaptor.0.weight"
    if state_key in src and state_key in dst:
        source, target = src[state_key], dst[state_key]
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(f"incompatible state adaptor shapes: {source.shape} -> {target.shape}")
        if source.shape[1] % 2 or target.shape[1] % 2:
            raise ValueError("state adaptor input width must contain state and indicator halves")
        source_dim, target_dim = source.shape[1] // 2, target.shape[1] // 2
        with torch.no_grad():
            for target_index, source_index in state_map.items():
                if target_index >= target_dim or source_index >= source_dim:
                    raise ValueError(
                        f"state mapping out of range: target={target_index}/{target_dim}, "
                        f"source={source_index}/{source_dim}"
                    )
                target[:, target_index] = source[:, source_index]
                target[:, target_dim + target_index] = source[:, source_dim + source_index]
        copied.add(state_key)

    action_key = "model.final_layer.ffn_final.fc2.weight"
    if action_key in src and action_key in dst:
        source, target = src[action_key], dst[action_key]
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError(f"incompatible action head shapes: {source.shape} -> {target.shape}")
        with torch.no_grad():
            for target_index, source_index in action_map.items():
                if target_index >= target.shape[0] or source_index >= source.shape[0]:
                    raise ValueError(
                        f"action mapping out of range: target={target_index}/{target.shape[0]}, "
                        f"source={source_index}/{source.shape[0]}"
                    )
                target[target_index] = source[source_index]
        copied.add(action_key)

    action_bias_key = "model.final_layer.ffn_final.fc2.bias"
    if action_bias_key in src and action_bias_key in dst:
        source, target = src[action_bias_key], dst[action_bias_key]
        if source.ndim != 1 or target.ndim != 1:
            raise ValueError(f"incompatible action bias shapes: {source.shape} -> {target.shape}")
        with torch.no_grad():
            for target_index, source_index in action_map.items():
                if target_index >= target.shape[0] or source_index >= source.shape[0]:
                    raise ValueError(
                        f"action bias mapping out of range: target={target_index}/{target.shape[0]}, "
                        f"source={source_index}/{source.shape[0]}"
                    )
                target[target_index] = source[source_index]
        copied.add(action_bias_key)

    if "state_adaptor.0.bias" in src and tuple(src["state_adaptor.0.bias"].shape) == tuple(dst["state_adaptor.0.bias"].shape):
        with torch.no_grad():
            dst["state_adaptor.0.bias"].copy_(src["state_adaptor.0.bias"])
        copied.add("state_adaptor.0.bias")
    return copied


def is_fresh_io_key(key: str, prefixes: Sequence[str] = DEFAULT_REINIT_PREFIXES) -> bool:
    return any(key.startswith(p) for p in prefixes)


def freeze_except_fresh_io(
    model: nn.Module,
    prefixes: Sequence[str] = DEFAULT_REINIT_PREFIXES,
) -> tuple[list[str], list[str]]:
    """Freeze inherited parameters and keep only rebuilt I/O trainable."""
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        is_fresh = is_fresh_io_key(name, prefixes)
        param.requires_grad_(is_fresh)
        (trainable if is_fresh else frozen).append(name)
    if not trainable:
        raise ValueError("no fresh I/O parameters matched the configured prefixes")
    return trainable, frozen


def load_compatible_pretrained(
    model: nn.Module,
    pretrained_path: str | Path,
    *,
    force_reinit_prefixes: Sequence[str] = DEFAULT_REINIT_PREFIXES,
    map_location: str = "cpu",
    init_seed: int = 0,
    partial_copy_map: str | Path | None = None,
) -> Tuple[list, list]:
    """Copy shape-matched weights; scale-init forced I/O layers from src std.

    Returns ``(loaded_keys, skipped)`` where skipped is a list of
    ``(key, reason)``.
    """
    ckpt_dir = resolve_checkpoint_dir(pretrained_path)
    src = _load_raw_state_dict(ckpt_dir, map_location=map_location)
    dst = model.state_dict()
    loaded: dict = {}
    skipped: list = []
    prefixes = tuple(force_reinit_prefixes)
    mapping = _load_partial_copy_map(partial_copy_map) if partial_copy_map else None

    for key, value in src.items():
        if is_fresh_io_key(key, prefixes):
            if mapping and key in {
                "state_adaptor.0.weight",
                "state_adaptor.0.bias",
                "model.final_layer.ffn_final.fc2.weight",
            }:
                continue
            skipped.append((key, "force_reinit"))
            continue
        if key not in dst:
            skipped.append((key, "missing_in_dst"))
            continue
        if tuple(dst[key].shape) != tuple(value.shape):
            skipped.append(
                (key, f"shape {tuple(value.shape)} -> {tuple(dst[key].shape)}")
            )
            continue
        loaded[key] = value

    model.load_state_dict(loaded, strict=False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(init_seed))
    inited: list[tuple[str, float]] = []
    for key, reason in skipped:
        if reason not in ("force_reinit",) and not str(reason).startswith("shape"):
            continue
        if key not in dst or key not in src:
            continue
        std = _assign_param(model, key, src[key], generator)
        inited.append((key, std))

    copied_partial: set[str] = set()
    if mapping:
        copied_partial = _copy_partial_io(model, src, model.state_dict(), mapping)
        logger.info(
            "Partial-copy I/O map %s: state=%d action=%d tensors=%s",
            partial_copy_map,
            len(mapping["state"]),
            len(mapping["action"]),
            sorted(copied_partial),
        )

    logger.info(
        "Compatible load from %s: loaded=%d skipped=%d scale_init=%d",
        ckpt_dir,
        len(loaded),
        len(skipped),
        len(inited),
    )
    reinit = [k for k, r in skipped if r == "force_reinit"]
    shape_skip = [k for k, r in skipped if r.startswith("shape")]
    if reinit:
        logger.info("Scale-init (not copied): %s", reinit)
    if inited:
        logger.info(
            "Scale-init stds: %s",
            ", ".join(f"{k}={s:.5f}" for k, s in inited),
        )
    if shape_skip:
        logger.info("Shape-skipped: %s", shape_skip[:8])
    return list(loaded.keys()), skipped
