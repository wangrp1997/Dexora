"""36-D -> 44-D compatible load: inherit same-shape I/O, scale-init fc2 / adaptor.0."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.compatible_load import (
    DEFAULT_REINIT_PREFIXES,
    freeze_except_fresh_io,
    is_fresh_io_key,
    load_compatible_pretrained,
)


class _Tiny(nn.Module):
    """Mirrors RDTRunner I/O names used by compatible_load prefixes."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.lang_adaptor = nn.Linear(8, 8)
        self.img_adaptor = nn.Linear(8, 8)
        self.state_adaptor = nn.Sequential(
            nn.Linear(in_dim * 2, 8),
            nn.GELU(),
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Linear(8, 8),
        )
        self.model = nn.Module()
        self.model.final_layer = nn.Module()
        self.model.final_layer.norm_final = nn.Linear(8, 8)
        self.model.final_layer.ffn_final = nn.Module()
        self.model.final_layer.ffn_final.fc1 = nn.Linear(8, 8)
        self.model.final_layer.ffn_final.fc2 = nn.Linear(8, out_dim)


def _fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim >= 2:
                p.fill_(value)
            else:
                p.zero_()


def test_fresh_io_prefixes_do_not_hit_later_layers() -> None:
    assert is_fresh_io_key("state_adaptor.0.weight")
    assert is_fresh_io_key("model.final_layer.ffn_final.fc2.bias")
    assert not is_fresh_io_key("state_adaptor.2.weight")
    assert not is_fresh_io_key("state_adaptor.4.weight")
    assert not is_fresh_io_key("model.final_layer.norm_final.weight")
    assert not is_fresh_io_key("model.final_layer.ffn_final.fc1.weight")
    assert not is_fresh_io_key("lang_adaptor.weight")
    assert DEFAULT_REINIT_PREFIXES == (
        "state_adaptor.0.",
        "model.final_layer.ffn_final.fc2.",
    )


def test_freeze_except_fresh_io_only_leaves_rebuilt_layers_trainable() -> None:
    model = _Tiny(44, 44)
    trainable, frozen = freeze_except_fresh_io(model)

    assert set(trainable) == {
        "state_adaptor.0.weight",
        "state_adaptor.0.bias",
        "model.final_layer.ffn_final.fc2.weight",
        "model.final_layer.ffn_final.fc2.bias",
    }
    assert frozen
    for name, param in model.named_parameters():
        assert param.requires_grad == is_fresh_io_key(name)


def test_compatible_load_inherits_same_shape_reinit_io(tmp_path) -> None:
    src = _Tiny(36, 36)
    _fill(src.backbone, 1.0)
    _fill(src.lang_adaptor, 1.5)
    _fill(src.img_adaptor, 1.6)
    with torch.no_grad():
        src.state_adaptor[0].weight.fill_(2.0)
        src.state_adaptor[2].weight.fill_(2.2)
        src.state_adaptor[4].weight.fill_(2.4)
        src.model.final_layer.norm_final.weight.fill_(3.1)
        src.model.final_layer.ffn_final.fc1.weight.fill_(3.2)
        src.model.final_layer.ffn_final.fc2.weight.normal_(0.0, 0.007)

    src_fc2_std = float(src.model.final_layer.ffn_final.fc2.weight.std().item())
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save(src.state_dict(), ckpt / "pytorch_model.bin")

    dst = _Tiny(44, 44)
    _fill(dst, 0.0)

    loaded, skipped = load_compatible_pretrained(dst, ckpt, init_seed=0)
    assert "backbone.weight" in loaded
    assert torch.allclose(dst.backbone.weight, torch.ones_like(dst.backbone.weight))
    assert torch.allclose(dst.state_adaptor[2].weight, src.state_adaptor[2].weight)
    assert torch.allclose(dst.state_adaptor[4].weight, src.state_adaptor[4].weight)
    assert torch.allclose(
        dst.model.final_layer.norm_final.weight, src.model.final_layer.norm_final.weight
    )
    assert torch.allclose(
        dst.model.final_layer.ffn_final.fc1.weight,
        src.model.final_layer.ffn_final.fc1.weight,
    )

    assert any(k.endswith("state_adaptor.0.weight") or k == "state_adaptor.0.weight" for k, _ in skipped)
    assert any("ffn_final.fc2.weight" in k for k, _ in skipped)
    assert torch.count_nonzero(dst.state_adaptor[0].weight) > 0
    dst_fc2_std = float(dst.model.final_layer.ffn_final.fc2.weight.float().std().item())
    assert dst_fc2_std > 1e-4
    assert abs(dst_fc2_std - src_fc2_std) / src_fc2_std < 0.5
    assert torch.count_nonzero(dst.model.final_layer.ffn_final.fc2.bias) == 0


def test_partial_copy_uses_only_explicit_state_and_action_mapping(tmp_path) -> None:
    src = _Tiny(3, 3)
    with torch.no_grad():
        src.state_adaptor[0].weight.copy_(torch.arange(48, dtype=torch.float32).reshape(8, 6))
        src.state_adaptor[0].bias.fill_(7.0)
        src.model.final_layer.ffn_final.fc2.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(3, 8))
        src.model.final_layer.ffn_final.fc2.bias.copy_(torch.tensor([8.0, 9.0, 10.0]))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save(src.state_dict(), ckpt / "pytorch_model.bin")
    mapping = tmp_path / "map.json"
    mapping.write_text(
        '{"state": {"0": 2, "2": 1}, "action": {"1": 0, "3": 2}}\n'
    )

    dst = _Tiny(4, 4)
    _fill(dst, 0.0)
    load_compatible_pretrained(dst, ckpt, partial_copy_map=mapping, init_seed=0)

    assert torch.allclose(dst.state_adaptor[0].weight[:, 0], src.state_adaptor[0].weight[:, 2])
    assert torch.allclose(dst.state_adaptor[0].weight[:, 2], src.state_adaptor[0].weight[:, 1])
    assert torch.allclose(dst.state_adaptor[0].weight[:, 4], src.state_adaptor[0].weight[:, 5])
    assert torch.allclose(dst.state_adaptor[0].bias, src.state_adaptor[0].bias)
    assert torch.allclose(dst.model.final_layer.ffn_final.fc2.weight[1], src.model.final_layer.ffn_final.fc2.weight[0])
    assert torch.allclose(dst.model.final_layer.ffn_final.fc2.weight[3], src.model.final_layer.ffn_final.fc2.weight[2])
    assert torch.allclose(dst.model.final_layer.ffn_final.fc2.bias[1], torch.tensor(8.0))
