import torch

from models.rdt.blocks import TimestepEmbedder


def test_timestep_embedder_follows_module_dtype() -> None:
    embedder = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8)
    embedder = embedder.to(dtype=torch.float32)
    output = embedder(torch.tensor([1]))
    assert output.dtype == torch.float32
