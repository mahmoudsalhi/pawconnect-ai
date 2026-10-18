import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from pawmatch.model import ArcFace, EmbeddingNet


def test_embedding_is_unit_norm_and_trains_one_step():
    model = EmbeddingNet("resnet18", embed_dim=32, pretrained=False)
    arc = ArcFace(32, num_classes=4)
    x = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 4, (8,))
    emb = model(x)
    assert emb.shape == (8, 32)
    assert torch.allclose(emb.norm(dim=1), torch.ones(8), atol=1e-4)
    loss = arc(emb, y)
    loss.backward()
    assert torch.isfinite(loss)
