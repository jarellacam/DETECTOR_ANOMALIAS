import pytest

torch = pytest.importorskip("torch")

from src.models.autoencoder import AutoencoderLSTM


def test_reconstruccion_misma_forma():
    modelo = AutoencoderLSTM(n_features=3, dim_latente=8, seq_len=30)
    x = torch.rand(4, 30, 3)
    assert modelo(x).shape == x.shape
