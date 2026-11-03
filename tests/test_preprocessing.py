import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import CARACT, calcular_caract, construir_secuencias, normalizar, pipeline


@pytest.fixture
def precios():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, 200))
    return pd.DataFrame({"Close": close}, index=pd.date_range("2024-01-01", periods=200))


def test_calcular_caract_elimina_nans(precios):
    df = calcular_caract(precios, ventana=20)
    assert list(df.columns) == CARACT
    assert not df.isna().any().any()
    assert len(df) == len(precios) - 19


def test_calcular_caract_sin_close():
    with pytest.raises(ValueError):
        calcular_caract(pd.DataFrame({"Open": [1, 2, 3]}))


def test_normalizar_rango_0_1(precios):
    df_norm, _ = normalizar(calcular_caract(precios))
    assert df_norm[CARACT].min().min() >= 0
    assert df_norm[CARACT].max().max() <= 1


def test_construir_secuencias_forma():
    datos = np.arange(50 * 3, dtype=float).reshape(50, 3)
    X = construir_secuencias(datos, seq=10)
    assert X.shape == (40, 10, 3)
    np.testing.assert_array_equal(X[0], datos[:10])


def test_construir_secuencias_demasiado_corta():
    with pytest.raises(ValueError):
        construir_secuencias(np.zeros((5, 3)), seq=10)


def test_pipeline_completo(precios):
    X, _ = pipeline(precios, ventana=20, seq=30)
    assert X.ndim == 3
    assert X.shape[1:] == (30, len(CARACT))
