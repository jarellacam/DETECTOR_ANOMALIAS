import numpy as np
import torch
import torch.nn as nn
import shap
from loguru import logger

CARACT = ["Return" , "ME20" , "Volatilidad"]

def calcular_errores(modelo: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Calcula el error de reconstrucción en cada secuencia
    """
    modelo.eval()
    X_tensor = torch.FloatTensor(X)

    # no nos interesa el backward en este proceso, por ello no incluimos gradientes
    with torch.no_grad():
        reconstruc = modelo(X_tensor)

    # sacamos MSE por secuencia
    errores = torch.mean( (X_tensor - reconstruc) ** 2, dim=(1,2))
    errores = errores.numpy()

    logger.info(
        f"Errores calculados | "
        f"Media: {errores.mean():.6f} | "
        f"Max: {errores.max():.6f} | "
        f"Min: {errores.min():.6f}"
    )
    return errores


def detectar_anomalias(errores: np.ndarray, n_sigmas: float=2.0) -> np.ndarray:
    """
    Detecta anomalías usando el umbral
    """
    umbral = errores.mean() + n_sigmas * errores.std()
    ind_anomalias = np.where(errores > umbral)[0]

    logger.info(
        f"Umbral: {umbral:.6f} | "
        f"Anomalías detectadas: {len(ind_anomalias)} de {len(errores)}"
    )
    return ind_anomalias

def funcion_error(X_flat: np.ndarray, modelo: nn.Module, seq: int, n_feature: int) -> np.ndarray:
    X_3d = X_flat.reshape(-1, seq, n_feature)
    X_tensor = torch.FloatTensor(X_3d)
    modelo.eval()
    with torch.no_grad():
        reconstruc = modelo(X_tensor)

    # sacamos MSE por secuencia
    errores = torch.mean( (X_tensor - reconstruc) ** 2, dim=(1,2))
    errores = errores.numpy()
    return errores

def explicar_anomalias(
        modelo: nn.Module,
        X: np.ndarray,
        ind_anomalias: np.ndarray,
        n_back: int = 50
) -> dict:
    """
    Saca valores SHAP para las anomalías detectadas (usa KernerExplainer)
    """
    seq = X.shape[1]
    n_feature = X.shape[2]

    logger.info(f"Calculando SHAP para {len(ind_anomalias)} anomalías")

    # Aplanamos X para SHAP, de 3D a 2D
    X_flat = X.reshape(X.shape[0], -1)
    
    back = X_flat[np.random.choice(len(X_flat), n_back, replace=False)]

    # Función que SHAP va a explicar
    def explicacion(x): return funcion_error(x, modelo, seq, n_feature)

    explica = shap.KernelExplainer(explicacion, back)

    # Sacamos SHAP solo para anomalías
    X_anomalias = X_flat[ind_anomalias]
    valores_shap = explica.shap_values(X_anomalias, nsamples=100)
    if isinstance(valores_shap, list):
        valores_shap = valores_shap[0]

    # Agurpamos por características (features)
    sharp_por_feat = valores_shap.reshape(len(ind_anomalias), seq, n_feature)
    contribuciones = np.abs(sharp_por_feat).mean(axis = (0,1))

    resultado = {
        "Valores_SHAP": sharp_por_feat,
        "Contribuciones":dict(zip(CARACT, contribuciones))
    }

    logger.info(f"Contribuciones medias: {resultado['Contribuciones']}")

    return resultado