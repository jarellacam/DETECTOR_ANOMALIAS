import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

CARACT = ["Return" , "ME20" , "Volatilidad"]
SEQ = 30

def calcular_caract(df: pd.DataFrame, ventana: int = 20) -> pd.DataFrame:
    """
    Calcula las caract a partir de los datos crudos
    """
    if "Close" not in df.columns:
        raise ValueError("El DataFrame debe tener la columna 'Close")
    
    logger.info(f"Calculando caract con ventana={ventana}")
    resultado = pd.DataFrame(index=df.index)

    # Usamos pct_change() para la fila Return
    resultado["Return"] = df["Close"].pct_change()

    # Sacamos ME20 con rolling y mean
    resultado["ME20"] = df["Close"].rolling(window=ventana).mean()

    # Volatilidad con rolling y std
    resultado["Volatilidad"] = df["Close"].rolling(window=ventana).std()

    # Eliminamos filas con NaN
    filas_antes = len(resultado)
    resultado = resultado.dropna()
    filas_despues = len(resultado)

    logger.info(
        f"Características calculadas: Filas: {filas_antes} -> {filas_despues}"
        f"(eliminadas {filas_antes} - {filas_despues} por NaN)"
    )

    return resultado

def normalizar(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normaliza las características a un rango [0,1] con MinMaxScaler
    """
    columnas_faltantes = [c for c in CARACT if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {columnas_faltantes}")
    
    logger.info("Normalización de características")
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[CARACT] = scaler.fit_transform(df[CARACT])
    return df_norm, scaler

def construir_secuencias(datos: np.ndarray, seq: int = SEQ) -> np.ndarray:
    """
    Construye secuencias temporales por medio de ventanas deslizantes
    array 2D a 3D
    """
    if len(datos) <= seq:
        raise ValueError(
            f"El array tiene {len(datos)} filas pero seq={seq}."
        )
    secuencias = []
    for i in range(len(datos) - seq):
        secuencias.append(datos[i : i + seq])
    resultado = np.array(secuencias)
    logger.info(
        f"Secuencias construidas: {resultado.shape[0]} secuencias "
        f"de {resultado.shape[1]} días × {resultado.shape[2]} features"
    )
    return resultado

def pipeline(
        df: pd.DataFrame,
        ventana: int = 20,
        seq: int = SEQ
) -> tuple[np.ndarray, MinMaxScaler]:
    logger.info("Iniciando pipeline de preprocesamiento...")
    df_features = calcular_caract(df, ventana=ventana)
    df_normalizado, scaler = normalizar(df_features)
    X = construir_secuencias(df_normalizado[CARACT].values, seq)
    logger.info(f"Pipeline completado. Tamaño final: {X.shape}")

    return X, scaler
