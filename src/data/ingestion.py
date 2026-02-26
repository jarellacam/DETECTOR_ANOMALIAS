import yfinance as yf
import pandas as pd
from pathlib import Path
from loguru import logger

DIR_DATA = Path(__file__).resolve().parents[2] / "data" / "raw"
# Columnas que nos interesan del API de Yahoo Finance
COLUMNAS = ["Open", "High", "Low", "Close", "Volume"]

def descargar_ticker(ticker, inicio, fin, guardar = True):
    """
    Descarga datos históricos de un activo financiero desde 
    Yahoo Finance.
    """
    logger.info(f"Descargando {ticker} desde {inicio} hasta {fin}...")
    
    df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True)


    if df.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")
    
    # yfinance devuelve MultiÍnidice al descragar un solo ticker
    df = aplanar_multiIndice(df)

    df.index.name = "Date"
    logger.info(f"Descargados {len(df)} registros para {ticker}.")

    if guardar:
        guardar_raw(df, ticker)
    return df

def aplanar_multiIndice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas de dos niveles en columnas simples de un solo nivel
    """
    if isinstance(df.columns, pd.MultiIndex):
        # get_level_values(0) se queda con el primer nivel: 'Open', 'High', etc.
        df.columns = df.columns.get_level_values(0)

    # Nos quedamos solo con las columnas que nos interesan
    return df[COLUMNAS]


def guardar_raw(df, ticker):
    """
    Guarda el DF en data/raw/ como CSV
    """
    DIR_DATA.mkdir(parents=True , exist_ok=True)
    path = DIR_DATA / f"{ticker}.csv"
    df.to_csv(path)
    logger.info(f"Datos guardados en {path}")

def cargar_raw(ticker):
    """
    Carga los datos ya descargados
    """
    path = DIR_DATA / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta download_ticker() primero."
        )

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df
