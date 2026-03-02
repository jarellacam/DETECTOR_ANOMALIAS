---
title: Anomaly Detector
emoji: 📈
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# Anomaly Detector — Financial Time Series
Sistema de detección de anomalías en series temporales financieras basado en un **Autoencoder LSTM** construido en PyTorch, con explicabilidad via SHAP y seguimiento de experimentos con MLflow.

## Motivación

Los modelos de detección de anomalías son críticos en finanzas para identificar comportamientos atípicos en precios o volúmenes. Este proyecto va más allá de la detección: **explica por qué** una observación es anómala, lo que lo hace aplicable en entornos reales donde la interpretabilidad es obligatoria.

## Stack tecnológico

| Área | Tecnología |
|---|---|
| Modelado | PyTorch (Autoencoder LSTM) |
| Datos | yfinance (Yahoo Finance API) |
| Explicabilidad | SHAP |
| Tracking de experimentos | MLflow |
| Despliegue | Docker + Streamlit |
| Testing | pytest |

## Estructura del proyecto

```
anomaly-detector/
├── data/
│   ├── raw/              # Datos originales de la API (no modificar)
│   └── processed/        # Datos listos para entrenar
├── notebooks/
│   └── exploration/      # EDA y experimentos
├── src/
│   ├── data/
│   │   ├── ingestion.py      # Descarga de datos via API
│   │   └── preprocessing.py  # Limpieza y preparación de secuencias
│   ├── models/
│   │   ├── autoencoder.py    # Arquitectura del modelo
│   │   └── trainer.py        # Loop de entrenamiento
│   ├── evaluation/
│   │   └── SHAP.py # Análisis SHAP
├── app.py                # Punto de entrada Streamlit
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/jarellacam/DETECTOR_ANOMALIAS.git
cd DETECTOR_ANOMALIAS

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Lanzar el dashboard
streamlit run app.py

# O con Docker
docker-compose up -d
```

## Cómo funciona

1. **Ingesta**: Se descargan series temporales reales de Yahoo Finance (precio, volumen, medias móviles).
2. **Preprocesamiento**: Normalización y construcción de ventanas temporales (secuencias).
3. **Entrenamiento**: Un Autoencoder LSTM aprende a reconstruir el comportamiento *normal* del activo.
4. **Detección**: Las observaciones con error de reconstrucción alto son candidatas a anomalía.
5. **Explicabilidad**: SHAP identifica qué features contribuyeron más a cada anomalía.
6. **Visualización**: Dashboard interactivo para explorar resultados en cualquier activo financiero.

## Resultados

Análisis sobre Apple (AAPL) 2020-2024:

- **957 secuencias** analizadas sobre 4 años de datos históricos
- **39 anomalías detectadas** (4.1%) con umbral σ=2.5
- Las anomalías coinciden con eventos reales: crash COVID (2020), corrección 2022, volatilidad 2023
- **Volatilidad** es la feature más determinante (60% del error SHAP), seguida de Return (25%) y MA20 (15%)

## Demo

[Prueba el dashboard en vivo](https://huggingface.co/spaces/jarellacam/DETECTOR_ANOMALIAS)

---

Proyecto desarrollado como parte del portfolio de Juan Arellano — Data Science & MLOps.
