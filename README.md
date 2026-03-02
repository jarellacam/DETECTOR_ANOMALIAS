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
│   │   └── explainability.py # Análisis SHAP
│   └── visualization/
│       └── dashboard.py      # Componentes Streamlit
├── tests/                # Tests unitarios
├── app.py                # Punto de entrada Streamlit
├── train.py              # Script de entrenamiento CLI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/anomaly-detector.git
cd anomaly-detector

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales si usas APIs de pago
```

## Uso

```bash
# Entrenar el modelo
python train.py --ticker AAPL --start 2020-01-01 --end 2024-01-01

# Lanzar el dashboard
streamlit run app.py

# Ejecutar tests
pytest tests/
```

## Cómo funciona

1. **Ingesta**: Se descargan series temporales reales de Yahoo Finance (precio, volumen, medias móviles).
2. **Preprocesamiento**: Normalización y construcción de ventanas temporales (secuencias).
3. **Entrenamiento**: Un Autoencoder LSTM aprende a reconstruir el comportamiento *normal* del activo.
4. **Detección**: Las observaciones con error de reconstrucción alto son candidatas a anomalía.
5. **Explicabilidad**: SHAP identifica qué features y en qué instantes temporales contribuyeron a la anomalía.
6. **Visualización**: Dashboard interactivo para explorar resultados.

## Resultados

*(Esta sección se completará conforme avance el proyecto)*

---

Proyecto desarrollado como parte del portfolio de Juan Arellano — Data Science & MLOps.
