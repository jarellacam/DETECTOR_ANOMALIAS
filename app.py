import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.ingestion import descargar_ticker, cargar_raw
from src.data.preprocessing import pipeline
from src.models.autoencoder import AutoencoderLSTM
from src.evaluation.SHAP import calcular_errores, detectar_anomalias, explicar_anomalias
from src.models.trainer import Entrenador

# Condifguración de la página
st.set_page_config(
    page_title="Detector de anomalías financieras",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estiloa: tema oscuro
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    .stApp { background-color: #0a0e1a; }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
        color: #00d4ff;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f1629 0%, #1a2340 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .anomaly-value { color: #ff4466; }
    .sidebar .sidebar-content { background-color: #0f1629; }
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)
# TÍTULO PRINCIPAL
st.markdown("# Detector de Anomalías Financieras")
st.markdown("*Autoencoder LSTM con explicabilidad SHAP*")
st.markdown("---")

# Descripción del proyecto 
st.markdown("""
    ### ¿Qué hace esta aplicación?
    Este dashboard detecta comportamientos anómalos en activos financieros usando un 
    **Autoencoder LSTM** entrenado con datos históricos. El modelo aprende qué es normal 
    y lanza una alerta cuando detecta algo inusual.

    **Pipeline completo:**
    - Descarga datos reales de Yahoo Finance
    - Calcula features: Returns, Media Móvil y Volatilidad  
    - Detecta anomalías con error de reconstrucción
    - Explica el motivo con análisis SHAP

    Configura el análisis en el panel izquierdo y pulsa **Analizar**.
    """)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("CONFIGURACIóN")    

    ticker = st.text_input(
        "Ticker del activo",
        value = "AAPL",
        help = "Ejemplos: AAPL, MSFT, BTC-USD, TSLA"
    ).upper()

    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Desde", value=pd.to_datetime("2020-01-01"))
    with col2:
        fecha_fin = st.date_input("Hasta", value=pd.to_datetime("2024-01-01"))

    st.markdown("Parámetros del modelo")

    n_sigmas = st.slider(
        "Sensibilidad del umbral",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="A menor valor, más anomalías detectadas."
    )

    dim_latente = st.selectbox(
        "Dimensión latente",
        options = [16,32,64],
        index=1,
        help="Tamaño de la representación comprimida de datos del Autoencoder (Vector latente)"
    )

    analizar = st.button("Analizar", use_container_width=True, type="primary")
# Lógica principal
if analizar:
    with st.spinner(f"Descargando datos de {ticker}..."):
        try:
            df = descargar_ticker(
                ticker,
                inicio=str(fecha_inicio),
                fin=str(fecha_fin),
                guardar=True
            )
        except ValueError as e:
            st.error(f"ERROR: {e}")
            st.stop()

    with st.spinner("Procesando datos y entrenando el modelo..."):
        X, scaler = pipeline(df)
        modelo = AutoencoderLSTM(n_features=3, dim_latente=dim_latente, seq_len=30)

        # Cargamos el modelo ya existente
        modelo_path = Path("models") / f"autoencoder_{ticker}.pt"
        if modelo_path.exists():
            modelo.load_state_dict(torch.load(str(modelo_path), weights_only=True))
        else:
            # Entrenamos rápido si no hay datos guardados
            entrenador = Entrenador(modelo, lr=0.001, batch_size=32, paciencia=10)
            entrenador.entrenar(X, epochs=50)
            Path("models").mkdir(exist_ok=True)
            torch.save(modelo.state_dict(), str(modelo_path))

    with st.spinner("Detectando anomalías..."):
        errores = calcular_errores(modelo, X)
        ind_anomalias = detectar_anomalias(errores, n_sigmas=n_sigmas)
        umbral = errores.mean() + n_sigmas * errores.std()

        # Fechas alineadas con las secuencias
        offset = len(df) - len(errores)
        fechas = df.index[offset:][:len(errores)]

    # Resumen Métricas
    st.markdown("### Resumen del análisis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Días analizados</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(errores)}</div>
            <div class="metric-label">Secuencias</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value anomaly-value">{len(ind_anomalias)}</div>
            <div class="metric-label">Anomalías detectadas</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        pct = len(ind_anomalias) / len(errores) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value anomaly-value">{pct:.1f}%</div>
            <div class="metric-label">% anómalo</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfica 1: Precio con anomalías

    st.markdown("### Precio de cierre y anomalías detectadas")

    fechas_anomalias = fechas[ind_anomalias]
    precios_anomalias = df.loc[fechas_anomalias, "Close"] if len(ind_anomalias) > 0 else []

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        name="Precio cierre",
        line=dict(color="#00d4ff", width=1.5)
    ))
    if len(ind_anomalias) > 0:
        fig1.add_trace(go.Scatter(
            x=fechas_anomalias,
            y=precios_anomalias,
            mode="markers",
            name="Anomalía",
            marker=dict(color="#ff4466", size=8, symbol="circle")
        ))
    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1629",
        plot_bgcolor="#0a0e1a",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfica 2: Error de reconstrucción

    st.markdown("### Error de reconstrucción del Autoencoder")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=fechas, y=errores,
        mode="lines",
        name="Error de reconstrucción",
        line=dict(color="#4488ff", width=1),
        fill="tozeroy",
        fillcolor="rgba(68,136,255,0.1)"
    ))
    fig2.add_hline(
        y=umbral,
        line_dash="dash",
        line_color="#ff4466",
        annotation_text=f"Umbral ({umbral:.4f})",
        annotation_position="top right"
    )
    if len(ind_anomalias) > 0:
        fig2.add_trace(go.Scatter(
            x=fechas[ind_anomalias],
            y=errores[ind_anomalias],
            mode="markers",
            name="Anomalía",
            marker=dict(color="#ff4466", size=8)
        ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1629",
        plot_bgcolor="#0a0e1a",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Análisis SHAP
    if len(ind_anomalias) > 0:
        st.markdown("### Explicabilidad SHAP — ¿Por qué son anómalas?")

        with st.spinner("Calculando contribuciones SHAP..."):
            resultado_shap = explicar_anomalias(modelo, X, ind_anomalias, n_back=30)

        contribuciones = resultado_shap["Contribuciones"]
        features = list(contribuciones.keys())
        valores = list(contribuciones.values())
        total = sum(valores)
        porcentajes = [v / total * 100 for v in valores]

        col1, col2 = st.columns(2)

        with col1:
            fig_bar = go.Figure(go.Bar(
                x=valores,
                y=features,
                orientation="h",
                marker=dict(
                    color=["#ff4466" if v == max(valores) else "#4488ff" for v in valores]
                ),
                text=[f"{v:.6f}" for v in valores],
                textposition="outside"
            ))
            fig_bar.update_layout(
                title="Contribución media por feature",
                template="plotly_dark",
                paper_bgcolor="#0f1629",
                plot_bgcolor="#0a0e1a",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = go.Figure(go.Pie(
                labels=features,
                values=porcentajes,
                hole=0.4,
                marker=dict(colors=["#ff4466", "#4488ff", "#00d4ff"])
            ))
            fig_pie.update_layout(
                title="Distribución porcentual",
                template="plotly_dark",
                paper_bgcolor="#0f1629",
                plot_bgcolor="#0a0e1a",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Interpretación automática
        feature_principal = max(contribuciones, key=contribuciones.get)
        pct_principal = contribuciones[feature_principal] / total * 100
        st.info(
            f" **Interpretación:** La feature más determinante en las anomalías detectadas es "
            f"**{feature_principal}**, que explica el **{pct_principal:.1f}%** del error de reconstrucción. "
            f"Esto sugiere que las anomalías se caracterizan principalmente por comportamientos "
            f"inusuales en {'la volatilidad del activo' if feature_principal == 'Volatilidad' else 'los movimientos de precio diarios' if feature_principal == 'Return' else 'la tendencia del precio'}."
        )
    else:
        st.success("No se detectaron anomalías con el umbral seleccionado. Prueba a reducir la sensibilidad (σ).")

else:
    # Estado inicial — instrucciones
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #8899aa;">
        <h2 style="color: #4488ff; font-family: 'IBM Plex Mono', monospace;">
            ¿Cómo usar este dashboard?
        </h2>
        <p style="font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.8;">
            1. Introduce el <b>ticker</b> del activo financiero que quieres analizar<br>
            2. Selecciona el <b>rango de fechas</b><br>
            3. Ajusta la <b>sensibilidad</b> del detector<br>
            4. Pulsa <b>Analizar</b> y espera los resultados
        </p>
        <br>
        <p style="color: #4466aa; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;">
            Tickers de ejemplo: AAPL · MSFT · TSLA · BTC-USD · AMZN · GOOGL
        </p>
    </div>
    """, unsafe_allow_html=True)
