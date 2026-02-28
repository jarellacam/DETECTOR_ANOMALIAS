FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero solo el requirements.txt para aprovechar la caché de Docker.
# Si el código cambia pero las dependencias no, Docker no reinstala todo.
COPY requirements.txt .

# Instalamos dependencias
# --no-cache-dir reduce el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Creamos las carpetas necesarias en tiempo de build
RUN mkdir -p data/raw data/processed models mlruns

# Puerto que expone Streamlit
EXPOSE 8501

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Comando de inicio
CMD ["streamlit", "run", "app.py"]