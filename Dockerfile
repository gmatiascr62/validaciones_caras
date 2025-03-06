FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de la app, incluyendo el archivo 'localidades.csv'
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir flask opencv-python numpy gunicorn mediapipe polars

# Exponer el puerto 8080
EXPOSE 8080

# Comando para ejecutar la app con Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
