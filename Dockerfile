# Imagen base más liviana
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema mínimas para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*  # Elimina archivos innecesarios

# Copia los archivos de la app
COPY . /app

# Instala las dependencias de Python directamente
RUN pip install --no-cache-dir flask opencv-python numpy gunicorn

# Expone el puerto 8080 para Cloud Run
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
