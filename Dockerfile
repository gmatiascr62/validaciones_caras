# Imagen base más liviana con Alpine
FROM python:3.10-alpine

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias necesarias para OpenCV y otras bibliotecas
RUN apk add --no-cache \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libjpeg-turbo \
    tiff \
    libpng \
    openblas \
    && rm -rf /var/cache/apk/*  # Limpia archivos innecesarios

# Copia los archivos de la app
COPY . /app

# Instala las dependencias de Python
RUN pip install --no-cache-dir flask opencv-python-headless numpy gunicorn

# Expone el puerto 8080 para Cloud Run
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
