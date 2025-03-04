import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

#Cargar el clasificador de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return jsonify({"face_detected": len(faces) > 0})


