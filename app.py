import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar el clasificador de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

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

@app.route('/detect_finger', methods=['POST'])
def detect_finger():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    # Convertir a RGB (porque MediaPipe usa RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Hands
    results = hands.process(image_rgb)

    def detectar_indice_levantado_solo(hand_landmarks):
        """ Detecta si SOLO el índice está levantado """
        y_indice = hand_landmarks.landmark[8].y  # Punta del índice
        y_base_indice = hand_landmarks.landmark[5].y  # Base del índice
        y_medio = hand_landmarks.landmark[12].y  # Punta del medio
        y_base_medio = hand_landmarks.landmark[9].y  # Base del medio
        y_anular = hand_landmarks.landmark[16].y
        y_menique = hand_landmarks.landmark[20].y

        if (y_indice < y_base_indice and  # Índice levantado
            y_medio > y_base_medio and  # Medio abajo
            y_anular > hand_landmarks.landmark[13].y and  # Anular abajo
            y_menique > hand_landmarks.landmark[17].y):  # Meñique abajo
            return True
        return False

    # Verificar si alguien tiene SOLO el índice levantado
    indice_levantado_solo = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detectar_indice_levantado_solo(hand_landmarks):
                indice_levantado_solo = True

    return jsonify({"index_finger_up": indice_levantado_solo})
