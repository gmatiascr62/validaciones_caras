
import cv2
import numpy as np
import mediapipe as mp
import polars as pl
import math
from flask import Flask, request, jsonify

app = Flask(__name__)

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Cargar los datos de las localidades
df_localidades = pl.read_csv('localidades.csv')

def haversine(lat1, lon1, lat2, lon2):
    """ Calcula la distancia entre dos coordenadas (lat, lon) en km usando la fórmula de Haversine """
    R = 6371  # Radio de la Tierra en kilómetros
    phi1, phi2 = map(math.radians, [lat1, lat2])
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    # Convertir a RGB (MediaPipe usa RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Face Detection
    results = face_detection.process(img_rgb)

    if results.detections:
        # Si detecta una cara y la confianza es suficiente, consideramos la cara como visible
        for detection in results.detections:
            if detection.score[0] < 0.9:  # Umbral de confianza ajustable
                return jsonify({"face_detected": False})  # Cara parcialmente visible
        return jsonify({"face_detected": True})  # Cara completamente visible

    return jsonify({"face_detected": False})  # No se detectó ninguna cara

@app.route('/detect_finger', methods=['POST'])
def detect_finger():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    
    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    def detectar_indice_levantado_solo(hand_landmarks):
        y_indice = hand_landmarks.landmark[8].y
        y_base_indice = hand_landmarks.landmark[5].y
        y_medio = hand_landmarks.landmark[12].y
        y_base_medio = hand_landmarks.landmark[9].y
        y_anular = hand_landmarks.landmark[16].y
        y_menique = hand_landmarks.landmark[20].y

        if (y_indice < y_base_indice and  
            y_medio > y_base_medio and  
            y_anular > hand_landmarks.landmark[13].y and  
            y_menique > hand_landmarks.landmark[17].y):  
            return True
        return False

    indice_levantado_solo = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detectar_indice_levantado_solo(hand_landmarks):
                indice_levantado_solo = True

    return jsonify({"index_finger_up": indice_levantado_solo})

@app.route('/detect_location', methods=['POST'])
def detect_nearest_location():
    data = request.get_json()
    if not data or 'lat' not in data or 'lon' not in data:
        return jsonify({"error": "Se requieren lat y lon"}), 400

    lat_usuario = data['lat']
    lon_usuario = data['lon']

    # Calcular la distancia a todas las localidades y agregar columna
    df = df_localidades.with_columns(
        pl.Series([haversine(lat_usuario, lon_usuario, lat, lon) for lat, lon in zip(df_localidades["centroide_lat"], df_localidades["centroide_lon"])]).alias("distancia")
    )

    # Obtener la localidad más cercana
    localidad_cercana = df.sort("distancia").row(0)

    return jsonify({
        "localidad": localidad_cercana[df.columns.index("nombre")],
        "municipio": localidad_cercana[df.columns.index("municipio_nombre")],
        "provincia": localidad_cercana[df.columns.index("provincia_nombre")],
        "distancia_km": localidad_cercana[df.columns.index("distancia")]
    })

