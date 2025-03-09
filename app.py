import cv2
import numpy as np
import mediapipe as mp
import polars as pl
import math
import requests
import gc
from flask import Flask, request, jsonify
#from insightface.app import FaceAnalysis
from io import BytesIO

app = Flask(__name__)

# Inicializar MediaPipe Face Detection y Hands
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Cargar datos de localidades
df_localidades = pl.read_csv('localidades.csv')
'''
def get_face_app():
    """Inicializa FaceAnalysis solo cuando se necesita."""
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app
'''
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = map(math.radians, [lat1, lat2])
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    image = cv2.resize(image, (640, 480))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    detected = any(detection.score[0] >= 0.9 for detection in results.detections) if results.detections else False
    
    gc.collect()  # Liberar memoria
    return jsonify({"face_detected": detected})

@app.route('/detect_finger', methods=['POST'])
def detect_finger():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    image = cv2.resize(image, (640, 480))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    def detectar_indice_levantado_solo(hand_landmarks):
        y_indice = hand_landmarks.landmark[8].y
        return (y_indice < hand_landmarks.landmark[5].y and  
                all(hand_landmarks.landmark[finger].y > hand_landmarks.landmark[base].y for finger, base in [(12, 9), (16, 13), (20, 17)]))

    index_finger_up = any(detectar_indice_levantado_solo(hl) for hl in results.multi_hand_landmarks) if results.multi_hand_landmarks else False
    
    gc.collect()  # Liberar memoria
    return jsonify({"index_finger_up": index_finger_up})

@app.route('/detect_location', methods=['POST'])
def detect_location():
    data = request.get_json()
    if not data or 'lat' not in data or 'lon' not in data:
        return jsonify({"error": "Se requieren lat y lon"}), 400

    lat_usuario, lon_usuario = data['lat'], data['lon']
    df = df_localidades.with_columns(
        pl.Series([haversine(lat_usuario, lon_usuario, lat, lon) for lat, lon in zip(df_localidades["centroide_lat"], df_localidades["centroide_lon"])]).alias("distancia")
    )

    localidad_cercana = df.sort("distancia").row(0)
    return jsonify({
        "localidad": localidad_cercana[df.columns.index("nombre")],
        "municipio": localidad_cercana[df.columns.index("municipio_nombre")],
        "provincia": localidad_cercana[df.columns.index("provincia_nombre")],
        "distancia_km": localidad_cercana[df.columns.index("distancia")]
    })
'''
def descargar_imagen(url):
    try:
        respuesta = requests.get(url, stream=True, timeout=10)
        if respuesta.status_code == 200:
            img_array = np.frombuffer(respuesta.content, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"⚠️ Error al descargar {url}: {e}")
    return None

def obtener_embedding(image):
    """Obtiene el embedding facial de una imagen."""
    if image is None:
        return None
    face_app = get_face_app()
    faces = face_app.get(image)
    return faces[0].normed_embedding if faces else None

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    if 'image' not in request.files or 'url' not in request.form:
        return jsonify({"error": "Debe enviar una imagen y una URL"}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image_local = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    url = request.form['url']
    image_url = descargar_imagen(url)

    emb1, emb2 = obtener_embedding(image_local), obtener_embedding(image_url)

    if emb1 is None or emb2 is None:
        return jsonify({"error": "No se detectaron caras en una o ambas imágenes"}), 400

    similarity = np.dot(emb1, emb2)
    gc.collect()  # Liberar memoria
    return jsonify({"same_person": similarity >= 0.4, "similarity": round(similarity, 4)})
'''

