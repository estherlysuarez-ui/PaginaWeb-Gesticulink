from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from deepface import DeepFace
import numpy as np
import base64
import serial

app = Flask(__name__)
CORS(app)  # <--- aquí habilitamos CORS

emociones_permitidas = ["angry", "sad", "happy", "surprise"]
porcentaje_minimo = 10

# Configura el puerto COM correspondiente a tu ESP32
SERIAL_PORT = 'COM3'  # Cambia según tu PC
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    ser = None
    print(f"No se pudo abrir el puerto serial: {e}")

def enviar_emocion_bluetooth(emocion):
    if ser and ser.is_open:
        try:
            ser.write((emocion + '\n').encode())
        except Exception as e:
            print(f"Error enviando por Bluetooth: {e}")

def detectar_emocion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion_scores = result[0]['emotion']
        emociones_filtradas = {k: v for k, v in emotion_scores.items() if k in emociones_permitidas}
        if emociones_filtradas:
            emotion = max(emociones_filtradas, key=lambda k: emociones_filtradas[k])
            if emociones_filtradas[emotion] < porcentaje_minimo:
                emotion = "neutral"
        else:
            emotion = "neutral"
        return emotion
    except Exception as e:
        print(f"Error detectando emoción: {e}")
        return "error"

@app.route('/emocion', methods=['POST'])
def emocion():
    data = request.json
    img_data = data['image']

    # Decodifica la imagen base64
    if ',' in img_data:
        img_data = img_data.split(",")[1]  # elimina prefijo data:image/png;base64, si existe

    try:
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detecta la emoción
        emocion_detectada = detectar_emocion(frame)

        # Imprime en consola para verificación
        print(f"Emoción recibida del HTML: {emocion_detectada}")

        # Envía al ESP32
        enviar_emocion_bluetooth(emocion_detectada)

        return jsonify({'emocion': emocion_detectada})
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return jsonify({'error': 'no se pudo procesar la imagen'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)