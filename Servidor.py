from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from deepface import DeepFace
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # habilita CORS para recibir peticiones desde cualquier origen

emociones_permitidas = ["angry", "sad", "happy", "surprise"]
porcentaje_minimo = 10

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
    print("Petición recibida")
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
        
        # Devuelve la emoción al cliente
        return jsonify({'emocion': emocion_detectada})
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return jsonify({'error': 'no se pudo procesar la imagen'}), 400

if __name__ == '__main__':
        app.run(host='192.168.0.101', port=5000, debug=True,
            ssl_context=('C:/Users/santi/Downloads/Noveno Semestre/Diseño Mecatronico/PaginaWeb-Gesticulink/flask_server.crt', 'C:/Users/santi/Downloads/Noveno Semestre/Diseño Mecatronico/PaginaWeb-Gesticulink/flask_server.key'))