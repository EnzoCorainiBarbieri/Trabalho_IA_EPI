from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Carregar modelo treinado
model = YOLO("runs/detect/epi_detector/weights/best.pt")

# Mapeamento de classes
class_names = {
    0: "capacete",
    1: "colete de segurança", 
    2: "luvas de proteção",
    3: "óculos de proteção",
    4: "sem epi"
}

# Configurações
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return jsonify({
        "message": "API ProtegeAÍ - Detector de EPIs",
        "status": "online",
        "endpoints": {
            "/detect": "POST - Detectar EPIs em imagem (envie JSON com {image: base64})"
        }
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Receber imagem em base64
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Nenhuma imagem fornecida"}), 400
            
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Falha ao decodificar imagem"}), 400
        
        # Salvar imagem (opcional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), img)
        
        # Fazer detecção
        results = model(img, conf=0.5)
        detections = []
        
        for result in results:
            for box in result.boxes:
                # Extrair informações
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detections.append({
                    "class": class_names[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id
                })
        
        # Verificar se há pessoas sem EPI completo
        has_helmet = any(d['class'] == 'capacete' for d in detections)
        has_vest = any(d['class'] == 'colete de segurança' for d in detections)
        has_gloves = any(d['class'] == 'luvas de proteção' for d in detections)
        has_goggles = any(d['class'] == 'óculos de proteção' for d in detections)
        no_epi = any(d['class'] == 'sem epi' for d in detections)
        
        safety_status = "SAFE"
        if no_epi or not (has_helmet and has_vest):
            safety_status = "UNSAFE"
        
        return jsonify({
            "detections": detections,
            "safety_status": safety_status,
            "timestamp": timestamp,
            "image_path": filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Iniciando API ProtegeAÍ...")
    print("Endpoints disponíveis:")
    print("  GET  /          - Informações da API")
    print("  GET  /status    - Status do serviço")
    print("  POST /detect    - Detectar EPIs em imagem")
    app.run(host='0.0.0.0', port=5000, debug=True)