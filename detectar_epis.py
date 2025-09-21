import os
import cv2
import numpy as np
import json
import time
from tensorflow import keras

# Verificar existência dos arquivos
if not os.path.exists('modelo_epis.h5') or not os.path.exists('classes.json'):
    print("Erro: Arquivos necessários não encontrados.")
    print("Certifique-se de que 'modelo_epis.h5' e 'classes.json' estão na pasta.")
    exit()

# Carregar modelo e classes
model = keras.models.load_model('modelo_epis.h5')
IMG_SIZE = (150, 150)

with open('classes.json', 'r') as f:
    CLASSES = json.load(f)
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# Cores por classe
CORES = {
    "capacete": (255, 255, 0),
    "oculos": (0, 255, 255),
    "luvas": (255, 0, 255),
    "colete": (0, 128, 255),
    "sem_epi": (0, 0, 255)
}

# Função de previsão
def prever_epi(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.expand_dims(img / 255.0, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx] * 100
    class_name = CLASS_NAMES[class_idx]
    return class_name, confidence, predictions

# Inicializar webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

print("Detector de EPIs iniciado. Pressione 'q' para sair.")

# Loop principal
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    classe, confianca, todas_predicoes = prever_epi(frame)
    cor = CORES.get(classe, (0, 255, 0))
    texto = f"{classe}: {confianca:.1f}%"
    cv2.putText(display_frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    # Alerta visual
    if classe == "sem_epi":
        cv2.rectangle(display_frame, (5, 5), (300, 80), (0, 0, 255), 2)
        cv2.putText(display_frame, "ALERTA: SEM EPI!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Mostrar todas as probabilidades
    y_offset = 100
    for i, prob in enumerate(todas_predicoes):
        nome = CLASS_NAMES[i]
        texto_prob = f"{nome}: {prob*100:.1f}%"
        cv2.putText(display_frame, texto_prob, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20

    # Mostrar FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    cv2.imshow('Detector de EPIs', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
