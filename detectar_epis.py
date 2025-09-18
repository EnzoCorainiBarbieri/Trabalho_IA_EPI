import cv2
import numpy as np
from tensorflow import keras
import json

# Verificar se o modelo existe antes de tentar carregar
import os
if not os.path.exists('modelo_epis.h5'):
    print("Erro: Modelo 'modelo_epis.h5' não encontrado.")
    print("Execute primeiro o script de treinamento.")
    exit()

# Carregar modelo
model = keras.models.load_model('modelo_epis.h5')
IMG_SIZE = (150, 150)

# Carregar mapeamento de classes
with open('classes.json', 'r') as f:
    CLASSES = json.load(f)

# Inverter o mapeamento (de número para nome)
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

# Função para prever EPIs
def prever_epi(frame):
    # Pré-processar frame
    img = cv2.resize(frame, IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fazer previsão
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Obter a classe com maior probabilidade
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx] * 100
    class_name = CLASS_NAMES[class_idx]
    
    return class_name, confidence, predictions

# Inicializar webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

print("Pressione 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espelhar a imagem (mais natural para webcam)
    frame = cv2.flip(frame, 1)
    
    # Fazer cópia para exibição
    display_frame = frame.copy()
    
    # Fazer previsão
    classe, confianca, todas_predicoes = prever_epi(frame)
    
    # Exibir resultado principal
    texto = f"{classe}: {confianca:.1f}%"
    
    # Definir cor baseado na classe
    if classe == "sem_epi":
        cor = (0, 0, 255)  # Vermelho para sem EPI
    else:
        cor = (0, 255, 0)  # Verde para com EPI
    
    cv2.putText(display_frame, texto, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    
    # Exibir todas as probabilidades
    y_offset = 60
    for i, prob in enumerate(todas_predicoes):
        classe_nome = CLASS_NAMES[i]
        texto_prob = f"{classe_nome}: {prob*100:.1f}%"
        cv2.putText(display_frame, texto_prob, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Mostrar frame
    cv2.imshow('Detector de EPIs', display_frame)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()