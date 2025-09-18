import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Carregar modelo
model = keras.models.load_model('modelo_epi.h5')
class_names = ["capacete", "oculos", "luvas", "colete", "sem_epi"]
IMG_SIZE = (224, 224)

def prever_frame(frame):
    # Pré-processar frame
    img = cv2.resize(frame, IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0
    
    # Fazer previsão
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Obter classe e confiança
    classe = class_names[tf.argmax(score)]
    confianca = 100 * tf.reduce_max(score)
    
    return classe, confianca

# Inicializar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Fazer cópia para exibição
    display_frame = frame.copy()
    
    # Fazer previsão
    classe, confianca = prever_frame(frame)
    
    # Exibir resultado
    texto = f"{classe}: {confianca:.2f}%"
    cv2.putText(display_frame, texto, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Destacar se não está usando EPI
    if classe == "sem_epi":
        cv2.putText(display_frame, "ALERTA: SEM EPI!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mostrar frame
    cv2.imshow('Detector de EPI', display_frame)
    
    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()