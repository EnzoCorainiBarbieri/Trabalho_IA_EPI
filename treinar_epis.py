import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

# Configurações
IMG_SIZE = (150, 150)
EPOCHS = 30
BATCH_SIZE = 16

# Mapeamento de classes baseado na sua estrutura de pastas
CLASSES = {
    'capacete': 0,
    'oculos': 1, 
    'luvas': 2,
    'colete': 3,
    'sem_epi': 4
}

# Função para carregar imagens de todas as classes
def carregar_imagens(base_dir):
    imagens = []
    labels = []
    
    for classe, idx in CLASSES.items():
        # Carregar imagens de treino
        pasta_treino = os.path.join(base_dir, 'treino', classe)
        
        if not os.path.exists(pasta_treino):
            print(f"Aviso: Pasta {pasta_treino} não encontrada")
            continue
            
        print(f"Carregando imagens de {classe}...")
        
        for arquivo in os.listdir(pasta_treino):
            if arquivo.endswith(('.jpg', '.jpeg', '.png')):
                # Carregar imagem
                img_path = os.path.join(pasta_treino, arquivo)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Erro ao carregar imagem: {img_path}")
                    continue
                
                # Redimensionar
                img = cv2.resize(img, IMG_SIZE)
                
                # Normalizar (0-1)
                img = img / 255.0
                
                imagens.append(img)
                labels.append(idx)
    
    return np.array(imagens), np.array(labels)

# Função para criar o modelo de múltiplas classes
def criar_modelo(num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Saída multi-classe
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Função principal
def main():
    base_dir = r"C:\Users\gabri\OneDrive\Documentos\unimax-ia\trabalho 4 - tentativa 2\dataset"
    
    # Carregar todas as imagens
    print("Carregando imagens...")
    X, y = carregar_imagens(base_dir)
    
    if len(X) == 0:
        print("Erro: Nenhuma imagem encontrada. Verifique os caminhos das pastas.")
        return
    
    # Converter labels para one-hot encoding
    y = to_categorical(y, num_classes=len(CLASSES))
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Total de imagens: {len(X)}")
    print(f"Imagens de treino: {len(X_train)}")
    print(f"Imagens de teste: {len(X_test)}")
    
    # Criar e treinar modelo
    print("Criando e treinando modelo...")
    model = criar_modelo(len(CLASSES))
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Salvar modelo
    model.save('modelo_epis.h5')
    print("Modelo salvo como 'modelo_epis.h5'")
    
    # Avaliar modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Acurácia no teste: {accuracy*100:.2f}%")
    
    # Salvar mapeamento de classes
    with open('classes.json', 'w') as f:
        json.dump(CLASSES, f)
    print("Mapeamento de classes salvo em 'classes.json'")

if __name__ == "__main__":
    main()