import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configurações
IMG_SIZE = (150, 150)
EPOCHS = 30
BATCH_SIZE = 16

# Mapeamento de classes
CLASSES = {
    'capacete': 0,
    'oculos': 1,
    'luvas': 2,
    'colete': 3,
    'sem_epi': 4
}

def carregar_imagens(base_dir):
    imagens = []
    labels = []
    for classe, idx in CLASSES.items():
        pasta_treino = os.path.join(base_dir, 'treino', classe)
        if not os.path.exists(pasta_treino):
            print(f"Aviso: Pasta {pasta_treino} não encontrada")
            continue
        print(f"Carregando imagens de {classe}...")
        for arquivo in os.listdir(pasta_treino):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pasta_treino, arquivo)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao carregar imagem: {img_path}")
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                imagens.append(img)
                labels.append(idx)
    return np.array(imagens), np.array(labels)

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
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plotar_metricas(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    base_dir = input("Digite o caminho da pasta base do dataset: ").strip()
    if not os.path.exists(base_dir):
        print("Erro: Caminho inválido.")
        return

    print("Carregando imagens...")
    X, y = carregar_imagens(base_dir)
    if len(X) == 0:
        print("Erro: Nenhuma imagem encontrada.")
        return

    y = to_categorical(y, num_classes=len(CLASSES))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Total de imagens: {len(X)}")
    print(f"Imagens de treino: {len(X_train)}")
    print(f"Imagens de teste: {len(X_test)}")

    print("Aplicando augmentação de dados...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    print("Criando e treinando modelo...")
    model = criar_modelo(len(CLASSES))
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("Salvando modelo...")
    model.save('modelo_epis.h5')
    with open('classes.json', 'w') as f:
        json.dump(CLASSES, f)
    print("Modelo e mapeamento salvos.")

    print("Avaliando modelo...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Acurácia no teste: {accuracy*100:.2f}%")

    print("Plotando métricas...")
    plotar_metricas(history)

if __name__ == "__main__":
    main()
