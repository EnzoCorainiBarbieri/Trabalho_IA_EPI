import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk

# Variáveis globais para armazenar o modelo e os índices das classes
model = None
class_indices = None
cap = None  # Variável global para a captura de vídeo

# Função para verificar a estrutura da pasta de imagens
def verificar_estrutura_pasta(pasta_imagens):
    if not os.path.exists(pasta_imagens):
        print(f"Erro: O diretório {pasta_imagens} não existe.")
        return False

    # Listar todas as subpastas
    subpastas = [f.path for f in os.scandir(pasta_imagens) if f.is_dir()]
    if len(subpastas) == 0:
        print(f"Erro: Não há subpastas em {pasta_imagens}. Certifique-se de ter pelo menos uma subpasta de classe.")
        return False

    # Verificar se as subpastas têm imagens
    for subpasta in subpastas:
        arquivos = [f for f in os.scandir(subpasta) if f.is_file()]
        if len(arquivos) == 0:
            print(f"Erro: A subpasta {subpasta} está vazia.")
            return False
    print(f"A estrutura de pastas em {pasta_imagens} está correta.")
    return True

# Função para treinar o modelo com data augmentation
def treinar_modelo(pasta_imagens):
    global model, class_indices  # Tornar as variáveis globais acessíveis

    # Verificar se a estrutura da pasta está correta
    if not verificar_estrutura_pasta(pasta_imagens):
        return  # Se a estrutura estiver errada, não tenta treinar

    # Criando o ImageDataGenerator com Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalizando as imagens
        rotation_range=40,  # Rotacionando as imagens aleatoriamente
        width_shift_range=0.2,  # Mudança horizontal
        height_shift_range=0.2,  # Mudança vertical
        shear_range=0.2,  # Cortar imagens
        zoom_range=0.2,  # Zoom nas imagens
        horizontal_flip=True,  # Flip horizontal
        fill_mode='nearest'  # Preenchendo as áreas criadas pelo flip e corte
    )

    # Carregando as imagens para treinamento
    train_generator = datagen.flow_from_directory(
        pasta_imagens,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Criando a rede neural
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')  # Camada de saída com número de classes
    ])

    # Compilando o modelo
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinando o modelo
    model.fit(train_generator, epochs=20)  # Aumentando o número de épocas

    # Salvando o modelo treinado
    model.save('meu_modelo.h5')

    # Salvando os índices das classes
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    print("Modelo treinado e salvo como 'meu_modelo.h5'")
    messagebox.showinfo("Treinamento Concluído", "Modelo treinado e salvo com sucesso!")

# Função para carregar o modelo e os índices de classe
def carregar_modelo():
    global model, class_indices
    try:
        model = load_model('meu_modelo.h5')
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        return True
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return False

# Função para reconhecer o objeto
def reconhecer_objeto(frame, class_indices):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    previsao = model.predict(image)
    classe_index = np.argmax(previsao)
    nome_classe = list(class_indices.keys())[list(class_indices.values()).index(classe_index)]
    return nome_classe

# Função para exibir a imagem da câmera dentro do Tkinter
def atualizar_imagem(label_img):
    global cap

    ret, frame = cap.read()
    if ret:
        # Converter a imagem para o formato correto para exibição no Tkinter
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Atualizar a imagem exibida na interface gráfica
        label_img.config(image=img_tk)
        label_img.image = img_tk

    # Continuar atualizando a imagem a cada 50ms (usando o método after)
    label_img.after(50, atualizar_imagem, label_img)

# Função para ativar a câmera e exibir na interface gráfica
def ativar_camera(label_img):
    global cap
    cap = cv2.VideoCapture(0)

    # Iniciar o loop de captura da câmera e exibição no Tkinter
    atualizar_imagem(label_img)

# Função para capturar uma imagem da câmera e realizar a análise
def capturar_e_analisar(label_img):
    ret, frame = cap.read()  # Captura um frame da câmera
    if ret:
        nome_classe = reconhecer_objeto(frame, class_indices)  # Análise com IA
        print('Previsão do modelo:', nome_classe)
        messagebox.showinfo("Resultado", f'O objeto é: {nome_classe}')
    else:
        messagebox.showerror("Erro", "Não foi possível capturar a imagem.")

# Função para selecionar pasta e treinar
def selecionar_pasta_e_treinar():
    pasta_imagens = filedialog.askdirectory()
    if pasta_imagens:
        treinar_modelo(pasta_imagens)
        habilitar_botao_analizar()

# Função para habilitar o botão "Analisar Objeto"
def habilitar_botao_analizar():
    if carregar_modelo():  # Tenta carregar o modelo após o treinamento
        btn_analizar.config(state=tk.NORMAL)  # Habilita o botão de análise
        print("Modelo carregado e botão de análise habilitado.")
    else:
        messagebox.showerror("Erro", "Falha ao carregar o modelo. Tente treinar novamente.")

# Interface gráfica
def interface_grafica():
    global btn_analizar  # Tornar o botão de análise acessível globalmente

    root = tk.Tk()
    root.title("Reconhecimento de Objetos com IA")

    # Botão para treinar o modelo
    btn_treinar = tk.Button(root, text="Treinar Modelo", command=selecionar_pasta_e_treinar)
    btn_treinar.pack(pady=10)

    # Criação de um label para exibir a imagem da câmera
    label_img = tk.Label(root)
    label_img.pack()

    # Botão para ativar a câmera e visualizar em tempo real
    btn_camera = tk.Button(root, text="Iniciar Câmera", command=lambda: ativar_camera(label_img))
    btn_camera.pack(pady=10)

    # Botão para capturar a imagem da câmera e realizar a análise
    btn_analizar = tk.Button(root, text="OK", state=tk.DISABLED, command=lambda: capturar_e_analisar(label_img))
    btn_analizar.pack(pady=10)

    # Rodar a interface
    root.mainloop()

# Iniciar
interface_grafica()