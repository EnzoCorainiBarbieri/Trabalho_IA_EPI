import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Função para pedir a descrição e imagens do usuário
def pedir_descricao_entrada():
    """
    Solicita ao usuário uma descrição do objeto e um diretório contendo imagens.
    Cria um novo diretório para armazenar essas imagens de acordo com a descrição fornecida.
    """
    descricao = input("Qual objeto você deseja ensinar ao modelo? (exemplo: xícara, carro): ")
    pasta_imagens = input(f"Por favor, forneça o diretório com imagens de {descricao}: ")

    # Verifica se o diretório existe
    if not os.path.exists(pasta_imagens):
        print(f"O diretório '{pasta_imagens}' não foi encontrado! Verifique o caminho.")
        return None

    # Cria o diretório para armazenar as imagens copiadas, se não existir
    pasta_destino = f'dados_de_treinamento/{descricao}'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    # Copiar as imagens fornecidas para a pasta de treinamento do objeto
    for arquivo in os.listdir(pasta_imagens):
        if arquivo.endswith(('.jpg', '.png', '.jpeg')):
            src = os.path.join(pasta_imagens, arquivo)
            dst = os.path.join(pasta_destino, arquivo)

            # Verifica se o arquivo já existe no diretório de destino
            if not os.path.exists(dst):
                shutil.copy(src, dst)
            else:
                print(f"A imagem '{arquivo}' já existe no diretório de destino. Ignorando a cópia.")

    print(f"Imagens de {descricao} foram adicionadas ao treinamento!")

    return descricao


# Classe para carregar imagens
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# Função para carregar imagens e associar com rótulos
def load_images_from_folder(folder, label, image_size=(64, 64)):
    image_paths = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels


# Função para criar o modelo
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # número de classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)  # número de saídas igual ao número de classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Logits para cada classe


# Função para treinar o modelo
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {correct/total}')


# Função para testar o modelo
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {correct / total * 100}%')


# Função principal para treino incremental
def treinar_com_novos_dados(model, descricao, numero_de_classes, transform, num_epochs=10):
    # Carregar as novas imagens da pasta correspondente
    image_paths, labels = load_images_from_folder(f'dados_de_treinamento/{descricao}', numero_de_classes)

    # Dividir os dados em treino e teste
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Criar os datasets
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=transform)

    # Criar os dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Atualizar o número de classes
    model.fc2 = nn.Linear(128, numero_de_classes)

    # Recompilar o modelo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinar o modelo com os novos dados
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    test_model(model, test_loader)


# Definir transformações para as imagens
transform = transforms.Compose([
    transforms.ToPILImage(),  # Converte para imagem PIL
    transforms.Resize((64, 64)),  # Redimensiona para o tamanho desejado
    transforms.ToTensor(),  # Converte para Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza as imagens
])

# Criar e treinar o modelo
model = SimpleCNN(num_classes=4)  # Número de classes iniciais

# Solicitar entrada do usuário
descricao = pedir_descricao_entrada()  # Solicita ao usuário uma descrição e carrega as imagens

if descricao:  # Verifica se a entrada foi válida
    num_classes = len(os.listdir('dados_de_treinamento'))  # Número de classes (pastas) no diretório
    treinar_com_novos_dados(model, descricao, num_classes, transform)