# baixar_epis.py
from bing_image_downloader import downloader
import os
import shutil

# Pastas principais
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Classes de EPIs
classes = ["capacete", "colete de segurança", "luvas de proteção", "óculos de proteção", "sem epi"]

# Quantidade de imagens
num_imagens = 100  # imagens por classe

# Cria pastas de treino e validação
for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

# Baixa imagens para cada classe
for c in classes:
    print(f"Baixando imagens da classe: {c}")
    downloader.download(
        c,
        limit=num_imagens,
        output_dir=train_dir,  # Baixa primeiro no treino
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

# Opcional: dividir 80% treino / 20% validação
for c in classes:
    classe_dir = os.path.join(train_dir, c)
    imagens = os.listdir(classe_dir)
    num_val = int(len(imagens) * 0.2)
    
    for img in imagens[:num_val]:
        src = os.path.join(classe_dir, img)
        dst = os.path.join(val_dir, c, img)
        shutil.move(src, dst)

print("Download e organização concluídos!")
