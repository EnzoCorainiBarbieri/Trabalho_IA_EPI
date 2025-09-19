import os
import cv2
import random
from pathlib import Path

# Configurações baseadas na SUA estrutura de pastas
dataset_base = 'dataset/train'
class_folders = {
    'capacete': 0,
    'colete de segurança': 1, 
    'luvas de proteção': 2,
    'óculos de proteção': 3,
    'sem epi': 4
}

# Criar diretórios de saída no formato YOLO
os.makedirs('yolo_dataset/images/train', exist_ok=True)
os.makedirs('yolo_dataset/labels/train', exist_ok=True)
os.makedirs('yolo_dataset/images/val', exist_ok=True)
os.makedirs('yolo_dataset/labels/val', exist_ok=True)

def create_automatic_annotations():
    """Cria anotações automáticas baseadas na estrutura de pastas"""
    
    all_images = []
    
    # Coletar todas as imagens com suas classes
    for class_name, class_id in class_folders.items():
        class_path = os.path.join(dataset_base, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append({
                        'path': os.path.join(class_path, img_file),
                        'class_id': class_id,
                        'filename': img_file
                    })
    
    # Embaralhar e dividir em treino/validação
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)  # 80% treino, 20% validação
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Processar imagens de treino
    print("Processando TREINO...")
    for img_info in train_images:
        process_image(img_info, 'train')
    
    # Processar imagens de validação
    print("Processando VALIDAÇÃO...")
    for img_info in val_images:
        process_image(img_info, 'val')
    
    print(f"Concluído! {len(train_images)} imagens de treino, {len(val_images)} de validação")

def process_image(img_info, dataset_type):
    """Processa uma imagem e cria anotações automáticas"""
    
    img_path = img_info['path']
    class_id = img_info['class_id']
    filename = img_info['filename']
    
    # Copiar imagem para pasta YOLO
    dest_image_path = os.path.join(f'yolo_dataset/images/{dataset_type}', filename)
    if not os.path.exists(dest_image_path):
        shutil.copy(img_path, dest_image_path)
    
    # Criar arquivo de anotação
    label_path = os.path.join(f'yolo_dataset/labels/{dataset_type}', 
                             os.path.splitext(filename)[0] + '.txt')
    
    # Se for "sem epi", detectar pessoas automaticamente
    if class_id == 4:  # sem epi
        create_person_annotations(img_path, label_path)
    else:
        # Para EPIs, criar anotação que cobre toda a imagem
        create_full_image_annotation(img_path, label_path, class_id)

def create_person_annotations(img_path, label_path):
    """Detecta pessoas automaticamente e cria anotações"""
    
    from ultralytics import YOLO
    
    # Carregar modelo para detectar pessoas
    model = YOLO('yolov8n.pt')
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    
    # Detectar pessoas (classe 0 no COCO)
    results = model(image, classes=[0], conf=0.5, verbose=False)
    
    annotations = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Converter para formato YOLO
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Classe 4 = "sem epi"
            annotations.append(f"4 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    # Salvar anotações
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(ann + '\n')

def create_full_image_annotation(img_path, label_path, class_id):
    """Cria anotação que cobre toda a imagem (para EPIs)"""
    
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    
    # Criar anotação que cobre 80% do centro da imagem
    cx, cy = 0.5, 0.5  # centro
    w, h = 0.8, 0.8    # 80% da imagem
    
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# Executar a geração automática
if __name__ == '__main__':
    import shutil
    create_automatic_annotations()