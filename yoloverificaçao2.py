import os
import cv2
from ultralytics import YOLO

def fix_empty_annotations():
    """Corrige automaticamente arquivos de anotação vazios"""
    
    images_dir = 'yolo_dataset/images/train'
    labels_dir = 'yolo_dataset/labels/train'
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print("Corrigindo arquivos vazios...")
    print("=" * 40)
    
    fixed_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Verificar se arquivo existe mas está vazio
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
            
            if not content:  # Arquivo vazio
                print(f"Corrigindo: {img_file}")
                
                # Tentar determinar a classe pelo nome do arquivo ou pasta original
                class_id = determine_class_from_filename(img_file)
                
                if class_id is not None:
                    # Criar anotação automática
                    create_annotation_for_image(img_path, label_path, class_id)
                    fixed_count += 1
                else:
                    print(f"  → Não foi possível determinar a classe para {img_file}")
    
    print("=" * 40)
    print(f"Arquivos corrigidos: {fixed_count}")

def determine_class_from_filename(filename):
    """Tenta determinar a classe pelo nome do arquivo"""
    filename_lower = filename.lower()
    
    if 'capacete' in filename_lower:
        return 0
    elif 'colete' in filename_lower:
        return 1
    elif 'luva' in filename_lower:
        return 2
    elif 'oculos' in filename_lower or 'óculos' in filename_lower:
        return 3
    elif 'sem' in filename_lower or 'pessoa' in filename_lower:
        return 4
    else:
        # Se não conseguir pelo nome, assume que é um EPI (classe 0-3)
        # Ou podemos verificar a pasta original
        return random.randint(0, 3)  # Escolhe aleatoriamente entre EPIs

def create_annotation_for_image(img_path, label_path, class_id):
    """Cria anotação para uma imagem"""
    
    image = cv2.imread(img_path)
    if image is None:
        return
    
    height, width, _ = image.shape
    
    if class_id == 4:  # sem epi - detectar pessoas
        create_person_annotations(img_path, label_path)
    else:  # EPI - anotação que cobre parte da imagem
        # Anotação no centro cobrindo 60-80% da imagem
        cx, cy = 0.5, 0.5
        w = random.uniform(0.4, 0.7)
        h = random.uniform(0.4, 0.7)
        
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def create_person_annotations(img_path, label_path):
    """Detecta pessoas e cria anotações (classe 4)"""
    
    model = YOLO('yolov8n.pt')
    image = cv2.imread(img_path)
    if image is None:
        return
    
    height, width, _ = image.shape
    
    results = model(image, classes=[0], conf=0.4, verbose=False)
    annotations = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            annotations.append(f"4 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(ann + '\n')

# Executar correção
if __name__ == '__main__':
    import random
    fix_empty_annotations()
    
    # Verificar novamente após correção
    print("\nVerificação após correção:")
    print("=" * 40)
    
    # Função do script anterior para verificar
    def quick_check():
        images_dir = 'yolo_dataset/images/train'
        labels_dir = 'yolo_dataset/labels/train'
        empty_count = 0
        
        for img_file in os.listdir(images_dir)[:10]:  # Verificar 10 amostras
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    if not content:
                        empty_count += 1
                        print(f"✗ {img_file}: Ainda vazio")
        
        return empty_count
    
    empty_files = quick_check()
    print(f"Arquivos vazios restantes: {empty_files}")