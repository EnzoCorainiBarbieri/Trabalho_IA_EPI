import os
import cv2
import numpy as np

def batch_verify_annotations():
    """Verifica várias imagens rapidamente sem mostrar janelas"""
    
    images_dir = 'yolo_dataset/images/train'
    labels_dir = 'yolo_dataset/labels/train'
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print("Verificando anotações automaticamente...")
    print("=" * 50)
    
    stats = {'total': 0, 'with_annotations': 0, 'empty': 0, 'missing': 0}
    
    for i, img_file in enumerate(image_files[:20]):  # Verificar apenas 20 amostras
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        stats['total'] += 1
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
            
            if content:
                stats['with_annotations'] += 1
                num_objects = len(content.split('\n'))
                print(f"✓ {img_file}: {num_objects} objetos")
            else:
                stats['empty'] += 1
                print(f"✗ {img_file}: Arquivo VAZIO")
        else:
            stats['missing'] += 1
            print(f"✗ {img_file}: Arquivo NÃO ENCONTRADO")
    
    print("=" * 50)
    print(f"Total: {stats['total']}")
    print(f"Com anotações: {stats['with_annotations']} ({stats['with_annotations']/stats['total']*100:.1f}%)")
    print(f"Vazios: {stats['empty']}")
    print(f"Faltantes: {stats['missing']}")

# Executar verificação em lote
batch_verify_annotations()