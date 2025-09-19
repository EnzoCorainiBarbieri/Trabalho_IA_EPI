import os
import shutil
from sklearn.model_selection import train_test_split

# Configurações
base_path = 'dataset'
classes = ['capacete', 'colete de segurança', 'luvas de proteção', 'óculos de proteção', 'sem epi']

# Criar estrutura de diretórios do YOLO
os.makedirs('yolo_dataset/images/train', exist_ok=True)
os.makedirs('yolo_dataset/images/val', exist_ok=True)
os.makedirs('yolo_dataset/labels/train', exist_ok=True)
os.makedirs('yolo_dataset/labels/val', exist_ok=True)

# Coletar todas as imagens
all_images = []
for class_name in classes:
    class_path = os.path.join(base_path, 'train', class_name)
    if os.path.exists(class_path):
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                all_images.append((class_name, img_file))

# Dividir em treino e validação
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Função para processar e mover os arquivos
def process_files(file_list, image_dest, label_dest):
    for class_name, img_file in file_list:
        # Mover imagem
        src_img = os.path.join(base_path, 'train', class_name, img_file)
        dest_img = os.path.join(image_dest, img_file)
        shutil.copy(src_img, dest_img)
        
        # Criar arquivo de label vazio (você precisará adicionar anotações reais)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        with open(os.path.join(label_dest, label_file), 'w') as f:
            # Aqui você precisaria adicionar as anotações reais no formato YOLO
            # Formato: <class_id> <center_x> <center_y> <width> <height>
            f.write('')

# Processar arquivos de treino e validação
process_files(train_files, 'yolo_dataset/images/train', 'yolo_dataset/labels/train')
process_files(val_files, 'yolo_dataset/images/val', 'yolo_dataset/labels/val')

print("Dataset convertido para formato YOLO!")