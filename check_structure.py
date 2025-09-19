import os
import yaml

def check_dataset_structure():
    """Verifica se a estrutura do dataset está correta"""
    
    # Verificar se data.yaml existe
    if not os.path.exists('data.yaml'):
        print("❌ ERRO: data.yaml não encontrado!")
        return False
    
    # Ler data.yaml
    with open('data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # Verificar chaves obrigatórias
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data:
            print(f"❌ ERRO: Chave '{key}' faltando no data.yaml!")
            return False
    
    # Verificar se os caminhos existem
    base_path = data['path']
    train_path = os.path.join(base_path, data['train'])
    val_path = os.path.join(base_path, data['val'])
    
    print(f"Base path: {base_path}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    
    if not os.path.exists(base_path):
        print("❌ ERRO: Caminho base não existe!")
        return False
    
    if not os.path.exists(train_path):
        print("❌ ERRO: Pasta de treino não existe!")
        return False
    
    if not os.path.exists(val_path):
        print("❌ ERRO: Pasta de validação não existe!")
        return False
    
    # Verificar se há imagens nas pastas
    train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Imagens de treino: {len(train_images)}")
    print(f"Imagens de validação: {len(val_images)}")
    
    if len(train_images) == 0:
        print("❌ ERRO: Nenhuma imagem encontrada na pasta de treino!")
        return False
    
    if len(val_images) == 0:
        print("❌ ERRO: Nenhuma imagem encontrada na pasta de validação!")
        return False
    
    print("✅ Estrutura do dataset está correta!")
    return True

# Executar verificação
if check_dataset_structure():
    print("\n🎉 Tudo pronto para treinar!")
else:
    print("\n❌ Corrija os erros acima antes de treinar!")