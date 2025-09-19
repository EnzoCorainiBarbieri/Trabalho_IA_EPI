import os
import yaml

def create_correct_data_yaml():
    """Cria um arquivo data.yaml correto automaticamente"""
    
    # Configurações - AJUSTE ESTES CAMINHOS SE NECESSÁRIO
    base_path = os.path.abspath('yolo_dataset')
    train_path = 'images/train'
    val_path = 'images/val'
    
    # Verificar se as pastas existem
    if not os.path.exists(os.path.join(base_path, train_path)):
        print("❌ Pasta de treino não encontrada!")
        return False
    
    if not os.path.exists(os.path.join(base_path, val_path)):
        print("❌ Pasta de validação não encontrada!")
        return False
    
    # Conteúdo do data.yaml
    data_yaml_content = {
        'path': base_path,
        'train': train_path,
        'val': val_path,
        'nc': 5,
        'names': ['capacete', 'colete de segurança', 'luvas de proteção', 'óculos de proteção', 'sem epi']
    }
    
    # Salvar arquivo
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ data.yaml criado com sucesso!")
    print(f"Path: {base_path}")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    return True

# Executar
if create_correct_data_yaml():
    print("\n🎉 Agora execute novamente: python check_structure.py")
else:
    print("\n❌ Erro ao criar data.yaml. Verifique a estrutura de pastas.")