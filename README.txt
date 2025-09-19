Dependências principais:

ultralytics (YOLOv8)

matplotlib (gráficos de métricas)

pandas (exportação de resultados)

flask (API web)


🚀 Como treinar o modelo

Coloque suas imagens organizadas na pasta dataset/train/:

Gere o data.yaml automaticamente: python create_data_yaml.py

Rode o treinamento: python train.py

para testar so rodar o detect.py