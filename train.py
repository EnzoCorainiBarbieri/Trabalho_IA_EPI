from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# Carregar modelo pré-treinado YOLOv8
model = YOLO("yolov8n.pt")

# Treinar o modelo
results = model.train(
    data="data.yaml",  # Arquivo de configuração do dataset
    epochs=50,
    imgsz=640,
    batch=8,
    name="epi_detector"
)

# Salvar métricas de treinamento
metrics = results.results_dict
pd.DataFrame([metrics]).to_csv("training_metrics.csv")

# Plotar gráficos de perda
plt.plot(results.history['train_loss'], label='Train Loss')
plt.plot(results.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig("loss_plot.png")