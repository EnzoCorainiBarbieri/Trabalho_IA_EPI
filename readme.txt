ProtegeAÍ — Monitor de EPI (YOLOv8 + Flask)
===========================================

Resumo
------
Aplicação web simples para DETECTAR EPIs (capacete, colete, máscara, óculos, luvas) em tempo real com YOLOv8.
Inclui:
- Página única com streaming da CÂMERA (iniciar/parar)
- Rótulos em PT-BR sobre o vídeo
- “Detectando agora” e “Itens essenciais ausentes”
- Botão SALVAR LOG e modo AUTO-SALVAR quando faltar capacete
- Logs em banco SQLite (protegeai.db) + imagens anotadas em static/frames
- Endpoints /predict (upload), /logs (JSON) e /health

Requisitos
----------
- Python 3.9+ (recomendado 3.10/3.11)
- Pip e venv
- Windows: câmera liberada (feche Teams/Zoom antes)
- Pesos do YOLO treinado (best.pt)

Instalação Rápida (PowerShell)
-------------------------------
# 1) (Opcional) criar venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) instalar dependências
pip install ultralytics flask sqlalchemy opencv-python numpy

Estrutura de Pastas 

-------------
# na raiz do projeto (com venv ativada)
python app.py

Abra no navegador:  http://127.0.0.1:8000/

-------------------------------
Pré-requisitos: Ultralytics instalado e dataset rotulado no formato YOLO.

para treina o modelo (demora quanto mais epochs colocar)
yolo detect train `  
>>   data=datasets/data.yaml `
>>   model=yolov8n.pt `
>>   epochs=60 `
>>   imgsz=640 `
>>   batch=-1 `
>>   workers=0 `
>>   cache=ram `
>>   project=runs/detect `
>>   name=train_fixlabels

para agustar o reconhecimento mais visualmente 

python scripts/review_fix_gui.py --dataset-root datasets/epi --split train --data
 datasets/data.yaml