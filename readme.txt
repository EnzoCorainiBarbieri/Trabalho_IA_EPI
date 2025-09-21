para ajustar a imagems
python scripts/review_fix_gui.py --dataset-root datasets/epi --split train --data
 datasets/data.yaml

para treina roda 

yolo detect train `
  data=datasets/data.yaml `
  model=yolov8n.pt `
  epochs=50 `
  imgsz=640 `
  batch=-1 `
  workers=0 `
  cache=ram
