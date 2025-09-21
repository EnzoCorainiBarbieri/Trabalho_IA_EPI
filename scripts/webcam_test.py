#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de inferência em tempo real (webcam) com YOLOv8.
- Mostra caixas, rótulos e FPS
- Teclas:
  Q  -> sair
  P  -> pausar/continuar
  S  -> salvar frame em ./captures/
  +/- -> ajustar confiança (conf threshold)

Uso:
  python scripts/webcam_test.py --weights runs/detect/train/weights/best.pt --data datasets/data.yaml
  # câmera alternativa:
  python scripts/webcam_test.py --source 1
  # arquivo de vídeo:
  python scripts/webcam_test.py --source "meu_video.mp4"
"""

import argparse
import time
from pathlib import Path
import yaml
import cv2
from ultralytics import YOLO

def load_names_from_yaml(data_yaml: Path):
    try:
        meta = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        names_dict = meta.get("names", {})
        # pode vir como dict {0:'helmet',1:'vest',...} ou lista
        if isinstance(names_dict, dict):
            names = [names_dict[i] for i in sorted(names_dict.keys())]
        elif isinstance(names_dict, list):
            names = names_dict
        else:
            names = []
        return names
    except Exception:
        return []

def put_text(img, text, org, color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="caminho do .pt treinado")
    ap.add_argument("--data", type=str, default="datasets/data.yaml", help="data.yaml para nomes das classes")
    ap.add_argument("--source", type=str, default="0", help="0=webcam primária, 1=secundária, caminho de vídeo, RTSP, etc.")
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem para inferência")
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima")
    ap.add_argument("--device", type=str, default=None, help="ex.: 'cpu', '0' (GPU 0). Se None, escolha automática.")
    args = ap.parse_args()

    # Carregar modelo
    model = YOLO(args.weights)
    # nomes das classes
    names = model.names if hasattr(model, "names") and model.names else load_names_from_yaml(args.data)
    if not names:
        names = []
    print(f"[INFO] Classes: {names if names else '(desconhecidas)'}")

    # Abrir fonte de vídeo
    source = args.source
    cap = None
    if source.isdigit():
        cam_index = int(source)
        # DirectShow no Windows ajuda a abrir algumas webcams
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap or not cap.isOpened():
        print(f"[ERRO] Não consegui abrir a fonte: {source}")
        return

    Path("captures").mkdir(exist_ok=True)
    paused = False
    conf = float(args.conf)

    print("[INFO] Rodando. Teclas: Q=sair, P=pausar, S=salvar frame, +/-=ajusta confiança")
    prev_t = time.time()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Sem frame (fim do vídeo ou câmera indisponível).")
                break

            # Inferência
            t0 = time.time()
            results = model.predict(frame, imgsz=args.imgsz, conf=conf, device=args.device, verbose=False)
            t1 = time.time()

            # Desenhar resultados (primeiro resultado do batch)
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                    cls = int(box.cls[0].item()) if box.cls is not None else -1
                    score = float(box.conf[0].item()) if box.conf is not None else 0.0

                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

                    label = f"{names[cls] if 0 <= cls < len(names) else cls} {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 220, 0), -1)
                    put_text(frame, label, (x1 + 3, y1 - 6), color=(0, 0, 0))

            # FPS
            dt = (t1 - t0)
            fps = 1.0 / (time.time() - prev_t + 1e-9)
            prev_t = time.time()
            put_text(frame, f"conf={conf:.2f} | infer={dt*1000:.1f}ms | FPS~{fps:.1f}", (10, 25))

            cv2.imshow("YOLOv8 - Webcam", frame)

        # Teclas
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):  # q ou ESC
            break
        elif k == ord('p'):
            paused = not paused
        elif k == ord('s'):
            # salvar último frame mostrado
            snap_path = Path("captures") / f"cap_{int(time.time())}.jpg"
            # Se pausado, precisamos do último frame do buffer da janela
            # Aqui simplificamos: pedimos outro frame se não estivermos pausados.
            # Em pause, o último frame ainda está em 'frame'.
            if 'frame' in locals():
                cv2.imwrite(str(snap_path), frame)
                print(f"[OK] Frame salvo em {snap_path}")
        elif k in (ord('+'), ord('=')):
            conf = min(0.99, conf + 0.05)
        elif k in (ord('-'), ord('_')):
            conf = max(0.01, conf - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
