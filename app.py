#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProtegeAÍ — Aplicação mínima (1 arquivo .py)

Recursos:
- Flask (servidor web) + Ultralytics YOLOv8 (detecção)
- Câmera ao vivo (start/stop) com streaming MJPEG
- Sobreposição de caixas e rótulos EM PT-BR no vídeo
- Painel “Detectando agora” e “Itens essenciais ausentes”
- Auto-salvar log quando faltar capacete + botão “Salvar log”
- Upload de imagem (/predict) e listagem de logs em JSON (/logs)

Como rodar (no terminal na raiz do projeto):
  pip install ultralytics flask sqlalchemy opencv-python numpy
  python app.py
Abrir no navegador: http://127.0.0.1:8000
"""

# =========================
# Importações e dependências
# =========================
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

from sqlalchemy import (create_engine, Column, Integer, String, Float, DateTime,
                        ForeignKey, Text)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from ultralytics import YOLO


# =========================
# Configurações (editar aqui)
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Caminho do melhor peso do YOLO (ajuste se necessário)
YOLO_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    str(BASE_DIR / "runs" / "detect" / "train_fixlabels" / "weights" / "best.pt")
)

# Confiança mínima para considerar uma detecção
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))

# Classes ESSENCIAIS (em PT-BR) — se faltar qualquer uma, marcamos alerta
ESSENCIAIS_PT = ["capacete", "colete", "máscara", "óculos", "luvas"]

# Pastas estáticas (para salvar frames anotados)
STATIC_DIR = BASE_DIR / "static"
FRAMES_DIR = STATIC_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Banco de dados SQLite
SQLITE_URL = os.getenv("SQLITE_URL", f"sqlite:///{BASE_DIR/'protegeai.db'}")


# =========================
# Banco de dados (SQLAlchemy)
# =========================
Base = declarative_base()

class FrameLog(Base):
    """Um registro por frame processado."""
    __tablename__ = "frame_logs"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)   # quando foi salvo
    source = Column(String(256))                             # ex.: live:0, upload, rtsp://...
    image_path = Column(String(512), nullable=True)          # caminho do frame anotado salvo
    total_dets = Column(Integer, default=0)                  # número total de detecções
    is_alert = Column(Integer, default=0)                    # 1 = alerta (faltou algum EPI essencial)
    summary_json = Column(Text)                              # resumo por classe (em PT), JSON em string
    boxes = relationship("Detection", back_populates="frame",
                         cascade="all, delete-orphan")

class Detection(Base):
    """Uma linha por caixa detectada no frame."""
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey("frame_logs.id", ondelete="CASCADE"))
    cls_name = Column(String(64))    # nome da classe (em PT no momento do salvamento)
    conf = Column(Float)             # confiança (0..1)
    x1 = Column(Integer); y1 = Column(Integer)
    x2 = Column(Integer); y2 = Column(Integer)
    frame = relationship("FrameLog", back_populates="boxes")

# Inicialização do banco
engine = create_engine(SQLITE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base.metadata.create_all(engine)


# =========================
# Modelo YOLO (carregado 1x)
# =========================
model = YOLO(YOLO_WEIGHTS)
CLASS_NAMES = model.names  # dict {id: 'helmet', ...} a partir do treino (geralmente em EN)

# Mapa EN -> PT apenas para exibição/relato (não precisa retreinar)
MAP_EN_PT = {
    "helmet":  "capacete",
    "vest":    "colete",
    "mask":    "mascara",
    "goggles": "oculos",
    "gloves":  "luvas",
}
def en_para_pt(nome: str) -> str:
    """Converte rótulo do modelo (EN) para PT-BR para exibir/salvar."""
    if not isinstance(nome, str):
        return str(nome)
    return MAP_EN_PT.get(nome.lower(), nome)

def inferir(frame_bgr, conf=CONF_THRESHOLD):
    """Roda YOLO no frame e retorna detecções com nomes em EN (originais do modelo)."""
    res = model.predict(frame_bgr, conf=conf, imgsz=640, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            cls_en = CLASS_NAMES.get(cls_id, str(cls_id))
            confv = float(b.conf[0].item()) if b.conf is not None else 0.0
            dets.append({"cls_en": cls_en, "conf": confv, "xyxy": (x1, y1, x2, y2)})
    return dets

def desenhar_e_salvar(frame_bgr, dets_en):
    """Desenha caixas com rótulos EM PT e salva JPEG em static/frames. Retorna caminho string."""
    img = frame_bgr.copy()
    for d in dets_en:
        x1, y1, x2, y2 = d["xyxy"]
        rotulo = f'{en_para_pt(d["cls_en"])} {d["conf"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 220, 0), -1)
        cv2.putText(img, rotulo, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    fname = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    out = FRAMES_DIR / fname
    cv2.imwrite(str(out), img)
    return str(out)

def resumir_e_sinalizar_pt(dets_en):
    """
    Constrói um resumo por classe EM PT e decide ALERTA:
    - Alerta quando faltar qualquer item em ESSENCIAIS_PT.
    """
    resumo_pt = {}
    for d in dets_en:
        cls_pt = en_para_pt(d["cls_en"])
        resumo_pt[cls_pt] = resumo_pt.get(cls_pt, 0) + 1
    faltantes = [c for c in ESSENCIAIS_PT if resumo_pt.get(c, 0) == 0]
    alerta = 1 if faltantes else 0
    return resumo_pt, alerta, faltantes


# =========================
# Estado da câmera ao vivo
# =========================
cap = None
live_lock = threading.Lock()
live_rodando = False
live_fonte = 0  # índice da webcam por padrão

# Último estado (para /live_status e /save_current)
ultimo_frame = None          # BGR
ultimo_dets_en = []          # lista de detecções (EN)
ultimo_resumo_pt = {}        # dict {classe_pt: contagem}
ultimo_alerta = False
ultimo_faltantes_pt = []

def abrir_camera(fonte=0):
    """Abre webcam (índice) ou URL RTSP/arquivo. Retorna True se deu certo."""
    global cap
    cap = cv2.VideoCapture(int(fonte), cv2.CAP_DSHOW) if str(fonte).isdigit() else cv2.VideoCapture(fonte)
    return cap is not None and cap.isOpened()

def fechar_camera():
    """Fecha a câmera se estiver aberta."""
    global cap
    if cap:
        cap.release()
    cap = None

def gerador_mjpeg():
    """
    Gera um stream MJPEG (multipart) já com caixas desenhadas EM PT-BR.
    Também atualiza os “últimos” valores para a página (status ao vivo).
    """
    global ultimo_frame, ultimo_dets_en, ultimo_resumo_pt, ultimo_alerta, ultimo_faltantes_pt
    while True:
        with live_lock:
            rodando = live_rodando
            c = cap
        if not rodando or c is None or not c.isOpened():
            # Placeholder amigável quando a câmera não está rodando
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Câmera parada. Clique em 'Ligar câmera'.", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", blank)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.3)
            continue

        ok, frame = c.read()
        if not ok:
            time.sleep(0.05)
            continue

        # Inferência (EN) + resumo/alerta (PT)
        dets_en = inferir(frame)
        resumo_pt, alerta, faltantes_pt = resumir_e_sinalizar_pt(dets_en)

        # Desenhar caixas com rótulos EM PT na cópia do frame
        overlay = frame.copy()
        for d in dets_en:
            x1, y1, x2, y2 = d["xyxy"]
            rotulo = f'{en_para_pt(d["cls_en"])} {d["conf"]:.2f}'
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 220, 0), -1)
            cv2.putText(overlay, rotulo, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)

        # Atualiza “últimos” para a UI
        with live_lock:
            ultimo_frame = frame
            ultimo_dets_en = dets_en
            ultimo_resumo_pt = resumo_pt
            ultimo_alerta = bool(alerta)
            ultimo_faltantes_pt = faltantes_pt

        ok, jpg = cv2.imencode(".jpg", overlay)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.01)  # folga de CPU/GPU


# =========================
# Flask (rotas web e API)
# =========================
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))

@app.get("/")
def pagina_inicial():
    """Renderiza a página única (index.html)."""
    return render_template("index.html")

# --------- Controle da câmera ao vivo ---------
@app.post("/start_live")
def iniciar_live():
    """Abre a câmera/stream e começa a produzir o MJPEG."""
    global live_rodando, live_fonte
    fonte = request.form.get("source") or request.args.get("source") or "0"
    live_fonte = int(fonte) if str(fonte).isdigit() else fonte
    with live_lock:
        if live_rodando:
            return jsonify({"status": "already_running", "source": live_fonte})
        if not abrir_camera(live_fonte):
            return jsonify({"status": "error", "message": "Não foi possível abrir a câmera"}), 400
        live_rodando = True
    return jsonify({"status": "running", "source": live_fonte})

@app.post("/stop_live")
def parar_live():
    """Interrompe a câmera/stream."""
    global live_rodando
    with live_lock:
        live_rodando = False
        fechar_camera()
    return jsonify({"status": "stopped"})

@app.get("/stream.mjpg")
def stream_mjpeg():
    """Endpoint do fluxo MJPEG consumido pelo <img> da página."""
    return Response(gerador_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/live_status")
def status_ao_vivo():
    """Retorna o resumo atual (EM PT) para preencher os chips da página."""
    with live_lock:
        return jsonify({
            "summary": ultimo_resumo_pt,
            "is_alert": bool(ultimo_alerta),
            "missing_essentials": ultimo_faltantes_pt
        })

@app.post("/save_current")
def salvar_atual():
    """
    Salva no banco o último frame visível (com anotação em disco).
    Use ?reason=helmet_missing para marcar manualmente como alerta de capacete.
    """
    motivo = request.form.get("reason") or request.args.get("reason")
    with live_lock:
        frame = None if ultimo_frame is None else ultimo_frame.copy()
        dets_en = list(ultimo_dets_en)
        resumo_pt = dict(ultimo_resumo_pt)
        alerta = bool(ultimo_alerta)

    if frame is None:
        return jsonify({"status": "error", "message": "Ainda não há frame ao vivo"}), 400
    if motivo == "helmet_missing":
        alerta = True  # força alerta manual

    img_path = desenhar_e_salvar(frame, dets_en) if dets_en or alerta else None

    with SessionLocal() as db:
        fl = FrameLog(
            source=f"live:{live_fonte}",
            image_path=img_path,
            total_dets=len(dets_en),
            is_alert=1 if alerta else 0,
            summary_json=json.dumps(resumo_pt, ensure_ascii=False)
        )
        db.add(fl); db.flush()
        # Salva cada caixa com NOME EM PT
        for d in dets_en:
            x1, y1, x2, y2 = d["xyxy"]
            db.add(Detection(frame_id=fl.id,
                             cls_name=en_para_pt(d["cls_en"]),
                             conf=d["conf"], x1=x1, y1=y1, x2=x2, y2=y2))
        db.commit()

    return jsonify({"status": "ok", "saved_image": img_path})

# --------- Upload simples (API) ---------
@app.post("/predict")
def prever_upload():
    """Recebe uma imagem via multipart/form-data e retorna o resultado (JSON) em PT."""
    if "file" not in request.files:
        return jsonify({"error": "Envie 'file' no form-data."}), 400
    fonte = request.form.get("source", "upload")

    img = cv2.imdecode(np.frombuffer(request.files["file"].read(), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Imagem inválida"}), 400

    dets_en = inferir(img)
    resumo_pt, alerta, faltantes_pt = resumir_e_sinalizar_pt(dets_en)
    img_path = desenhar_e_salvar(img, dets_en) if dets_en or alerta else None

    with SessionLocal() as db:
        fl = FrameLog(
            source=fonte,
            image_path=img_path,
            total_dets=len(dets_en),
            is_alert=alerta,
            summary_json=json.dumps(resumo_pt, ensure_ascii=False)
        )
        db.add(fl); db.flush()
        for d in dets_en:
            x1, y1, x2, y2 = d["xyxy"]
            db.add(Detection(frame_id=fl.id,
                             cls_name=en_para_pt(d["cls_en"]),
                             conf=d["conf"], x1=x1, y1=y1, x2=x2, y2=y2))
        db.commit()

    return jsonify({
        "total": len(dets_en),
        "is_alert": bool(alerta),
        "missing_essentials": faltantes_pt,  # EM PT
        "summary": resumo_pt,                # EM PT
        "image_annotated": img_path
    })

# --------- Logs e saúde ---------
@app.get("/logs")
def listar_logs():
    """Lista os logs recentes em JSON (com nomes EM PT)."""
    limite = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    with SessionLocal() as db:
        q = db.query(FrameLog).order_by(FrameLog.id.desc())
        total = q.count()
        itens = q.offset(offset).limit(limite).all()
        saida = []
        for it in itens:
            saida.append({
                "id": it.id,
                "timestamp": it.timestamp.isoformat(),
                "source": it.source,
                "image_path": it.image_path,
                "total_dets": it.total_dets,
                "is_alert": bool(it.is_alert),
                "summary": json.loads(it.summary_json) if it.summary_json else {}
            })
    return jsonify({"total": total, "items": saida})

@app.get("/health")
def saude():
    return {"status": "ok"}


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Rode: python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
