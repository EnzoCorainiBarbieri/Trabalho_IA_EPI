#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autolabel com OWL-ViT (zero-shot) para dataset YOLOv8.

Fluxo:
- Lê imagens em: datasets/epi/images/{train,val,test}/{classe}/...
- Usa prompts PT/EN por classe (helmet/vest/goggles/mask/gloves)
- Gera labels YOLO em: datasets/epi/labels/{train,val,test}/...
- Cria previews com caixas: datasets/epi/autolabeled_preview/{train,val,test}/...

Requisitos:
    pip install transformers>=4.43 torch pillow numpy
"""

from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ----------------- Config -----------------
DEFAULT_CLASSES = ["helmet", "vest", "goggles", "mask", "gloves"]

PROMPTS = {
    "helmet": [
        "capacete de segurança", "capacete EPI", "hard hat", "safety helmet"
    ],
    "vest": [
        "colete refletivo", "colete de segurança", "reflective safety vest", "high-visibility vest"
    ],
    "goggles": [
        "óculos de proteção", "óculos de segurança", "safety goggles", "protective eyewear"
    ],
    "mask": [
        "máscara de proteção respiratória", "máscara PFF2", "respirator mask", "surgical mask", "N95 mask"
    ],
    "gloves": [
        "luvas de proteção", "luvas EPI", "safety gloves", "work gloves"
    ],
}

MAX_BOXES_PER_CLASS = 5   # limitar número de caixas por classe/imagem (sanidade)
PREVIEW_MAX_SIDE = 1200   # limite do maior lado no preview (só visual)

# ----------------- Utils -----------------
def yolo_norm(xmin, ymin, xmax, ymax, W, H):
    """Converte box absoluta para YOLO (cx,cy,w,h) normalizados [0..1]."""
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    cx = xmin + w / 2.0
    cy = ymin + h / 2.0
    return cx / W, cy / H, w / W, h / H

def iou(box, boxes):
    """IOU de 1 caixa contra N caixas. box e boxes no formato [xmin,ymin,xmax,ymax]."""
    if len(boxes) == 0:
        return np.array([])
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def nms(boxes, scores, iou_thr=0.5):
    """Non-Max Suppression simples."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | None):
    """Obtém tamanho do texto de forma compatível com diferentes versões do Pillow."""
    try:
        # Pillow 8.0+: textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            # Pillow 10+: textlength (altura aproximada)
            w = draw.textlength(text, font=font)
            h = (font.size + 4) if font else 14
            return int(w), int(h)
        except Exception:
            # Fallback
            return max(60, len(text) * 7), (font.size + 4 if font else 14)

def draw_boxes_pil(img_pil: Image.Image, dets, class_names, font=None):
    """Desenha caixas (cls_id, [xmin,ymin,xmax,ymax], score) no PIL.Image."""
    draw = ImageDraw.Draw(img_pil)
    for cls_id, (xmin, ymin, xmax, ymax), score in dets:
        label = f"{class_names[cls_id]} {score:.2f}"
        # caixa
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=3)
        # rótulo
        tw, th = text_size(draw, label, font)
        y0 = max(0, ymin - th - 2)
        draw.rectangle([xmin, y0, xmin + tw + 6, y0 + th + 2], fill=(0, 255, 0))
        draw.text((xmin + 3, y0 + 1), label, fill=(0, 0, 0), font=font)

# ----------------- Core -----------------
def run_autolabel(dataset_root: Path,
                  classes_order,
                  device: str,
                  score_thr: float,
                  nms_iou_thr: float):
    """Executa o autolabel em todas as imagens do dataset."""
    classes_order = [c.lower() for c in classes_order]
    name_to_id = {name: i for i, name in enumerate(classes_order)}

    # Modelo
    print("[INFO] Carregando OWL-ViT (google/owlvit-base-patch32)...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()

    # Prompts: lista de listas (uma lista por classe)
    texts = [PROMPTS.get(c, [c]) for c in classes_order]

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    preview_dir = dataset_root / "autolabeled_preview"

    for split in ["train", "val", "test"]:
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        (preview_dir / split).mkdir(parents=True, exist_ok=True)
        split_dir = images_dir / split
        if not split_dir.exists():
            continue

        # Fonte (opcional; se não achar arial.ttf, usa default)
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = None

        # Percorrer subpastas por classe (como estão organizadas)
        for cls_name in classes_order:
            cls_img_dir = split_dir / cls_name
            if not cls_img_dir.exists():
                continue

            img_paths = [p for p in cls_img_dir.rglob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            if not img_paths:
                continue

            print(f"[INFO] Split={split} Classe={cls_name} Imagens={len(img_paths)}")
            for img_path in img_paths:
                # Carregar imagem
                try:
                    im = Image.open(img_path).convert("RGB")
                except Exception:
                    # pular arquivo defeituoso
                    continue
                W, H = im.size

                # Zero-shot com prompts de TODAS as classes
                inputs = processor(text=texts, images=im, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                target_sizes = torch.tensor([[H, W]], device=device)
                results = processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=score_thr
                )[0]

                boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) else np.zeros((0, 4))
                scores = results["scores"].cpu().numpy() if len(results["scores"]) else np.zeros((0,))
                labels_prompt = results["labels"].cpu().numpy() if len(results["labels"]) else np.zeros((0,), dtype=np.int32)

                # Montar detecções (cid, box, score)
                dets = []
                for b, s, lp in zip(boxes, scores, labels_prompt):
                    cid = int(lp)  # índice da lista de prompts -> índice da classe
                    xmin, ymin, xmax, ymax = [float(v) for v in b.tolist()]
                    xmin = max(0.0, min(W - 1.0, xmin))
                    ymin = max(0.0, min(H - 1.0, ymin))
                    xmax = max(0.0, min(W - 1.0, xmax))
                    ymax = max(0.0, min(H - 1.0, ymax))
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    dets.append((cid, [xmin, ymin, xmax, ymax], float(s)))

                # NMS por classe + limitar quantidade
                final_dets = []
                for cid in range(len(classes_order)):
                    cls_boxes = [d[1] for d in dets if d[0] == cid]
                    cls_scores = [d[2] for d in dets if d[0] == cid]
                    if not cls_boxes:
                        continue
                    keep = nms(cls_boxes, cls_scores, iou_thr=nms_iou_thr)
                    kept = [(cid, cls_boxes[i], cls_scores[i]) for i in keep]
                    kept.sort(key=lambda x: x[2], reverse=True)
                    final_dets.extend(kept[:MAX_BOXES_PER_CLASS])

                # (Opcional) fallback fraco: se não achou nada e a imagem está na pasta da classe X,
                # cria 1 bbox central (score 0.10) só para facilitar revisão.
                if not final_dets and (img_path.parent.name in name_to_id):
                    cid = name_to_id[img_path.parent.name]
                    w0, h0 = int(W * 0.6), int(H * 0.6)
                    xmin = (W - w0) // 2
                    ymin = (H - h0) // 2
                    xmax = xmin + w0
                    ymax = ymin + h0
                    final_dets.append((cid, [float(xmin), float(ymin), float(xmax), float(ymax)], 0.10))

                # Salvar YOLO label
                label_rel = img_path.relative_to(images_dir / split).with_suffix(".txt")
                label_out = labels_dir / split / label_rel
                label_out.parent.mkdir(parents=True, exist_ok=True)

                lines = []
                for cid, (xmin, ymin, xmax, ymax), s in final_dets:
                    cx, cy, ww, hh = yolo_norm(xmin, ymin, xmax, ymax, W, H)
                    if ww <= 0 or hh <= 0:
                        continue
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                # Cria .txt (vazio = sem objetos)
                label_out.write_text("\n".join(lines), encoding="utf-8")

                # Preview (apenas visual)
                im_prev = im.copy()
                draw_boxes_pil(im_prev, final_dets, classes_order, font=font)
                PW, PH = im_prev.size
                scale = 1.0
                if max(PW, PH) > PREVIEW_MAX_SIDE:
                    scale = PREVIEW_MAX_SIDE / float(max(PW, PH))
                    im_prev = im_prev.resize((int(PW * scale), int(PH * scale)), Image.LANCZOS)

                prev_rel = img_path.relative_to(images_dir / split)
                prev_out = (preview_dir / split / prev_rel).with_suffix(".jpg")
                prev_out.parent.mkdir(parents=True, exist_ok=True)
                # garantir modo compatível
                if im_prev.mode in ("RGBA", "P"):
                    im_prev = im_prev.convert("RGB")
                im_prev.save(prev_out, quality=90)

    print("\n[OK] Autolabel concluído.")
    print("Revise as imagens em: datasets/epi/autolabeled_preview/{train,val,test}/")
    print("Se necessário, edite os .txt correspondentes em: datasets/epi/labels/{train,val,test}/")

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Autolabel de EPI usando OWL-ViT (zero-shot).")
    parser.add_argument("--dataset-root", type=str, default="datasets/epi",
                        help="Raiz do dataset (onde existem images/ e labels/).")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Ordem das classes (define os IDs YOLO).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositivo: cuda ou cpu.")
    parser.add_argument("--score-thr", type=float, default=0.20,
                        help="Confiança mínima por detecção (0.0–1.0).")
    parser.add_argument("--nms-iou-thr", type=float, default=0.50,
                        help="IoU do NMS por classe (0.0–1.0).")
    args = parser.parse_args()

    run_autolabel(
        dataset_root=Path(args.dataset_root),
        classes_order=args.classes,
        device=args.device,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou_thr
    )

if __name__ == "__main__":
    main()
