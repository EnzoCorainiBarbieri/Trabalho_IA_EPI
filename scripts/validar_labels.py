#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de labels YOLO para datasets Ultralytics.

Verifica:
- imagem ↔ label correspondente
- formato das linhas (class cx cy w h)
- classes dentro do intervalo
- valores normalizados [0,1]
- w,h > 0
- caixas muito pequenas (limiar configurável)
- IoU alto entre caixas da MESMA classe (possíveis duplicatas)

Gera:
- resumo no terminal
- CSV detalhado em scripts/validar_labels_report.csv
- (opcional) move arquivos problemáticos para quarantine/

Uso:
    python scripts/validar_labels.py --root datasets/epi --classes helmet vest goggles mask gloves
    # ou
    python scripts/validar_labels.py --root datasets/epi --classes-file classes.txt

Opções úteis:
    --min-wh 0.01           # considera caixa com w<0.01 ou h<0.01 como muito pequena
    --dup-iou 0.9           # IoU acima disso sinaliza duplicidade
    --quarantine            # move imagem + label com problema para datasets/epi/quarantine/<split>/
"""

from pathlib import Path
import csv
import argparse
from typing import List, Tuple, Optional
import math

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def load_classes(args) -> List[str]:
    if args.classes_file:
        txt = Path(args.classes_file).read_text(encoding="utf-8").strip().splitlines()
        names = [line.strip() for line in txt if line.strip()]
        if not names:
            raise ValueError("classes.txt vazio!")
        return names
    if args.classes:
        return [c.strip() for c in args.classes]
    # fallback
    return ["helmet","vest","goggles","mask","gloves"]

def safe_read_label(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]

def parse_line(line: str) -> Optional[Tuple[int,float,float,float,float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
    except Exception:
        return None
    return cid, cx, cy, w, h

def within01(*vals) -> bool:
    return all(0.0 <= v <= 1.0 for v in vals)

def iou_yolo(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    # a,b = (cx,cy,w,h) normalizados
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, a[2]) * max(0.0, a[3])
    area_b = max(0.0, b[2]) * max(0.0, b[3])
    union = area_a + area_b - inter + 1e-12
    return inter / union if union > 0 else 0.0

def scan_split(root: Path, split: str, class_count: int, min_wh: float, dup_iou: float):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.exists():
        return [], {"images":0,"labels_missing":0,"labels_empty":0,"boxes":0,"bad":0}

    problems = []
    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    stats = {"images":len(images),"labels_missing":0,"labels_empty":0,"boxes":0,"bad":0}

    for img in images:
        rel = img.relative_to(img_dir)
        label = lbl_dir / rel.with_suffix(".txt")
        lines = safe_read_label(label)

        if not label.exists():
            problems.append((split, "MISSING_LABEL", str(rel), "label não encontrado para a imagem"))
            stats["labels_missing"] += 1
            continue

        if len(lines) == 0:
            problems.append((split, "EMPTY_LABEL", str(rel), "arquivo .txt vazio (sem objetos)"))
            stats["labels_empty"] += 1
            continue

        # validar sintaxe/semântica
        parsed = []
        for i, ln in enumerate(lines, 1):
            if not ln.strip():
                problems.append((split, "EMPTY_LINE", f"{rel}#L{i}", "linha vazia"))
                stats["bad"] += 1
                continue
            tup = parse_line(ln)
            if tup is None:
                problems.append((split, "BAD_FORMAT", f"{rel}#L{i}", f"esperado: 'cls cx cy w h' — obtido: '{ln}'"))
                stats["bad"] += 1
                continue
            cid, cx, cy, w, h = tup
            if cid < 0 or cid >= class_count:
                problems.append((split, "BAD_CLASS_ID", f"{rel}#L{i}", f"classe {cid} fora de [0,{class_count-1}]"))
                stats["bad"] += 1
            if not within01(cx, cy, w, h):
                problems.append((split, "OUT_OF_RANGE", f"{rel}#L{i}", f"valores fora de [0,1]: {cx},{cy},{w},{h}"))
                stats["bad"] += 1
            if w <= 0 or h <= 0:
                problems.append((split, "ZERO_SIZE", f"{rel}#L{i}", f"w/h <= 0: {w},{h}"))
                stats["bad"] += 1
            if w < min_wh or h < min_wh:
                problems.append((split, "SMALL_BOX", f"{rel}#L{i}", f"caixa muito pequena (w={w:.4f}, h={h:.4f} < {min_wh})"))
                # não conta como 'bad' fatal; é aviso

            parsed.append((cid, cx, cy, w, h))

        # IoU alto entre caixas da MESMA classe (suspeita de duplicata)
        # (não marca 'bad' fatal; é aviso)
        for i in range(len(parsed)):
            for j in range(i+1, len(parsed)):
                ci, cxi, cyi, wi, hi = parsed[i]
                cj, cxj, cyj, wj, hj = parsed[j]
                if ci != cj:
                    continue
                iou = iou_yolo((cxi,cyi,wi,hi), (cxj,cyj,wj,hj))
                if iou >= dup_iou:
                    problems.append((split, "DUP_IOU", str(rel), f"duas caixas da classe {ci} com IoU={iou:.2f} ≥ {dup_iou}"))

        stats["boxes"] += sum(1 for x in parsed if x is not None)

    return problems, stats

def write_csv(report_path: Path, rows: List[Tuple[str,str,str,str]]):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","code","file_or_ref","detail"])
        for r in rows:
            w.writerow(r)

def maybe_quarantine(root: Path, problems: List[Tuple[str,str,str,str]]):
    """Move imagem + label com problemas para quarantine/<split>/ mantendo subpastas."""
    # considerar apenas problemas ligados a um arquivo específico
    targets = {}
    for split, code, file_or_ref, _ in problems:
        # file_or_ref pode vir como "sub/arquivo.jpg" ou "sub/arquivo.jpg#L3"
        rel_img = file_or_ref.split("#")[0]
        targets.setdefault(split, set()).add(rel_img)

    for split, files in targets.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        q_img = root / "quarantine" / split / "images"
        q_lbl = root / "quarantine" / split / "labels"
        for rel in files:
            img_path = img_dir / rel
            lbl_path = lbl_dir / Path(rel).with_suffix(".txt")
            if img_path.exists():
                (q_img / Path(rel).parent).mkdir(parents=True, exist_ok=True)
                img_path.replace(q_img / rel)
            if lbl_path.exists():
                (q_lbl / Path(rel).parent).mkdir(parents=True, exist_ok=True)
                lbl_path.replace(q_lbl / Path(rel).with_suffix(".txt"))

def main():
    ap = argparse.ArgumentParser(description="Validador de labels YOLO.")
    ap.add_argument("--root", type=str, default="datasets/epi", help="raiz do dataset (contém images/ e labels/)")
    ap.add_argument("--classes", nargs="*", help="nomes das classes na ordem dos IDs (0..N-1)")
    ap.add_argument("--classes-file", type=str, help="arquivo classes.txt (um nome por linha)")
    ap.add_argument("--min-wh", type=float, default=0.01, help="limiar mínimo de w/h (normalizados) para alertar 'caixa pequena'")
    ap.add_argument("--dup-iou", type=float, default=0.90, help="IoU para sinalizar duplicidade dentro da mesma classe")
    ap.add_argument("--report", type=str, default="scripts/validar_labels_report.csv", help="caminho do CSV de saída")
    ap.add_argument("--quarantine", action="store_true", help="mover arquivos com problemas para quarantine/")
    args = ap.parse_args()

    root = Path(args.root)
    classes = load_classes(args)
    print(f"[INFO] Classes ({len(classes)}): {classes}")

    all_problems = []
    grand_stats = {"images":0,"labels_missing":0,"labels_empty":0,"boxes":0,"bad":0}
    for split in ["train","val","test"]:
        probs, stats = scan_split(root, split, len(classes), args.min_wh, args.dup_iou)
        all_problems.extend(probs)
        for k in grand_stats:
            grand_stats[k] += stats.get(k, 0)

        # resumo por split
        print(f"\n== {split.upper()} ==")
        print(f"Imagens: {stats['images']}")
        print(f"Labels faltando: {stats['labels_missing']}")
        print(f"Labels vazios:   {stats['labels_empty']}")
        print(f"Caixas válidas:  {stats['boxes']}")
        print(f"Problemas 'bad': {stats['bad']} (formato/range/classe/zerosize)")
        # contagem de códigos por split
        if probs:
            codes = {}
            for _, code, *_ in probs:
                codes[code] = codes.get(code, 0) + 1
            top = ", ".join([f"{c}:{n}" for c,n in sorted(codes.items(), key=lambda x: -x[1])[:8]])
            print(f"Avisos/detalhes: {top}")

    # salvar CSV
    write_csv(Path(args.report), all_problems)
    print(f"\n[INFO] Relatório salvo em: {args.report}")

    # mover problemáticos (opcional)
    if args.quarantine and all_problems:
        maybe_quarantine(root, all_problems)
        print("[INFO] Arquivos problemáticos movidos para quarantine/ (verifique antes de treinar).")

    # resumo geral
    print("\n== RESUMO GERAL ==")
    for k,v in grand_stats.items():
        print(f"{k}: {v}")
    if not all_problems:
        print("\n[OK] Nenhum problema encontrado. Pode treinar tranquilo! ✅")
    else:
        print("\n[INFO] Existem itens a revisar (ver CSV). Corrija e rode novamente o validador.")

if __name__ == "__main__":
    main()
