#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para buscar imagens na web, baixar, dividir em train/val/test
e gerar o data.yaml para treino com YOLOv8.

Uso básico:
    python scripts/baixar_criar_dataset.py \
        --dataset-root datasets/epi \
        --classes helmet vest goggles mask gloves \
        --per-class 200 \
        --train 0.7 --val 0.2 --test 0.1 \
        --lang pt

Requisitos:
    pip install -r requirements.txt

Notas:
- O script usa DuckDuckGo Images (lib 'ddgs') -> sem API key.
- Para DETECÇÃO, você ainda precisará criar as anotações em YOLO
  dentro de datasets/epi/labels/* (este script já cria as pastas).
"""

import argparse
import os
import random
import re
import shutil
import string
import sys
from pathlib import Path
from typing import List, Tuple, Iterable

import requests
from ddgs import DDGS
from PIL import Image
from tqdm import tqdm
import yaml

# ---------- Configs gerais ----------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
TIMEOUT = 20
VALID_CONTENT = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# ---------- Helpers ----------
def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"

def ensure_dirs(root: Path):
    for sub in [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def guess_ext_from_headers(headers: dict) -> str:
    ctype = headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ctype == "image/jpeg" or ctype == "image/jpg":
        return ".jpg"
    if ctype == "image/png":
        return ".png"
    if ctype == "image/webp":
        return ".webp"
    return ""  # deixamos vazio e tentamos pela URL

def safe_open_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def random_suffix(n=6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

# ---------- Busca de imagens ----------
def ddg_image_urls(query: str, max_results: int, safe_on: bool = True) -> list[str]:
    """
    Retorna URLs de imagens usando DuckDuckGo, compatível com variações da lib ddgs.
    Algumas versões usam images(query, ...), outras aceitam keywords=...
    """
    results = []
    safesearch = "moderate" if safe_on else "off"
    with DDGS() as ddgs:
        try:
            # Tentativa 1: assinatura que usa 'query' posicional (versões antigas)
            gen = ddgs.images(
                query,
                max_results=max_results,
                safesearch=safesearch,
                size=None,
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
            )
        except TypeError:
            # Tentativa 2: assinatura nova que usa 'keywords='
            gen = ddgs.images(
                keywords=query,
                max_results=max_results,
                safesearch=safesearch,
                size=None,
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
            )

        for r in gen:
            url = r.get("image")
            if url and isinstance(url, str):
                results.append(url)
    return results

# ---------- Download ----------
def download_one(url: str, out_dir: Path, basename: str) -> Path | None:
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        # Validar tipo
        ctype = resp.headers.get("Content-Type", "").split(";")[0].lower()
        if ctype and ctype not in VALID_CONTENT:
            # Tentar mesmo assim se URL tem extensão válida
            ext_from_ct = guess_ext_from_headers(resp.headers)
        else:
            ext_from_ct = guess_ext_from_headers(resp.headers)

        # Ext pela URL (fallback)
        ext_from_url = ""
        m = re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", url, flags=re.I)
        if m:
            ext_from_url = "." + m.group(1).lower()

        ext = ext_from_ct or ext_from_url or ".jpg"
        out_path = out_dir / f"{basename}{ext}"

        # Salvar stream
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Verificar arquivo de imagem válido
        if not safe_open_image(out_path):
            out_path.unlink(missing_ok=True)
            return None

        return out_path
    except Exception:
        return None

# ---------- Split ----------
def split_paths(paths: List[Path], train: float, val: float, test: float) -> Tuple[List[Path], List[Path], List[Path]]:
    assert abs((train + val + test) - 1.0) < 1e-6, "Proporções de split devem somar 1.0"
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * train)
    n_val = int(n * val)
    train_set = paths[:n_train]
    val_set = paths[n_train:n_train + n_val]
    test_set = paths[n_train + n_val:]
    return train_set, val_set, test_set

# ---------- Data.yaml ----------
def write_data_yaml(root: Path, names: List[str]):
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names)}
    }
    out = root.parent / "data.yaml"  # na raiz do projeto sugerida
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[OK] data.yaml criado em: {out}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Baixar imagens e montar dataset YOLOv8.")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Caminho para a raiz do dataset (ex.: datasets/epi)")
    parser.add_argument("--classes", nargs="+", required=True,
                        help="Lista de classes (ex.: helmet vest goggles mask gloves)")
    parser.add_argument("--per-class", type=int, default=150,
                        help="Número alvo de imagens por classe (default=150)")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--lang", type=str, default="pt",
                        help="Língua dos termos de busca (pt ou en).")
    parser.add_argument("--query-extra", type=str, default="EPI segurança trabalho",
                        help="Texto extra a ser adicionado na busca")
    parser.add_argument("--override", action="store_true",
                        help="Se setado, apaga e recria a pasta dataset-root")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if args.override and root.exists():
        print(f"[WARN] Limpando diretório: {root}")
        shutil.rmtree(root)

    ensure_dirs(root)

    # Monta queries por classe, em pt/en
    lang_terms = {
        "pt": {
            "helmet": "capacete",
            "vest": "colete refletivo",
            "goggles": "óculos de proteção",
            "mask": "máscara proteção respiratória",
            "gloves": "luvas de proteção",
        },
        "en": {
            "helmet": "safety helmet",
            "vest": "reflective safety vest",
            "goggles": "safety goggles",
            "mask": "respiratory safety mask",
            "gloves": "safety gloves",
        },
    }
    # Fallback simples: usa a própria classe se não tiver mapeamento
    def term_for(cls: str) -> str:
        m = lang_terms.get(args.lang.lower(), {})
        return m.get(cls.lower(), cls)

    all_downloaded: List[Path] = []
    for cls in args.classes:
        cls_slug = slugify(cls)
        target_dir = root / "images" / "train_candidates" / cls_slug
        target_dir.mkdir(parents=True, exist_ok=True)

        query = f"{term_for(cls)} {args.query_extra}".strip()
        print(f"🔎 Buscando imagens de: {query} (classe: {cls_slug})")

        urls = ddg_image_urls(query=query, max_results=args.per_class * 2, safe_on=True)
        urls = list(dict.fromkeys(urls))  # dedup
        if not urls:
            print(f"[WARN] Nenhum resultado para {cls_slug}.")
            continue

        pbar = tqdm(total=args.per_class, desc=f"Baixando {cls_slug}")
        count = 0
        tried = 0
        for url in urls:
            tried += 1
            outfile = download_one(url, target_dir, f"{cls_slug}_{random_suffix()}")
            if outfile:
                count += 1
                all_downloaded.append(outfile)
                pbar.update(1)
            if count >= args.per_class:
                break
            if tried >= args.per_class * 10:  # limite de tentativas
                break
        pbar.close()
        print(f"[OK] Classe '{cls_slug}': {count} imagens baixadas.")

    # Se nada baixou, encerra
    img_candidates_root = root / "images" / "train_candidates"
    if not any(img_candidates_root.rglob("*.*")):
        print("[ERRO] Nenhuma imagem foi baixada. Tente ajustar as classes/queries.")
        sys.exit(1)

    # Agora vamos juntar todas as imagens baixadas e fazer o split,
    # mas mantendo subpastas por classe para facilitar anotação futura.
    print("[INFO] Organizando em train/val/test ...")
    for cls_dir in sorted(img_candidates_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_slug = cls_dir.name
        imgs = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        if not imgs:
            continue
        train_set, val_set, test_set = split_paths(imgs, args.train, args.val, args.test)

        for subset, subset_paths in [("train", train_set), ("val", val_set), ("test", test_set)]:
            out_cls_dir = root / "images" / subset / cls_slug
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for p in subset_paths:
                dest = out_cls_dir / p.name
                if not dest.exists():
                    shutil.move(str(p), dest)

    # Remover candidatos vazios
    try:
        shutil.rmtree(img_candidates_root)
    except Exception:
        pass

    # Criar data.yaml (na raiz sugerida do projeto)
    write_data_yaml(root, [slugify(c) for c in args.classes])

    # Criar pastas labels (já criado no ensure_dirs) e avisar
    print("[OK] Pastas de labels criadas (você deve anotar bounding boxes em YOLO).")
    print(f"[DONE] Dataset pronto em: {root.resolve()}")
    print("\nDica: Para treinar (após anotar labels):")
    print("  yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640")
    print("Ou para classificar (sem bbox):")
    print("  yolo cls train data=datasets/epi images/train=... # (configuração de classificação)")
    print("\nFerramentas de anotação recomendadas: LabelImg, CVAT ou Roboflow.")
    print("Formato YOLO: um arquivo .txt por imagem com linhas: class x_center y_center width height (normalizados).")

if __name__ == "__main__":
    random.seed(42)
    main()
