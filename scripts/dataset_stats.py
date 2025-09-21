#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

CLASSES = ["helmet","vest","goggles","mask","gloves"]

def count_lines(p):
    if not p.exists(): return 0
    s = p.read_text(encoding="utf-8").strip()
    return 0 if not s else len(s.splitlines())

def main(root="datasets/epi"):
    root = Path(root)
    for split in ["train","val","test"]:
        img_dir = root/"images"/split
        lbl_dir = root/"labels"/split
        if not img_dir.exists(): 
            continue
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".webp")]
        total_imgs = len(imgs)
        empty = 0
        per_class = {i:0 for i in range(len(CLASSES))}
        for img in imgs:
            rel = img.relative_to(img_dir).with_suffix(".txt")
            lpath = lbl_dir/rel
            if not lpath.exists():
                empty += 1
                continue
            lines = lpath.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5: 
                    continue
                try:
                    cid = int(parts[0])
                except:
                    continue
                if cid in per_class:
                    per_class[cid] += 1

        print(f"\n== {split.upper()} ==")
        print(f"Imagens: {total_imgs}")
        print(f"Sem rótulo (ou .txt vazio): {empty} ({(empty/total_imgs*100 if total_imgs else 0):.1f}%)")
        for cid, cnt in per_class.items():
            print(f"- {CLASSES[cid]}: {cnt} boxes")

if __name__ == "__main__":
    main()
