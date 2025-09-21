#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revisor Visual de Labels YOLO (PT-BR, UI limpa)
- Varre images/<split>/... e edita labels/<split>/...
- UI minimalista com painel translúcido de status/ajuda
- Comandos em PT-BR e destaque do item (classe) selecionado
"""

from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
from dataclasses import dataclass

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ------------------ Modelo de caixa ------------------
@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, W, H):
        x1 = int((self.cx - self.w/2) * W)
        y1 = int((self.cy - self.h/2) * H)
        x2 = int((self.cx + self.w/2) * W)
        y2 = int((self.cy + self.h/2) * H)
        return x1, y1, x2, y2

    @staticmethod
    def from_xyxy(cls, x1, y1, x2, y2, W, H):
        x1, y1 = max(0, min(W-1, x1)), max(0, min(H-1, y1))
        x2, y2 = max(0, min(W-1, x2)), max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            # caixa vazia -> cria algo mínimo
            x2 = x1 + 2
            y2 = y1 + 2
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        cx = (x1 + x2) / (2*W)
        cy = (y1 + y2) / (2*H)
        return Box(cls, np.clip(cx,0,1), np.clip(cy,0,1), np.clip(w,1e-6,1), np.clip(h,1e-6,1))

# ------------------ Utilidades ------------------
def load_names_from_yaml(data_yaml: Path):
    try:
        meta = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        names = meta.get("names", {})
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        elif isinstance(names, list):
            return names
        return []
    except Exception:
        return []

def load_classes(args):
    if args.data:
        names = load_names_from_yaml(Path(args.data))
        if names:
            return names
    if args.classes_file:
        txt = Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        names = [t.strip() for t in txt if t.strip()]
        if names:
            return names
    if args.classes:
        return args.classes
    return ["helmet","vest","goggles","mask","gloves"]

def list_images(dataset_root: Path, split: str):
    ims = []
    splits = ["train","val","test"] if split == "all" else [split]
    for sp in splits:
        img_dir = dataset_root/"images"/sp
        if not img_dir.exists(): 
            continue
        ims.extend([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    ims.sort(key=lambda p: str(p).lower())
    return ims

def read_labels_for(img_path: Path, dataset_root: Path):
    # img_path: .../images/<split>/subpasta/.../arquivo.ext
    parts = img_path.parts
    if "images" not in parts:
        return [], None
    idx = parts.index("images")
    split = parts[idx+1] if idx+1 < len(parts) else None
    rel_after_split = Path(*parts[idx+2:])
    lbl_dir = dataset_root/"labels"/split
    lbl_path = lbl_dir/rel_after_split.with_suffix(".txt")
    boxes = []
    if lbl_path.exists():
        for ln in lbl_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip(): 
                continue
            ss = ln.split()
            if len(ss) != 5:
                continue
            try:
                cls = int(ss[0]); cx,cy,w,h = map(float, ss[1:])
                boxes.append(Box(cls, cx, cy, w, h))
            except:
                pass
    return boxes, lbl_path

def save_labels(lbl_path: Path, boxes):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        lbl_path.write_text("", encoding="utf-8")
    else:
        s = "\n".join(f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes)
        lbl_path.write_text(s, encoding="utf-8")

def class_color(index: int):
    # paleta consistente por classe (BGR)
    rng = np.random.default_rng(seed=index * 9973 + 123)
    c = rng.integers(80, 230, size=3).tolist()
    return int(c[2]), int(c[1]), int(c[0])  # para contraste no OpenCV

# ------------------ Estado da UI ------------------
class UI:
    def __init__(self, class_names, window="Revisor YOLO"):
        self.names = class_names
        self.win = window
        self.img = None
        self.view = None
        self.scale = 1.0
        self.W = 0
        self.H = 0
        self.boxes = []
        self.sel = -1
        self.mode = "idle"      # idle|drag|resize|draw
        self.drag_off = (0,0)
        self.rs_corner = None
        self.draw_start = None
        self.help_on = True
        self.undo_stack = []
        self.path_info = ""
        self.index_info = ""

    def set_image(self, img_bgr, boxes, path_info="", index_info=""):
        self.img = img_bgr
        self.H, self.W = img_bgr.shape[:2]
        # fit sutil no maior lado (1280)
        s = min(1.0, 1280.0 / max(self.W, self.H))
        self.scale = s
        if s != 1.0:
            self.view = cv2.resize(img_bgr, (int(self.W*s), int(self.H*s)), interpolation=cv2.INTER_AREA)
        else:
            self.view = img_bgr.copy()
        self.boxes = [Box(b.cls,b.cx,b.cy,b.w,b.h) for b in boxes]
        self.sel = -1
        self.mode = "idle"
        self.undo_stack.clear()
        self.path_info = path_info
        self.index_info = index_info

    # --------- desenho ----------
    def overlay_panel(self, canvas, text_lines, x=10, y=10, alpha=0.20, pad=10):
        # painel translúcido
        if not text_lines:
            return
        w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0] for t in text_lines) + pad*2
        h = (len(text_lines)*24) + pad*2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,0,0), -1)
        cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)
        yp = y + pad + 18
        for t in text_lines:
            cv2.putText(canvas, t, (x+pad, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
            yp += 24

    def draw(self):
        canvas = self.view.copy()

        # desenhar caixas
        for i, b in enumerate(self.boxes):
            x1,y1,x2,y2 = b.to_xyxy(self.W, self.H)
            if self.scale != 1.0:
                x1 = int(x1*self.scale); y1 = int(y1*self.scale)
                x2 = int(x2*self.scale); y2 = int(y2*self.scale)
            col = class_color(b.cls)
            thick = 3 if i != self.sel else 5
            cv2.rectangle(canvas, (x1,y1), (x2,y2), col, thick)
            # etiqueta discreta
            label = self.names[b.cls] if 0 <= b.cls < len(self.names) else f"cls {b.cls}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(canvas, (x1, max(0, y1-th-8)), (x1+tw+10, y1), col, -1)
            cv2.putText(canvas, label, (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2, cv2.LINE_AA)

            # cantos para resize (apenas selecionada)
            if i == self.sel:
                for (cx,cy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    cv2.circle(canvas, (cx,cy), 6, col, -1)

        # painel de status (topo-esquerdo)
        sel_txt = "-"
        if self.sel >= 0 and 0 <= self.boxes[self.sel].cls < len(self.names):
            sel_txt = self.names[self.boxes[self.sel].cls]
        stats = [
            f"Arquivo: {self.path_info}",
            f"Imagem: {self.index_info}",
            f"Selecionado: {sel_txt}",
        ]
        self.overlay_panel(canvas, stats, x=10, y=10, alpha=0.25)

        # painel de ajuda (inferior-esquerdo)
        if self.help_on:
            help_lines = [
                "Mouse: mover/redimensionar | Clique para selecionar",
                "A: nova caixa  |  M: mudar classe  |  0–9: definir classe",
                "D: apagar  |  U: desfazer  |  S: salvar",
                "Enter: salvar + próxima  |  Backspace: salvar + anterior",
                "H: ajuda on/off  |  Q/Esc: sair"
            ]
            h = canvas.shape[0]
            self.overlay_panel(canvas, help_lines, x=10, y=h-10-(len(help_lines)*24+20), alpha=0.22)

        return canvas

    # --------- mouse ----------
    def on_mouse(self, event, x, y, flags, _userdata=None):
        X = int(x / self.scale); Y = int(y / self.scale)

        def inside(i, X, Y):
            x1,y1,x2,y2 = self.boxes[i].to_xyxy(self.W, self.H)
            return x1<=X<=x2 and y1<=Y<=y2

        def near_corner(i, X, Y, r=10):
            x1,y1,x2,y2 = self.boxes[i].to_xyxy(self.W, self.H)
            corners = [("tl",(x1,y1)),("tr",(x2,y1)),("bl",(x1,y2)),("br",(x2,y2))]
            for name,(cx,cy) in corners:
                if abs(X-cx)<=r and abs(Y-cy)<=r:
                    return name
            return None

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "draw":
                self.draw_start = (X,Y)
            else:
                # prioriza canto da selecionada
                if self.sel >= 0:
                    c = near_corner(self.sel, X, Y)
                    if c:
                        self.mode = "resize"
                        self.rs_corner = c
                        self.push_undo()
                        return
                # seleção/drag
                sel = -1
                for i in range(len(self.boxes)-1, -1, -1):
                    if inside(i, X, Y):
                        sel = i
                        break
                self.sel = sel
                if self.sel >= 0:
                    c = near_corner(self.sel, X, Y)
                    if c:
                        self.mode = "resize"
                        self.rs_corner = c
                        self.push_undo()
                    else:
                        self.mode = "drag"
                        x1,y1,_,_ = self.boxes[self.sel].to_xyxy(self.W,self.H)
                        self.drag_off = (X - x1, Y - y1)
                        self.push_undo()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "drag" and self.sel >= 0:
                b = self.boxes[self.sel]
                x1,y1,x2,y2 = b.to_xyxy(self.W,self.H)
                w = x2 - x1; h = y2 - y1
                nx1 = int(X - self.drag_off[0]); ny1 = int(Y - self.drag_off[1])
                nx2 = nx1 + w; ny2 = ny1 + h
                nx1 = np.clip(nx1, 0, self.W-1); ny1 = np.clip(ny1, 0, self.H-1)
                nx2 = np.clip(nx2, 0, self.W-1); ny2 = np.clip(ny2, 0, self.H-1)
                if nx2 > nx1 and ny2 > ny1:
                    self.boxes[self.sel] = Box.from_xyxy(b.cls, nx1, ny1, nx2, ny2, self.W, self.H)

            elif self.mode == "resize" and self.sel >= 0:
                b = self.boxes[self.sel]
                x1,y1,x2,y2 = b.to_xyxy(self.W,self.H)
                if self.rs_corner == "tl": x1,y1 = X,Y
                elif self.rs_corner == "tr": x2,y1 = X,Y
                elif self.rs_corner == "bl": x1,y2 = X,Y
                elif self.rs_corner == "br": x2,y2 = X,Y
                x1 = np.clip(x1, 0, self.W-1); y1 = np.clip(y1, 0, self.H-1)
                x2 = np.clip(x2, 0, self.W-1); y2 = np.clip(y2, 0, self.H-1)
                if x2 > x1 and y2 > y1:
                    self.boxes[self.sel] = Box.from_xyxy(b.cls, x1,y1,x2,y2, self.W, self.H)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "draw" and self.draw_start:
                x0,y0 = self.draw_start
                x1,y1 = min(x0,X), min(y0,Y)
                x2,y2 = max(x0,X), max(y0,Y)
                if (x2-x1) > 4 and (y2-y1) > 4:
                    cls = self.boxes[self.sel].cls if self.sel>=0 else 0
                    self.push_undo()
                    self.boxes.append(Box.from_xyxy(cls, x1,y1,x2,y2, self.W, self.H))
                    self.sel = len(self.boxes)-1
            self.mode = "idle"
            self.draw_start = None
            self.rs_corner = None

    # --------- ações ----------
    def toggle_help(self): self.help_on = not self.help_on
    def start_draw(self): self.mode = "draw"
    def cycle_class(self):
        if self.sel >= 0 and self.names:
            self.boxes[self.sel].cls = (self.boxes[self.sel].cls + 1) % len(self.names)
    def set_class_num(self, cid):
        if self.sel >= 0 and self.names:
            self.boxes[self.sel].cls = int(np.clip(cid, 0, len(self.names)-1))
    def delete_sel(self):
        if self.sel >= 0:
            self.push_undo()
            self.boxes.pop(self.sel); self.sel = -1
    def push_undo(self):
        snap = [Box(b.cls,b.cx,b.cy,b.w,b.h) for b in self.boxes]
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 32: self.undo_stack.pop(0)
    def undo(self):
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()

# ------------------ Loop principal ------------------
def main():
    ap = argparse.ArgumentParser(description="Revisor visual YOLO (PT-BR, UI limpa)")
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train","val","test","all"])
    ap.add_argument("--data", type=str, default=None, help="data.yaml para nomes de classes")
    ap.add_argument("--classes-file", type=str, default=None, help="classes.txt (um por linha)")
    ap.add_argument("--classes", nargs="*", help="nomes das classes")
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    names = load_classes(args)
    imgs = list_images(dataset_root, args.split)
    if not imgs:
        print("[ERRO] Nenhuma imagem encontrada.")
        return

    ui = UI(names, window="Revisor YOLO")
    cv2.namedWindow(ui.win, cv2.WINDOW_NORMAL)

    i = np.clip(args.start_index, 0, len(imgs)-1)
    while 0 <= i < len(imgs):
        img_path = imgs[i]
        boxes, lbl_path = read_labels_for(img_path, dataset_root)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Não consegui abrir: {img_path}")
            i += 1
            continue

        # info
        # split é a pasta após 'images'
        parts = img_path.parts
        split = parts[parts.index("images")+1] if "images" in parts else "?"
        rel_after_split = Path(*parts[parts.index("images")+2:]) if "images" in parts else img_path.name
        path_info = f"{split}/{rel_after_split}"
        index_info = f"{i+1}/{len(imgs)}"

        ui.set_image(bgr, boxes, path_info=path_info, index_info=index_info)
        cv2.setMouseCallback(ui.win, ui.on_mouse)

        while True:
            canvas = ui.draw()
            cv2.imshow(ui.win, canvas)
            k = cv2.waitKey(12) & 0xFF

            if k in (ord('q'), 27):  # q / esc
                return
            elif k == ord('h'): ui.toggle_help()
            elif k == ord('a'): ui.start_draw()
            elif k == ord('m'): ui.cycle_class()
            elif k == ord('d'): ui.delete_sel()
            elif k == ord('u'): ui.undo()
            elif k == ord('s'):
                save_labels(lbl_path, ui.boxes)
                print(f"[OK] Salvo: {lbl_path}")
            elif k == 13:  # Enter
                save_labels(lbl_path, ui.boxes)
                i += 1
                break
            elif k == 8:   # Backspace
                save_labels(lbl_path, ui.boxes)
                i -= 1
                break
            elif k in (ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),
                       ord('5'),ord('6'),ord('7'),ord('8'),ord('9')):
                ui.set_class_num(int(chr(k)))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
