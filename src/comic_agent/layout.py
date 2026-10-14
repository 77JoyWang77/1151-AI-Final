"""繪本排版（Pillow，CPU）：台詞 → 圖上的對話框，旁白 → 圖下方的旁白區，輸出每頁 PNG 與 PDF。

台詞不交給 diffusion 生成（中文會亂碼），一律在這裡後製。
對話框的尾巴指向說話者頭部：說話者的框由呼叫端提供（例如評估時 Grounding DINO 偵測到的主角框），
沒有框的說話者（配角）則不畫尾巴。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 依序嘗試：Windows 內建 Noto Sans TC／微軟正黑體、Linux（Colab 需 apt install fonts-noto-cjk）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/NotoSansTC-VF.ttf",
    "C:/Windows/Fonts/msjh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
FONT_WEIGHT = 500  # 可變字型的粗細；預設實例常是最細的 Thin，童書需要較粗的字

# 不可出現在行首的標點（避頭點）
NO_LINE_START = set("，。、！？；：」』）》〉…—,.!?;:)")

PAPER = (250, 246, 236)  # 旁白區底色：米白
INK = (40, 36, 32)
BUBBLE_FILL = (255, 255, 255)
BUBBLE_LINE = (30, 30, 30)


def load_font(size: int, path: str | None = None) -> ImageFont.FreeTypeFont:
    for p in [path] if path else FONT_CANDIDATES:
        if p and Path(p).exists():
            font = ImageFont.truetype(p, size)
            try:
                font.set_variation_by_axes([FONT_WEIGHT])  # 只對可變字型有效
            except (OSError, AttributeError):
                pass
            return font
    raise FileNotFoundError(f"找不到繁體中文字型，請以 font_path 指定（已嘗試：{FONT_CANDIDATES}）")


def wrap_text(text: str, font, max_width: float) -> list[str]:
    """逐字斷行（中文沒有空白可斷）；標點若會落在行首，就跟著前一個字一起換到下一行。"""
    lines, line = [], ""
    for ch in text:
        if line and font.getlength(line + ch) > max_width:
            if ch in NO_LINE_START and len(line) > 1:
                lines.append(line[:-1])
                line = line[-1] + ch
            else:
                lines.append(line)
                line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


BREAK_AFTER = set("，、；：。！？,;:.!?")  # 中文最自然的斷點：在這些標點之後換行


def wrap_balanced(text: str, font, max_width: float) -> list[str]:
    """斷行，優先在逗號等標點之後斷；否則讓各行長度接近（避免最後一行只剩一兩個字）。"""
    lines = wrap_text(text, font, max_width)
    if len(lines) < 2:
        return lines
    if len(lines) == 2:  # 兩行時：找兩段都放得下、且最長一行最短的標點斷點
        splits = [(text[:i + 1], text[i + 1:]) for i, ch in enumerate(text[:-1]) if ch in BREAK_AFTER]
        fits = [s for s in splits if max(font.getlength(s[0]), font.getlength(s[1])) <= max_width]
        if fits:
            return list(min(fits, key=lambda s: max(font.getlength(s[0]), font.getlength(s[1]))))
    lo, hi = font.getlength(text) / len(lines), max_width  # 在保持行數不變下，找最窄的寬度
    for _ in range(12):
        mid = (lo + hi) / 2
        if len(wrap_text(text, font, mid)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return wrap_text(text, font, hi)


def _text_block(lines, font, spacing):
    height = sum(font.getbbox(ln)[3] - font.getbbox(ln)[1] for ln in lines) + spacing * (len(lines) - 1)
    return max(font.getlength(ln) for ln in lines), height


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def place_bubble(size, img_size, head=None, avoid=(), margin=28):
    """選對話框位置：候選為左上、右上，以及說話者頭部的左側、右側；
    挑與「要避開的區域」重疊最少者，平手時挑離頭部最近者。回傳 (x0, y0, x1, y1)。"""
    bw, bh = size
    W, H = img_size
    cands = [(margin, margin), (W - margin - bw, margin)]
    if head is not None:
        hx, hy = head
        cands += [(hx - bw - 60, hy - bh - 20), (hx + 60, hy - bh - 20)]
    rects = []
    for x, y in cands:
        x = min(max(x, margin), W - margin - bw)
        y = min(max(y, margin), H - margin - bh)
        rects.append((x, y, x + bw, y + bh))

    def cost(r):
        dist = 0 if head is None else ((r[0] + r[2]) / 2 - head[0]) ** 2 + ((r[1] + r[3]) / 2 - head[1]) ** 2
        return (sum(_overlap(r, a) for a in avoid), dist)
    return min(rects, key=cost)


def bubble_tail(rect, head, pad, half, max_len):
    """對話框尾巴的三角形頂點；頭部在框內（或沒有頭部）時回傳 None。

    從框上最靠近頭部的邊緣點出發，朝頭部延伸（最多 max_len，且停在頭部前方）；
    底邊兩點放在框內 0.8 個 pad 深處，讓尾巴與框重疊、看起來連在一起。
    """
    if head is None:
        return None
    x0, y0, x1, y1 = rect
    hx, hy = head
    nx, ny = min(max(hx, x0), x1), min(max(hy, y0), y1)  # 框上離頭部最近的點
    dx, dy = hx - nx, hy - ny
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1:
        return None
    step = min(dist * 0.6, max_len) / dist
    tip = (nx + dx * step, ny + dy * step)
    inset = pad * 0.8
    if abs(dy) >= abs(dx):  # 從上／下邊伸出
        bx = min(max(nx, x0 + pad + half), x1 - pad - half)
        by = ny - inset if dy > 0 else ny + inset
        return [(bx - half, by), (bx + half, by), tip]
    by = min(max(ny, y0 + pad + half), y1 - pad - half)  # 從左／右邊伸出
    bx = nx - inset if dx > 0 else nx + inset
    return [(bx, by - half), (bx, by + half), tip]


def draw_bubble(img: Image.Image, text: str, head=None, avoid=(), font=None, max_width_ratio=0.42):
    """在圖上畫一個對話框（含指向 head 的尾巴），回傳對話框外框，供下一個對話框避開。"""
    font = font or load_font(round(img.width * 0.04))
    pad, spacing = round(font.size * 0.7), round(font.size * 0.35)
    lines = wrap_balanced(text, font, img.width * max_width_ratio)
    tw, th = _text_block(lines, font, spacing)
    rect = place_bubble((tw + 2 * pad, th + 2 * pad), img.size, head, avoid)
    draw = ImageDraw.Draw(img)
    line_w = max(3, round(font.size * 0.08))
    tail = bubble_tail(rect, head, pad, font.size * 0.45, max_len=img.width * 0.12)
    if tail:
        draw.polygon(tail, fill=BUBBLE_FILL, outline=BUBBLE_LINE, width=line_w)
    draw.rounded_rectangle(rect, radius=pad, fill=BUBBLE_FILL, outline=BUBBLE_LINE, width=line_w)
    if tail:  # 只填色不描邊再蓋一次：抹掉框線與尾巴的交接處，讓兩者連成一體
        draw.polygon(tail, fill=BUBBLE_FILL)
    y = rect[1] + pad
    for ln in lines:
        top, bottom = font.getbbox(ln)[1], font.getbbox(ln)[3]
        draw.text((rect[0] + pad, y - top), ln, font=font, fill=INK)
        y += bottom - top + spacing
    return rect


def head_point(box) -> tuple[float, float]:
    """由角色框估計頭部位置：框的水平中央、上緣往下 12%。"""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2, y0 + (y1 - y0) * 0.12)


def compose_page(image: Image.Image, panel: dict, speaker_boxes: dict | None = None,
                 page_no: int | None = None, font_path: str | None = None) -> Image.Image:
    """一格 → 一頁：圖上畫角色台詞的對話框，圖下方旁白區放旁白與頁碼。

    speaker_boxes：{角色 id: (x0, y0, x1, y1)}，用來決定對話框尾巴指向與要避開的區域。
    """
    img = image.convert("RGB").copy()
    W = img.width
    speaker_boxes = speaker_boxes or {}
    bubble_font = load_font(round(W * 0.04), font_path)
    # 要避開的區域：每個已知角色的上半身（臉通常在這裡）
    avoid = [(b[0], b[1], b[2], b[1] + (b[3] - b[1]) * 0.45) for b in speaker_boxes.values()]
    for d in panel["dialogue"]:
        if d["speaker"] == "narrator":
            continue
        box = speaker_boxes.get(d["speaker"])
        rect = draw_bubble(img, d["text"], head_point(box) if box else None, avoid, bubble_font)
        avoid.append(rect)

    narration = [d["text"] for d in panel["dialogue"] if d["speaker"] == "narrator"]
    cap_font = load_font(round(W * 0.042), font_path)
    pad, spacing = round(W * 0.05), round(cap_font.size * 0.5)
    lines = [ln for t in narration for ln in wrap_balanced(t, cap_font, W - 2 * pad)]
    text_h = _text_block(lines, cap_font, spacing)[1] if lines else 0
    band_h = max(round(W * 0.16), text_h + 2 * pad)
    page = Image.new("RGB", (W, img.height + band_h), PAPER)
    page.paste(img, (0, 0))
    draw = ImageDraw.Draw(page)
    y = img.height + (band_h - text_h) / 2
    for ln in lines:
        top, bottom = cap_font.getbbox(ln)[1], cap_font.getbbox(ln)[3]
        draw.text(((W - cap_font.getlength(ln)) / 2, y - top), ln, font=cap_font, fill=INK)
        y += bottom - top + spacing
    if page_no is not None:
        num_font = load_font(round(W * 0.025), font_path)
        label = str(page_no)
        draw.text((W - pad - num_font.getlength(label), page.height - pad * 0.8), label,
                  font=num_font, fill=(140, 130, 120))
    return page


def compose_cover(image: Image.Image, title: str, size: tuple[int, int] | None = None,
                  font_path: str | None = None) -> Image.Image:
    """封面：第 1 格的圖鋪滿，下方旁白區放標題；尺寸與內頁相同，PDF 每頁大小一致。"""
    img = image.convert("RGB")
    W = img.width
    H = size[1] if size else round(img.height * 1.16)
    page = Image.new("RGB", (W, H), PAPER)
    page.paste(img, (0, 0))
    font = load_font(round(W * 0.06), font_path)
    lines = wrap_balanced(title, font, W * 0.9)
    _, th = _text_block(lines, font, round(font.size * 0.3))
    draw = ImageDraw.Draw(page)
    y = img.height + (H - img.height - th) / 2
    for ln in lines:
        top, bottom = font.getbbox(ln)[1], font.getbbox(ln)[3]
        draw.text(((W - font.getlength(ln)) / 2, y - top), ln, font=font, fill=INK)
        y += bottom - top + round(font.size * 0.3)
    return page


def save_book(pages: list[Image.Image], out_dir: str | Path) -> Path:
    """每頁存 PNG（page_00 為封面），並合成一份 PDF；回傳 PDF 路徑。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pages):
        p.save(out / f"page_{i:02d}.png")
    pdf = out / "book.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    return pdf
