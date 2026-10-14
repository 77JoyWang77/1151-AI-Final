"""Gradio 介面背後的邏輯（不 import gradio，可離線測試）。

生圖透過 Renderer 介面抽換：
- DemoRenderer（本機，無 GPU）：範例故事讀 Drive 上已生成的圖；新故事或重生時畫占位圖
- 之後在 Colab 換成 SDXL Renderer，介面不用改
"""
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import layout
from .evaluate import character_boxes

PLACEHOLDER_COLORS = [(232, 222, 205), (214, 228, 222), (226, 215, 230), (230, 226, 206)]


# ==== 分鏡 ↔ 介面 ====

def storyboard_markdown(board: dict) -> str:
    """分鏡摘要：標題、一句話故事、角色、每格的劇情與鏡頭。"""
    lines = [f"### 《{board['title']}》", board.get("logline", ""), "", "**角色**"]
    lines += [f"- `{c['id']}` {c['name']}：{c['appearance']}" for c in board["characters"]]
    lines += ["", "**分鏡**"]
    lines += [f"{p['index']}. （{p['shot_type']}）{p['scene']}" for p in board["panels"]]
    return "\n".join(lines)


def dialogue_rows(board: dict) -> list[list]:
    """台詞 → 可編輯表格的列：[格, 說話者 id, 台詞]。"""
    return [[p["index"], d["speaker"], d["text"]] for p in board["panels"] for d in p["dialogue"]]


def apply_dialogue_rows(board: dict, rows: list[list]) -> dict:
    """把編輯後的表格寫回分鏡（回傳新的 dict）；台詞空白的列會被刪除。

    說話者必須是角色 id 或 narrator，格數必須存在，否則拋 ValueError 讓介面顯示。
    """
    ids = {c["id"] for c in board["characters"]} | {"narrator"}
    indices = {p["index"] for p in board["panels"]}
    per_panel = {i: [] for i in indices}
    for row in rows:
        if not row or len(row) < 3 or not str(row[2] or "").strip():
            continue
        idx, speaker, text = int(row[0]), str(row[1]).strip(), str(row[2]).strip()
        if idx not in indices:
            raise ValueError(f"沒有第 {idx} 格")
        if speaker not in ids:
            raise ValueError(f"第 {idx} 格的說話者 '{speaker}' 不是角色 id 或 narrator（可用：{sorted(ids)}）")
        per_panel[idx].append({"speaker": speaker, "text": text})
    new = json.loads(json.dumps(board))  # 深拷貝，不改到原本的分鏡
    for p in new["panels"]:
        p["dialogue"] = per_panel[p["index"]]
    return new


# ==== 生圖（可抽換）====

def placeholder_image(board: dict, panel_index: int, attempt: int = 0, size: int = 1024) -> Image.Image:
    """示範用占位圖：顯示格數與劇情；重生時換底色，方便看出有換。"""
    panel = board["panels"][panel_index - 1]
    img = Image.new("RGB", (size, size), PLACEHOLDER_COLORS[attempt % len(PLACEHOLDER_COLORS)])
    draw = ImageDraw.Draw(img)
    big, small = layout.load_font(round(size * 0.07)), layout.load_font(round(size * 0.04))
    draw.text((size * 0.08, size * 0.1), f"第 {panel_index} 格（占位圖）", font=big, fill=(90, 80, 70))
    y = size * 0.25
    for ln in layout.wrap_text(panel["scene"], small, size * 0.84):
        draw.text((size * 0.08, y), ln, font=small, fill=(60, 55, 50))
        y += small.size * 1.5
    if attempt:
        draw.text((size * 0.08, size * 0.88), f"重生第 {attempt} 次", font=small, fill=(150, 60, 60))
    return img


class DemoRenderer:
    """本機示範：範例故事（帶 _story_no）讀 Drive 上已生成的圖，其餘情況畫占位圖。"""

    def __init__(self, render_root: str | Path, scheme: str = "B_s0.3", cache_dir: str | Path | None = None):
        self.render_root, self.scheme = Path(render_root), scheme
        self.cache_dir = cache_dir  # 評估快取（角色位置）；None 表示不畫對話框尾巴

    def _path(self, board, panel_index):
        story_no = board.get("_story_no")
        if story_no is None:
            return None
        png = self.render_root / self.scheme / f"story_{story_no}" / f"panel_{panel_index:02d}.png"
        return png if png.exists() else None

    def panel(self, board: dict, panel_index: int, attempt: int = 0) -> Image.Image:
        png = self._path(board, panel_index) if attempt == 0 else None
        return Image.open(png).convert("RGB") if png else placeholder_image(board, panel_index, attempt)

    def speaker_boxes(self, board: dict, panel_index: int, attempt: int = 0) -> dict:
        """各角色的位置框（取自評估快取：主角 boxes.json、配角 char_boxes.json）；沒有就回傳空 dict。"""
        png = self._path(board, panel_index) if attempt == 0 else None
        return character_boxes(png, board, self.cache_dir) if png and self.cache_dir else {}


# ==== 排版 ====

def build_book(board: dict, images: dict, boxes: dict, out_dir: str | Path | None = None):
    """images／boxes：{格數: 圖}／{格數: {角色 id: 框}} → (頁面圖 list, PDF 路徑)。"""
    pages = [layout.compose_page(images[p["index"]], p, boxes.get(p["index"], {}), page_no=p["index"])
             for p in board["panels"]]
    cover = layout.compose_cover(images[board["panels"][0]["index"]], board["title"], size=pages[0].size)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="book_"))
    pdf = layout.save_book([cover] + pages, out)
    return [cover] + pages, pdf
