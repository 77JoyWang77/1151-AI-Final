"""把生成好的圖排成繪本：每格一頁（對話框 + 旁白），加封面，輸出 PNG 與 PDF。

用法：
    python scripts/make_book.py --scheme B_s0.3
    python scripts/make_book.py --scheme A --stories 1 3

對話框尾巴要指向主角，需要主角的位置：取自評估快取 .cache/eval/boxes.json
（先跑過 scripts/run_eval.py）；沒有快取時對話框照常排版，只是不畫尾巴。
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from comic_agent import render  # noqa: E402
from comic_agent.evaluate import image_key  # noqa: E402
from comic_agent.layout import compose_cover, compose_page, save_book  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--scheme", default="B_s0.3")
    ap.add_argument("--stories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="outputs/books")
    args = ap.parse_args()

    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    boxes_file = ROOT / ".cache" / "eval" / "boxes.json"
    boxes = json.loads(boxes_file.read_text(encoding="utf-8")) if boxes_file.exists() else {}
    for story_no in args.stories:
        board = boards[story_no]
        hero = render.protagonist(board)["id"]
        story_dir = Path(args.render_root) / args.scheme / f"story_{story_no}"
        pages, tails = [], 0
        for panel in board["panels"]:
            png = story_dir / f"panel_{panel['index']:02d}.png"
            if not png.exists():
                sys.exit(f"缺圖：{png}")
            box = boxes.get(image_key(png), [None])[0]
            tails += box is not None
            pages.append(compose_page(Image.open(png), panel, {hero: box} if box else {}, page_no=panel["index"]))
        cover = compose_cover(Image.open(story_dir / "panel_01.png"), board["title"], size=pages[0].size)
        pdf = save_book([cover] + pages, Path(args.out) / args.scheme / f"story_{story_no}")
        print(f"[OK] story {story_no}《{board['title']}》→ {pdf}（主角位置 {tails}/{len(pages)} 頁）")


if __name__ == "__main__":
    main()
