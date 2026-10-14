"""偵測配角位置（Grounding DINO，CPU），給排版時對話框的尾巴指向配角。

用法：
    python scripts/detect_characters.py --schemes A B_s0.3

只偵測每格 characters 中的非主角；主角位置沿用評估快取（需先跑 scripts/run_eval.py）。
結果快取在 .cache/eval/char_boxes.json，之後 make_book.py 與 Gradio 介面會自動使用。
"""
import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import render  # noqa: E402
from comic_agent.evaluate import detect_side_characters  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--schemes", nargs="+", default=["A", "B_s0.3"])
    ap.add_argument("--stories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    t0 = time.time()
    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    cache = detect_side_characters(args.render_root, args.schemes, args.stories, boards,
                                   ROOT / ".cache" / "eval", ROOT / ".cache" / "eval_models",
                                   log=lambda m: print(f"[{time.time() - t0:4.0f}s] {m}", flush=True))
    found = sum(v[0] is not None for per_img in cache.values() for v in per_img.values())
    total = sum(len(per_img) for per_img in cache.values())
    print(f"配角位置：找到 {found}/{total}")


if __name__ == "__main__":
    main()
