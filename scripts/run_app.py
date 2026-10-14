"""啟動 Gradio 介面（本機示範模式：不需 GPU）。

用法：
    python scripts/run_app.py                 # 範例故事讀 Drive 上的方案 B 圖
    python scripts/run_app.py --scheme A
    python scripts/run_app.py --share         # 產生公開網址（demo 用）

「產生分鏡」會呼叫 OpenAI（需 .env 的 OPENAI_API_KEY，結果有快取）；
新故事在本機沒有 GPU，會以占位圖示範流程。
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import generate_storyboard, render  # noqa: E402
from comic_agent.app import build_app  # noqa: E402
from comic_agent.app_logic import DemoRenderer  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"


def load_examples() -> dict:
    """凍結分鏡 → {顯示名稱: 分鏡}；加上 _story_no 讓 DemoRenderer 找得到對應的圖。"""
    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    return {f"{n}. {b['title']}": {**b, "_story_no": n} for n, b in boards.items()}


def make_storyboard(topic: str, n_panels: int) -> dict:
    sb = generate_storyboard(topic, n_panels=n_panels)
    return sb.model_dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--scheme", default="B_s0.3")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    renderer = DemoRenderer(args.render_root, args.scheme, ROOT / ".cache" / "eval" / "boxes.json")
    app = build_app(renderer, load_examples(), make_storyboard)
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
