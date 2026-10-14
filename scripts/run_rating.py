"""人工抽查評分介面（盲測，Gradio，本機）。

用法：
    python scripts/run_rating.py                      # 評 A 與 B_s0.3 的同樣 8 格，共 16 張
    python scripts/run_rating.py --schemes A B_s0.3 B_s0.3_R

每評一張就存到 data/human_eval/ratings.csv，關掉再開會從沒評過的繼續。
評完後用 scripts/analyze_ratings.py 計算平均分與人 vs GPT 的一致率。
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from comic_agent import human_eval as he  # noqa: E402
from comic_agent import render  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"
RATINGS = ROOT / "data" / "human_eval" / "ratings.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--schemes", nargs="+", default=["A", "B_s0.3"])
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--ratings", default=str(RATINGS), help="評分檔（測試時可指定暫存檔）")
    args = ap.parse_args()
    ratings_file = Path(args.ratings)

    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    queue = he.rating_queue(args.schemes, he.sample_positions(boards))
    root = Path(args.render_root)

    def png(it, panel=None):
        return root / it["scheme"] / f"story_{it['story']}" / f"panel_{(panel or it['panel']):02d}.png"

    def view(rater):
        """顯示下一個還沒評的項目；全部評完時顯示完成訊息。"""
        todo = he.pending(queue, he.load_ratings(ratings_file), rater)
        done = len(queue) - len(todo)
        if not todo:
            return (None, f"## 全部完成（{done}/{len(queue)}）\n請執行 `python scripts/analyze_ratings.py`", "", [], None)
        it = todo[0]
        panel = boards[it["story"]]["panels"][it["panel"] - 1]
        others = [Image.open(png(it, p["index"])) for p in boards[it["story"]]["panels"] if p["index"] != it["panel"]]
        info = (f"### 第 {done + 1} / {len(queue)} 張\n"
                f"**劇情**：{panel['scene']}\n\n**主角外觀**：{boards[it['story']]['characters'][0]['appearance']}")
        return it, info, "", others, Image.open(png(it))

    def save(rater, it, identity, plot, note):
        if not rater.strip():
            raise gr.Error("請先填評分者名稱")
        if it is None:
            raise gr.Error("已全部完成")
        if identity is None or plot is None:
            raise gr.Error("兩題都要評分")
        he.append_rating(ratings_file, rater.strip(), it, int(identity), int(plot), note or "")
        return (*view(rater.strip()), None, None)

    with gr.Blocks(title="人工抽查") as app:
        gr.Markdown("# 人工抽查（盲測）\n大圖是要評的那一格；下方是同一故事的其他格，用來判斷角色是否一致。不會顯示是哪個方案。")
        current = gr.State(None)
        with gr.Row():
            rater = gr.Textbox(label="評分者", value="Joy")
            btn_start = gr.Button("開始／繼續", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                target = gr.Image(label="要評的格", height=560)
            with gr.Column(scale=2):
                info = gr.Markdown()
                identity = gr.Radio([1, 2, 3, 4, 5], label=he.QUESTIONS["identity"])
                plot = gr.Radio([1, 2, 3, 4, 5], label=he.QUESTIONS["plot"])
                note = gr.Textbox(label="備註（選填）")
                btn_save = gr.Button("儲存並下一張", variant="primary")
        strip = gr.Gallery(label="同一故事的其他格", columns=5, height=220)

        btn_start.click(view, rater, [current, info, note, strip, target])
        btn_save.click(save, [rater, current, identity, plot, note], [current, info, note, strip, target, identity, plot])
    app.launch(server_port=args.port)


if __name__ == "__main__":
    main()
