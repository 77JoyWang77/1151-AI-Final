"""Reflection 看圖檢查：GPT 逐格檢查已生成的圖，輸出每格的動作（通過／改寫 prompt／換 seed）與理由。

用法：
    python scripts/run_reflect.py --scheme B_s0.3
    python scripts/run_reflect.py --scheme A --stories 1 2

結果有快取（.cache/llm/），同一張圖重跑不花錢；實際的重新生圖在 Colab 進行。
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import llm, render  # noqa: E402
from comic_agent.reflect import REVIEW_MODEL, decide, review_panel  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"
# 每百萬 tokens 的單價（美金，估算用；以 OpenAI 官方價目表為準）
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--scheme", default="B_s0.3")
    ap.add_argument("--stories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="outputs/reflect")
    args = ap.parse_args()

    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    rows = []
    for story_no in args.stories:
        board = boards[story_no]
        for panel in board["panels"]:
            png = Path(args.render_root) / args.scheme / f"story_{story_no}" / f"panel_{panel['index']:02d}.png"
            if not png.exists():
                print(f"[略過] 缺圖 {png}")
                continue
            review = review_panel(png, panel, board, model=args.model)
            action, reasons = decide(review)
            seen = [a for a in review.attributes if a.status != "not_visible"]
            rows.append({
                "scheme": args.scheme, "story": story_no, "panel": panel["index"], "action": action,
                "attr_match": f"{sum(a.status == 'match' for a in seen)}/{len(seen)}",
                "missing": "；".join(e.element for e in review.elements if e.status == "absent" and e.kind != "expression"),
                "problems": "；".join(f"{p.kind}：{p.detail}" for p in review.problems),
                "description": review.description,
                "revised_image_prompt": review.revised_image_prompt,
            })
            print(f"story {story_no} panel {panel['index']}: {action:13} {'；'.join(reasons)}")

    out = Path(args.out) / f"reflect_{args.scheme}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["action"] for r in rows)
    u = llm.USAGE
    price_in, price_out = PRICES.get(args.model or REVIEW_MODEL, (float("nan"), float("nan")))
    cost = u["input_tokens"] / 1e6 * price_in + u["output_tokens"] / 1e6 * price_out
    print(f"\n{args.scheme}：{len(rows)} 格 → " + "、".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"API：{u['calls']} 次呼叫（其餘為快取），輸入 {u['input_tokens']} / 輸出 {u['output_tokens']} tokens，"
          f"估計 {cost:.4f} 美金")
    print(f"已存檔：{out}")


if __name__ == "__main__":
    main()
