"""評估已生成的圖（CPU），輸出逐格與逐故事的指標 CSV。

用法：
    python scripts/run_eval.py --schemes A B_s0.3
    python scripts/run_eval.py --schemes A B_s0.2 B_s0.3 B_s0.4 --stories 1 3

預設從 Google Drive 電腦版（G:）讀 Colab 的生成結果；首次執行會下載評估模型（約 2 GB）。
一次只載入一個模型（峰值約 1–2 GB），結果快取在 .cache/eval/，中途停掉重跑會續算。
需先安裝：pip install -r requirements-eval.txt（Windows 請在 PYTHONUTF8=1 下安裝）。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import render  # noqa: E402
from comic_agent.evaluate import collect_items, run_passes, summarize  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"
METRICS = ["detect_rate", "id_consistency", "id_consistency_fg", "id_to_ref", "id_to_ref_fg",
           "clip_score", "diversity", "ref_copy"]


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟不亂碼
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    ap.add_argument("--schemes", nargs="+", default=["A"])
    ap.add_argument("--stories", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    t0 = time.time()
    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    items = collect_items(args.render_root, args.schemes, args.stories, boards)
    if not items:
        sys.exit(f"找不到任何圖：請確認 --render-root（目前 {args.render_root}）存在，且有 {args.schemes} 的結果")
    print(f"共 {len(items)} 張圖（含參考圖）")
    cache = run_passes(items, ROOT / ".cache" / "eval", ROOT / ".cache" / "eval_models",
                       log=lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True))
    panels, stories = summarize(items, cache)

    for s in stories:
        print(f"{s['scheme']:7} story {s['story']}: " + "  ".join(f"{k}={s[k]}" for k in METRICS))

    out = Path(args.out)
    tag = "_".join(args.schemes)
    write_csv(out / f"panels_{tag}.csv", panels)
    write_csv(out / f"stories_{tag}.csv", stories)

    print("\n方案平均（各故事平均）：")
    print(f"{'scheme':8}" + "".join(f"{k:>16}" for k in METRICS))
    for scheme in args.schemes:
        rows = [s for s in stories if s["scheme"] == scheme]
        cells = []
        for k in METRICS:
            vals = [r[k] for r in rows if r[k] is not None]
            cells.append(f"{sum(vals) / len(vals):16.4f}" if vals else f"{'-':>16}")
        print(f"{scheme:8}" + "".join(cells))
    print(f"\n已存檔：{out / f'panels_{tag}.csv'}、{out / f'stories_{tag}.csv'}（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
