"""分析人工抽查：各方案平均、人 vs GPT 一致率、人評 vs 自動指標的等級相關。

用法：python scripts/analyze_ratings.py
需要：data/human_eval/ratings.csv（run_rating.py 產生）、outputs/reflect/reflect_<scheme>.csv（run_reflect.py）、
評估快取 .cache/eval/（run_eval.py）。缺少的部分會略過並提示。
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import evaluate, render  # noqa: E402
from comic_agent import human_eval as he  # noqa: E402

DEFAULT_RENDER_ROOT = "G:/我的雲端硬碟/02 課程/1151-AI-Final/outputs/render"
GOOD = 4  # 人評 ≥ 4 視為「合格」


def per_panel_metrics(schemes, stories, boards):
    """每格的去背角色一致（與同故事其他格的平均 cosine）與 CLIP score，取自評估快取。"""
    items = evaluate.collect_items(DEFAULT_RENDER_ROOT, schemes, stories, boards)
    cache = evaluate.load_cache(ROOT / ".cache" / "eval")
    out, groups = {}, defaultdict(list)
    for it in items:
        if it["panel"] > 0:
            groups[(it["scheme"], it["story"])].append(it)
    for (scheme, story), group in groups.items():
        embs = {it["panel"]: cache["dino_fg"].get(it["key"]) for it in group}
        for it in group:
            e = embs[it["panel"]]
            others = [v for p, v in embs.items() if p != it["panel"] and v is not None]
            ident = sum(float(e @ v) for v in others) / len(others) if e is not None and others else None
            clip = cache["clip"].get(f"{it['key']}|{it['text']}")
            out[(scheme, story, it["panel"])] = {"identity_fg": ident, "clip": clip}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "data" / "human_eval" / "ratings.csv"))
    ratings = he.load_ratings(ap.parse_args().ratings)
    if not ratings:
        sys.exit("還沒有評分：先執行 python scripts/run_rating.py")
    # 同一格有多位評分者時取平均
    agg = defaultdict(lambda: {"identity": [], "plot": []})
    for r in ratings:
        k = (r["scheme"], r["story"], r["panel"])
        agg[k]["identity"].append(r["identity"])
        agg[k]["plot"].append(r["plot"])
    human = {k: {q: sum(v) / len(v) for q, v in d.items()} for k, d in agg.items()}
    schemes = sorted({k[0] for k in human})
    print(f"評分 {len(ratings)} 筆，評分者 {sorted({r['rater'] for r in ratings})}，方案 {schemes}\n")

    print("## 各方案人評平均（1–5）")
    for sc in schemes:
        ks = [k for k in human if k[0] == sc]
        mi = sum(human[k]["identity"] for k in ks) / len(ks)
        mp = sum(human[k]["plot"] for k in ks) / len(ks)
        print(f"- {sc}：角色一致 {mi:.2f}、劇情符合 {mp:.2f}（{len(ks)} 張）")

    print("\n## 人 vs GPT：劇情是否符合（人 ≥ 4 vs GPT 沒有回報缺漏元素）")
    pairs = []
    for sc in schemes:
        f = ROOT / "outputs" / "reflect" / f"reflect_{sc}.csv"
        if not f.exists():
            print(f"- {sc}：沒有 GPT 檢查結果（先跑 run_reflect.py --scheme {sc}），略過")
            continue
        with f.open(encoding="utf-8-sig") as fh:
            gpt = {(int(r["story"]), int(r["panel"])): not r["missing"].strip() for r in csv.DictReader(fh)}
        for k, v in human.items():
            if k[0] == sc and (k[1], k[2]) in gpt:
                pairs.append((v["plot"] >= GOOD, gpt[(k[1], k[2])]))
    if pairs:
        hs, gs = [p[0] for p in pairs], [p[1] for p in pairs]
        agree = sum(a == b for a, b in pairs) / len(pairs)
        kappa = he.cohen_kappa(hs, gs)
        print(f"- 一致率 {agree:.0%}（{len(pairs)} 張），Cohen's kappa {kappa if kappa is None else round(kappa, 2)}")
        print(f"- 人說合格、GPT 說缺漏：{sum(h and not g for h, g in pairs)}；人說不合格、GPT 說沒缺漏：{sum(g and not h for h, g in pairs)}")

    print("\n## 人評 vs 自動指標（Spearman 等級相關）")
    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    metrics = per_panel_metrics(schemes, sorted({k[1] for k in human}), boards)
    for q, m, label in [("identity", "identity_fg", "角色一致 vs 去背 DINOv2"), ("plot", "clip", "劇情符合 vs CLIP score")]:
        xs, ys = [], []
        for k, v in human.items():
            val = metrics.get(k, {}).get(m)
            if val is not None:
                xs.append(v[q])
                ys.append(val)
        rho = he.spearman(xs, ys)
        print(f"- {label}：ρ = {rho if rho is None else round(rho, 2)}（{len(xs)} 張）")
    print("\n解讀提示：kappa > 0.6 算高度一致、0.4–0.6 中等；ρ 為正且越大，代表自動指標越能反映人的判斷。")


if __name__ == "__main__":
    main()
