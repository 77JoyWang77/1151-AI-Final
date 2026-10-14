"""人工抽查（企畫書：每方案抽 8 張，兩題 1–5 分，並與 GPT 判定計算一致率）。

- 抽樣：固定 seed 從 30 格抽 8 個位置，所有方案評同樣的位置（成對比較）
- 盲測：評分順序隨機、介面不顯示方案名稱
- 存檔：每評一張就附加到 CSV，可中斷續評
- 分析：各方案平均、人與 GPT 的一致率與 Cohen's kappa、人評角色一致與指標的等級相關
"""
import csv
import random
from datetime import datetime
from pathlib import Path

SAMPLE_SEED = 42
N_PER_SCHEME = 8
FIELDS = ["rater", "scheme", "story", "panel", "identity", "plot", "note", "time"]
QUESTIONS = {
    "identity": "角色一致：這格的主角，和同一故事其他格的主角像同一個角色嗎？（1＝完全不像 … 5＝一看就是同一個）",
    "plot": "劇情符合：這格畫出了劇情描述的內容（人物、動作、物件）嗎？（1＝完全沒有 … 5＝完全符合）",
}


def sample_positions(boards: dict, n: int = N_PER_SCHEME, seed: int = SAMPLE_SEED) -> list[tuple[int, int]]:
    """從所有 (故事, 格數) 中固定抽 n 個；所有方案共用。"""
    positions = [(s, p["index"]) for s, b in sorted(boards.items()) for p in b["panels"]]
    return sorted(random.Random(seed).sample(positions, n))


def rating_queue(schemes: list[str], positions: list[tuple[int, int]], seed: int = SAMPLE_SEED) -> list[dict]:
    """方案 × 位置，打亂順序（盲測）。順序由 seed 與方案清單決定，可重現。"""
    items = [{"scheme": sc, "story": s, "panel": p} for sc in schemes for s, p in positions]
    random.Random(f"{seed}|{'|'.join(schemes)}").shuffle(items)
    return items


def load_ratings(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{**r, "story": int(r["story"]), "panel": int(r["panel"]),
                 "identity": int(r["identity"]), "plot": int(r["plot"])} for r in csv.DictReader(f)]


def append_rating(path: str | Path, rater: str, item: dict, identity: int, plot: int, note: str = ""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"rater": rater, "scheme": item["scheme"], "story": item["story"], "panel": item["panel"],
                    "identity": int(identity), "plot": int(plot), "note": note.strip(),
                    "time": datetime.now().isoformat(timespec="seconds")})


def pending(queue: list[dict], ratings: list[dict], rater: str) -> list[dict]:
    """還沒被這位評分者評過的項目（依佇列順序）。"""
    done = {(r["scheme"], r["story"], r["panel"]) for r in ratings if r["rater"] == rater}
    return [it for it in queue if (it["scheme"], it["story"], it["panel"]) not in done]


# ==== 分析 ====

def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """兩位評審（例如人與 GPT）二元判斷的 Cohen's kappa；扣掉偶然一致後的一致程度。"""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def spearman(x: list[float], y: list[float]) -> float | None:
    """等級相關（同分取平均等級）；少於 3 個點回傳 None。"""
    if len(x) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx == 0 or sy == 0 else cov / (sx * sy)
