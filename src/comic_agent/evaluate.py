"""評估指標（CPU 可跑，不需要 GPU）。

| 指標 | 模型 | 量什麼 |
| 角色一致性 id_consistency | Grounding DINO 裁主角 → DINOv2 | 同一故事內，主角裁切兩兩 cosine 平均 |
| 與參考圖相似 id_to_ref | 同上 | 每格主角裁切與參考圖主角裁切的 cosine 平均（方案 A 以第 1 格為錨） |
| 圖文對齊 clip_score | CLIP ViT-B/32 | 100 × cos(圖, 該格 image_prompt) |
| 構圖多樣性 diversity | DreamSim | 整張圖兩兩 DreamSim 距離平均；越低代表各格越像（照抄） |
| 參考圖照抄 ref_copy | DreamSim | 每格整張圖與參考圖的 DreamSim 相似度（1 − 距離）平均 |

省記憶體設計：分成 4 個 pass（偵測 → CLIP → DINOv2 → DreamSim），**一次只載入一個模型**，
用完即釋放；每個 pass 的結果寫入磁碟快取（以檔案路徑 + 修改時間 + 大小為 key），
中途停掉可續跑，圖重新生成也會自動重算。彙總（summarize）是純計算，可離線測試。
"""
import gc
import json
import os
from pathlib import Path

CLIP_ID = "openai/clip-vit-base-patch32"
DINO_ID = "facebook/dinov2-base"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
DREAMSIM_TYPE = "dino_vitb16"  # 單一模型版，記憶體約為 ensemble 的 1/3
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PASSES = ("boxes", "clip", "dino", "dreamsim")


# ==== 純計算 helper（不 import torch）====

def detection_query(appearance: str) -> str:
    """Grounding DINO 的文字查詢：取外觀描述的第一個 tag（例如 'young girl'），小寫並以句點結尾。"""
    return appearance.split(",")[0].strip().lower() + "."


def pairwise_mean(vectors) -> float | None:
    """已正規化向量的兩兩 cosine 平均（不含自己）；少於 2 個回傳 None。"""
    n = len(vectors)
    if n < 2:
        return None
    total = sum(float(vectors[i] @ vectors[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def reference_path(render_root: str | Path, scheme: str, story_no: int) -> Path | None:
    """從該方案第 1 格的紀錄讀出參考圖路徑；方案 A 沒有參考圖回傳 None。"""
    record = Path(render_root) / scheme / f"story_{story_no}" / "panel_01.json"
    if not record.exists():
        return None
    ref = json.loads(record.read_text(encoding="utf-8")).get("ref_image")
    return Path(render_root) / ref if ref else None


def image_key(path: str | Path) -> str:
    """快取 key：路徑 + 修改時間 + 大小；圖重新生成後 key 會變，自動重算。"""
    st = os.stat(path)
    return f"{Path(path).as_posix()}|{st.st_mtime_ns}|{st.st_size}"


def collect_items(render_root, schemes, stories, boards) -> list[dict]:
    """列出要評估的每張圖（各格與參考圖）。參考圖的 panel 記為 0。"""
    items = []
    for scheme in schemes:
        for story_no in stories:
            board = boards[story_no]
            query = detection_query(board["characters"][0]["appearance"])
            story_dir = Path(render_root) / scheme / f"story_{story_no}"
            for panel in board["panels"]:
                png = story_dir / f"panel_{panel['index']:02d}.png"
                if png.exists():
                    items.append({"scheme": scheme, "story": story_no, "panel": panel["index"],
                                  "path": str(png), "query": query, "text": panel["image_prompt"]})
            ref = reference_path(render_root, scheme, story_no)
            if ref is not None and ref.exists():
                items.append({"scheme": scheme, "story": story_no, "panel": 0,
                              "path": str(ref), "query": query, "text": None})
    for it in items:
        it["key"] = image_key(it["path"])
    return items


def summarize(items: list[dict], cache: dict) -> tuple[list[dict], list[dict]]:
    """由各 pass 的快取計算逐格紀錄與逐故事摘要（純計算）。

    cache = {"boxes": {key: [box|None, score]}, "clip": {key|text: score},
             "dino": {key: vec}, "dreamsim": {key: vec}}；向量為已正規化的 numpy array。
    """
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    groups = {}
    for it in items:
        groups.setdefault((it["scheme"], it["story"]), []).append(it)

    rows, summaries = [], []
    for (scheme, story), group in groups.items():
        panels = sorted((it for it in group if it["panel"] > 0), key=lambda it: it["panel"])
        ref = next((it for it in group if it["panel"] == 0), None)
        crops, full = [], []
        for it in panels:
            box, det = cache["boxes"][it["key"]]
            if box is not None:
                crops.append(cache["dino"][it["key"]])
            full.append(cache["dreamsim"][it["key"]])
            rows.append({"scheme": scheme, "story": story, "panel": it["panel"],
                         "clip_score": round(cache["clip"][f"{it['key']}|{it['text']}"], 2),
                         "detected": box is not None, "det_score": round(det, 3)})
        if ref is not None:
            ref_box = cache["boxes"][ref["key"]][0]
            ref_crop = cache["dino"][ref["key"]] if ref_box is not None else None
            ref_full = cache["dreamsim"][ref["key"]]
            others = crops
        else:  # 方案 A：以第 1 格的主角為錨，和其餘各格比
            ref_crop, ref_full, others = (crops[0] if crops else None), None, crops[1:]
        div = pairwise_mean(full)
        idc = pairwise_mean(crops)
        summaries.append({
            "scheme": scheme, "story": story, "panels": len(panels),
            "detect_rate": round(len(crops) / len(panels), 2) if panels else None,
            "id_consistency": round(idc, 4) if idc is not None else None,
            "id_to_ref": mean([float(e @ ref_crop) for e in others]) if ref_crop is not None else None,
            "clip_score": mean([r["clip_score"] for r in rows if (r["scheme"], r["story"]) == (scheme, story)]),
            "diversity": round(1 - div, 4) if div is not None else None,
            "ref_copy": mean([float(e @ ref_full) for e in full]) if ref_full is not None else None,
        })
    return rows, summaries


# ==== 磁碟快取 ====

def _cache_file(cache_dir, name) -> Path:
    return Path(cache_dir) / f"{name}.json"


def load_cache(cache_dir) -> dict:
    """讀各 pass 的快取；向量轉回 numpy array。"""
    import numpy as np
    cache = {}
    for name in PASSES:
        f = _cache_file(cache_dir, name)
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if name in ("dino", "dreamsim"):
            data = {k: np.asarray(v, dtype="float32") for k, v in data.items()}
        cache[name] = data
    return cache


def _save(cache_dir, name, data):
    f = _cache_file(cache_dir, name)
    f.parent.mkdir(parents=True, exist_ok=True)
    plain = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    f.write_text(json.dumps(plain), encoding="utf-8")


# ==== 模型（每個 pass 只載入一個）====

def _load(name, model_dir):
    """回傳單張圖的計算函式；呼叫端用完後丟掉參照即可釋放記憶體。"""
    import torch

    if name == "boxes":
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).eval()
        proc = AutoProcessor.from_pretrained(GDINO_ID)

        def fn(image, it):
            inputs = proc(images=image, text=it["query"], return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            kw = dict(text_threshold=TEXT_THRESHOLD, target_sizes=[image.size[::-1]])
            try:
                res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=BOX_THRESHOLD, **kw)[0]
            except TypeError:  # 較舊版 transformers 的參數名稱
                res = proc.post_process_grounded_object_detection(out, inputs.input_ids, box_threshold=BOX_THRESHOLD, **kw)[0]
            if len(res["scores"]) == 0:
                return [None, 0.0]
            best = int(res["scores"].argmax())
            return [[float(v) for v in res["boxes"][best]], float(res["scores"][best])]
        return fn

    if name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model, proc = CLIPModel.from_pretrained(CLIP_ID).eval(), CLIPProcessor.from_pretrained(CLIP_ID)

        def fn(image, it):
            inputs = proc(text=[it["text"]], images=image, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():  # 用完整 forward 取投影後的嵌入：get_*_features 的回傳型別在 transformers 5 改了
                out = model(**inputs)
            img, txt = out.image_embeds, out.text_embeds
            img, txt = img / img.norm(dim=-1, keepdim=True), txt / txt.norm(dim=-1, keepdim=True)
            return max(100 * float((img * txt).sum()), 0.0)
        return fn

    if name == "dino":
        from transformers import AutoImageProcessor, AutoModel
        model, proc = AutoModel.from_pretrained(DINO_ID).eval(), AutoImageProcessor.from_pretrained(DINO_ID)

        def fn(image, it):
            with torch.no_grad():
                v = model(**proc(images=image.crop(tuple(it["box"])), return_tensors="pt")).pooler_output[0]
            return (v / v.norm()).numpy()
        return fn

    if name == "dreamsim":
        from dreamsim import dreamsim
        model, preprocess = dreamsim(pretrained=True, device="cpu", cache_dir=str(model_dir), dreamsim_type=DREAMSIM_TYPE)
        model.eval()

        def fn(image, it):
            with torch.no_grad():
                v = model.embed(preprocess(image))[0]
            return (v / v.norm()).numpy()
        return fn

    raise ValueError(name)


def run_passes(items, cache_dir, model_dir, log=print) -> dict:
    """依序跑 4 個 pass，只計算快取裡沒有的部分；回傳完整快取。"""
    from PIL import Image

    cache = load_cache(cache_dir)
    for name in PASSES:
        if name == "clip":
            todo = [it for it in items if it["text"] and f"{it['key']}|{it['text']}" not in cache["clip"]]
        elif name == "dino":  # 只對偵測到主角的圖算裁切嵌入
            todo = []
            for it in items:
                box = cache["boxes"][it["key"]][0]
                if box is not None and it["key"] not in cache["dino"]:
                    todo.append({**it, "box": box})
        else:
            todo = [it for it in items if it["key"] not in cache[name]]
        # 同一張圖（例如多個 scale 共用的參考圖）只算一次
        todo = list({(it["key"], it["text"] if name == "clip" else None): it for it in todo}.values())
        log(f"[{name}] 需計算 {len(todo)} 張（其餘已在快取）")
        if not todo:
            continue
        fn = _load(name, model_dir)
        for k, it in enumerate(todo, 1):
            value = fn(Image.open(it["path"]).convert("RGB"), it)
            cache[name][f"{it['key']}|{it['text']}" if name == "clip" else it["key"]] = value
            if k % 10 == 0 or k == len(todo):
                _save(cache_dir, name, cache[name])
                log(f"[{name}] {k}/{len(todo)}")
        del fn
        gc.collect()  # 釋放這個 pass 的模型，再載入下一個
    return cache
