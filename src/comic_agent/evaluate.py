"""評估指標（CPU 可跑，不需要 GPU）。

| 指標 | 模型 | 量什麼 |
| 角色一致性 id_consistency | Grounding DINO 裁主角 → DINOv2 | 同一故事內，主角裁切兩兩 cosine 平均 |
| 與參考圖相似 id_to_ref | 同上 | 每格主角裁切與參考圖主角裁切的 cosine 平均（方案 A 以第 1 格為錨） |
| 去背版 *_fg | Grounding DINO 框 → SAM 去背 → DINOv2 | 同上兩項，但背景填成中性灰，只比角色本身 |
| 圖文對齊 clip_score | CLIP ViT-B/32 | 100 × cos(圖, 該格 image_prompt) |
| 構圖多樣性 diversity | DreamSim | 整張圖兩兩 DreamSim 距離平均；越低代表各格越像（照抄） |
| 參考圖照抄 ref_copy | DreamSim | 每格整張圖與參考圖的 DreamSim 相似度（1 − 距離）平均 |

省記憶體設計：分成多個 pass（偵測 → CLIP → DINOv2 → SAM 去背 → 去背 DINOv2 → DreamSim），**一次只載入一個模型**，
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
SAM_ID = "facebook/sam-vit-base"
FG_FILL = (128, 128, 128)  # 去背後的背景色：中性灰，避免背景顏色影響 DINOv2
PASSES = ("boxes", "clip", "dino", "masks", "dino_fg", "dreamsim")
VECTOR_PASSES = ("dino", "dino_fg", "dreamsim")


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
        detected = 0
        for it in panels:
            box, det = cache["boxes"][it["key"]]
            detected += box is not None
            rows.append({"scheme": scheme, "story": story, "panel": it["panel"],
                         "clip_score": round(cache["clip"][f"{it['key']}|{it['text']}"], 2),
                         "detected": box is not None, "det_score": round(det, 3)})

        def identity(name):
            """某種主角嵌入（dino = 框裁切、dino_fg = 去背）的 (一致性, 與參考圖相似)。"""
            embs = [cache[name][it["key"]] for it in panels if it["key"] in cache.get(name, {})]
            if ref is not None:
                anchor, others = cache.get(name, {}).get(ref["key"]), embs
            else:  # 方案 A：以第 1 格的主角為錨，和其餘各格比
                anchor, others = (embs[0] if embs else None), embs[1:]
            idc = pairwise_mean(embs)
            return (round(idc, 4) if idc is not None else None,
                    mean([float(e @ anchor) for e in others]) if anchor is not None else None)

        full = [cache["dreamsim"][it["key"]] for it in panels]
        ref_full = cache["dreamsim"][ref["key"]] if ref is not None else None
        div = pairwise_mean(full)
        idc, idr = identity("dino")
        idc_fg, idr_fg = identity("dino_fg")
        summaries.append({
            "scheme": scheme, "story": story, "panels": len(panels),
            "detect_rate": round(detected / len(panels), 2) if panels else None,
            "id_consistency": idc, "id_to_ref": idr,
            "id_consistency_fg": idc_fg, "id_to_ref_fg": idr_fg,
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
        if name in VECTOR_PASSES:
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

    if name == "masks":  # SAM：以 Grounding DINO 的框為提示分割主角，存框內的遮罩
        import hashlib

        from PIL import Image
        from transformers import SamModel, SamProcessor
        model, proc = SamModel.from_pretrained(SAM_ID).eval(), SamProcessor.from_pretrained(SAM_ID)
        post = getattr(proc, "post_process_masks", None) or proc.image_processor.post_process_masks

        def fn(image, it):
            inputs = proc(image, input_boxes=[[it["box"]]], return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs, multimask_output=True)
            masks = post(out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
            best = int(out.iou_scores[0, 0].argmax())
            x0, y0, x1, y1 = (int(round(v)) for v in it["box"])
            crop = masks[0, best].numpy()[y0:y1, x0:x1]
            if crop.size == 0 or crop.mean() < 0.05:  # 框內幾乎沒有前景：視為分割失敗
                return None
            name_ = hashlib.sha1(it["key"].encode()).hexdigest()[:16] + ".png"
            Path(it["mask_dir"]).mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop.astype("uint8") * 255).save(Path(it["mask_dir"]) / name_)
            return name_
        return fn

    if name == "dino_fg":  # 去背：框內非主角的像素填中性灰，再算 DINOv2
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        model, proc = AutoModel.from_pretrained(DINO_ID).eval(), AutoImageProcessor.from_pretrained(DINO_ID)

        def fn(image, it):
            crop = image.crop(tuple(int(round(v)) for v in it["box"]))
            mask = Image.open(it["mask_path"]).convert("L").resize(crop.size)
            fg = Image.composite(crop, Image.new("RGB", crop.size, FG_FILL), mask)
            with torch.no_grad():
                v = model(**proc(images=fg, return_tensors="pt")).pooler_output[0]
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
    """依序跑各 pass（見 PASSES），只計算快取裡沒有的部分；回傳完整快取。"""
    from PIL import Image

    cache = load_cache(cache_dir)
    mask_dir = Path(cache_dir) / "masks"
    for name in PASSES:
        if name == "clip":
            todo = [it for it in items if it["text"] and f"{it['key']}|{it['text']}" not in cache["clip"]]
        elif name in ("dino", "masks"):  # 只對偵測到主角的圖
            todo = []
            for it in items:
                box = cache["boxes"][it["key"]][0]
                if box is not None and it["key"] not in cache[name]:
                    todo.append({**it, "box": box, "mask_dir": str(mask_dir)})
        elif name == "dino_fg":  # 只對分割成功的圖
            todo = []
            for it in items:
                mask = cache["masks"].get(it["key"])
                if mask is not None and it["key"] not in cache["dino_fg"]:
                    todo.append({**it, "box": cache["boxes"][it["key"]][0], "mask_path": str(mask_dir / mask)})
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


# ==== 配角位置（給排版的對話框尾巴用）====

SIDE_CACHE = "char_boxes"  # {image_key: {角色 id: [box|None, score]}}
SAME_OBJECT_IOU = 0.6  # 配角框與主角框重疊超過這個值，視為偵測到的其實是主角


def _read_json(cache_dir, name) -> dict:
    """只讀單一快取檔（不像 load_cache 會載入所有向量）。"""
    f = _cache_file(cache_dir, name)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def iou(a, b) -> float:
    inter = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_side_characters(render_root, schemes, stories, boards, cache_dir, model_dir, log=print) -> dict:
    """對每格 characters 中的配角跑 Grounding DINO，結果快取在 char_boxes.json；回傳整個快取。

    需要先跑過評估（boxes.json 有主角框），用來排除「其實抓到主角」的配角框。
    """
    from PIL import Image

    f = _cache_file(cache_dir, SIDE_CACHE)
    cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    hero_boxes = _read_json(cache_dir, "boxes")
    todo = []
    for scheme in schemes:
        for story_no in stories:
            board = boards[story_no]
            looks = {c["id"]: c["appearance"] for c in board["characters"]}
            hero = board["characters"][0]["id"]
            for panel in board["panels"]:
                png = Path(render_root) / scheme / f"story_{story_no}" / f"panel_{panel['index']:02d}.png"
                if not png.exists():
                    continue
                key = image_key(png)
                for cid in panel["characters"]:
                    if cid != hero and cid not in cache.get(key, {}):
                        todo.append({"key": key, "path": str(png), "cid": cid, "query": detection_query(looks[cid])})
    log(f"[配角] 需偵測 {len(todo)} 個（其餘已在快取）")
    if todo:
        fn = _load("boxes", model_dir)
        for k, it in enumerate(todo, 1):
            box, score = fn(Image.open(it["path"]).convert("RGB"), it)
            hero_box = hero_boxes.get(it["key"], [None])[0]
            if box is not None and hero_box is not None and iou(box, hero_box) > SAME_OBJECT_IOU:
                box, score = None, 0.0
            cache.setdefault(it["key"], {})[it["cid"]] = [box, score]
            log(f"[配角] {k}/{len(todo)} {Path(it['path']).parent.name}/{Path(it['path']).name} {it['cid']}："
                + ("找到" if box else "未找到"))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cache), encoding="utf-8")
        del fn
        gc.collect()
    return cache


def character_boxes(png, board: dict, cache_dir) -> dict:
    """某張圖中各角色的位置框 {角色 id: box}：主角取自 boxes.json，配角取自 char_boxes.json。"""
    key = image_key(png)
    out = {}
    hero = _read_json(cache_dir, "boxes").get(key, [None])[0]
    if hero is not None:
        out[board["characters"][0]["id"]] = hero
    side = _read_json(cache_dir, SIDE_CACHE).get(key, {})
    out.update({cid: v[0] for cid, v in side.items() if v[0] is not None})
    return out
