"""不呼叫 API 的離線測試：檢查 schema、驗證邏輯與 prompt 組合。

用法：python scripts/test_offline.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import Storyboard, build_sd_prompt, llm, render, validate_storyboard  # noqa: E402
from comic_agent.storyboard import MAX_CLIP_TOKENS, estimate_clip_tokens  # noqa: E402

SAMPLE = {
    "title": "雨天的小貓",
    "logline": "小美在雨天找到迷路的小貓，幫牠回家。",
    "art_style": "children's picture book, soft watercolor, warm colors",
    "characters": [
        {"id": "mei", "name": "小美", "appearance": "young girl, short black bob hair, yellow raincoat, red boots", "personality": "勇敢、好奇"},
        {"id": "kitten", "name": "小橘", "appearance": "small orange tabby kitten, green eyes, blue collar", "personality": "膽小"},
    ],
    "panels": [
        {"index": 1, "scene": "小美在雨中散步", "characters": ["mei"], "shot_type": "wide",
         "image_prompt": "mei walking on a rainy street, puddles, gray sky",
         "dialogue": [{"speaker": "narrator", "text": "下雨的星期天。"}]},
        {"index": 2, "scene": "小美發現紙箱裡的小貓", "characters": ["mei", "kitten"], "shot_type": "close_up",
         "image_prompt": "mei crouching, looking at kitten inside a cardboard box, surprised",
         "dialogue": [{"speaker": "mei", "text": "你迷路了嗎？"}]},
    ],
}


def main():
    sb = Storyboard.model_validate(SAMPLE)
    validate_storyboard(sb, n_panels=2)
    p2 = build_sd_prompt(sb, sb.panels[1])
    assert "yellow raincoat" in p2 and "orange tabby" in p2 and "close up shot" in p2
    assert p2.index("orange tabby") < p2.index("crouching"), "外觀要在場景前，避免被 CLIP 截斷"
    print("[OK] schema / 驗證 / prompt 組合正常")
    print("第 2 格 SD prompt：", p2)

    bad = Storyboard.model_validate({**SAMPLE, "panels": [{**SAMPLE["panels"][0], "characters": ["dog"]}]})
    try:
        validate_storyboard(bad)
        raise AssertionError("應該要擋下未定義角色")
    except ValueError as e:
        print("[OK] 能擋下錯誤：", str(e).splitlines()[1])

    long_panel = {**SAMPLE["panels"][1], "image_prompt": ", ".join(["mei looking at kitten"] * 15)}
    too_long = Storyboard.model_validate({**SAMPLE, "panels": [SAMPLE["panels"][0], long_panel]})
    try:
        validate_storyboard(too_long)
        raise AssertionError("應該要擋下過長的 SD prompt")
    except ValueError as e:
        assert "tokens" in str(e)
        print("[OK] 能擋下過長 prompt：", str(e).splitlines()[1])

    crowded_sample = {
        **SAMPLE,
        "characters": SAMPLE["characters"] + [
            {"id": "dog", "name": "小黑", "appearance": "black puppy", "personality": "活潑"}],
        "panels": [{**SAMPLE["panels"][1], "characters": ["mei", "kitten", "dog"]}],
    }
    try:
        validate_storyboard(Storyboard.model_validate(crowded_sample))
        raise AssertionError("應該要擋下 3 個角色同框")
    except ValueError as e:
        assert "同框" in str(e)
        print("[OK] 能擋下過多角色同框：", str(e).splitlines()[1])

    test_cache_only_stores_valid(bad)
    test_render_jobs()
    test_eval_helpers()
    test_layout()
    test_reflect()
    test_app_logic()


def test_app_logic():
    """介面邏輯：台詞表 ↔ 分鏡、錯誤提示、示範生圖（找得到圖 vs 占位圖）、組成整本書。"""
    from PIL import Image

    from comic_agent import app_logic

    rows = app_logic.dialogue_rows(SAMPLE)
    assert rows == [[1, "narrator", "下雨的星期天。"], [2, "mei", "你迷路了嗎？"]]
    edited = app_logic.apply_dialogue_rows(SAMPLE, [[1, "narrator", "改過的旁白"], [2, "mei", ""], [2, "kitten", "喵"]])
    assert edited["panels"][0]["dialogue"][0]["text"] == "改過的旁白"
    assert edited["panels"][1]["dialogue"] == [{"speaker": "kitten", "text": "喵"}]  # 空白台詞被刪除
    assert SAMPLE["panels"][0]["dialogue"][0]["text"] == "下雨的星期天。"  # 不改到原本的分鏡
    for bad in ([[1, "dog", "汪"]], [[9, "mei", "嗨"]]):
        try:
            app_logic.apply_dialogue_rows(SAMPLE, bad)
            raise AssertionError("應該要擋下不存在的說話者或格數")
        except ValueError:
            pass
    assert "雨天的小貓" in app_logic.storyboard_markdown(SAMPLE)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "B_s0.3" / "story_1"
        d.mkdir(parents=True)
        Image.new("RGB", (1024, 1024), (10, 200, 10)).save(d / "panel_01.png")
        r = app_logic.DemoRenderer(tmp, "B_s0.3")
        example = {**SAMPLE, "_story_no": 1}
        assert r.panel(example, 1).getpixel((5, 5)) == (10, 200, 10)  # 範例故事：讀已生成的圖
        assert r.panel(example, 2).getpixel((5, 5)) == app_logic.PLACEHOLDER_COLORS[0]  # 缺圖：占位圖
        assert r.panel(example, 1, attempt=1).getpixel((5, 5)) == app_logic.PLACEHOLDER_COLORS[1]  # 重生：換占位圖
        assert r.panel(SAMPLE, 1).getpixel((5, 5)) == app_logic.PLACEHOLDER_COLORS[0]  # 新故事：占位圖
        images = {p["index"]: r.panel(example, p["index"]) for p in SAMPLE["panels"]}
        pages, pdf = app_logic.build_book(SAMPLE, images, {}, Path(tmp) / "book")
        assert len(pages) == 3 and pdf.exists() and pdf.stat().st_size > 0  # 封面 + 2 頁
    print("[OK] 介面邏輯：台詞表 ↔ 分鏡、錯誤提示、示範生圖、組成整本書")


def test_reflect():
    """reflection：decide() 的規則，以及看圖呼叫會附上圖、圖的內容是快取 key 的一部分。"""
    from PIL import Image

    from comic_agent import reflect
    from comic_agent.reflect import AttributeCheck as A, ElementCheck as E, PanelReview

    def review(attrs=("match",) * 4, present=(True, True), problems=(), revised=""):
        return PanelReview(attributes=[A(attribute=f"a{i}", status=s) for i, s in enumerate(attrs)],
                           elements=[E(element=f"e{i}", kind="object", status="present" if p else "absent")
                                     for i, p in enumerate(present)],
                           problems=list(problems), revised_image_prompt=revised)

    assert reflect.decide(review())[0] == "pass"
    assert reflect.decide(review(present=(True, False), revised="lily building sandcastle"))[0] == "revise_prompt"
    assert reflect.decide(review(present=(True, False)))[0] == "reseed"  # 缺漏但沒給改寫
    assert reflect.decide(review(attrs=("match", "mismatch", "mismatch", "match")))[0] == "reseed"  # 2/4 > 0.3
    assert reflect.decide(review(attrs=("mismatch", "not_visible", "not_visible", "match", "match", "match", "match")))[0] == "pass"  # 1/5
    assert reflect.decide(review(problems=[reflect.Problem(kind="duplicate_character", detail="兩隻小貓")]))[0] == "reseed"
    smile = PanelReview(attributes=[], elements=[E(element="微笑", kind="expression", status="absent"),
                                               E(element="螃蟹", kind="object", status="cannot_tell")],
                        problems=[], revised_image_prompt="")
    assert reflect.decide(smile)[0] == "pass"  # 表情與看不到的元素不觸發重生
    assert reflect.attribute_checklist("young girl, curly brown hair, ") == ["young girl", "curly brown hair"]

    with tempfile.TemporaryDirectory() as tmp:
        llm.CACHE_DIR = Path(tmp) / "llm"
        imgs = []
        for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):
            imgs.append(Path(tmp) / f"img{i}.png")
            Image.new("RGB", (1024, 1024), color).save(imgs[-1])
        sent = []

        class Recorder(_FakeClient):
            def parse(self, **kwargs):
                sent.append(kwargs["input"][1]["content"])
                return super().parse(**kwargs)

        good = review()
        llm._client = Recorder([good, good])
        for img in imgs + imgs[:1]:  # 第 3 次與第 1 次同圖 → 讀快取
            llm.structured_call("sys", "user", PanelReview, model="fake", images=[img])
        assert len(sent) == 2, "不同圖要各呼叫一次，同一張圖要讀快取"
        part = sent[0][1]
        assert part["type"] == "input_image" and part["image_url"].startswith("data:image/png;base64,")
    llm._client = None
    print("[OK] reflection：動作規則、看圖呼叫附圖、圖的內容納入快取 key")

    # reflection_loop：用假的生圖與檢查函式驗證流程
    job = {"panel_index": 2, "prompt": "orig", "seed": 102}
    ok = review()
    missing = review(present=(True, False), revised="mei building a sandcastle")
    bad_attr = lambda n: review(attrs=("mismatch",) * n + ("match",) * (4 - n))  # noqa: E731

    def run(reviews):
        calls, seq = [], iter(reviews)
        out = reflect.reflection_loop(job, SAMPLE, "orig.png",
                                      generate=lambda j, k: calls.append(j) or f"try{k}.png",
                                      review=lambda png: next(seq))
        return out, calls

    out, calls = run([ok])
    assert out["chosen"] == 0 and not calls  # 原圖通過：不重生
    out, calls = run([missing, ok])
    assert out["chosen"] == 1 and calls[0]["seed"] == 102  # 改 prompt：seed 不變
    assert "sandcastle" in calls[0]["prompt"] and "yellow raincoat" in calls[0]["prompt"]  # 外觀仍由程式組合
    out, calls = run([bad_attr(3), bad_attr(4), bad_attr(2), bad_attr(3)])
    assert [c["seed"] for c in calls] == [1102, 2102, 3102]  # 換 seed，最多 3 次
    assert len(out["attempts"]) == 4 and out["chosen"] == 2  # 都沒通過：挑外觀不符最少的
    print("[OK] reflection loop：通過不重生、改 prompt 保留 seed、換 seed、最多 3 次並挑最佳")


def test_layout():
    """排版：避頭點、對話框尾巴幾何、避開臉部、頁面尺寸一致。"""
    from PIL import Image

    from comic_agent import layout

    font = layout.load_font(40)
    lines = layout.wrap_text("媽媽，我們一起蓋城堡吧！", font, font.getlength("媽媽，我們一起蓋城堡"))
    assert len(lines) == 2 and all(ln[0] not in layout.NO_LINE_START for ln in lines), lines

    rect = (100, 100, 300, 180)
    tail = layout.bubble_tail(rect, head=(200, 400), pad=20, half=10, max_len=200)
    assert tail[2][1] > rect[3] and tail[2][1] < 400  # 尖端在框下方、頭部之前
    assert layout.bubble_tail(rect, head=(150, 150), pad=20, half=10, max_len=200) is None  # 頭在框內

    face = (0, 0, 300, 300)  # 左上角是臉，對話框應放右上
    r = layout.place_bubble((200, 80), (1024, 1024), head=(150, 100), avoid=[face])
    assert layout._overlap(r, face) == 0, r

    panel = {"dialogue": [{"speaker": "mei", "text": "你迷路了嗎？"}, {"speaker": "narrator", "text": "下雨的星期天。"}]}
    img = Image.new("RGB", (1024, 1024), (90, 140, 200))
    p1 = layout.compose_page(img, panel, {"mei": (400, 300, 700, 1000)}, page_no=1)
    p2 = layout.compose_page(img, {"dialogue": []}, page_no=2)
    cover = layout.compose_cover(img, "雨天的小貓", size=p1.size)
    assert p1.size == p2.size == cover.size and p1.height > 1024
    print("[OK] 排版：避頭點、對話框尾巴、避開臉部、頁面尺寸一致")


def test_eval_helpers():
    """評估的純計算部分（不載入模型）。"""
    import json as _json

    import numpy as np

    from comic_agent import evaluate

    assert evaluate.detection_query("Young Girl, curly brown hair") == "young girl."
    v = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    assert abs(evaluate.pairwise_mean(v) - 1 / 3) < 1e-9  # (0 + 1 + 0) / 3
    assert evaluate.pairwise_mean(v[:1]) is None
    with tempfile.TemporaryDirectory() as tmp:
        for scheme, rec in [("A", {}), ("B_s0.3", {"ref_image": "refs/story_1/panel_05.png"})]:
            d = Path(tmp) / scheme / "story_1"
            d.mkdir(parents=True)
            (d / "panel_01.json").write_text(_json.dumps(rec), encoding="utf-8")
        assert evaluate.reference_path(tmp, "A", 1) is None
        assert evaluate.reference_path(tmp, "B_s0.3", 1) == Path(tmp) / "refs/story_1/panel_05.png"
        assert evaluate.reference_path(tmp, "B_s0.3", 2) is None

        # collect_items + summarize：用假圖檔與假快取，不載入任何模型
        from PIL import Image
        board = {"characters": [{"appearance": "young girl, red hat"}],
                 "panels": [{"index": i, "image_prompt": f"scene {i}"} for i in (1, 2, 3)]}
        for scheme in ("A", "B_s0.3"):
            for i in (1, 2, 3):
                Image.new("RGB", (4, 4)).save(Path(tmp) / scheme / "story_1" / f"panel_{i:02d}.png")
        (Path(tmp) / "refs/story_1").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(Path(tmp) / "refs/story_1/panel_05.png")
        items = evaluate.collect_items(tmp, ["A", "B_s0.3"], [1], {1: board})
        assert len(items) == 7 and sum(it["panel"] == 0 for it in items) == 1  # 6 格 + 1 張參考圖

        e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        cache = {"boxes": {}, "clip": {}, "dino": {}, "dino_fg": {}, "dreamsim": {}}
        for it in items:
            missed = it["scheme"] == "A" and it["panel"] == 3  # A 第 3 格偵測不到主角
            cache["boxes"][it["key"]] = [None, 0.0] if missed else [[0, 0, 2, 2], 0.9]
            if not missed:
                cache["dino"][it["key"]] = e1 if it["scheme"] == "B_s0.3" or it["panel"] == 1 else e2
                cache["dino_fg"][it["key"]] = e1  # 去背後全部相同：模擬「差異其實來自背景」
            cache["dreamsim"][it["key"]] = e1
            if it["text"]:
                cache["clip"][f"{it['key']}|{it['text']}"] = 30.0
        rows, summ = evaluate.summarize(items, cache)
        a, b = {s["scheme"]: s for s in summ}["A"], {s["scheme"]: s for s in summ}["B_s0.3"]
        assert len(rows) == 6 and a["detect_rate"] == 0.67 and b["detect_rate"] == 1.0
        assert a["id_consistency"] == 0.0 and a["id_to_ref"] == 0.0 and a["ref_copy"] is None
        assert b["id_consistency"] == 1.0 and b["id_to_ref"] == 1.0 and b["ref_copy"] == 1.0
        assert b["diversity"] == 0.0 and b["clip_score"] == 30.0
        assert a["id_consistency_fg"] == 1.0 and a["id_to_ref_fg"] == 1.0  # 去背版不受背景差異影響
    print("[OK] 評估 helper：偵測查詢、兩兩相似度、參考圖路徑、項目收集與彙總")


def test_render_jobs():
    """凍結分鏡 → 生圖工作清單：30 格、seed 不重複、prompt 用凍結版本。"""
    boards = render.load_frozen_storyboards(ROOT / "data" / "storyboards")
    assert sorted(boards) == [1, 2, 3, 4, 5]
    jobs = list(render.iter_jobs(boards, scheme="A"))
    assert len(jobs) == 30
    assert len({j["seed"] for j in jobs}) == 30
    assert render.panel_seed(3, 2) == 302
    assert jobs[0]["prompt"] == boards[1]["sd_prompts"][0]
    png, js = render.panel_paths("/out", jobs[7])
    assert png.as_posix().endswith("A/story_2/panel_02.png") and js.suffix == ".json"
    print(f"[OK] 凍結分鏡 → {len(jobs)} 個生圖工作，seed 不重複")

    refs = list(render.reference_jobs(boards, n_candidates=4))
    assert len(refs) == 20 and len({r["seed"] for r in refs} | {j["seed"] for j in jobs}) == 50
    assert [render.protagonist(boards[n])["id"] for n in range(1, 6)] == ["lily", "bear", "boy", "fox", "girl"]
    assert all(r["prompt"].startswith(boards[r["story_no"]]["art_style"]) for r in refs)
    assert max(estimate_clip_tokens(r["prompt"]) for r in refs) <= MAX_CLIP_TOKENS
    print(f"[OK] 參考圖候選 {len(refs)} 張，主角正確，seed 與分鏡不重疊，prompt 長度在上限內")


class _FakeClient:
    """依序回傳預先準備的結果，模擬 responses.parse，不打 API。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.responses = self

    def parse(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(output_parsed=self.results.pop(0), output_text="")


def test_cache_only_stores_valid(bad):
    good = Storyboard.model_validate(SAMPLE)
    check = lambda s: validate_storyboard(s, n_panels=2)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        llm.CACHE_DIR = Path(tmp)
        call = lambda: llm.structured_call("sys", "user", Storyboard, model="fake", validate=check)  # noqa: E731

        # 1. 第一次不合格 → 自動重試一次 → 合格的結果才寫入快取
        llm._client = _FakeClient([bad, good])
        assert call() == good and llm._client.calls == 2
        assert len(list(Path(tmp).iterdir())) == 1

        # 2. 舊快取不合格 → 視為沒有快取，重新呼叫
        next(Path(tmp).iterdir()).write_text(bad.model_dump_json(), encoding="utf-8")
        llm._client = _FakeClient([good])
        assert call() == good and llm._client.calls == 1

        # 3. 重試後仍不合格 → 拋錯，且不寫入快取
        for f in Path(tmp).iterdir():
            f.unlink()
        llm._client = _FakeClient([bad, bad])
        try:
            call()
            raise AssertionError("應該要拋出驗證錯誤")
        except ValueError:
            pass
        assert not list(Path(tmp).iterdir())
    llm._client = None
    print("[OK] 快取只存通過驗證的結果；不合格會重試一次")


if __name__ == "__main__":
    main()
