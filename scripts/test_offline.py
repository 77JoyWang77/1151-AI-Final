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
