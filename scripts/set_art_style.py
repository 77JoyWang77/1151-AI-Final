"""統一替換凍結分鏡的畫風，並重新產生 sd_prompts（不呼叫 LLM，劇情與角色不變）。

用法：python scripts/set_art_style.py "children's picture book, gouache illustration, flat bold colors"
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_agent import Storyboard, build_sd_prompt, validate_storyboard  # noqa: E402

DATA_DIR = ROOT / "data" / "storyboards"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    art_style = sys.argv[1].strip()
    for path in sorted(DATA_DIR.glob("story_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sb = Storyboard.model_validate(data).model_copy(update={"art_style": art_style})
        validate_storyboard(sb, n_panels=len(sb.panels))  # 換畫風後 prompt 長度仍要在上限內
        data["art_style"] = art_style
        data["sd_prompts"] = [build_sd_prompt(sb, p) for p in sb.panels]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {path.name}《{sb.title}》")


if __name__ == "__main__":
    main()
