"""產生分鏡並存成 JSON，同時印出每格的 SD prompt。

用法：
    python scripts/run_storyboard.py "小女孩在雨天找到一隻迷路的小貓" --panels 6
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comic_agent import NEGATIVE_PROMPT, build_sd_prompt, generate_storyboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--audience", default="3–6 歲兒童")
    ap.add_argument("--style", default="溫暖的水彩繪本風")
    ap.add_argument("--characters", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="outputs/storyboard.json")
    args = ap.parse_args()

    sb = generate_storyboard(
        args.topic, n_panels=args.panels, audience=args.audience, style=args.style,
        characters=args.characters, model=args.model, use_cache=not args.no_cache,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = sb.model_dump()
    data["sd_prompts"] = [build_sd_prompt(sb, p) for p in sb.panels]
    data["negative_prompt"] = NEGATIVE_PROMPT
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"《{sb.title}》— {sb.logline}\n")
    for c in sb.characters:
        print(f"[角色] {c.id} / {c.name}：{c.appearance}")
    for p, prompt in zip(sb.panels, data["sd_prompts"]):
        print(f"\n--- 第 {p.index} 格（{p.shot_type}）---\n{p.scene}")
        for d in p.dialogue:
            print(f"  {d.speaker}：{d.text}")
        print(f"  SD prompt: {prompt}")
    print(f"\n已存檔：{out}")


if __name__ == "__main__":
    main()
