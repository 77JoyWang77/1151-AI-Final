"""生圖的設定、seed 規則與紀錄（純 Python，不 import torch）。

Colab notebook 與本機離線測試共用；四種一致性方案（A–D）都從這裡取 seed 與 prompt，
確保方案之間只差在一致性做法本身。
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenConfig:
    """實驗中固定的生圖參數（見 docs/proposal.md「逐格生圖」）。"""

    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"  # 原版 SDXL VAE 在 fp16 會溢位出黑圖
    scheduler: str = "DPM++ 2M Karras"
    steps: int = 30
    cfg: float = 6.5
    width: int = 1024
    height: int = 1024


def panel_seed(story_no: int, panel_index: int) -> int:
    """seed = 故事編號 × 100 + 格數，四種方案共用同一組。"""
    return story_no * 100 + panel_index


def load_frozen_storyboards(data_dir: str | Path) -> dict[int, dict]:
    """讀 data/storyboards/story_N.json，回傳 {故事編號: 分鏡 dict}。"""
    boards = {}
    for path in sorted(Path(data_dir).glob("story_*.json")):
        story_no = int(path.stem.split("_")[1])
        boards[story_no] = json.loads(path.read_text(encoding="utf-8"))
    return boards


def iter_jobs(boards: dict[int, dict], scheme: str):
    """列出每一格要生的圖。prompt 直接用凍結的 sd_prompts，不重新組合。"""
    for story_no, sb in sorted(boards.items()):
        for panel, prompt in zip(sb["panels"], sb["sd_prompts"]):
            yield {
                "scheme": scheme,
                "story_no": story_no,
                "panel_index": panel["index"],
                "characters": panel["characters"],
                "prompt": prompt,
                "negative_prompt": sb["negative_prompt"],
                "seed": panel_seed(story_no, panel["index"]),
            }


def panel_paths(out_root: str | Path, job: dict) -> tuple[Path, Path]:
    """輸出路徑：<out_root>/<scheme>/story_N/panel_XX.png 與同名 .json。"""
    base = Path(out_root) / job["scheme"] / f"story_{job['story_no']}" / f"panel_{job['panel_index']:02d}"
    return base.with_suffix(".png"), base.with_suffix(".json")


def write_record(json_path: Path, job: dict, config: GenConfig, **extra) -> None:
    """每張圖旁邊存一份 json，記下重現所需的一切（prompt、seed、參數、耗時等）。"""
    record = {**job, "config": asdict(config), **extra}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
