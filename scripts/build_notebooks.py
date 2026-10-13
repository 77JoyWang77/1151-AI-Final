"""產生 notebooks/01_sdxl_baseline.ipynb（notebook 的唯一來源，不要直接改 .ipynb）。

用法：python scripts/build_notebooks.py
寫入前會檢查每個 code cell 的 Python 語法（`!` 開頭的 shell 行略過）。
注意：cell 內容是一般 ''' 字串，要在 cell 裡寫字面上的 \\n 必須寫成 \\\\n。
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 終端機預設 cp950，印不出部分符號
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_sdxl_baseline.ipynb"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md('''
# 01 SDXL 逐格生圖：方案 A（Prompt only baseline）

讀取 `data/storyboards/` 的 5 份凍結分鏡（共 30 格），用 SDXL 逐格生圖。這是一致性實驗的對照組，也用來驗證整條生圖流程。

**執行前**：「執行階段 → 變更執行階段類型」選 **T4 GPU**。

- 生圖參數固定在 `comic_agent.render.GenConfig`（SDXL base 1.0、fp16、DPM++ 2M Karras、30 步、CFG 6.5、1024×1024）
- seed = 故事編號 × 100 + 格數，四種方案共用
- 每張圖旁邊存一份 `.json`，記錄 prompt、seed、參數、耗時、git commit
- 結果存到 Google Drive，**斷線後重跑會略過已完成的格**；若 prompt 改變（例如換畫風），該格會重新生成並覆蓋
''')

code('''
# ==== 設定 ====
REPO_URL = "https://github.com/77JoyWang77/1151-AI-Final.git"
REPO_DIR = "/content/1151-AI-Final"
DRIVE_ROOT = "/content/drive/MyDrive/1151-AI-Final"  # 生成結果存這裡
SCHEME = "A"               # 方案 A：prompt only
STORIES = [1, 2, 3, 4, 5]  # 先試跑可改成 [1]
CPU_OFFLOAD = False        # 若出現 CUDA out of memory 改成 True（較慢）

from google.colab import drive
drive.mount("/content/drive")
''')

md('''
### 取得程式碼

repo 目前是 public，不用設定。若改成 **private**：左側欄「鑰匙」圖示（Secrets）新增 `GH_TOKEN`，值為 GitHub personal access token（只需 repo 讀取權限），並開啟「Notebook access」。

若重啟過執行階段（`/content` 會被清空），要從第一格重新執行。
''')

code('''
import base64
import os
import subprocess


def clone(*git_opts):
    return subprocess.run(["git", *git_opts, "clone", "-q", REPO_URL, REPO_DIR], capture_output=True, text=True)


if os.path.isdir(f"{REPO_DIR}/.git"):
    !git -C {REPO_DIR} pull -q --ff-only
else:
    !rm -rf {REPO_DIR}  # 清掉上次 clone 失敗留下的殘骸
    result = clone()  # public repo 不需要 token
    if result.returncode != 0:
        try:
            from google.colab import userdata
            token = (userdata.get("GH_TOKEN") or "").strip()
        except Exception:
            token = ""
        # 不印出 token 內容；檢查前綴，避免誤填成 OpenAI key 等其他密鑰
        if not token.startswith(("ghp_", "github_pat_")):
            raise RuntimeError("clone 失敗，且 Secrets 沒有有效的 GH_TOKEN（GitHub token 以 ghp_ 或 github_pat_ 開頭）\\n"
                               + result.stderr)
        # token 放在 HTTP header 而不是網址，git 的錯誤訊息不會把它印出來
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        result = clone("-c", f"http.extraHeader=Authorization: Basic {auth}")
        if result.returncode != 0:
            raise RuntimeError("用 GH_TOKEN clone 仍失敗：\\n" + result.stderr)
assert os.path.isdir(f"{REPO_DIR}/src/comic_agent"), "clone 失敗：src/comic_agent 不存在"
!git -C {REPO_DIR} log --oneline -1
''')

code('''
!pip install -q diffusers transformers accelerate python-dotenv
''')

code('''
import importlib
import json
import subprocess
import sys
import time

if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
importlib.invalidate_caches()  # 同一個 kernel 若在 clone 前 import 失敗過，Python 會記住「路徑不存在」
from comic_agent import render
from comic_agent.storyboard import estimate_clip_tokens

CONFIG = render.GenConfig()
OUT_ROOT = f"{DRIVE_ROOT}/outputs/render"
GIT_COMMIT = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()

boards = render.load_frozen_storyboards(f"{REPO_DIR}/data/storyboards")
jobs = [j for j in render.iter_jobs(boards, SCHEME) if j["story_no"] in STORIES]
print(CONFIG)
print(f"{len(jobs)} 格，commit {GIT_COMMIT}，輸出到 {OUT_ROOT}/{SCHEME}")
''')

md('''
### 載入 SDXL

第一次執行會下載約 7 GB 權重，需要幾分鐘。VAE 換成 `sdxl-vae-fp16-fix`，原版 VAE 在 fp16 下會溢位產生黑圖。
''')

code('''
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

vae = AutoencoderKL.from_pretrained(CONFIG.vae_id, dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    CONFIG.model_id, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True,
)
# DPM++ 2M Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True,
)
if CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(torch.cuda.get_device_name(0), f"{torch.cuda.memory_allocated() / 2**30:.1f} GB")
''')

md('''
### 用真正的 CLIP tokenizer 檢查 prompt 長度

本機只能用 `estimate_clip_tokens()` 粗估。CLIP 上限 77 tokens（含開頭與結尾符號），超過的部分會被截掉。這格同時校正粗估值：若實際值普遍比粗估多，要調低 `MAX_CLIP_TOKENS`。
''')

code('''
rows = []
for j in jobs:
    actual = max(len(pipe.tokenizer(j["prompt"]).input_ids), len(pipe.tokenizer_2(j["prompt"]).input_ids))
    rows.append((j["story_no"], j["panel_index"], estimate_clip_tokens(j["prompt"]), actual))

over = [r for r in rows if r[3] > 77]
ratio = sum(r[3] / r[2] for r in rows) / len(rows)
print(f"實際 tokens：最長 {max(r[3] for r in rows)}，超過 77 的格數 {len(over)} / {len(rows)}")
print(f"實際 / 粗估 平均比例 {ratio:.2f}")
for s, p, est, act in over:
    print(f"  story {s} panel {p}: 粗估 {est}，實際 {act}（尾端會被截斷）")
neg = len(pipe.tokenizer(jobs[0]["negative_prompt"]).input_ids)
print(f"negative prompt：{neg} tokens")
''')

code('''
from PIL import Image


def generate(job, out_root=OUT_ROOT, skip_existing=True):
    """生成一格並存圖與紀錄；已存在且 prompt 相同就略過（斷線續跑）。

    prompt 不同（例如換了畫風）會重新生成並覆蓋，避免沿用舊圖。
    """
    png, js = render.panel_paths(out_root, job)
    if skip_existing and png.exists() and js.exists():
        if json.loads(js.read_text(encoding="utf-8"))["prompt"] == job["prompt"]:
            return png, False
    generator = torch.Generator("cpu").manual_seed(job["seed"])  # CPU generator：跨機器較可重現
    t0 = time.time()
    image = pipe(
        prompt=job["prompt"],
        negative_prompt=job["negative_prompt"],
        num_inference_steps=CONFIG.steps,
        guidance_scale=CONFIG.cfg,
        width=CONFIG.width,
        height=CONFIG.height,
        generator=generator,
    ).images[0]
    seconds = round(time.time() - t0, 1)
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    render.write_record(js, job, CONFIG, seconds=seconds, git_commit=GIT_COMMIT,
                        gpu=torch.cuda.get_device_name(0))
    return png, True
''')

md('''
### 試跑一格並檢查可重現性

同一個 seed 再生一次，比較兩張圖的像素差。GPU 運算有微小的非確定性，平均差值接近 0 即可視為可重現。
''')

code('''
import numpy as np

png, _ = generate(jobs[0])
again, _ = generate(jobs[0], out_root="/content/repro_check", skip_existing=False)
diff = np.abs(np.asarray(Image.open(png), dtype=float) - np.asarray(Image.open(again), dtype=float)).mean()
print(f"同 seed 兩次生成的平均像素差：{diff:.2f}（0–255）")
display(Image.open(png).resize((512, 512)))
''')

md('''
### 批次生成

每格約 30–40 秒（T4），30 格約 20 分鐘。斷線後重新執行上面各格與這一格即可續跑。
''')

code('''
for k, job in enumerate(jobs, 1):
    png, new = generate(job)
    status = "生成" if new else "已存在，略過"
    print(f"[{k}/{len(jobs)}] story {job['story_no']} panel {job['panel_index']}：{status}")
''')

md('''
### 每篇故事的總覽圖與耗時

總覽圖存成 `<scheme>/story_N/contact.png`。方案 A 沒有任何一致性機制，預期主角在各格的長相會有明顯差異；這正是後續方案 B–D 要改善的地方。
''')

code('''
from pathlib import Path

THUMB = 384
for story_no in STORIES:
    story_dir = Path(OUT_ROOT) / SCHEME / f"story_{story_no}"
    pngs = sorted(story_dir.glob("panel_*.png"))
    if not pngs:
        continue
    sheet = Image.new("RGB", (THUMB * 3, THUMB * ((len(pngs) + 2) // 3)), "white")
    for i, p in enumerate(pngs):
        sheet.paste(Image.open(p).resize((THUMB, THUMB)), ((i % 3) * THUMB, (i // 3) * THUMB))
    sheet.save(story_dir / "contact.png")
    print(f"story {story_no}《{boards[story_no]['title']}》")
    display(sheet)

secs = [json.loads(p.read_text(encoding="utf-8"))["seconds"]
        for p in (Path(OUT_ROOT) / SCHEME).glob("story_*/panel_*.json")]
print(f"已完成 {len(secs)} 格，平均每格 {sum(secs) / len(secs):.1f} 秒")
''')

md('''
### （選用）畫風比較

同一格分鏡、同一個 seed，只替換 `art_style`，比較不同畫風。角色外觀與場景完全不變，`current` 那一列應與方案 A 的圖相同。

- 測試 2 格：故事 1 第 3 格（多色外觀的人類）、故事 2 第 1 格（單色系動物）
- 結果存到 `outputs/style_compare/`，重跑會略過已生成的圖
- 可自行增刪 `STYLES`；不要放在世畫家或工作室名稱（企畫書倫理原則）
- 選定後，畫風會統一寫回 5 份凍結分鏡並重跑方案 A
''')

code('''
from PIL import ImageDraw

from comic_agent import Storyboard, build_sd_prompt

STYLES = {  # key 用英文，當作檔名與圖上標籤
    "current": None,  # None = 用凍結分鏡原本的 prompt（各篇 art_style 字序略有不同）
    "watercolor_cool": "children's picture book, soft watercolor, muted pastel colors, light airy palette",
    "colored_pencil": "children's picture book, colored pencil illustration, soft textures",
    "gouache": "children's picture book, gouache illustration, flat bold colors",
    "flat_vector": "children's picture book, flat vector illustration, clean shapes, minimal shading",
    "crayon": "children's picture book, crayon drawing, textured paper, playful",
    "digital_paint": "children's picture book, digital painting, soft lighting, storybook illustration",
}
TEST_PANELS = [(1, 3), (2, 1)]  # (故事編號, 格數)
STYLE_DIR = Path(DRIVE_ROOT) / "outputs" / "style_compare"


def style_job(story_no, panel_index, style):
    board = boards[story_no]
    if style is None:
        prompt = board["sd_prompts"][panel_index - 1]
    else:
        sb = Storyboard.model_validate(board).model_copy(update={"art_style": style})
        prompt = build_sd_prompt(sb, sb.panels[panel_index - 1])
    return {"prompt": prompt, "negative_prompt": board["negative_prompt"],
            "seed": render.panel_seed(story_no, panel_index)}


THUMB = 384
grid = Image.new("RGB", (THUMB * len(TEST_PANELS), (THUMB + 28) * len(STYLES)), "white")
draw = ImageDraw.Draw(grid)
for row, (key, style) in enumerate(STYLES.items()):
    for col, (story_no, panel_index) in enumerate(TEST_PANELS):
        png = STYLE_DIR / f"{key}_s{story_no}p{panel_index}.png"
        if not png.exists():
            job = style_job(story_no, panel_index, style)
            image = pipe(
                prompt=job["prompt"], negative_prompt=job["negative_prompt"],
                num_inference_steps=CONFIG.steps, guidance_scale=CONFIG.cfg,
                width=CONFIG.width, height=CONFIG.height,
                generator=torch.Generator("cpu").manual_seed(job["seed"]),
            ).images[0]
            png.parent.mkdir(parents=True, exist_ok=True)
            image.save(png)
        y = row * (THUMB + 28)
        grid.paste(Image.open(png).resize((THUMB, THUMB)), (col * THUMB, y + 28))
    draw.text((6, y + 6), key, fill="black")
    print(f"{key}: {style or '（各篇凍結分鏡原本的 art_style）'}")

grid.save(STYLE_DIR / "style_compare.png")
display(grid)
''')

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
for c in nb["cells"]:
    if c["cell_type"] == "code":  # 語法檢查：把 IPython 的 `!` shell 行換成 pass
        ast.parse(re.sub(r"^(\s*)!.*$", r"\1pass", c["source"], flags=re.M))
    lines = c["source"].split("\n")  # ipynb 的 source 慣例為逐行 list
    c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"[OK] 寫入 {OUT.relative_to(OUT.parents[1])}，{len(cells)} 個 cell，語法檢查通過")
