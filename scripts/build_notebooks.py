"""產生 notebooks/*.ipynb（notebook 的唯一來源，不要直接改 .ipynb）。

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
NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

notebooks = {}  # 檔名 → cells
cells = []


def start(name):
    """之後的 md() / code() 都寫進這份 notebook。"""
    global cells
    cells = notebooks[name] = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


# ==== 共用 cell（01、02 都用）====

def settings_code(extra):
    return '''
# ==== 設定 ====
REPO_URL = "https://github.com/77JoyWang77/1151-AI-Final.git"
REPO_DIR = "/content/1151-AI-Final"
DRIVE_ROOT = "/content/drive/MyDrive/02 課程/1151-AI-Final"  # 生成結果存這裡
''' + extra.strip("\n") + '''
CPU_OFFLOAD = False        # 若出現 CUDA out of memory 改成 True（較慢）

from google.colab import drive
drive.mount("/content/drive")
'''


CLONE_MD = '''
### 取得程式碼

repo 目前是 public，不用設定。若改成 **private**：左側欄「鑰匙」圖示（Secrets）新增 `GH_TOKEN`，值為 GitHub personal access token（只需 repo 讀取權限），並開啟「Notebook access」。

若重啟過執行階段（`/content` 會被清空），要從第一格重新執行。
'''

CLONE_CODE = '''
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
'''

PIP_CODE = '''
!pip install -q diffusers transformers accelerate python-dotenv
'''


def imports_code(extra):
    return '''
import importlib
import json
import subprocess
import sys
import time

if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
importlib.invalidate_caches()  # 同一個 kernel 若在 clone 前 import 失敗過，Python 會記住「路徑不存在」
from comic_agent import render

CONFIG = render.GenConfig()
OUT_ROOT = f"{DRIVE_ROOT}/outputs/render"
GIT_COMMIT = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()

boards = render.load_frozen_storyboards(f"{REPO_DIR}/data/storyboards")
''' + extra


LOAD_SDXL_MD = '''
### 載入 SDXL

第一次執行會下載約 7 GB 權重，需要幾分鐘。VAE 換成 `sdxl-vae-fp16-fix`，原版 VAE 在 fp16 下會溢位產生黑圖。
'''

LOAD_SDXL_CODE = '''
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
'''

GENERATE_CODE = '''
from pathlib import Path

from PIL import Image, ImageDraw


def generate(job, out_root=OUT_ROOT, skip_existing=True, pipe_kwargs=None, record=None):
    """生成一格並存圖與紀錄；已存在且設定相同就略過（斷線續跑）。

    「設定」= prompt 加上 record 的欄位（例如 IP-Adapter scale、參考圖）。
    任一項改變（例如換了畫風或參考圖）都會重新生成並覆蓋，避免沿用舊圖。
    """
    record = record or {}
    png, js = render.panel_paths(out_root, job)
    if skip_existing and png.exists() and js.exists():
        old = json.loads(js.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in {"prompt": job["prompt"], **record}.items()):
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
        **(pipe_kwargs or {}),
    ).images[0]
    seconds = round(time.time() - t0, 1)
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)
    render.write_record(js, job, CONFIG, seconds=seconds, git_commit=GIT_COMMIT,
                        gpu=torch.cuda.get_device_name(0), **record)
    return png, True


def contact_sheet(pngs, cols=3, thumb=384, labels=None):
    """把多張圖排成總覽圖；labels 為每一列的標題。缺圖的位置留白。"""
    rows = (len(pngs) + cols - 1) // cols
    head = 28 if labels else 0
    sheet = Image.new("RGB", (thumb * cols, (thumb + head) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(pngs):
        r, c = divmod(i, cols)
        if p is not None and Path(p).exists():
            sheet.paste(Image.open(p).resize((thumb, thumb)), (c * thumb, r * (thumb + head) + head))
    for r, text in enumerate(labels or []):
        draw.text((6, r * (thumb + head) + 6), text, fill="black")
    return sheet
'''


# ==== 01：方案 A ====

start("01_sdxl_baseline.ipynb")

md('''
# 01 SDXL 逐格生圖：方案 A（Prompt only baseline）

讀取 `data/storyboards/` 的 5 份凍結分鏡（共 30 格），用 SDXL 逐格生圖。這是一致性實驗的對照組，也用來驗證整條生圖流程。

**執行前**：「執行階段 → 變更執行階段類型」選 **T4 GPU**。

- 生圖參數固定在 `comic_agent.render.GenConfig`（SDXL base 1.0、fp16、DPM++ 2M Karras、30 步、CFG 6.5、1024×1024）
- seed = 故事編號 × 100 + 格數，四種方案共用
- 每張圖旁邊存一份 `.json`，記錄 prompt、seed、參數、耗時、git commit
- 結果存到 Google Drive，**斷線後重跑會略過已完成的格**；若 prompt 改變（例如換畫風），該格會重新生成並覆蓋
''')

code(settings_code('''
SCHEME = "A"               # 方案 A：prompt only
STORIES = [1, 2, 3, 4, 5]  # 先試跑可改成 [1]
'''))
md(CLONE_MD)
code(CLONE_CODE)
code(PIP_CODE)
code(imports_code('''from comic_agent.storyboard import estimate_clip_tokens

jobs = [j for j in render.iter_jobs(boards, SCHEME) if j["story_no"] in STORIES]
print(CONFIG)
print(f"{len(jobs)} 格，commit {GIT_COMMIT}，輸出到 {OUT_ROOT}/{SCHEME}")
'''))
md(LOAD_SDXL_MD)
code(LOAD_SDXL_CODE)

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

code(GENERATE_CODE)

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

每格約 25 秒（T4），30 格約 13 分鐘。斷線後重新執行上面各格與這一格即可續跑。
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
for story_no in STORIES:
    story_dir = Path(OUT_ROOT) / SCHEME / f"story_{story_no}"
    pngs = sorted(story_dir.glob("panel_*.png"))
    if not pngs:
        continue
    sheet = contact_sheet(pngs)
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
from comic_agent import Storyboard, build_sd_prompt

STYLES = {  # key 用英文，當作檔名與圖上標籤
    "current": None,  # None = 用凍結分鏡原本的 prompt
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


pngs = []
for key, style in STYLES.items():
    for story_no, panel_index in TEST_PANELS:
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
        pngs.append(png)
    print(f"{key}: {style or '（凍結分鏡目前的 art_style）'}")

grid = contact_sheet(pngs, cols=len(TEST_PANELS), labels=list(STYLES))
grid.save(STYLE_DIR / "style_compare.png")
display(grid)
''')


# ==== 02：方案 B ====

start("02_ip_adapter.ipynb")

md('''
# 02 方案 B：IP-Adapter（參考圖約束角色外觀）

方案 A 只靠文字描述外觀，人類角色的臉每格都不同。方案 B 先為每篇主角生成一張參考圖，再用 IP-Adapter 把參考圖的 image embedding 注入 cross-attention，讓 6 格都參照同一個角色。

**執行前**：「執行階段 → 變更執行階段類型」選 **T4 GPU**。

**步驟**
1. 生成主角參考圖候選（每篇 `N_REF` 張正面全身立繪）
2. **人工挑選**每篇的參考圖（填 `REF_CHOICE`）
3. 載入 IP-Adapter（`h94/IP-Adapter` 的 `ip-adapter-plus_sdxl_vit-h`）
4. scale 掃描：在 `SCAN_STORIES` 上比較 `SCALES`，與方案 A 並排
5. 選定 `BEST_SCALE`，跑完 5 篇 30 格
6. 方案 A／B 並排比較

**設計**
- 只對主角（`characters[0]`）用參考圖；故事 3 的小貓仍只靠 prompt（範圍以單一主角為底線）
- seed、prompt、生圖參數都與方案 A 相同，差異只來自 IP-Adapter
- 參考圖用中性站姿、白背景，降低 IP-Adapter 照抄參考圖姿勢與構圖的傾向
- 輸出：`outputs/render/refs/`（參考圖候選）、`outputs/render/B_s{scale}/`（各 scale 的結果）
''')

code(settings_code('''
STORIES = [1, 2, 3, 4, 5]
N_REF = 8                  # 每個主角的參考圖候選數（多色外觀的人類常需要較多候選才有顏色正確的）
SCALES = [0.2, 0.3, 0.4]   # IP-Adapter scale 掃描（首輪 0.5 以上已明顯照抄參考圖）
SCAN_STORIES = [1, 3]      # 掃描用：方案 A 一致性最差的兩篇（多色外觀的人類）
'''))
md(CLONE_MD)
code(CLONE_CODE)
code(PIP_CODE)
code(imports_code('''
ref_jobs = [r for r in render.reference_jobs(boards, N_REF) if r["story_no"] in STORIES]
print(CONFIG)
print(f"參考圖候選 {len(ref_jobs)} 張，commit {GIT_COMMIT}，輸出到 {OUT_ROOT}")
'''))
md(LOAD_SDXL_MD)
code(LOAD_SDXL_CODE)
code(GENERATE_CODE)

md('''
### 1. 生成主角參考圖候選

每篇 `N_REF` 張，約 25 秒一張（5 篇 × 8 張 ≈ 17 分鐘；已生成的會略過）。**必須在載入 IP-Adapter 之前執行**：載入後每次生圖都需要參考圖。若之後要補生參考圖，請重啟執行階段再從頭跑（已生成的會略過）。
''')

code('''
for k, job in enumerate(ref_jobs, 1):
    _, new = generate(job)
    print(f"[{k}/{len(ref_jobs)}] story {job['story_no']} 候選 #{job['panel_index']}：{'生成' if new else '已存在，略過'}")

for story_no in STORIES:
    hero = render.protagonist(boards[story_no])
    cands = [render.panel_paths(OUT_ROOT, j)[0] for j in ref_jobs if j["story_no"] == story_no]
    print(f"\\nstory {story_no} 主角 {hero['id']}：{hero['appearance']}")
    display(contact_sheet(cands, cols=N_REF, thumb=256, labels=["#1 ~ #%d (left to right)" % N_REF]))
''')

md('''
### 2. 挑選參考圖

每篇填一個候選編號（由左至右 1 開始）。挑選原則：
- 外觀符合描述（髮型、服裝、配色都對）
- 正面、全身、背景乾淨、只有一個角色
- 姿勢中性（越特殊的姿勢越容易被照抄到每一格）
- **背景素色**：參考圖的背景也會被照抄，有場景的參考圖會讓每一格都變成同一個場景
- 顏色要對：參考圖的顏色會壓過 prompt，顏色錯的參考圖會讓每一格「一致地錯」
''')

code('''
REF_CHOICE = {1: None, 2: 1, 3: None, 4: 4, 5: 2}  # 故事編號 → 候選編號；None = 尚未選定

missing = [s for s in STORIES if REF_CHOICE.get(s) is None]
assert not missing, f"請先看上面的候選圖，為故事 {missing} 填入 REF_CHOICE"
ref_paths = {}
for story_no in STORIES:
    job = next(j for j in ref_jobs if j["story_no"] == story_no and j["panel_index"] == REF_CHOICE[story_no])
    ref_paths[story_no] = render.panel_paths(OUT_ROOT, job)[0]
(Path(OUT_ROOT) / "refs" / "choice.json").write_text(
    json.dumps({str(s): REF_CHOICE[s] for s in STORIES}, indent=2), encoding="utf-8")
display(contact_sheet(list(ref_paths.values()), cols=len(ref_paths), thumb=256,
                      labels=["selected refs: story " + ", ".join(map(str, STORIES))]))
''')

md('''
### 3. 載入 IP-Adapter

`plus` 版保留較多服裝與配色細節；image encoder 是 ViT-H（`models/image_encoder`），與 SDXL 資料夾內預設的 ViT-bigG 不同，要明確指定。
''')

code('''
IP_REPO, IP_SUBFOLDER, IP_WEIGHT = "h94/IP-Adapter", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.safetensors"
pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUBFOLDER, weight_name=IP_WEIGHT,
                     image_encoder_folder="models/image_encoder")
print("IP-Adapter 已載入", f"{torch.cuda.memory_allocated() / 2**30:.1f} GB")


def run_scheme_b(scale, stories):
    """以指定 scale 生成方案 B；結果存在 B_s{scale}/，換參考圖或 scale 會自動重生。"""
    pipe.set_ip_adapter_scale(scale)
    scheme = f"B_s{scale}"
    todo = [j for j in render.iter_jobs(boards, scheme) if j["story_no"] in stories]
    for k, job in enumerate(todo, 1):
        ref = ref_paths[job["story_no"]]
        _, new = generate(
            job,
            pipe_kwargs={"ip_adapter_image": Image.open(ref).convert("RGB")},
            record={"ip_adapter": IP_WEIGHT, "ip_scale": scale,
                    "ref_image": f"refs/story_{job['story_no']}/{ref.name}"},
        )
        print(f"[{scheme} {k}/{len(todo)}] story {job['story_no']} panel {job['panel_index']}："
              f"{'生成' if new else '已存在，略過'}")
    return scheme


def compare(story_no, schemes):
    """每列一個方案、每欄一格，與參考圖一起顯示。"""
    n = len(boards[story_no]["panels"])
    pngs = [render.panel_paths(OUT_ROOT, {"scheme": s, "story_no": story_no, "panel_index": i})[0]
            for s in schemes for i in range(1, n + 1)]
    print(f"story {story_no}《{boards[story_no]['title']}》 參考圖：")
    display(Image.open(ref_paths[story_no]).resize((192, 192)))
    sheet = contact_sheet(pngs, cols=n, thumb=256, labels=list(schemes))
    sheet.save(Path(OUT_ROOT) / f"compare_story_{story_no}.png")
    display(sheet)
''')

md('''
### 4. scale 掃描

scale 越高越像參考圖，但越容易壓過 prompt（動作、場景不符）或照抄參考圖姿勢。首輪試跑 0.3／0.5／0.7：0.5 起每格幾乎都是參考圖的站姿與背景，0.7 六格近乎同一張圖，因此改在 0.2–0.4 間找平衡點。2 篇 × 6 格 × 3 個 scale ≈ 16 分鐘。看並排圖時注意：
- 臉、髮型、服裝是否跨格一致（一致性）
- 每格的動作與場景是否還符合分鏡（圖文對齊）
- 是否每格都是參考圖的站姿（copy-paste）
''')

code('''
for scale in SCALES:
    run_scheme_b(scale, SCAN_STORIES)
for story_no in SCAN_STORIES:
    compare(story_no, ["A"] + [f"B_s{s}" for s in SCALES])
''')

md('''
### 5. 以選定的 scale 跑完 5 篇

掃描用過的故事會直接沿用，不會重生。
''')

code('''
BEST_SCALE = 0.3  # 依掃描結果修改
scheme = run_scheme_b(BEST_SCALE, STORIES)

for story_no in STORIES:
    compare(story_no, ["A", scheme])

secs = [json.loads(p.read_text(encoding="utf-8"))["seconds"]
        for p in (Path(OUT_ROOT) / scheme).glob("story_*/panel_*.json")]
print(f"{scheme}：已完成 {len(secs)} 格，平均每格 {sum(secs) / len(secs):.1f} 秒")
''')


# ==== 寫檔 ====

for name, nb_cells in notebooks.items():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": nb_cells,
    }
    for c in nb_cells:
        if c["cell_type"] == "code":  # 語法檢查：把 IPython 的 `!` shell 行換成 pass
            ast.parse(re.sub(r"^(\s*)!.*$", r"\1pass", c["source"], flags=re.M))
        lines = c["source"].split("\n")  # ipynb 的 source 慣例為逐行 list
        c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    out = NB_DIR / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[OK] 寫入 notebooks/{name}，{len(nb_cells)} 個 cell，語法檢查通過")
