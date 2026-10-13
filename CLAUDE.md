# CLAUDE.md

生成式 AI 課程期末專案：**繪本生成 Agent**。完整企畫書見 `docs/proposal.md`（目標、架構、評估、時程都以它為準）。

## 一句話

使用者輸入故事主題 → LLM 產生分鏡 → Diffusion 逐格生圖 → 角色一致性（IP-Adapter / LoRA）→ CLIP + GPT 品質檢查、不合格重生 → 排版輸出。

## 執行環境（重要）

- 本機：Windows 筆電，**只有 Intel Iris Xe 內顯，沒有 NVIDIA GPU**，不能跑 diffusion。
- 本機只做：分鏡 Agent、LLM 呼叫、純 Python 邏輯、單元測試。
- Colab（免費 T4）做：SDXL / SD 1.5 生圖、IP-Adapter、LoRA 訓練、CLIP / DINOv2 評估、Gradio（`share=True`）。
- Colab 程式寫成 `notebooks/*.ipynb`；需要 GPU 的程式不要寫成假設本機能跑的形式。
- 專案資料夾在 OneDrive，Colab 讀不到；Colab 透過 GitHub `git clone` 取得 `src/comic_agent`。
- Colab 上的模型快取、LoRA 權重、生成結果存 Google Drive，以便斷線後續跑。

## 技術選型

- LLM：OpenAI API，模型由 `.env` 的 `OPENAI_MODEL` 控制（預設 `gpt-4.1-mini`）。預算約 5 美金，避免不必要的呼叫。
- 結構化輸出：`client.responses.parse(..., text_format=PydanticModel)`。
- 所有 LLM 呼叫經過 `src/comic_agent/llm.py` 的 `structured_call()`，自帶磁碟快取 `.cache/llm/`。
- 圖像：HuggingFace `diffusers`、`peft`（LoRA）、`open_clip`。

## 專案結構

```
src/comic_agent/
  schema.py       Character / Dialogue / Panel / Storyboard（pydantic）
  llm.py          OpenAI 結構化輸出 + 磁碟快取
  storyboard.py   generate_storyboard / validate_storyboard / build_sd_prompt
  render.py       生圖設定 GenConfig、seed 規則、凍結分鏡 → 生圖工作、輸出路徑與紀錄（不 import torch）
notebooks/
  01_sdxl_baseline.ipynb  Colab：方案 A（prompt only）SDXL 逐格生圖，結果存 Drive、可續跑
scripts/
  run_storyboard.py   CLI：產生分鏡並存 outputs/storyboard.json
  test_offline.py     不呼叫 API 的測試
  build_notebooks.py  產生 notebooks/*.ipynb 的唯一來源（含語法檢查）；改 notebook 請改這裡再重跑
runs/                 Colab「儲存至 GitHub」存回的已執行 notebook（含輸出，供檢視結果）
docs/proposal.md      企畫書（主版本，線上版已停止更新）
docs/research_notes.md  相關研究調查、採用／不採用理由、待確認清單
data/storyboards/     凍結的 5 份實驗分鏡（納入版本控制，供 Colab 讀取；不要修改或重生）
```

## 設計原則

- **角色外觀只定義一次**：外觀寫在 `Character.appearance`；`Panel.image_prompt` 只寫場景與動作、以角色 id 指稱；最終 SD prompt 由 `build_sd_prompt()` 組合。不要讓 LLM 在每格重寫外觀。`appearance` 只放整個故事不會變的特徵（表情、道具、淋濕等狀態寫在 `image_prompt`）；組合時外觀放在場景前，避免被 CLIP 77 tokens 截斷。
- 台詞不交給 diffusion 生成（中文會亂碼），由 Pillow 後製；negative prompt 排除文字。
- 實驗要可重現：固定 seed、記錄 scheduler、步數、CFG scale。
- 評估數字要附解讀（比較、比例、代表的取捨），不要只列分數。

## 工作慣例

- 使用者偏好：**最小 diff、漸進式修改**，不要整檔重寫；錯誤直接承認。
- 溝通用繁體中文，技術名詞中英並列。
- 改完執行 `python scripts/test_offline.py` 確認沒壞。
- 程式自己印的訊息不要用 `✓` 等 cp950 編不出的符號（Windows 終端機會報 `UnicodeEncodeError`）；scripts 開頭保留 `sys.stdout.reconfigure(encoding="utf-8")` 處理 LLM 輸出。
- 不要提交 `.env`、`.cache/`、`outputs/`、`*.safetensors`。
- Colab 存回 GitHub 時，路徑開頭不能有空格（OneDrive 無法同步，需用 `git mv` 修正）；Colab Secrets 的 `GH_TOKEN` 只能放 GitHub token（repo 是 public，通常不需要）。

## 目前進度（2026-10-13）

| 階段 | 狀態 |
|---|---|
| 1. 分鏡 Agent | **完成**：5 個實驗主題皆產出合法分鏡並凍結於 `data/storyboards/`。驗證涵蓋格數、未定義角色、每格最多 2 角色、SD prompt 粗估 ≤ 75 tokens；不合格自動重試一次、只快取合格結果 |
| 2. Colab 跑通 SDXL 逐格生圖 | notebook `01_sdxl_baseline.ipynb` 已寫好（語法檢查通過），**尚未在 Colab 實際執行** |
| 3. IP-Adapter、LoRA | 未開始（課程 W12，11/24 後） |
| 4. Reflection loop | 未開始 |
| 5. 排版、Gradio、一致性實驗 | 未開始 |

已知待處理：

- SD prompt 長度用 `estimate_clip_tokens()` 粗估（凍結分鏡最長約 57），需在 Colab 用 CLIP tokenizer 實測；若實際超過 77，調整 `MAX_CLIP_TOKENS` 或改用 `compel`。
- 固定服裝會與劇情衝突（例如室內仍穿雨衣）；實驗階段維持固定服裝，報告列為限制。
- 配角（如故事 1 的媽媽）LLM 不一定定義成角色，程式無法從中文劇情偵測；主實驗只量測主角一致性。
GitHub：<https://github.com/77JoyWang77/1151-AI-Final>（`main`），Colab 以 `git clone` 取得程式與 `data/storyboards/`。

下一步：notebook 已推上 GitHub，在 Colab 執行 `01_sdxl_baseline.ipynb`（建議先 `STORIES = [1]` 試跑）；確認 tokenizer 實測長度、同 seed 可重現性、每格耗時，以及方案 A 的 30 張圖（里程碑 10/28–11/16）。
