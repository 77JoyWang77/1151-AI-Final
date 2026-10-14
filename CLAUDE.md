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
- 所有 LLM 呼叫經過 `src/comic_agent/llm.py` 的 `structured_call()`，自帶磁碟快取 `.cache/llm/`；可附圖（`images=`）、可設 temperature。
- Reflection 評審用 `gpt-4.1`（mini 會漏看明顯物件）；分鏡仍用 `.env` 的 `OPENAI_MODEL`。累計 API 花費約 0.7 美金（2026-10-14）。
- 圖像：HuggingFace `diffusers`、`peft`（LoRA）、`open_clip`。

## 專案結構

```
src/comic_agent/
  schema.py       Character / Dialogue / Panel / Storyboard（pydantic）
  llm.py          OpenAI 結構化輸出 + 磁碟快取
  storyboard.py   generate_storyboard / validate_storyboard / build_sd_prompt
  app.py          Gradio 介面（主題／範例 → 分鏡與台詞表 → 生圖 → 單格重生 → 排版下載）；renderer 可抽換
  app_logic.py    介面背後的邏輯（不 import gradio）：台詞表 ↔ 分鏡、DemoRenderer（本機無 GPU：讀 Drive 圖或占位圖）、組書
  human_eval.py   人工抽查：固定抽 8 個位置（各方案相同）、盲測順序、存檔續評、Cohen's kappa、Spearman
  evaluate.py     評估指標（CPU）：分 pass 一次載入一個模型、磁碟快取；含 SAM 去背的角色一致性
  layout.py       排版（Pillow）：台詞 → 對話框（尾巴指向說話者頭部、避開臉）、旁白 → 圖下方旁白區、封面、PDF
  reflect.py      Reflection：GPT（gpt-4.1）看圖回報事實，decide() 依規則決定 pass／revise_prompt／reseed；reflection_loop() 為不含模型的重生流程
  render.py       生圖設定 GenConfig、seed 規則、凍結分鏡 → 生圖工作、參考圖候選、輸出路徑與紀錄（不 import torch）
notebooks/
  01_sdxl_baseline.ipynb  Colab：方案 A（prompt only）SDXL 逐格生圖，結果存 Drive、可續跑
  02_ip_adapter.ipynb     Colab：方案 B，主角參考圖候選 → 人工挑選 → IP-Adapter scale 掃描 → 30 格
  03_reflect_regen.ipynb  Colab：reflection loop，GPT 檢查 → 改 prompt／換 seed 重生（每格最多 3 次）→ 輸出 <方案>_R；需 Secrets 的 OPENAI_API_KEY
scripts/
  run_storyboard.py   CLI：產生分鏡並存 outputs/storyboard.json
  test_offline.py     不呼叫 API 的測試
  run_rating.py       人工抽查評分介面（盲測；存 data/human_eval/ratings.csv，可續評）
  analyze_ratings.py  各方案人評平均、人 vs GPT 一致率與 kappa、人評 vs 指標等級相關
  run_app.py          啟動 Gradio（本機示範模式；--share 產生公開網址）
  detect_characters.py  偵測配角位置（Grounding DINO，CPU）→ .cache/eval/char_boxes.json，讓配角的對話框也有尾巴
  make_book.py        排版 CLI：讀 G: 上的圖 + 評估快取的主角框 → outputs/books/<scheme>/story_N/（PNG + PDF）
  run_reflect.py      Reflection CLI：逐格看圖檢查，輸出 outputs/reflect/reflect_<scheme>.csv
  run_eval.py         評估 CLI：直接讀 G: 上的 Colab 結果，輸出 outputs/eval/*.csv
  build_notebooks.py  產生所有 notebooks/*.ipynb 的唯一來源（共用 cell 只寫一次、含語法檢查）；改 notebook 請改這裡再重跑
  set_art_style.py    統一替換凍結分鏡的 art_style 並重建 sd_prompts（不呼叫 LLM）
runs/                 Colab「儲存至 GitHub」存回的已執行 notebook（含輸出，供檢視結果）
docs/proposal.md      企畫書（主版本，線上版已停止更新）
docs/research_notes.md  相關研究調查、採用／不採用理由、待確認清單
data/human_eval/      人工抽查評分（ratings.csv，研究資料，納入版本控制）
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

## 目前進度（2026-10-14）

| 階段 | 狀態 |
|---|---|
| 1. 分鏡 Agent | **完成**：5 個實驗主題皆產出合法分鏡並凍結於 `data/storyboards/`。驗證涵蓋格數、未定義角色、每格最多 2 角色、SD prompt 粗估 ≤ 75 tokens；不合格自動重試一次、只快取合格結果 |
| 2. Colab 跑通 SDXL 逐格生圖 | 流程已在 Colab 跑通（watercolor 版 30 格：同 seed 像素差 0、每格 25 秒、實際 tokens 最長 61、實際/粗估 1.09 → `MAX_CLIP_TOKENS=68`）。畫風比較後改用 gouache，**方案 A 需以新畫風重跑** |
| 3. IP-Adapter、LoRA | IP-Adapter（方案 B）提前進行：`02_ip_adapter.ipynb` 已在 Colab 跑完 30 格（scale 0.3），待依挑選規則重選參考圖。LoRA 未開始 |
| 4. Reflection loop | 看圖檢查完成（gpt-4.1，A／B 各 30 格，抽查動作判斷合理 94%）；重生 notebook `03_reflect_regen.ipynb` 已寫好並以假 GPU／假 GPT 乾跑通過，**尚未在 Colab 執行** |
| 5. 排版、Gradio、一致性實驗 | 排版完成（`make_book.py`，A 與 B@0.3 各 5 本）；一致性實驗 A vs B 完成；Gradio 骨架完成（本機示範模式，已用 gradio_client 走完整流程）；SDXL renderer 未接（需 GPU） |

已知待處理：

- SD prompt 長度用 `estimate_clip_tokens()` 粗估（凍結分鏡最長約 57），需在 Colab 用 CLIP tokenizer 實測；若實際超過 77，調整 `MAX_CLIP_TOKENS` 或改用 `compel`。
- 固定服裝會與劇情衝突（例如室內仍穿雨衣）；實驗階段維持固定服裝，報告列為限制。
- 配角（如故事 1 的媽媽）LLM 不一定定義成角色，程式無法從中文劇情偵測；主實驗只量測主角一致性。
GitHub：<https://github.com/77JoyWang77/1151-AI-Final>（`main`），Colab 以 `git clone` 取得程式與 `data/storyboards/`。

評估（CPU，`scripts/run_eval.py`）已完成 A vs B@0.3 與 scale 掃描：去背後角色一致性 0.572 → 0.717（+25%，4/5 篇提升），代價為 CLIP −8.6%、構圖多樣性 −38%；細節見 `docs/research_notes.md` 7.1、7.2。

待使用者做：`python scripts/run_rating.py` 評 16 張（A 與 B_s0.3 的同樣 8 格），再跑 `analyze_ratings.py`。

下一步：依挑選規則重選故事 1、2、3 的參考圖後重跑 B（需 GPU）；Colab 執行 `03_reflect_regen.ipynb`（先對重跑後的 B，預估 0.5–0.75 美金），比較 `<方案>` 與 `<方案>_R` 做 reflection 消融；視需要加「畫風一致性」指標與 InstantStyle。

方案 B 已在 Colab 跑完（參考圖 6／7／5／8／8、掃描 0.2–0.4、30 格 @0.3；結果在 Drive，本機可由 `G:\我的雲端硬碟\02 課程\1151-AI-Final\outputs\render`（Colab 為 `/content/drive/MyDrive/02 課程/1151-AI-Final`） 直接讀取）。0.3 讓臉、服裝、畫風跨格一致；參考圖的構圖、背景（含素色底色）、顏色、姿勢都會被照抄並壓過 prompt。細節與參考圖挑選規則見 `docs/research_notes.md` 第七節；建議依規則重選故事 1、2、3 的參考圖後重跑（需 GPU）。

watercolor 版觀察（待 gouache 重跑後確認）：單色系動物角色光靠 prompt 就相當一致，多色外觀的人類角色有屬性綁定錯置（顏色互換）；劇情關鍵物件常漏畫；shot_type 幾乎被忽略。
