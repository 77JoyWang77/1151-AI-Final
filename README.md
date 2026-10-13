# 繪本生成 Agent（Picture Book Agent）

生成式 AI：文字與圖像生成的原理與實務｜期末專案

LLM 寫故事分鏡 → Diffusion 逐格生圖 → 角色一致性（IP-Adapter / LoRA）→ 品質檢查 → 排版輸出

## 進度

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| 1 | 分鏡 Agent（OpenAI 結構化輸出） | ✅ 骨架完成 |
| 2 | Colab 跑通 SDXL 逐格生圖 | ⏳ |
| 3 | 角色一致性：IP-Adapter、LoRA | ⏳（W12 後） |
| 4 | Reflection：CLIP 分數 ＋ GPT 看圖檢查、自動重生 | ⏳ |
| 5 | 排版 ＋ Gradio 介面、一致性實驗 | ⏳ |

## 快速開始

```bash
pip install -r requirements.txt
copy .env.example .env        # 填入 OPENAI_API_KEY
python scripts/test_offline.py                      # 不花錢，檢查程式
python scripts/run_storyboard.py "小女孩在雨天找到一隻迷路的小貓" --panels 6
```

輸出在 `outputs/storyboard.json`，包含每格的 `sd_prompts`，第二階段直接拿去生圖。

## 專案結構

``` text
src/comic_agent/
  schema.py       分鏡資料結構（Character / Panel / Storyboard）
  llm.py          OpenAI 結構化輸出 ＋ 磁碟快取
  storyboard.py   分鏡生成、邏輯驗證、SD prompt 組合
scripts/
  run_storyboard.py   CLI
  test_offline.py     離線測試
```

## 設計重點

- **外觀只定義一次**：角色外觀寫在 `Character.appearance`，每格 `image_prompt` 只寫場景與動作，最終 prompt 由 `build_sd_prompt()` 組合。這是 prompt 層的一致性策略，也是之後實驗的 baseline。
- **快取**：相同輸入直接讀 `.cache/llm/`，不重複花 API 費用；要重生請加 `--no-cache`。
- **驗證**：schema 擋不住的邏輯錯誤（格數不對、引用不存在的角色）由 `validate_storyboard()` 檢查。
