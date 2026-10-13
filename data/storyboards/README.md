# 凍結的實驗分鏡

一致性實驗（方案 A–D）共用的 5 份分鏡，**不要修改或重新生成**，以免 LLM 隨機性混入方案之間的比較。

- 凍結日期：2026-10-13
- 產生方式：`scripts/run_storyboard.py "<主題>" --panels 6`，模型 `gpt-4.1-mini`，其餘參數用預設值（3–6 歲兒童、溫暖的水彩繪本風）
- 每個檔案除了 `Storyboard` 欄位，還附有 `sd_prompts`（`build_sd_prompt()` 組合結果）與 `negative_prompt`
- 畫風：5 份統一為 `children's picture book, gouache illustration, flat bold colors`（2026-10-13 由畫風比較選定，以 `scripts/set_art_style.py` 寫入；原本是 LLM 各自寫的 warm watercolor）。只改 `art_style` 與 `sd_prompts`，劇情與角色不變

| 檔案 | 主題 | 角色 |
| --- | --- | --- |
| `story_1.json` | 小女孩第一次去海邊 | lily |
| `story_2.json` | 小熊找回遺失的帽子 | bear |
| `story_3.json` | 男孩在雨天幫助迷路的小貓 | boy, kitten |
| `story_4.json` | 小狐狸第一次自己做生日蛋糕 | fox |
| `story_5.json` | 女孩夜晚和月亮說話 | girl |

已知限制：故事 1 的媽媽、故事 2 的蝴蝶與青蛙沒有定義成角色，各格外觀可能不同；主實驗只量測主角一致性。
