# 相關研究調查與企畫書修訂紀錄

> 2026-10-13 調查。用途：日後修改企畫書、實作或寫報告時參考。
> 企畫書本體見 `proposal.md`；本檔記錄「查到什麼、採用什麼、為什麼沒採用什麼」。

## 一、調查結論（TL;DR）

1. **整體方向正確**：「LLM 分鏡 → diffusion 生圖 → 一致性模組 → VLM 檢查 → 修正」是 2025–2026 年故事視覺化 Agent 的主流架構（MM-StoryAgent、Audit & Repair、CANVAS 都屬這一型）。本專案的辨識度在於**同一設定下的量化比較**，不在提出新方法。
2. **最重要的補強是 copy-paste 問題**：一致性分數高，可能只是照抄參考圖的姿勢與構圖。ViStoryBench 為此專門設了 Copy-Paste Detection 指標。企畫書已加入「構圖多樣性」指標與假設 H4。
3. **人臉類 ID 方法不適合繪本**：InstantID、PuLID、IP-Adapter-FaceID 以及 ViStoryBench 的 CIDS 都依賴人臉特徵（ArcFace／AdaFace／FaceNet），動物與卡通角色會失效。維持 IP-Adapter plus 加 DINOv2 的選擇。
4. **Reflection 可借鏡 Audit & Repair**：讓 VLM 逐項比對屬性清單，而不是只問「像不像」；只有單一屬性錯時，局部 inpainting 比整張重生划算。
5. **可加的低成本對照組**：StoryDiffusion（免訓練、外掛 SDXL）與 `gpt-image-1-mini` 商用參照組（30 張約 0.35 美金），都列為彈性目標。

## 二、相關研究速查表

| 名稱 | 年份／出處 | 一句話 | 對本專案的用處 | 連結 |
| --- | --- | --- | --- | --- |
| ViStoryBench | CVPR 2026 | 故事視覺化評測基準：80 段故事、344 個角色、12 項自動指標，有 full／lite 版 | 指標設計參考（CIDS、CSD、PromptAlign、Copy-Paste）；角色指標用人臉特徵，不適合我們直接套用 | [arXiv](https://arxiv.org/abs/2505.24862)、[code](https://github.com/vistorybench/vistorybench) |
| Audit & Repair | arXiv 2506.18900（2025） | 四個 Agent（初始化／稽核／修正／總監）；GPT-4 描述各格屬性並比對不一致，以 Flux-ControlNet 局部修正 | reflection 的屬性清單比對、局部修正；指標含 CLIP-I、DINO、LPIPS、TIFA 與前景遮罩版 | [arXiv](https://arxiv.org/html/2506.18900v1) |
| CANVAS | arXiv 2604.13452（2026） | 角色／場景／道具三種記憶，維持長程連續性；底模 Gemini-3-pro-image | 未來工作：多場景、道具狀態連續性；有 VLM 評估器 ContinuityEval | [arXiv](https://arxiv.org/html/2604.13452v1) |
| MM-StoryAgent | arXiv 2503.05242（2025） | 兒童故事書影片的多 Agent 系統（文字＋圖＋音），開源 | 相關系統比較；它做影片與語音，本專案明確不做（Won't） | [arXiv](https://arxiv.org/pdf/2503.05242) |
| StoryDiffusion | NeurIPS 2024 | Consistent Self-Attention，免訓練、可外掛 SD1.5／SDXL，以滑動視窗降低記憶體 | 彈性方案 E | [code](https://github.com/HVision-NKU/StoryDiffusion) |
| 1Prompt1Story | ICLR 2025 Spotlight | 把所有格的描述串成一個 prompt，以 SVR 與 IPCA 維持身分 | 另一個免訓練候選；比 StoryDiffusion 新，但整合成本未知 | [code](https://github.com/byliutao/1Prompt1Story) |
| Story2Board | 2025 | Latent Panel Anchoring 加 RAVM，強調構圖多樣性，提出 Scene Diversity 指標 | 「一致性 vs 多樣性」取捨的論據 | [page](https://daviddinkevich.github.io/Story2Board/) |
| ConsiStory | SIGGRAPH 2024 | 免訓練、共享 subject attention | 相關研究引用 | arXiv 2402.03286 |
| DreamSim | NeurIPS 2023 | 貼近人類感知的影像相似度（對構圖、姿勢、顏色敏感） | 輔助一致性指標，以及構圖多樣性指標 | — |

## 三、對企畫書做的修改（本次）

| 位置 | 修改 | 依據 |
| --- | --- | --- |
| 相關研究 | 新增免訓練一致注意力、VLM 稽核 Agent、故事視覺化 Agent、評測基準四列；加入 copy-paste 問題說明 | ViStoryBench、Story2Board、Audit & Repair、CANVAS |
| 目標與範圍 Could | 加方案 E、商用參照組 | StoryDiffusion、gpt-image-1-mini 價格 |
| 方法 §3 | IP-Adapter 選型理由（不用人臉類）、姿勢外洩對策、彈性方案 E 與商用參照組 | 人臉類方法對卡通失效；content leakage 相關研究 |
| 方法 §4 | GPT 檢查附參考圖與屬性清單，逐項勾選；彈性局部 inpainting | Audit & Repair |
| 評估 | 新增「構圖多樣性」指標、H4、DreamSim；說明不直接套 ViStoryBench 角色指標的理由；人工抽查數字統一為 32 張 | ViStoryBench、DreamSim |
| 成本 | 加商用參照組約 0.35 美金，總估 1–1.5 美金 | OpenAI 圖片價格（第三方整理，需再確認） |
| 參考文獻 | 第 13–19 篇 | — |

## 四、考慮過但沒採用

| 選項 | 不採用的原因 |
| --- | --- |
| 改用 FLUX（FLUX.1-Kontext 等）當底模 | 12B 參數，T4 16GB 需量化且很慢；課程內容以 SD／SDXL 為主。可在報告「未來工作」提 |
| InstantID／PuLID／IP-Adapter-FaceID | 依賴人臉偵測，動物與卡通角色會失效；FaceID-Plus 有風格退化問題 |
| 直接跑完整 ViStoryBench | 故事多為多角色、寫實或電影風格，與「單一主角繪本」不符；權重超過 10GB，角色指標用人臉特徵。可挑 lite 版中單一主角的故事當外部驗證（未列入計畫） |
| CANVAS 式三種記憶 | 範圍太大，屬於多場景長故事；6–8 格繪本用不到道具狀態追蹤 |
| ControlNet 姿勢控制 | 需要姿勢來源（另一個生成步驟），增加複雜度；若 copy-paste 嚴重可再考慮 |

## 五、實務數據與經驗（可信度標註）

- **SDXL LoRA 在 Colab T4**：官方 `train_dreambooth_lora_sdxl.py` 加 `--gradient_checkpointing --use_8bit_adam --mixed_precision=fp16`（以及 xformers）可在 16GB 內訓練，但多篇教學都說「勉強跑得動」。〔多個教學來源，可信度中〕
- **IP-Adapter scale**：社群經驗以 0.6–0.7 為一致性與可控性的平衡點；有一篇部落格做了 600 格評估，IP-Adapter（0.65）的同角色通過率 84%，LoRA 為 78%，但底模是 FLUX Kontext、評估是人工判斷。〔單篇部落格，可信度低，只當起點〕
- **`gpt-image-1-mini` 價格**：1024×1024 為 low 0.005／medium 0.011／high 0.036 美金。〔第三方整理，使用前到 OpenAI 官方價目表確認〕
- **CLIP-I 的弱點**：對外觀相近的不同角色區分力差；DINO 較著重主體本身，更適合衡量身分一致性。〔多篇論文的共同說法〕

## 六、待確認或可追的線索

- [ ] StoryDiffusion 在 T4 跑 SDXL 的實際記憶體用量（官方 repo 沒寫，需自己試）
- [ ] 1Prompt1Story 是否有 diffusers 相容實作，可否取代 E
- [ ] `gpt-image-1-mini` 的 edits API 是否支援參考圖保角色（需實測）
- [ ] ViStoryBench lite 中是否有單一主角、繪本風格的故事可當外部驗證
- [ ] Audit & Repair 的屬性比對 prompt 細節（可讀全文）

## 七、自家實驗觀察（2026-10-13～14，目測，待評估指標量化）

> 設定：SDXL base 1.0、DPM++ 2M Karras、30 步、CFG 6.5、1024²；5 份凍結分鏡共 30 格；seed = 故事 × 100 + 格數。

**方案 A（prompt only）**

- 單色系動物角色（小熊、小狐狸）光靠 prompt 就相當一致；多色外觀的人類角色（故事 1、3）有**屬性綁定錯置**：顏色在帽子、泳裝、雨衣、雨靴之間互換（CLIP attribute binding）。
- 人類角色的**臉每格不同**（年齡、臉型、膚色）：外觀 tags 只描述髮型與服裝，沒有定義臉，seed 不同就是不同的人。
- 同一故事內**畫風會漂移**：每格的光線與場景詞（sunset、golden light）把畫風拉往不同方向。
- 劇情關鍵物件常漏畫（沙堡、青蛙、倒麵糊），且有劇情邏輯錯誤（「帽子遺失」卻戴著帽子）；shot_type 幾乎被忽略。
- 畫風比較（2 格 × 7 種）：水彩系較常漏畫具體物件，色塊明確的畫風（gouache、flat vector、crayon）較會畫出沙堡。樣本極少，只作傾向。最終選 gouache。

**方案 B（IP-Adapter plus ViT-H）**

- scale 0.3 時 5 篇的臉、服裝、畫風都明顯一致，是 A → B 最大的改善；畫風統一是**混淆變因**，需另量畫風一致性。
- scale 取捨（印證 H4）：0.2 保留較多劇情但服裝顏色仍會漂；0.3 平衡；0.4 起構圖趨同；0.5 起每格照抄參考圖站姿與背景；0.7 六格近乎同一張。
- **參考圖會被整體照抄**，不只角色：
  - 構圖：半身參考圖 → 每格變半身特寫（故事 1）；側身參考圖 → scale 高時每格側身走路（故事 3）。
  - 背景：有場景的參考圖 → 每格被拉回該場景；**連素色的粉紅底色都會擴散**（故事 5 戶外月夜變粉紅背景）。
  - 顏色：參考圖顏色壓過 prompt，顏色錯的參考圖造成「一致地錯」（故事 2 參考圖是紅吊帶褲，6 格都紅吊帶褲，與描述的藍吊帶褲矛盾）。
  - 姿勢：站姿參考圖讓「躺在床上」變成坐著（故事 5）。
- 參考圖候選本身也有屬性綁定問題：多色外觀的人類 8 張候選中常無一完全正確。
- **參考圖挑選規則（固定優先序，報告需寫明以利重現）**：① 屬性正確 → ② 白或淺灰素色背景 → ③ 正面全身、中性站姿。依此規則建議重選：故事 1 → #5、故事 2 → #5、故事 3 → #6（首輪實際用 6／7／5／8／8）。

**對評估設計的影響**

- 一致性指標必須搭配**構圖多樣性**與**參考圖 copy-paste** 指標，否則 scale 越高分數越好，會誤判。
- 圖文對齊需涵蓋「劇情物件是否出現」，CLIP score 對漏畫小物件不敏感，需 GPT 看圖補足。
- 建議新增**畫風一致性**指標，與角色一致性分開。
