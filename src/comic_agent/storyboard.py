"""分鏡 Agent：使用者需求 → Storyboard → 每格的 SD prompt。"""
import re

from .llm import structured_call
from .schema import Panel, Storyboard

SYSTEM_PROMPT = """你是專業的繪本分鏡師。根據使用者的需求，產生一份完整分鏡。

規則：
1. 故事要有起承轉合，最後一格要有收尾。
2. 角色外觀只在 characters[].appearance 寫一次，用英文 tags，要具體且有辨識度
   （髮型、髮色、服裝、配色、配件），之後每格都靠它維持一致。
   appearance 會原封不動貼到每一格，所以只能寫「整個故事都不會變」的身分特徵；
   表情、手上的道具（如雨傘）、天氣造成的狀態（如淋濕）會隨劇情改變，不要寫進 appearance。
   appearance 不超過 12 個英文詞。
3. 每格的 image_prompt 用英文，只描述場景、動作、表情、手持道具、構圖與光線；
   長度上限：單一角色的格不超過 25 個英文詞，2 個角色同框的格不超過 15 個英文詞
   （外觀描述會依同框人數加長，總長不能超過 CLIP 的 77 tokens）；
   提到角色時直接用角色 id（例如 "mei is jumping over a puddle"），
   不要重寫任何角色的外觀（連局部也不行，例如不要寫 "gray kitten with blue eyes"，寫 "kitten" 即可）；
   不要寫鏡頭類型（shot_type 會由程式另外加上），也不要寫畫風詞（如 watercolor，art_style 會另外加上）。
4. 每格最多 2 個角色同框；鏡頭類型 shot_type 要有變化，避免連續相同。
   出現在 2 格以上、或有台詞的人物（例如爸爸、朋友）都必須定義在 characters 並列入該格 characters；
   只出現一次的路人可以不定義，但 image_prompt 只能用簡短類別稱呼（如 "a woman"）。
5. scene、角色 name、台詞與旁白都使用 {lang}，台詞文字中提到角色時用 name 稱呼；每格 0–2 句，簡短口語。
   dialogue.speaker 一律填角色 id（英文代號）或 "narrator"，不要填 name。
6. art_style 是整本共用的英文畫風 tags。"""

USER_TEMPLATE = """主題：{topic}
目標讀者：{audience}
格數：剛好 {n_panels} 格
畫風偏好：{style}
角色設定（可為空，空的話由你設計）：{characters}"""


def generate_storyboard(
    topic: str,
    n_panels: int = 6,
    audience: str = "3–6 歲兒童",
    style: str = "溫暖的水彩繪本風",
    characters: str = "",
    lang: str = "繁體中文",
    model: str | None = None,
    use_cache: bool = True,
) -> Storyboard:
    sb = structured_call(
        system=SYSTEM_PROMPT.format(lang=lang),
        user=USER_TEMPLATE.format(
            topic=topic, audience=audience, n_panels=n_panels,
            style=style, characters=characters or "（無）",
        ),
        schema=Storyboard,
        model=model,
        use_cache=use_cache,
        validate=lambda s: validate_storyboard(s, n_panels),
    )
    return sb


MAX_CLIP_TOKENS = 75  # CLIP 上限 77，扣掉 BOS / EOS
MAX_CHARS_PER_PANEL = 2  # 範圍以單一主角為底線，多角色同框是彈性目標


def estimate_clip_tokens(text: str) -> int:
    """粗估 CLIP token 數：一個英文詞或一個標點算 1。

    本機沒有 CLIP tokenizer，實際 BPE 會切得更細（偏低估），Colab 上需實測校正。
    """
    return len(re.findall(r"[A-Za-z']+|[^\sA-Za-z']", text))


def validate_storyboard(sb: Storyboard, n_panels: int | None = None) -> None:
    """檢查 LLM 輸出的邏輯錯誤（schema 擋不住的部分）。"""
    ids = {c.id for c in sb.characters}
    errors = []
    if n_panels is not None and len(sb.panels) != n_panels:
        errors.append(f"格數 {len(sb.panels)} ≠ 要求的 {n_panels}")
    for p in sb.panels:
        unknown = set(p.characters) - ids
        if unknown:
            errors.append(f"第 {p.index} 格引用未定義角色 {unknown}")
        if len(p.characters) > MAX_CHARS_PER_PANEL:
            errors.append(f"第 {p.index} 格有 {len(p.characters)} 個角色同框，上限 {MAX_CHARS_PER_PANEL}")
        for d in p.dialogue:
            if d.speaker != "narrator" and d.speaker not in ids:
                errors.append(f"第 {p.index} 格說話者 '{d.speaker}' 未定義")
        n = estimate_clip_tokens(build_sd_prompt(sb, p))
        if n > MAX_CLIP_TOKENS:
            errors.append(f"第 {p.index} 格 SD prompt 約 {n} tokens，超過 {MAX_CLIP_TOKENS}")
    if errors:
        raise ValueError("分鏡驗證失敗：\n- " + "\n- ".join(errors))


def build_sd_prompt(sb: Storyboard, panel: Panel) -> str:
    """組合最終 SD prompt：畫風 ＋ 鏡頭 ＋ 角色外觀（固定） ＋ 本格場景。

    把 prompt 中的角色 id 換成「名字 (外觀)」，讓每格的外觀描述完全相同。
    外觀放在場景之前：CLIP 只吃 77 tokens，超過會從尾端截斷，外觀不能被截掉。
    """
    shot = panel.shot_type.replace("_", " ") + " shot"
    scene = panel.image_prompt
    looks = []
    for c in sb.characters:
        if c.id in panel.characters:
            looks.append(f"{c.id}: {c.appearance}")
    parts = [sb.art_style, shot] + looks + [scene]
    return ", ".join(p.strip().rstrip(",") for p in parts if p.strip())


NEGATIVE_PROMPT = (
    "text, watermark, signature, speech bubble, extra fingers, deformed hands, "
    "blurry, low quality, multiple views"
)
