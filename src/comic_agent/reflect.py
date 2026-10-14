"""Reflection：GPT 看圖檢查每一格，程式依規則決定 通過／改寫 prompt／換 seed。

分工：GPT 只回報「事實」（各屬性看不看得到、劇情元素有沒有畫出來、有沒有異常），
動作由 decide() 依固定規則決定，比讓 GPT 直接下結論更好控制、也方便調整門檻。
改寫的 image_prompt 仍遵守分鏡設計原則：只寫場景與動作、以角色 id 指稱、不重寫外觀。
"""
from typing import Literal

from pydantic import BaseModel, Field

from .llm import structured_call

# 評審用 gpt-4.1：實測 gpt-4.1-mini 會漏看明顯物件（大螃蟹）、誤用 problems 類型；4.1 每格約 0.009 美金
REVIEW_MODEL = "gpt-4.1"
ATTR_MISMATCH_LIMIT = 0.3  # 看得到的外觀屬性中，不符的比例超過這個值 → 換 seed


class AttributeCheck(BaseModel):
    attribute: str = Field(description="外觀屬性原文（英文 tag）")
    status: Literal["match", "mismatch", "not_visible"] = Field(
        description="match＝畫面中看得到且相符；mismatch＝看得到但不符（例如顏色錯）；not_visible＝被遮住、背對或出鏡")


class ElementCheck(BaseModel):
    element: str = Field(description="這一格劇情需要的關鍵元素，用繁體中文簡述")
    kind: Literal["object", "action", "character", "expression"] = Field(
        description="object＝物件、action＝動作、character＝角色、expression＝表情或情緒")
    status: Literal["present", "absent", "cannot_tell"] = Field(
        description="present＝畫出來了（動作大致符合即可）；absent＝明確沒畫；cannot_tell＝被遮住或角度看不到（例如背對鏡頭看不到表情）")


class Problem(BaseModel):
    kind: Literal["duplicate_character", "extra_character", "deformity", "text_in_image", "plot_contradiction"] = Field(
        description="只限這五類真正的缺陷：主角重複出現、多出不該有的角色、肢體或臉部變形、畫面出現文字、與劇情矛盾")
    detail: str = Field(description="繁體中文簡述")


class PanelReview(BaseModel):
    # 放在第一個欄位：結構化輸出依欄位順序生成，先描述畫面再檢查，可減少視覺誤判
    description: str = Field(default="", description="先用 2–3 句繁體中文客觀描述畫面中看得到的角色、動作、物件與角色朝向（正面／側面／背面）")
    attributes: list[AttributeCheck] = Field(description="主角外觀屬性清單，逐項檢查，順序與輸入相同")
    elements: list[ElementCheck] = Field(description="從分鏡找出 2–4 個關鍵元素，逐項檢查是否畫出來")
    problems: list[Problem] = Field(description="外觀屬性以外的真正缺陷；沒有就空清單。看不到、無法確認、大小或細節略有差異都不是缺陷")
    revised_image_prompt: str = Field(
        description="若有關鍵元素沒畫出來：改寫後的英文 image_prompt，只寫場景與動作、以角色 id 指稱、"
                    "不寫外觀與畫風、不超過 25 個英文詞，缺漏的元素放在最前面；沒有缺漏則留空字串")


SYSTEM_PROMPT = """你是繪本插畫的品質檢查員。根據分鏡檢查一張生成的插畫，只回報看到的事實，不要放水也不要過度挑剔。

規則：
1. 外觀屬性：逐項判斷。看不到的（背對鏡頭、被遮住、不在畫面內）一律填 not_visible，不算錯。
2. 關鍵元素：從「劇情」與「image_prompt」找出 2–4 個這一格最重要、讀者一定要看到的**物件、動作或角色**
   （例如沙堡、青蛙、把麵糊倒進模具的動作），逐項判斷有沒有畫出來。
   不要把表情、情緒、光線、氣氛當成關鍵元素。判斷前先仔細看整張圖，包括前景與角落的小物件。
   只判斷「有沒有畫出來」：大小、數量、顏色、姿勢細節與描述略有差異（例如描述是小螃蟹、畫成大螃蟹）仍算有畫出來；
   動作大致符合即可（例如伸手靠近螃蟹可算「指著螃蟹」）。看不到的填 cannot_tell，不要填 absent。
3. problems 只寫五類真正的缺陷：主角重複出現、多出不該有的角色、肢體或臉部變形、畫面出現文字、
   與劇情矛盾（例如劇情說帽子不見了，角色卻戴著帽子）。外觀屬性的結果已經在 attributes，不要重複寫；
   「看不到」「無法確認」「大小或細節略有差異」都不是缺陷，不要寫進 problems。
4. 若有關鍵元素沒畫出來，提供 revised_image_prompt；否則留空字串。
5. element 與 problems 一律用繁體中文。"""

USER_TEMPLATE = """劇情（這一格）：{scene}
image_prompt：{image_prompt}
鏡頭：{shot}
主角 id：{hero_id}
主角外觀屬性清單：{attributes}
本格出現的角色：{characters}"""


def attribute_checklist(appearance: str) -> list[str]:
    """外觀 tags → 屬性清單（第一個 tag 是角色類別，例如 'young girl'，也保留）。"""
    return [t.strip() for t in appearance.split(",") if t.strip()]


def review_panel(image_path, panel: dict, board: dict, model: str | None = None,
                 use_cache: bool = True) -> PanelReview:
    hero = board["characters"][0]
    user = USER_TEMPLATE.format(
        scene=panel["scene"], image_prompt=panel["image_prompt"], shot=panel["shot_type"],
        hero_id=hero["id"], attributes=attribute_checklist(hero["appearance"]),
        characters=", ".join(panel["characters"]) or "（無）",
    )
    return structured_call(SYSTEM_PROMPT, user, PanelReview, model=model or REVIEW_MODEL, use_cache=use_cache,
                           images=[image_path], temperature=0)  # 檢查要穩定：同一張圖每次判斷相同


def decide(review: PanelReview) -> tuple[str, list[str]]:
    """回傳 (動作, 理由)。動作：pass / revise_prompt / reseed。

    優先序：關鍵元素缺漏 → revise_prompt（換 seed 通常補不回漏掉的物件）；
    外觀不符比例過高或有其他問題 → reseed；其餘 pass。
    """
    # 只有「明確沒畫」且不是表情的元素才算缺漏：表情常因角度看不到，換 seed 或改 prompt 也難保證
    missing = [e.element for e in review.elements if e.status == "absent" and e.kind != "expression"]
    if missing and review.revised_image_prompt.strip():
        return "revise_prompt", [f"缺：{m}" for m in missing]
    visible = [a for a in review.attributes if a.status != "not_visible"]
    wrong = [a.attribute for a in visible if a.status == "mismatch"]
    reasons = [f"外觀不符：{w}" for w in wrong] + [f"{p.kind}：{p.detail}" for p in review.problems]
    if missing:  # 有缺漏但模型沒給改寫：只能換 seed
        reasons = [f"缺：{m}" for m in missing] + reasons
        return "reseed", reasons
    if (visible and len(wrong) / len(visible) > ATTR_MISMATCH_LIMIT) or review.problems:
        return "reseed", reasons
    return "pass", reasons


# ==== Reflection loop：依 decide() 的結果重生，每格最多 max_tries 次 ====

MAX_TRIES = 3
RESEED_STEP = 1000  # 換 seed：seed + 1000 × 嘗試次數，不與分鏡 seed（故事 × 100 + 格數）重疊


def review_score(review: PanelReview) -> tuple[int, int, int]:
    """越小越好：(缺漏元素數, 外觀不符數, 問題數)；用來在多次嘗試中挑最好的一張。"""
    missing = sum(e.status == "absent" and e.kind != "expression" for e in review.elements)
    wrong = sum(a.status == "mismatch" for a in review.attributes)
    return missing, wrong, len(review.problems)


def revised_prompt(board: dict, panel_index: int, image_prompt: str) -> str:
    """把改寫後的 image_prompt 放回分鏡，再用 build_sd_prompt 組合（外觀仍只定義一次）。"""
    from .schema import Storyboard
    from .storyboard import build_sd_prompt

    sb = Storyboard.model_validate(board)
    panel = sb.panels[panel_index - 1].model_copy(update={"image_prompt": image_prompt})
    return build_sd_prompt(sb, panel)


def reflection_loop(job: dict, board: dict, original_png, generate, review, max_tries: int = MAX_TRIES) -> dict:
    """單格的 reflection loop（不含模型，生圖與檢查由呼叫端傳入，方便離線測試）。

    generate(job, attempt) -> 圖檔路徑；review(png) -> PanelReview（一律對照原本的分鏡檢查）。
    revise_prompt：用評審改寫的 image_prompt、seed 不變（只改 prompt 一個變因）；
    reseed：prompt 不變、seed 加 RESEED_STEP × 嘗試次數。
    回傳 {"attempts": [...], "chosen": 選中的嘗試編號}；第 0 次是原圖。
    """
    rv = review(original_png)
    action, reasons = decide(rv)
    attempts = [{"attempt": 0, "path": str(original_png), "action": action, "reasons": reasons,
                 "score": review_score(rv), "prompt": job["prompt"], "seed": job["seed"]}]
    current = dict(job)
    while action != "pass" and len(attempts) <= max_tries:
        k = len(attempts)
        if action == "revise_prompt":
            current = {**current, "prompt": revised_prompt(board, job["panel_index"], rv.revised_image_prompt)}
        else:
            current = {**current, "seed": job["seed"] + RESEED_STEP * k}
        png = generate(current, k)
        rv = review(png)
        action, reasons = decide(rv)
        attempts.append({"attempt": k, "path": str(png), "action": action, "reasons": reasons,
                         "score": review_score(rv), "prompt": current["prompt"], "seed": current["seed"]})
    chosen = min(attempts, key=lambda a: (a["action"] != "pass", a["score"], a["attempt"]))
    return {"attempts": attempts, "chosen": chosen["attempt"]}
