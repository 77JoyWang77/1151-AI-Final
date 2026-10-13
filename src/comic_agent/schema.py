"""分鏡資料結構 Storyboard schema。

設計原則：角色外觀只在 Character.appearance 定義一次（single source of truth），
每格的 image_prompt 只描述「場景與動作」，最終給 SD 的 prompt 由程式組合，
避免 LLM 每格重寫外觀造成不一致。
"""
from typing import Literal

from pydantic import BaseModel, Field

ShotType = Literal["wide", "medium", "close_up", "extreme_close_up", "over_the_shoulder"]


class Character(BaseModel):
    id: str = Field(description="簡短英文代號，例如 'mei'，供分鏡引用")
    name: str = Field(description="角色名稱（故事語言）")
    appearance: str = Field(
        description="給圖像模型的固定外觀描述，英文、逗號分隔的 tags，"
        "例如 'young girl, short black bob hair, yellow raincoat, red boots'"
    )
    personality: str = Field(description="個性簡述（故事語言）")


class Dialogue(BaseModel):
    speaker: str = Field(description="說話角色的 id；旁白用 'narrator'")
    text: str = Field(description="台詞（故事語言），一句不超過 25 字")


class Panel(BaseModel):
    index: int = Field(description="格數，從 1 開始")
    scene: str = Field(description="這一格發生什麼事（故事語言），給人看的說明")
    characters: list[str] = Field(description="本格出現的角色 id；可為空")
    shot_type: ShotType
    image_prompt: str = Field(
        description="給圖像模型的英文 prompt：只寫場景、動作、表情、構圖、光線，"
        "不要重寫角色外觀，以角色 id 指稱角色"
    )
    dialogue: list[Dialogue]


class Storyboard(BaseModel):
    title: str
    logline: str = Field(description="一句話故事摘要")
    art_style: str = Field(description="整本共用的英文畫風 tags，例如 'children's picture book, watercolor'")
    characters: list[Character]
    panels: list[Panel]
