"""OpenAI 呼叫封裝：結構化輸出 ＋ 磁碟快取（相同輸入不重複花錢）。"""
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # 延遲 import，離線測試不需安裝

        _client = OpenAI()
    return _client


USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}  # 本次執行實際呼叫 API 的累計用量（快取命中不算）


def _cache_key(model: str, system: str, user: str, schema: type[BaseModel], image_digest: str = "") -> str:
    payload = {"m": model, "s": system, "u": user, "schema": schema.model_json_schema()}
    if image_digest:  # 只在有圖時加入，舊的純文字快取 key 不變
        payload["img"] = image_digest
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _image_parts(images, max_side: int) -> tuple[list[dict], str]:
    """把圖縮到最長邊 max_side（省 token），轉成 Responses API 的 input_image；回傳 (parts, 內容摘要)。"""
    import base64
    import io

    from PIL import Image

    parts, digest = [], hashlib.sha256()
    for p in images:
        img = Image.open(p).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        digest.update(data)
        parts.append({"type": "input_image", "detail": "high",
                      "image_url": "data:image/png;base64," + base64.b64encode(data).decode()})
    return parts, digest.hexdigest()[:16]


def structured_call(
    system: str,
    user: str,
    schema: type[T],
    model: str | None = None,
    use_cache: bool = True,
    validate: Callable[[T], None] | None = None,
    retries: int = 1,
    images: list | None = None,
    image_max_side: int = 768,
    temperature: float | None = None,
) -> T:
    """呼叫 LLM 並解析成 pydantic 物件。

    validate 不通過（拋 ValueError）時重新呼叫，最多 retries 次；
    只有通過驗證的結果才寫入快取，避免壞結果被永久快取。
    images：要一起給模型看的圖檔路徑（例如 reflection 的看圖檢查）；圖的內容也是快取 key 的一部分。
    temperature：None 表示用模型預設值；需要穩定輸出時（例如品質檢查）設 0。
    """
    model = model or DEFAULT_MODEL
    image_parts, digest = _image_parts(images, image_max_side) if images else ([], "")
    extra = {} if temperature is None else {"temperature": temperature}
    path = CACHE_DIR / f"{_cache_key(model, system + (f'|t={temperature}' if extra else ''), user, schema, digest)}.json"
    user_content = [{"type": "input_text", "text": user}] + image_parts if image_parts else user

    if use_cache and path.exists():
        cached = schema.model_validate_json(path.read_text(encoding="utf-8"))
        try:
            if validate:
                validate(cached)
            return cached
        except ValueError:
            pass  # 舊快取不合格（例如驗證規則變嚴），重新呼叫

    last_error: ValueError | None = None
    for _ in range(retries + 1):
        resp = _get_client().responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            text_format=schema,
            **extra,
        )
        usage = getattr(resp, "usage", None)
        USAGE["calls"] += 1
        USAGE["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        USAGE["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        result = resp.output_parsed
        if result is None:
            raise RuntimeError(f"模型沒有回傳可解析的結果：{resp.output_text[:200]}")
        try:
            if validate:
                validate(result)
        except ValueError as e:
            last_error = e
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    raise last_error
