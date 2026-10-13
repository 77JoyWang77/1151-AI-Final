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


def _cache_key(model: str, system: str, user: str, schema: type[BaseModel]) -> str:
    raw = json.dumps(
        {"m": model, "s": system, "u": user, "schema": schema.model_json_schema()},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def structured_call(
    system: str,
    user: str,
    schema: type[T],
    model: str | None = None,
    use_cache: bool = True,
    validate: Callable[[T], None] | None = None,
    retries: int = 1,
) -> T:
    """呼叫 LLM 並解析成 pydantic 物件。

    validate 不通過（拋 ValueError）時重新呼叫，最多 retries 次；
    只有通過驗證的結果才寫入快取，避免壞結果被永久快取。
    """
    model = model or DEFAULT_MODEL
    path = CACHE_DIR / f"{_cache_key(model, system, user, schema)}.json"

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
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
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
