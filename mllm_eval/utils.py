import base64
import io
import json
import os
import re
import string
from typing import Any, Dict

from PIL import Image


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pil_to_data_url(image: Image.Image) -> str:
    payload = base64.b64encode(pil_to_png_bytes(image)).decode("utf-8")
    return "data:image/png;base64,{0}".format(payload)


def pil_to_base64(image: Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(image)).decode("utf-8")


def normalize_text(value: Any, lowercase: bool = True, strip_punctuation: bool = True) -> str:
    text = str(value).strip()
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
