import os
from typing import Any, Dict

from PIL import Image

from .utils import pil_to_base64, pil_to_data_url


class BaseAdapter:
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.generation_cfg = generation_cfg

    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        super().__init__(model_cfg, generation_cfg)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install the 'openai' package.") from exc

        api_key = os.environ.get(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise EnvironmentError("Missing OpenAI API key in environment.")

        self.client = OpenAI(api_key=api_key)

    def generate(self, image: Image.Image, prompt: str) -> str:
        request_kwargs = {
            "model": self.model_cfg["model"],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": pil_to_data_url(image)},
                    ],
                }
            ],
            "temperature": self.generation_cfg.get("temperature", 0.0),
            "max_output_tokens": self.generation_cfg.get("max_new_tokens", 64),
        }
        if self.model_cfg.get("reasoning_effort"):
            request_kwargs["reasoning"] = {"effort": self.model_cfg["reasoning_effort"]}

        response = self.client.responses.create(
            **request_kwargs
        )
        return response.output_text.strip()


class OpenAICompatibleAdapter(BaseAdapter):
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        super().__init__(model_cfg, generation_cfg)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Please install the 'openai' package.") from exc

        api_key_env = model_cfg.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else model_cfg.get("api_key", "EMPTY")
        base_url = model_cfg.get("base_url")
        if not base_url:
            raise ValueError("OpenAI-compatible providers require a base_url.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, image: Image.Image, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_cfg["model"],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": pil_to_data_url(image)},
                    ],
                }
            ],
            temperature=self.generation_cfg.get("temperature", 0.0),
            max_output_tokens=self.generation_cfg.get("max_new_tokens", 64),
        )
        return response.output_text.strip()


class AnthropicAdapter(BaseAdapter):
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        super().__init__(model_cfg, generation_cfg)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Please install the 'anthropic' package.") from exc

        api_key = os.environ.get(model_cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            raise EnvironmentError("Missing Anthropic API key in environment.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, image: Image.Image, prompt: str) -> str:
        request_kwargs = {
            "model": self.model_cfg["model"],
            "max_tokens": self.model_cfg.get("max_tokens", self.generation_cfg.get("max_new_tokens", 64)),
            "temperature": self.generation_cfg.get("temperature", 0.0),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": pil_to_base64(image),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if self.model_cfg.get("thinking_budget_tokens"):
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.model_cfg["thinking_budget_tokens"],
            }

        response = self.client.messages.create(**request_kwargs)
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()


class GeminiAdapter(BaseAdapter):
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        super().__init__(model_cfg, generation_cfg)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Please install the 'google-genai' package.") from exc

        api_key = os.environ.get(model_cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not api_key:
            raise EnvironmentError("Missing Gemini API key in environment.")

        self.client = genai.Client(api_key=api_key)

    def generate(self, image: Image.Image, prompt: str) -> str:
        from google.genai import types

        generate_config = {
            "temperature": self.generation_cfg.get("temperature", 0.0),
            "max_output_tokens": self.generation_cfg.get("max_new_tokens", 64),
        }
        if "thinking_budget" in self.model_cfg:
            generate_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.model_cfg["thinking_budget"]
            )

        response = self.client.models.generate_content(
            model=self.model_cfg["model"],
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image.tobytes() if self.model_cfg.get("raw_bytes") else image_to_png_bytes(image),
                    mime_type="image/png",
                ),
            ],
            config=types.GenerateContentConfig(**generate_config),
        )
        return (response.text or "").strip()


def image_to_png_bytes(image: Image.Image) -> bytes:
    from .utils import pil_to_png_bytes

    return pil_to_png_bytes(image)


class HuggingFaceLocalAdapter(BaseAdapter):
    def __init__(self, model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]):
        super().__init__(model_cfg, generation_cfg)
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError("Please install 'torch' and 'transformers'.") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_cfg["model"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

        torch_dtype = model_cfg.get("torch_dtype", "auto")
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"

        model_kwargs = {
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
        }
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype

        if model_cfg.get("device", "auto") == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = self._load_model(model_cfg["model"], model_kwargs)

        if "device_map" not in model_kwargs:
            device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

    @staticmethod
    def _load_model(model_id: str, model_kwargs: Dict[str, Any]):
        from transformers import AutoModelForImageTextToText

        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        except Exception:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)

    def generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt_text = prompt

        model_inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        )

        for key, value in model_inputs.items():
            if hasattr(value, "to"):
                model_inputs[key] = value.to(self.model.device)

        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.generation_cfg.get("max_new_tokens", 64),
            temperature=self.generation_cfg.get("temperature", 0.0),
        )

        input_length = model_inputs["input_ids"].shape[-1] if "input_ids" in model_inputs else 0
        generated_tokens = generated[:, input_length:]
        decoded = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded[0].strip()


def build_adapter(model_cfg: Dict[str, Any], generation_cfg: Dict[str, Any]) -> BaseAdapter:
    provider = model_cfg["provider"]
    if provider == "openai":
        return OpenAIAdapter(model_cfg, generation_cfg)
    if provider == "openai_compatible" or provider == "vllm_server":
        return OpenAICompatibleAdapter(model_cfg, generation_cfg)
    if provider == "anthropic":
        return AnthropicAdapter(model_cfg, generation_cfg)
    if provider == "gemini":
        return GeminiAdapter(model_cfg, generation_cfg)
    if provider == "huggingface_local":
        return HuggingFaceLocalAdapter(model_cfg, generation_cfg)
    raise ValueError("Unsupported provider: {0}".format(provider))
