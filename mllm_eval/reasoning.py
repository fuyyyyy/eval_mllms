import re
from typing import Any, Dict, List, Optional

from .utils import deep_merge


def load_reasoning_modes(reasoning_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not reasoning_config:
        return [{"name": "default"}]
    modes = reasoning_config.get("reasoning_modes", [])
    if not modes:
        return [{"name": "default"}]
    return modes


def build_mode_prompt(base_prompt: str, mode_cfg: Dict[str, Any]) -> str:
    template = mode_cfg.get("prompt_template")
    if not template:
        return base_prompt
    return template.format(base_prompt=base_prompt)


def resolve_generation_cfg(base_generation_cfg: Dict[str, Any], mode_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(dict(base_generation_cfg), mode_cfg.get("generation", {}))


def resolve_request_kwargs(model_cfg: Dict[str, Any], mode_cfg: Dict[str, Any]) -> Dict[str, Any]:
    provider = model_cfg["provider"]
    model_defaults = model_cfg.get("request_defaults", {})
    mode_defaults = mode_cfg.get("request_overrides", {})

    resolved = {}
    if "all" in model_defaults:
        resolved = deep_merge(resolved, model_defaults["all"])
    if provider in model_defaults:
        resolved = deep_merge(resolved, model_defaults[provider])

    if "all" in mode_defaults:
        resolved = deep_merge(resolved, mode_defaults["all"])
    if provider in mode_defaults:
        resolved = deep_merge(resolved, mode_defaults[provider])
    return resolved


def _extract_first_match(patterns: List[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def parse_response(raw_text: str, mode_cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    parser_cfg = mode_cfg.get("parser", {})
    answer_regexes = parser_cfg.get("answer_regexes", [])
    reasoning_regexes = parser_cfg.get("reasoning_regexes", [])

    parsed_answer = _extract_first_match(answer_regexes, raw_text) if answer_regexes else None
    parsed_reasoning = _extract_first_match(reasoning_regexes, raw_text) if reasoning_regexes else None

    return {
        "parsed_answer": parsed_answer or raw_text.strip(),
        "parsed_reasoning": parsed_reasoning,
    }
