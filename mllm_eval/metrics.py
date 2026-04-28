from typing import Any, Dict, Iterable, List, Optional

from .utils import normalize_text


def canonicalize_label(
    text: str,
    label_names: List[str],
    aliases: Optional[Dict[str, str]] = None,
    lowercase: bool = True,
    strip_punctuation: bool = True,
) -> str:
    normalized = normalize_text(text, lowercase=lowercase, strip_punctuation=strip_punctuation)
    if aliases and normalized in aliases:
        normalized = normalize_text(
            aliases[normalized],
            lowercase=lowercase,
            strip_punctuation=strip_punctuation,
        )

    label_map = {
        normalize_text(label, lowercase=lowercase, strip_punctuation=strip_punctuation): label
        for label in label_names
    }

    if normalized in label_map:
        return label_map[normalized]

    for norm_label, original_label in label_map.items():
        if normalized and normalized in norm_label:
            return original_label
        if norm_label and norm_label in normalized:
            return original_label

    return text.strip()


def compute_accuracy(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    accuracy = (correct / total) if total else 0.0
    return {
        "num_samples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 6),
    }
