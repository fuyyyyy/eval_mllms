from typing import Any, Dict, List, Optional

from datasets import ClassLabel, load_dataset
from PIL import Image


def _guess_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _resolve_image_value(row: Dict[str, Any], image_column: str, image_index: int) -> Image.Image:
    image_value = row[image_column]

    if isinstance(image_value, list):
        if not image_value:
            raise ValueError("Image list column '{0}' is empty.".format(image_column))
        if image_index >= len(image_value):
            raise IndexError(
                "Configured image_index={0} but column '{1}' only has {2} images.".format(
                    image_index,
                    image_column,
                    len(image_value),
                )
            )
        image_value = image_value[image_index]

    if not isinstance(image_value, Image.Image):
        raise TypeError("Expected a PIL image in column '{0}'.".format(image_column))

    return image_value.convert("RGB")


def resolve_label_names(dataset_split, label_column: str, configured_names: Optional[List[str]]) -> List[str]:
    if configured_names:
        return list(configured_names)

    feature = dataset_split.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return list(feature.names)

    names = getattr(feature, "names", None)
    if names:
        return list(names)

    return []


def load_hf_samples(dataset_cfg: Dict[str, Any], override_max_samples: Optional[int] = None):
    dataset = load_dataset(
        path=dataset_cfg["path"],
        name=dataset_cfg.get("name"),
        split=dataset_cfg.get("split", "test"),
    )

    columns = list(dataset.column_names)
    image_column = dataset_cfg.get("image_column") or _guess_column(columns, ["image", "images", "img"])
    label_column = dataset_cfg.get("label_column") or _guess_column(columns, ["label", "answer", "target"])
    question_column = dataset_cfg.get("question_column") or _guess_column(
        columns,
        ["question", "problem", "prompt", "instruction", "query"],
    )
    image_index = dataset_cfg.get("image_index", 0)

    if image_column is None:
        raise ValueError("Could not determine image column. Please set dataset.image_column.")
    if label_column is None:
        raise ValueError("Could not determine label column. Please set dataset.label_column.")

    label_names = resolve_label_names(dataset, label_column, dataset_cfg.get("label_names"))
    max_samples = override_max_samples or dataset_cfg.get("max_samples")

    samples = []
    for index, row in enumerate(dataset):
        if max_samples is not None and index >= max_samples:
            break

        label_value = row[label_column]
        if label_names and isinstance(label_value, int) and 0 <= label_value < len(label_names):
            label_text = label_names[label_value]
        else:
            label_text = str(label_value)

        sample = {
            "id": row.get("id", index),
            "image": _resolve_image_value(row, image_column, image_index),
            "label": label_text,
            "question": row.get(question_column) if question_column else None,
            "raw_row": row,
        }
        samples.append(sample)

    return samples, label_names, {
        "image_column": image_column,
        "image_index": image_index,
        "label_column": label_column,
        "question_column": question_column,
        "num_loaded_samples": len(samples),
    }
