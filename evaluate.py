import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from mllm_eval.adapters import build_adapter
from mllm_eval.dataset import load_hf_samples
from mllm_eval.metrics import canonicalize_label, compute_accuracy
from mllm_eval.utils import dump_json, ensure_dir, normalize_text


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def build_prompt(sample: Dict[str, Any], dataset_cfg: Dict[str, Any], label_names: List[str]) -> str:
    prompt_template = dataset_cfg["prompt_template"]
    label_space = "\n".join("- {0}".format(label) for label in label_names) if label_names else ""
    question = sample.get("question") or ""
    return prompt_template.format(label_space=label_space, question=question)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_one_model(
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    matching_cfg: Dict[str, Any],
    samples: List[Dict[str, Any]],
    label_names: List[str],
    dataset_tag: str,
) -> Dict[str, Any]:
    generation_cfg = dict(eval_cfg.get("generation", {}))
    adapter = build_adapter(model_cfg, generation_cfg)

    output_dir = os.path.join(eval_cfg["output_root"], dataset_tag, model_cfg["name"])
    ensure_dir(output_dir)

    predictions = []
    save_every = eval_cfg.get("save_every", 20)

    for index, sample in enumerate(tqdm(samples, desc=model_cfg["name"])):
        prompt = build_prompt(sample, dataset_cfg, label_names)
        raw_prediction = adapter.generate(sample["image"], prompt)
        canonical_prediction = canonicalize_label(
            raw_prediction,
            label_names=label_names,
            aliases=matching_cfg.get("aliases"),
            lowercase=matching_cfg.get("lowercase", True),
            strip_punctuation=matching_cfg.get("strip_punctuation", True),
        )
        gold_label = sample["label"]
        is_correct = normalize_text(
            canonical_prediction,
            lowercase=matching_cfg.get("lowercase", True),
            strip_punctuation=matching_cfg.get("strip_punctuation", True),
        ) == normalize_text(
            gold_label,
            lowercase=matching_cfg.get("lowercase", True),
            strip_punctuation=matching_cfg.get("strip_punctuation", True),
        )

        record = {
            "id": sample["id"],
            "question": sample.get("question"),
            "gold_label": gold_label,
            "raw_prediction": raw_prediction,
            "canonical_prediction": canonical_prediction,
            "is_correct": is_correct,
        }
        predictions.append(record)

        if save_every and (index + 1) % save_every == 0:
            write_jsonl(os.path.join(output_dir, "predictions.jsonl"), predictions)

    metrics = compute_accuracy(predictions)
    metrics["model_name"] = model_cfg["name"]
    metrics["model_id"] = model_cfg["model"]
    metrics["provider"] = model_cfg["provider"]

    write_jsonl(os.path.join(output_dir, "predictions.jsonl"), predictions)
    dump_json(os.path.join(output_dir, "metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-source and closed-source MLLMs.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--models", required=True, help="Path to model YAML config.")
    parser.add_argument("--only", nargs="*", default=None, help="Only run the specified model names.")
    parser.add_argument(
        "--only-source",
        choices=["open_source", "closed_source"],
        default=None,
        help="Only run models from one source category.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Override the configured max samples.")
    args = parser.parse_args()

    dataset_config = load_yaml(args.config)
    models_config = load_yaml(args.models)

    dataset_cfg = dataset_config["dataset"]
    eval_cfg = dataset_config.get("eval", {})
    matching_cfg = dataset_config.get("matching", {})
    eval_cfg.setdefault("output_root", "outputs")

    override_max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples")
    samples, label_names, dataset_info = load_hf_samples(
        dataset_cfg,
        override_max_samples=override_max_samples,
    )
    dataset_tag = dataset_cfg["path"].replace("/", "__")

    model_items = models_config["models"]
    if args.only:
        only_set = set(args.only)
        model_items = [model for model in model_items if model["name"] in only_set]
    if args.only_source:
        model_items = [model for model in model_items if model.get("source_type") == args.only_source]

    if not model_items:
        raise ValueError("No models selected for evaluation.")

    summary = {
        "dataset": dataset_cfg["path"],
        "dataset_info": dataset_info,
        "label_names": label_names,
        "results": [],
    }

    for model_cfg in model_items:
        metrics = evaluate_one_model(
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            eval_cfg=eval_cfg,
            matching_cfg=matching_cfg,
            samples=samples,
            label_names=label_names,
            dataset_tag=dataset_tag,
        )
        summary["results"].append(metrics)

    summary_dir = os.path.join(eval_cfg["output_root"], dataset_tag)
    ensure_dir(summary_dir)
    dump_json(os.path.join(summary_dir, "summary.json"), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
