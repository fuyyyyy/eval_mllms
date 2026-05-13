import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from mllm_eval.adapters import build_adapter
from mllm_eval.dataset import load_hf_samples
from mllm_eval.metrics import canonicalize_label, compute_accuracy
from mllm_eval.reasoning import (
    build_mode_prompt,
    load_reasoning_modes,
    parse_response,
    resolve_generation_cfg,
    resolve_request_kwargs,
)
from mllm_eval.utils import dump_csv, dump_json, ensure_dir, normalize_text


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
    mode_cfg: Dict[str, Any],
    samples: List[Dict[str, Any]],
    label_names: List[str],
    dataset_tag: str,
) -> Dict[str, Any]:
    mode_name = mode_cfg["name"]
    generation_cfg = resolve_generation_cfg(eval_cfg.get("generation", {}), mode_cfg)
    adapter = build_adapter(model_cfg, generation_cfg)
    request_kwargs = resolve_request_kwargs(model_cfg, mode_cfg)

    output_dir = os.path.join(eval_cfg["output_root"], dataset_tag, model_cfg["name"], mode_name)
    ensure_dir(output_dir)

    predictions = []
    save_every = eval_cfg.get("save_every", 20)

    for index, sample in enumerate(tqdm(samples, desc=model_cfg["name"] + ":" + mode_name)):
        base_prompt = build_prompt(sample, dataset_cfg, label_names)
        prompt = build_mode_prompt(base_prompt, mode_cfg)
        raw_prediction = adapter.generate(sample["image"], prompt, dict(request_kwargs))
        parsed_response = parse_response(raw_prediction, mode_cfg)
        parsed_answer = parsed_response["parsed_answer"] or raw_prediction

        canonical_prediction = canonicalize_label(
            parsed_answer,
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
            "reasoning_mode": mode_name,
            "prompt": prompt,
            "raw_prediction": raw_prediction,
            "parsed_answer": parsed_answer,
            "parsed_reasoning": parsed_response.get("parsed_reasoning"),
            "canonical_prediction": canonical_prediction,
            "is_correct": is_correct,
        }
        predictions.append(record)

        if save_every and (index + 1) % save_every == 0:
            write_jsonl(os.path.join(output_dir, "predictions.jsonl"), predictions)

    metrics = compute_accuracy(predictions)
    metrics["model_name"] = model_cfg["name"]
    metrics["model_family"] = model_cfg.get("model_family")
    metrics["model_id"] = model_cfg["model"]
    metrics["provider"] = model_cfg["provider"]
    metrics["source_type"] = model_cfg.get("source_type")
    metrics["reasoning_mode"] = mode_name

    write_jsonl(os.path.join(output_dir, "predictions.jsonl"), predictions)
    dump_json(os.path.join(output_dir, "metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-source and closed-source MLLMs.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--models", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--reasoning-config",
        default=None,
        help="Path to reasoning-mode YAML config. If omitted, runs a single default mode.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Only run the specified model names.")
    parser.add_argument("--only-mode", nargs="*", default=None, help="Only run the specified reasoning modes.")
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
    reasoning_config = load_yaml(args.reasoning_config) if args.reasoning_config else None

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

    mode_items = load_reasoning_modes(reasoning_config)
    if args.only_mode:
        only_mode_set = set(args.only_mode)
        mode_items = [mode for mode in mode_items if mode["name"] in only_mode_set]
    if not mode_items:
        raise ValueError("No reasoning modes selected.")

    summary = {
        "dataset": dataset_cfg["path"],
        "dataset_info": dataset_info,
        "label_names": label_names,
        "reasoning_modes": [mode["name"] for mode in mode_items],
        "results": [],
    }

    for model_cfg in model_items:
        disabled_modes = set(model_cfg.get("disabled_modes", []))
        for mode_cfg in mode_items:
            if mode_cfg["name"] in disabled_modes:
                continue
            metrics = evaluate_one_model(
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                eval_cfg=eval_cfg,
                matching_cfg=matching_cfg,
                mode_cfg=mode_cfg,
                samples=samples,
                label_names=label_names,
                dataset_tag=dataset_tag,
            )
            summary["results"].append(metrics)

    summary_dir = os.path.join(eval_cfg["output_root"], dataset_tag)
    ensure_dir(summary_dir)
    dump_json(os.path.join(summary_dir, "summary.json"), summary)
    dump_csv(
        os.path.join(summary_dir, "summary.csv"),
        summary["results"],
        fieldnames=[
            "model_name",
            "model_family",
            "model_id",
            "provider",
            "source_type",
            "reasoning_mode",
            "num_samples",
            "num_correct",
            "accuracy",
        ],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
