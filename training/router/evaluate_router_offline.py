"""Desktop Voice Orchestrator — Router Offline & Endpoint Evaluation (M4 Slice C).

Evaluates schema adherence, exact plan match, action F1, false positive rejection,
hallucinated app IDs, and p50/p95 latency against ground truth test sets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

# Support importing local schema whether run directly or as package
try:
    from .schema import (
        build_system_prompt,
        strip_markdown_code_fence,
        validate_router_plan,
    )
except ImportError:
    from schema import (
        build_system_prompt,
        strip_markdown_code_fence,
        validate_router_plan,
    )


def action_to_tuple(action: Mapping[str, Any]) -> tuple:
    """Convert an action dictionary into an immutable comparable tuple."""
    return (
        action.get("type"),
        action.get("volume_delta_percent"),
        action.get("search_query"),
        action.get("search_type"),
        action.get("search_artist"),
        action.get("app_id"),
    )


def compute_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute all evaluation metrics across a list of evaluation records.

    Each record must have:
      - 'ground_truth': dict with 'actions'
      - 'predicted_raw': str or dict
      - 'allowed_app_ids': list[str]
      - 'latency_ms': float (optional)
    """
    total = len(records)
    if total == 0:
        return {
            "total_samples": 0,
            "schema_valid_rate": 0.0,
            "exact_plan_match_rate": 0.0,
            "action_sequence_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_action_rate": 0.0,
            "hallucinated_app_id_rate": 0.0,
            "total_negatives": 0,
            "false_positive_count": 0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        }

    schema_valid_count = 0
    exact_match_count = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_negatives = 0
    false_positive_count = 0

    total_pred_app_actions = 0
    hallucinated_app_count = 0

    latencies: list[float] = []

    for r in records:
        gt_plan = r.get("ground_truth", {})
        allowed_ids = set(r.get("allowed_app_ids", []))
        raw_pred = r.get("predicted_raw", "")
        latency = r.get("latency_ms")
        if latency is not None and isinstance(latency, (int, float)):
            latencies.append(float(latency))

        gt_actions = gt_plan.get("actions", [])
        is_negative_sample = len(gt_actions) == 0
        if is_negative_sample:
            total_negatives += 1

        pred_plan: dict[str, Any] | None = None
        is_schema_valid = False

        if isinstance(raw_pred, dict):
            pred_plan = raw_pred
            valid, _ = validate_router_plan(pred_plan, allowed_app_ids=allowed_ids)
            is_schema_valid = valid
        elif isinstance(raw_pred, str):
            clean_str = strip_markdown_code_fence(raw_pred)
            try:
                parsed = json.loads(clean_str)
                if isinstance(parsed, dict):
                    valid, _ = validate_router_plan(parsed, allowed_app_ids=allowed_ids)
                    if valid:
                        pred_plan = parsed
                        is_schema_valid = True
                    else:
                        pred_plan = parsed
            except Exception:
                pred_plan = None

        if is_schema_valid:
            schema_valid_count += 1

        pred_actions = pred_plan.get("actions", []) if (pred_plan and isinstance(pred_plan.get("actions"), list)) else []

        # False positive rejection check:
        if is_negative_sample:
            if len(pred_actions) > 0:
                false_positive_count += 1

        # Hallucinated app id check:
        for act in pred_actions:
            if isinstance(act, dict) and act.get("type") == "app.open":
                total_pred_app_actions += 1
                app_id = act.get("app_id")
                if not app_id or app_id not in allowed_ids:
                    hallucinated_app_count += 1

        # Exact match check:
        if len(pred_actions) == len(gt_actions):
            matched = True
            for pa, ga in zip(pred_actions, gt_actions):
                if action_to_tuple(pa) != action_to_tuple(ga):
                    matched = False
                    break
            if matched:
                exact_match_count += 1

        # Action sequence precision/recall/F1 (multiset comparison):
        gt_tuples = [action_to_tuple(a) for a in gt_actions]
        pred_tuples = [action_to_tuple(a) for a in pred_actions]
        gt_counter = Counter(gt_tuples)
        pred_counter = Counter(pred_tuples)

        all_keys = set(gt_counter.keys()) | set(pred_counter.keys())
        sample_tp = 0
        sample_fp = 0
        sample_fn = 0
        for k in all_keys:
            c_gt = gt_counter[k]
            c_pred = pred_counter[k]
            sample_tp += min(c_gt, c_pred)
            if c_pred > c_gt:
                sample_fp += c_pred - c_gt
            if c_gt > c_pred:
                sample_fn += c_gt - c_pred

        total_tp += sample_tp
        total_fp += sample_fp
        total_fn += sample_fn

    schema_valid_rate = schema_valid_count / total
    exact_match_rate = exact_match_count / total

    # Precision, Recall, F1:
    if total_tp + total_fp == 0:
        precision = 1.0 if total_fn == 0 else 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        recall = 1.0 if total_fp == 0 else 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    false_positive_rate = (
        (false_positive_count / total_negatives) if total_negatives > 0 else 0.0
    )
    hallucinated_app_rate = (
        (hallucinated_app_count / total_pred_app_actions)
        if total_pred_app_actions > 0
        else 0.0
    )

    p50_ms = None
    p95_ms = None
    if latencies:
        latencies.sort()
        idx_50 = int(math.floor(len(latencies) * 0.50))
        idx_95 = int(math.floor(len(latencies) * 0.95))
        p50_ms = latencies[min(idx_50, len(latencies) - 1)]
        p95_ms = latencies[min(idx_95, len(latencies) - 1)]

    return {
        "total_samples": total,
        "schema_valid_rate": round(schema_valid_rate, 4),
        "exact_plan_match_rate": round(exact_match_rate, 4),
        "action_sequence_f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_action_rate": round(false_positive_rate, 4),
        "hallucinated_app_id_rate": round(hallucinated_app_rate, 4),
        "total_negatives": total_negatives,
        "false_positive_count": false_positive_count,
        "total_pred_app_actions": total_pred_app_actions,
        "hallucinated_app_count": hallucinated_app_count,
        "latency_p50_ms": round(p50_ms, 2) if p50_ms is not None else None,
        "latency_p95_ms": round(p95_ms, 2) if p95_ms is not None else None,
    }


def print_evaluation_report(metrics: dict[str, Any]) -> None:
    """Print a clean Markdown formatted evaluation report."""
    print("\n" + "=" * 64)
    print("      Desktop Voice Orchestrator — Router Evaluation Report")
    print("=" * 64)
    print(f"Total Evaluated Samples        : {metrics['total_samples']}")
    print(f"Schema Adherence Rate          : {metrics['schema_valid_rate'] * 100:.2f}%")
    print(f"Exact Plan Match Rate          : {metrics['exact_plan_match_rate'] * 100:.2f}%")
    print(f"Action Sequence F1 Score       : {metrics['action_sequence_f1'] * 100:.2f}%")
    print(f"  - Precision                  : {metrics['precision'] * 100:.2f}%")
    print(f"  - Recall                     : {metrics['recall'] * 100:.2f}%")
    print("-" * 64)
    print(
        f"False Positive Rate (Target 0%): {metrics['false_positive_action_rate'] * 100:.2f}% "
        f"({metrics['false_positive_count']}/{metrics['total_negatives']})"
    )
    print(
        f"Hallucinated App ID Rate       : {metrics['hallucinated_app_id_rate'] * 100:.2f}% "
        f"({metrics['hallucinated_app_count']}/{metrics['total_pred_app_actions']})"
    )
    print("-" * 64)
    if metrics["latency_p50_ms"] is not None:
        print(f"Inference Latency p50          : {metrics['latency_p50_ms']} ms")
        print(f"Inference Latency p95          : {metrics['latency_p95_ms']} ms")
    else:
        print("Inference Latency              : N/A (offline predictions evaluation)")
    print("=" * 64 + "\n")


def query_openai_endpoint(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: float = 5.0,
    api_key: str = "local-no-key",
) -> tuple[str, float]:
    """Send chat completions request to OpenAI-compatible endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    parsed = json.loads(body)
    content = parsed["choices"][0]["message"]["content"]
    return content, elapsed_ms


def run_self_test() -> int:
    """Execute unit tests on metric calculations and validation."""
    print("=== [Self-Test] 1. Perfect Match Test ===")
    records_perfect = [
        {
            "ground_truth": {
                "actions": [
                    {
                        "type": "audio.volume.adjust",
                        "volume_delta_percent": -5,
                        "search_query": None,
                        "search_type": None,
                        "search_artist": None,
                        "app_id": None,
                    }
                ]
            },
            "predicted_raw": '{"actions":[{"type":"audio.volume.adjust","volume_delta_percent":-5,"search_query":null,"search_type":null,"search_artist":null,"app_id":null}]}',
            "allowed_app_ids": ["steam", "clash"],
            "latency_ms": 25.0,
        },
        {
            "ground_truth": {"actions": []},
            "predicted_raw": '{"actions":[]}',
            "allowed_app_ids": ["steam", "clash"],
            "latency_ms": 20.0,
        },
    ]
    m_perfect = compute_metrics(records_perfect)
    assert m_perfect["schema_valid_rate"] == 1.0, f"Expected 1.0, got {m_perfect['schema_valid_rate']}"
    assert m_perfect["exact_plan_match_rate"] == 1.0, f"Expected 1.0, got {m_perfect['exact_plan_match_rate']}"
    assert m_perfect["false_positive_action_rate"] == 0.0, f"Expected 0.0, got {m_perfect['false_positive_action_rate']}"
    assert m_perfect["action_sequence_f1"] == 1.0, f"Expected 1.0, got {m_perfect['action_sequence_f1']}"
    print("✅ Perfect match test passed.")

    print("\n=== [Self-Test] 2. False Trigger on Negative Test ===")
    records_ft = [
        {
            "ground_truth": {"actions": []},
            "predicted_raw": '{"actions":[{"type":"media.next","volume_delta_percent":null,"search_query":null,"search_type":null,"search_artist":null,"app_id":null}]}',
            "allowed_app_ids": ["steam"],
            "latency_ms": 30.0,
        }
    ]
    m_ft = compute_metrics(records_ft)
    assert m_ft["false_positive_action_rate"] == 1.0, f"Expected 1.0, got {m_ft['false_positive_action_rate']}"
    assert m_ft["exact_plan_match_rate"] == 0.0
    print("✅ False trigger test passed.")

    print("\n=== [Self-Test] 3. Hallucinated App ID Test ===")
    records_hal = [
        {
            "ground_truth": {"actions": []},
            "predicted_raw": '{"actions":[{"type":"app.open","volume_delta_percent":null,"search_query":null,"search_type":null,"search_artist":null,"app_id":"photoshop"}]}',
            "allowed_app_ids": ["steam", "clash"],
            "latency_ms": 15.0,
        }
    ]
    m_hal = compute_metrics(records_hal)
    assert m_hal["hallucinated_app_id_rate"] == 1.0, f"Expected 1.0, got {m_hal['hallucinated_app_id_rate']}"
    print("✅ Hallucinated app id test passed.")

    print("\n=== [Self-Test] 4. Markdown Code Fence Stripping & Schema Error Test ===")
    fenced_raw = "```json\n{\"actions\":[]}\n```"
    stripped = strip_markdown_code_fence(fenced_raw)
    assert stripped == '{"actions":[]}', f"Markdown stripping failed: {stripped}"

    records_fence = [
        {
            "ground_truth": {"actions": []},
            "predicted_raw": fenced_raw,
            "allowed_app_ids": [],
            "latency_ms": 12.0,
        },
        {
            "ground_truth": {"actions": []},
            "predicted_raw": "Not a valid JSON string at all",
            "allowed_app_ids": [],
            "latency_ms": 10.0,
        },
    ]
    m_fence = compute_metrics(records_fence)
    assert m_fence["schema_valid_rate"] == 0.5, f"Expected 0.5, got {m_fence['schema_valid_rate']}"
    print("✅ Code fence and schema error handling test passed.")

    print("\n=== [Self-Test] 5. Partial Mismatch F1 Test ===")
    records_partial = [
        {
            "ground_truth": {
                "actions": [
                    {
                        "type": "media.play",
                        "volume_delta_percent": None,
                        "search_query": None,
                        "search_type": None,
                        "search_artist": None,
                        "app_id": None,
                    },
                    {
                        "type": "media.next",
                        "volume_delta_percent": None,
                        "search_query": None,
                        "search_type": None,
                        "search_artist": None,
                        "app_id": None,
                    },
                ]
            },
            "predicted_raw": '{"actions":[{"type":"media.play","volume_delta_percent":null,"search_query":null,"search_type":null,"search_artist":null,"app_id":null}]}',
            "allowed_app_ids": [],
            "latency_ms": 40.0,
        }
    ]
    m_partial = compute_metrics(records_partial)
    assert m_partial["exact_plan_match_rate"] == 0.0
    assert m_partial["precision"] == 1.0
    assert m_partial["recall"] == 0.5
    assert math.isclose(m_partial["action_sequence_f1"], 2 * 1.0 * 0.5 / 1.5, rel_tol=1e-3)
    print("✅ Partial mismatch F1 test passed.")

    print_evaluation_report(m_perfect)
    print("=== [Self-Test] Summary: ALL CHECKS PASSED (exit code 0) ===")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Router Offline or via OpenAI-Compatible Endpoint")
    parser.add_argument("--self-test", action="store_true", help="Run offline unit tests and exit")
    parser.add_argument("--val-file", type=str, default="dataset/router/val.jsonl", help="Ground truth validation JSONL")
    parser.add_argument("--predictions", type=str, default=None, help="Path to offline predictions JSONL")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL (e.g. http://127.0.0.1:11434/v1)")
    parser.add_argument("--model", type=str, default="dvo-router", help="Model name for endpoint completions")
    parser.add_argument("--output-report", type=str, default=None, help="Path to export JSON evaluation report")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of validation samples to evaluate")

    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    val_path = Path(args.val_file)
    if not val_path.is_file():
        print(f"Error: val file '{val_path}' does not exist.")
        sys.exit(1)

    val_samples: list[dict[str, Any]] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line))

    if args.max_samples and len(val_samples) > args.max_samples:
        val_samples = val_samples[:args.max_samples]

    records: list[dict[str, Any]] = []

    if args.predictions:
        pred_path = Path(args.predictions)
        if not pred_path.is_file():
            print(f"Error: predictions file '{pred_path}' does not exist.")
            sys.exit(1)

        preds: list[Any] = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))

        for i, val in enumerate(val_samples):
            pred_item = preds[i] if i < len(preds) else ""
            raw_pred = pred_item.get("prediction", pred_item) if isinstance(pred_item, dict) else pred_item
            latency = pred_item.get("latency_ms") if isinstance(pred_item, dict) else None

            gt_json = val["messages"][2]["content"]
            gt_plan = json.loads(gt_json)
            allowed_ids = val.get("app_ids", [])

            records.append(
                {
                    "ground_truth": gt_plan,
                    "predicted_raw": raw_pred,
                    "allowed_app_ids": allowed_ids,
                    "latency_ms": latency,
                }
            )

    elif args.base_url:
        print(f"Connecting to endpoint {args.base_url} (model={args.model}) ...")
        for i, val in enumerate(val_samples):
            sys_prompt = val["messages"][0]["content"]
            user_prompt = val["messages"][1]["content"]
            gt_json = val["messages"][2]["content"]
            gt_plan = json.loads(gt_json)
            allowed_ids = val.get("app_ids", [])

            try:
                raw_pred, latency = query_openai_endpoint(
                    base_url=args.base_url,
                    model=args.model,
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as e:
                print(f"Warning: Sample #{i} endpoint error: {e}")
                raw_pred = ""
                latency = None

            records.append(
                {
                    "ground_truth": gt_plan,
                    "predicted_raw": raw_pred,
                    "allowed_app_ids": allowed_ids,
                    "latency_ms": latency,
                }
            )
    else:
        print("Error: Either --predictions or --base-url must be provided.")
        sys.exit(1)

    metrics = compute_metrics(records)
    print_evaluation_report(metrics)

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
