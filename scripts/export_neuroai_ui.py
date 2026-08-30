"""Export compact frozen experiment data for the static NeuroAI interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from competitive_architectures.graphs import (
    rewire_signed_masks,
    structured_signed_masks,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "ui/data/neuroai-atlas.json"


def graph_payload(mode: str, seed: int = 23) -> dict[str, object]:
    """Return the exact signed mask used by the 64-channel model."""
    if mode == "standard":
        groups = np.repeat(np.arange(4), 16)
        return {"nodes": node_positions(groups), "edges": []}
    masks = structured_signed_masks(
        channels=64,
        groups=4,
        cooperative_degree=4,
        competitive_degree=4,
        seed=seed,
    )
    if mode == "random_signed":
        masks = rewire_signed_masks(masks, seed=seed + 1)
    groups = masks.groups.cpu().numpy()
    edges = []
    for sign, mask in (("cooperative", masks.cooperative), ("competitive", masks.competitive)):
        targets, sources = torch.where(mask > 0)
        edges.extend(
            {"source": int(source), "target": int(target), "sign": sign}
            for target, source in zip(targets, sources, strict=True)
        )
    return {"nodes": node_positions(groups), "edges": edges}


def node_positions(groups: np.ndarray) -> list[dict[str, float | int]]:
    """Lay out representation channels in four visibly separated modules."""
    centers = ((0.27, 0.28), (0.73, 0.28), (0.27, 0.72), (0.73, 0.72))
    positions = []
    for group, (cx, cy) in enumerate(centers):
        members = np.flatnonzero(groups == group)
        for offset, channel in enumerate(members):
            angle = 2 * np.pi * offset / len(members)
            radius = 0.115 + 0.025 * (offset % 2)
            positions.append({
                "id": int(channel), "group": group,
                "x": round(float(cx + radius * np.cos(angle)), 6),
                "y": round(float(cy + radius * np.sin(angle)), 6),
            })
    return sorted(positions, key=lambda item: item["id"])


def mean_transition_curves(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = ("old_group_profile_cosine_drift", "old_accuracy_change", "old_margin_change")
    output: dict[str, object] = {}
    for mode in ("random_signed", "structured_signed"):
        selected = [run for run in runs if run["mode"] == mode]
        curve = []
        for experience in range(2, 6):
            rows = [
                transition
                for run in selected
                for transition in run["transitions"]
                if transition["after_experience"] == experience
            ]
            curve.append({"after_experience": experience, **{
                metric: float(np.mean([row[metric] for row in rows])) for metric in metrics
            }})
        output[mode] = curve
    return output


def build_payload() -> dict[str, object]:
    confirmation = json.loads((RESULTS / "split_cifar10_replay_confirmation.json").read_text())
    bottleneck = json.loads((RESULTS / "bottleneck_development_screen.json").read_text())
    drift = json.loads((RESULTS / "bottleneck_drift_screen.json").read_text())
    selective = json.loads((RESULTS / "selective_consolidation_screen.json").read_text())
    runs = [{
        key: run[key] for key in (
            "mode", "seed", "class_order", "experiences", "accuracy_matrix",
            "average_incremental_accuracy", "final_average_accuracy",
            "average_forgetting", "mean_new_experience_accuracy", "parameter_count",
        )
    } for run in confirmation["runs"]]
    return {
        "schema_version": "1.0.0",
        "title": "Competitive Architectures Experiment Atlas",
        "provenance": {
            "confirmation": "ten untouched paired class-order seeds",
            "mechanism": "five exposed development seeds; exploratory",
            "dataset": confirmation["config"]["dataset"],
            "task": "five two-class Split CIFAR-10 experiences with replay",
        },
        "graphs": {mode: graph_payload(mode) for mode in ("standard", "random_signed", "structured_signed")},
        "confirmation": {
            "seeds": confirmation["summary"]["seeds"],
            "condition_means": confirmation["summary"]["condition_means"],
            "primary_contrast": confirmation["summary"]["structured_minus_random"]["average_incremental_accuracy"],
            "runs": runs,
        },
        "bottleneck": {
            "summary": bottleneck["summary"],
            "runs": [{
                key: run[key] for key in (
                    "mode", "seed", "post_signed_edge_tuning_gap", "post_group_tuning_gap",
                    "pathway_accuracy_effect", "pathway_prediction_disagreement",
                    "final_average_accuracy", "average_forgetting",
                )
            } for run in bottleneck["runs"]],
        },
        "drift": {
            "condition_means": drift["summary"]["condition_means"],
            "transition_curves": mean_transition_curves(drift["runs"]),
        },
        "interventions": selective["summary"],
    }


def main() -> None:
    payload = build_payload()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Wrote {len(payload['confirmation']['runs'])} confirmation runs to {OUTPUT}")


if __name__ == "__main__":
    main()
