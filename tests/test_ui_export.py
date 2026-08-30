import numpy as np

from scripts.export_neuroai_ui import (
    graph_payload,
    mean_transition_curves,
    node_positions,
)


def test_node_layout_is_deterministic_and_bounded() -> None:
    groups = np.repeat(np.arange(4), 4)
    first = node_positions(groups)
    assert first == node_positions(groups)
    assert len(first) == 16
    assert all(0 <= node["x"] <= 1 and 0 <= node["y"] <= 1 for node in first)


def test_exported_structured_graph_obeys_group_sign_rule() -> None:
    graph = graph_payload("structured_signed")
    groups = {node["id"]: node["group"] for node in graph["nodes"]}
    for edge in graph["edges"]:
        same_group = groups[edge["source"]] == groups[edge["target"]]
        assert same_group is (edge["sign"] == "cooperative")


def test_transition_curve_averages_by_condition_and_experience() -> None:
    runs = []
    for mode, offset in (("random_signed", 0.0), ("structured_signed", 1.0)):
        runs.append({"mode": mode, "transitions": [{
            "after_experience": experience,
            "old_group_profile_cosine_drift": offset + experience,
            "old_accuracy_change": offset - experience,
            "old_margin_change": offset - 2 * experience,
        } for experience in range(2, 6)]})
    curves = mean_transition_curves(runs)
    assert curves["structured_signed"][0]["old_group_profile_cosine_drift"] == 3.0
    assert curves["random_signed"][-1]["old_accuracy_change"] == -5.0
