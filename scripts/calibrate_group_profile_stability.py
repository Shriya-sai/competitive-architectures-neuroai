"""Calibrate old-class group-profile stability on development seed 23."""

import json

from competitive_architectures.drift import run_profile_stability_calibration


def main() -> None:
    result = run_profile_stability_calibration(
        data_dir="data/cifar10",
        output_path="results/group_profile_stability_calibration.json",
    )
    summaries = []
    for run in result["runs"]:
        summaries.append(
            {
                "weight": run["profile_stability_weight"],
                "mode": run["mode"],
                "mean_group_drift": sum(
                    item["old_group_profile_cosine_drift"]
                    for item in run["transitions"]
                )
                / len(run["transitions"]),
                "acquisition": run["mean_new_experience_accuracy"],
                "final_accuracy": run["final_average_accuracy"],
            }
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
