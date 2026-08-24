"""Execute the frozen ten-seed Split CIFAR-10 confirmation."""

from competitive_architectures.multiseed import run_frozen_multiseed


def main() -> None:
    run_frozen_multiseed(
        config_path="configs/split_cifar10_replay_confirmation.json",
        data_dir="data/cifar10",
        output_path="results/split_cifar10_replay_confirmation.json",
    )


if __name__ == "__main__":
    main()
