# Competitive Architectures for Brain-Inspired Deep Learning

Can structured cooperative–competitive interactions inspired by biological
brain networks improve DNN computation, and do they make the resulting
representations more brain-like?

This repository is a NeuroAI child project motivated by
[`competitive-connectomes-adaptive-dynamics`](https://github.com/Shriya-sai/competitive-connectomes-adaptive-dynamics).
The parent project studies how the organization of cooperative and competitive
interactions shapes whole-brain model dynamics. This project tests whether a
carefully controlled computational analogue transfers to artificial networks.

## Current status

The first controlled DNN experiment is complete. A small CIFAR-10 CNN was
tested under three capacity-matched conditions: no lateral interaction,
degree-preserving randomly rewired signed interactions, and structured signed
interactions. The signed mechanism passed synthetic construct-validity tests
before entering a five-experience Split CIFAR-10 replay experiment.

A frozen paired confirmation across ten fresh class-order seeds found no
reliable advantage of structured over random signed topology. The primary
structured-minus-random difference in average incremental accuracy was 0.00066
(95% bootstrap interval: -0.00257 to 0.00446; exact paired sign-flip
`p = 0.764`).

A subsequent causal diagnosis showed that the residual signed pathway altered
the backbone representation by only about 1.2%. Bypassing it after training
changed fewer than 0.7% of predictions and changed accuracy by less than 0.1
percentage point. A non-bypassable signed bottleneck corrected this measurement
failure. Across five exploratory development seeds, structured bottlenecks
reliably expressed cooperative--competitive class-tuning organization, whereas
matched random bottlenecks did not.

That engagement did not produce a clear computational advantage. New-class
acquisition was matched, but the structured models showed lower final retention
and greater forgetting on average. The next analysis asks whether structured
topology increases representational interference across sequential experiences.

## Planned experimental comparison

All conditions will share the same task, data, optimizer, training budget, and
parameter budget as closely as possible:

1. standard DNN;
2. DNN with magnitude-matched random competition;
3. DNN with structured competition.

Primary computational endpoints will be frozen before training. RSA against
brain data is a later stage and remains logically separate from computational
benefit: improved performance and improved biological alignment need not agree.

## Repository map

- `docs/STRUCTURED_COMPETITION_CONSTRUCT.md` — foundational construct and decision gates
- `docs/PROJECT_ROADMAP.md` — staged project plan
- `docs/FIRST_DNN_EXPERIMENT.md` — frozen confirmation and pathway diagnosis
- `configs/` — frozen experiment specifications
- `src/competitive_architectures/` — reusable mechanisms and evaluation code
- `tests/` — unit and construct-validity tests
- `scripts/` — executable experiment entry points
- `notebooks/` — exploratory analysis only
- `data/` — local-data instructions; datasets are never committed
- `results/` — generated outputs; large artifacts are never committed

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

The repository is private during development.
