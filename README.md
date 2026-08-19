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

The project is in **Stage 0: construct definition**. No DNN architecture,
dataset, or competitive mechanism has been selected yet. Those choices will be
made only after the construct, controls, endpoints, and falsification criteria
are frozen.

The leading construct is now an explicit signed lateral interaction prior with
predefined structural channel populations and emergent functional roles. The
leading computational task is class-incremental visual learning, with average
incremental accuracy as the provisional primary endpoint. Both remain subject
to the Stage 0 validity and literature gates before confirmation.

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
