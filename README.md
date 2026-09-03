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
and greater forgetting on average. Experience-wise drift analysis found no
excess global centroid movement in structured models. Instead, old classes
consistently redistributed activity across the imposed structural groups more
strongly, with the clearest additional decision loss appearing late in the
sequence. This is a mechanistic signature, not yet evidence that module
reallocation causes forgetting. A matched replay regularizer subsequently
reduced structured group-profile drift in every development seed but did not
improve final retention and reduced new-class acquisition on average. This
supports a stability--plasticity trade-off rather than group drift as a
sufficient cause of forgetting. Weighting protection by replay-memory
consolidation avoided the final-accuracy cost of global freezing, but did not
outperform the unregularized structured model.

## Completed controlled comparison

The initial experiment held the task, data, optimizer, training budget and
parameter budget as closely matched as possible across:

1. standard DNN;
2. DNN with magnitude-matched random competition;
3. DNN with structured competition.

The computational endpoints were frozen before confirmation training. As
reported above, structured topology did not reliably outperform the matched
random signed control, and the subsequent bottleneck experiments established
pathway engagement while exposing a stability--plasticity trade-off. RSA
against brain data remains a later stage and is logically separate from
computational benefit: improved performance and improved biological alignment
need not agree.

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
- `ui/` — dependency-free interactive experiment atlas

## Interactive experiment atlas

The repository includes a static browser interface with three linked views:

1. the exact standard, degree-preserving random-signed and structured-signed
   interaction masks;
2. experience-wise class retention for every frozen confirmation seed;
3. exploratory pathway-engagement, representational-drift and selective-
   consolidation results.

Run it from the repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/ui/`. The committed 79 KB payload makes the atlas
work immediately after cloning. After reproducing the local experiments,
regenerate it with:

```bash
PYTHONPATH=src python scripts/export_neuroai_ui.py
```

The interface labels confirmatory and exploratory evidence separately and does
not include uncommitted experiments. See [`ui/README.md`](ui/README.md) for its
data contract.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

## Licence

Project-authored code is released under the [MIT License](LICENSE). CIFAR-10,
PyTorch and other dependencies retain their own licences and attribution
requirements. Generated datasets, checkpoints and result files are not
distributed in this repository.
