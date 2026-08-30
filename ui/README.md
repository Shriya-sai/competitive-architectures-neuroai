# Experiment Atlas

A dependency-free browser interface for inspecting the repository's frozen
NeuroAI experiments. It provides three linked views:

- the exact 64-channel signed masks used by the controlled models;
- experience-wise class retention for every frozen confirmation seed;
- exploratory pathway-engagement, drift and intervention results.

Generate the committed interface dataset from local result files:

```bash
PYTHONPATH=src python scripts/export_neuroai_ui.py
```

Run from the repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/ui/`.

The architecture view depicts exact graph masks but uses a schematic layout.
The continual-learning view uses ten untouched paired confirmation seeds. The
mechanism view uses five exposed development seeds and is explicitly labelled
exploratory. These evidence levels are not pooled.
