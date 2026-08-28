# Contributing

Bug reports, reproducibility checks and focused pull requests are welcome.
Please open an issue before proposing a substantial change to the scientific
design.

Create the Python 3.12 development environment described in the README and run:

```bash
pytest
ruff check src scripts tests
```

Do not commit datasets, generated results, model checkpoints, credentials or
local machine paths. Preserve the distinction between exploratory development
seeds and frozen confirmation seeds when changing experiments.
