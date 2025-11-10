# GRN ML Pipeline Package

Python package housing the reusable components that power the GRN regression workflow:

- `config.py` – dataclasses describing filesystem layout, training hyperparameters, and model selections.
- `cli.py` – entrypoint for command-line execution (`python -m ml_grn_pipeline.cli`).
- `data.py`, `training.py`, `evaluation.py`, `metrics.py` – data handling, model training loops, and evaluation logic.
- `visualization.py` – plotting utilities for diagnostic figures.

Install in editable mode for development:

```bash
pip install -e .
```

Run `python -m ml_grn_pipeline.cli --help` for the full list of pipeline options.
