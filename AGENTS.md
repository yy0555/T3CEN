# Repository guidance

This repository contains the T3CEN implementation for *A Hypertoroidal
Covering for Perfect Color Equivariance*. The release contains T3CEN only.

## Layout

- `modules/`: core T3CEN transforms, group layers, networks, and Lightning model.
- `dataloader/`: dataset classes and loader factories.
- `utils/`: shared paths, training configuration helpers, analysis utilities, and tests.
- `data/`: local datasets; never commit their contents.
- `outputs/`: all generated logs, weights, figures, results, environments, and caches; never commit their contents.
- `exp/`: training entrypoint and experiment definitions.

Keep root project files limited to `uv.lock`, `pyproject.toml`, `python.version`,
`README.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore`, and `LICENSE`.
Git metadata and environment-managed hidden directories are not source files.
`CLAUDE.md` contains only `@AGENTS.md`.

## Development

Run commands from the repository root. Use `python -m exp.train --gpus N`
(default `1`; `0` for CPU; more than one enables DDP). Use package imports
instead of modifying `sys.path`.
Select experiments with `--exp_class <dataset> --exp_name h<H>s<S>l<L>`.
Each `exp/configs/<dataset>.json` maps lifting-order keys to experiment settings;
the 10 dataset files contain 60 selectable presets. For example, run
`python -m exp.train --exp_class cifar10 --exp_name h4s1l1 --gpus 1`.
Keep one seed (`1999`) and use datasets and lifting orders referenced by the paper.
Run and log names combine the dataset and lifting key as `<dataset>_h<H>s<S>l<L>`.
Use `utils.constants` for paths, which are anchored at the repository root.
Keep import and model forward paths free of file-writing side effects.

Use Python 3.11. For uv commands set `UV_PROJECT_ENVIRONMENT` to
`$PWD/outputs/.venv`, `UV_CACHE_DIR` to `$PWD/outputs/.cache/uv`, and
`PYTHONPYCACHEPREFIX` to `$PWD/outputs/.cache/pycache`. Install with
`uv sync --locked --python "$(cat python.version)"`; see README for setup.
Keep `uv.lock` consistent with `pyproject.toml` when changing dependencies.

Run `python -m unittest discover -s utils/tests -v` for relevant smoke checks.
Do not download datasets or start full training just to validate a layout change.
Keep tests under `utils/tests/` and generated test artifacts under `outputs/` or
a temporary directory.

Distinguish runtime cleanup from changes to the scientific protocol when editing
dataset splits, lifting transforms, losses, or experiment presets. Document any
change that affects reported results.
Preserve local datasets and past experiment artifacts during maintenance.
