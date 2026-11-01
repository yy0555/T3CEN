# A Hypertoroidal Covering for Perfect Color Equivariance

Official T3CEN code for **A Hypertoroidal Covering for Perfect Color Equivariance** (ICML 2026).

[Paper on arXiv](https://arxiv.org/abs/2603.04256)

T3CEN implements the hypertoroidal color equivariant network introduced in the paper. The central idea is to build color equivariance in a way that respects the topology of color space. Hue is naturally cyclic, but saturation and luminance are bounded interval-valued quantities. Treating those intervals as translations forces clipping or padding at the boundary, which introduces artifacts and prevents exact equivariance.

Instead, T3CEN uses a double-cover lifting for saturation and luminance. The interval is lifted to a circle, discretized as a cyclic group, and combined with the hue group to form a product color group. An input image is lifted to a function on this hue-saturation-luminance group, group convolutions propagate features by cyclically permuting the learned filter orbit, and group pooling produces an invariant representation for classification when desired.

This gives a color equivariant architecture that is exactly equivariant to hue, saturation, and luminance shifts under the lifted group action. In the paper, this removes the approximation artifacts of prior saturation and luminance equivariant models, lowers equivariance error by several orders of magnitude, and improves robustness under color shift on synthetic benchmarks, fine-grained recognition datasets, and Camelyon17.

## Repository layout

```text
modules/     Core T3CEN transforms, group layers, networks, Lightning model
  hsgroup/
  networks/
dataloader/  Dataset definitions and loader factories
utils/       Shared paths, configuration loading, analysis tools, smoke tests
data/        Local datasets (ignored by Git)
outputs/     Logs, checkpoints, results, caches, and local environments (ignored)
exp/         Training entrypoint, defaults, experiment configs
```

The core implementation is in [`modules/hsgroup/`](modules/hsgroup/), with
backbones in [`modules/networks/`](modules/networks/). This release contains
T3CEN models and experiments only.

## Setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```bash
mkdir -p data outputs
export UV_PROJECT_ENVIRONMENT="$PWD/outputs/.venv"
export UV_CACHE_DIR="$PWD/outputs/.cache/uv"
export PYTHONPYCACHEPREFIX="$PWD/outputs/.cache/pycache"
uv sync --locked --python "$(cat python.version)"
```

Repeat the exports in each new shell. Dependencies are declared in
[`pyproject.toml`](pyproject.toml) and resolved in [`uv.lock`](uv.lock).
The requested `python.version` filename is read explicitly by these commands;
uv normally discovers `.python-version` automatically. The Python 3.11 constraint
is also enforced by `pyproject.toml`.

The lock uses the PyPI builds of PyTorch 2.7.1 and TorchVision 0.22.1. GPU runs
need a compatible NVIDIA driver. It was generated from the installed environment's
direct dependency versions; it is not a recovered environment from the paper's runs.

## Running experiments

Each dataset has one JSON file, `exp/configs/<dataset>.json`. It maps lifting
keys such as `h4s1l1` to experiment settings that override
[`exp/default_params.json`](exp/default_params.json). Keys follow
`h<H>s<S>l<L>`, where `H`, `S`, and `L` are the hue, saturation, and luminance
lifting orders. All presets use seed `1999`.

The 10 dataset files contain 60 training presets covering the paper's
classification results and hue-MNIST cardinality study. The catalogue follows
[Tables 1–7 and Appendix D](https://arxiv.org/html/2603.04256#S5.T1):

| Dataset (`--exp_class`) | Lifting orders | Paper reference |
| --- | --- | --- |
| `caltech101`, `cifar10`, `cifar100`, `oxford_pets`, `stanford_cars`, `stl10` | H4, S4, L4, H4S4, H4L4, S4L4 for each dataset | Tables 6–7 |
| `shapes3d` | H3, H4, S3, H3S3L3, H4S4L4 | Tables 1, 2, 4 |
| `smallnorb` | L3, L4, L8, L16 | Table 3 |
| `camelyon17` | H4, S4, L4, H4S4, S4L4, H4S4L4 | Table 5 |
| `hue_mnist` | H1–H8, H20 | Figure 9 and Appendix C.1 |

Inactive axes have order one: S4 is `h1s4l1`, and H4L4 is `h4s1l4`.
For example, use `--exp_class smallnorb --exp_name h1s1l8` or
`--exp_class camelyon17 --exp_name h1s4l1`.
H4L4 is included for each of the six real-image datasets because it appears in
Table 6's luminance-shift results. Figure 9 plots MNIST orders 1–8; H20 is
reported separately in the accompanying text.

Training settings follow Appendix D: ResNet44/Adam/10,000 steps for Shapes3D;
ResNet18/Adam/300 epochs for SmallNORB and the four larger real-image datasets;
ResNet44/SGD/300 epochs for CIFAR; ResNet50/Adam/10,000 steps for Camelyon17;
and Z2CNN/SGD/5 epochs for hue-MNIST. Learning rates and batch sizes are explicit
in each preset. All runs use the single release seed, `1999`.

The paper also contains equivariance-error, lifting coverage, RGB/scale, pixel
statistics, and computational-cost analyses. Those are separate from this
classification-training catalogue; runnable analysis entrypoints are not included.

Run the training module from the repository root with the dataset filename stem,
lifting key, and GPU count:

```bash
uv run --locked python -m exp.train --exp_class cifar10 --exp_name h4s1l1 --gpus 1
uv run --locked python -m exp.train --exp_class caltech101 --exp_name h4s1l1 --gpus 2
```

`--exp_class cifar10` selects `exp/configs/cifar10.json`, and `--exp_name h4s1l1`
selects the nested `h4s1l1` settings. The resulting run name is `cifar10_h4s1l1`.

`--gpus` defaults to `1`. Use `--gpus=0` for CPU training; values greater than
one enable distributed GPU training with DDP. The count applies to CUDA GPUs
visible to the process, so `CUDA_VISIBLE_DEVICES` can select which GPUs to use.
A request exceeding the available GPU count fails before loading data.

Exactly one of `n_epochs` and `n_iters` must be set. Missing luminance group counts
default to one. Training tests the best validation checkpoint when one was saved,
and otherwise tests the current model, for every GPU count.

All dataset paths are anchored at this checkout's `data/`, and logs and
checkpoints go to `outputs/logs/<dataset>_h<H>s<S>l<L>/version_<N>/`.
Existing run artifacts were preserved under `outputs/logs/`,
`outputs/lightning_logs/`, and `outputs/results/`.

The loaders can download CIFAR-10, STL-10, Oxford Pets, Camelyon17, and MNIST.
Other loaders expect data to be prepared locally.

- Shapes3D expects `data/shapes3d/3dshapes.h5`. Hue experiments construct the
  warm/cold subsets from the original labels. Saturation and HSL experiments
  require pre-generated shifted HDF5 files in `data/shapes3d/validation/saturation/`
  and `data/shapes3d/validation/hsl/`, respectively, with the original image/label
  ordering. The paper does not fully specify the random-shift distributions;
  this loader does not generate replacement shifted datasets.
- SmallNORB expects `smallnorb_train.parquet` and `smallnorb_test.parquet` under
  `data/smallnorb/`. Test loaders return medium, low, and high lighting conditions.
- Hue-MNIST trains with hues in 0–120 degrees and tests with hues in 120–360
  degrees. The loader reserves 10% of the official training set for validation.
- Camelyon17 uses the WILDS training, validation, and test splits.
- The six real-image datasets evaluate the test variants listed in each preset
  under `dataset_kwargs.test_shifts`: original, saturation reduced by 0.5,
  and/or luminance reduced by 0.5. H4S4 uses original and saturation; H4L4 uses
  luminance; the other orders use all three. Validation always uses the original images.

Shapes3D's 80/20 split and MNIST's 90/10 training/validation split are explicit
implementation choices; the paper does not specify these fractions. See
[`dataloader/dataloaders.py`](dataloader/dataloaders.py) for dataset options and
[`dataloader/custom_datasets.py`](dataloader/custom_datasets.py) for dataset classes.

Analysis utilities run as `python -m utils.h5` and `python -m utils.parq`; their
outputs are written under `outputs/` and importing them does not read datasets.

## Checks

Run the CPU smoke checks without downloading datasets:

```bash
uv run --locked python -m unittest discover -s utils/tests -v
```

## Citation

```bibtex
@inproceedings{yang2026hypertoroidal,
  title = {A Hypertoroidal Covering for Perfect Color Equivariance},
  author = {Yang, Yulong and Xu, Zhikun and Li, Yaojun and Allen-Blanchette, Christine},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {306},
  year = {2026}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
