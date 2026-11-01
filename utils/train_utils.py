"""Load experiment settings and validate trainer limits before allocating data."""

import json
import math

from utils import constants as c


def get_default_params():
    with c.DEFAULT_PARAMS_DIR.open() as source:
        return json.load(source)


def get_experiment_params(exp_class, exp_name):
    """Select a preset from a dataset JSON file and merge it over defaults."""
    for name, value in (("exp_class", exp_class), ("exp_name", exp_name)):
        if (
            not isinstance(value, str) or not value or value in (".", "..")
            or any(character in value for character in ("/", "\\"))
        ):
            raise ValueError(f"{name} must be a filename stem, without a directory.")
    config_path = c.EXPERIMENTS_PATH / f"{exp_class}.json"
    if not config_path.is_file():
        available = ", ".join(path.stem for path in sorted(c.EXPERIMENTS_PATH.glob("*.json")))
        raise ValueError(f"Unknown dataset {exp_class!r}. Available: {available}")
    with config_path.open() as source:
        experiments = json.load(source)
    if not isinstance(experiments, dict):
        raise ValueError(f"{config_path.name} must map experiment names to configuration objects.")
    if exp_name not in experiments:
        available = ", ".join(experiments)
        raise ValueError(
            f"Unknown experiment {exp_name!r} in {config_path.name}. Available: {available}"
        )
    settings = experiments[exp_name]
    if not isinstance(settings, dict) or any(
        isinstance(value, dict) and key != "dataset_kwargs"
        for key, value in settings.items()
    ):
        raise ValueError(
            f"{config_path.name}[{exp_name!r}] must contain an experiment object."
        )
    params = get_default_params()
    params.update(settings)
    if not isinstance(params.get("dataset_kwargs"), dict):
        raise ValueError("dataset_kwargs must be an object of dataset options.")
    if "ours" in params or "ours" in params.get("dataset_kwargs", {}):
        raise ValueError("The 'ours' option was removed; all experiments now use T3CEN.")
    params["experiment_name"] = f"{exp_class}_{exp_name}"
    limits = [key for key in ("n_epochs", "n_iters") if key in params]
    if len(limits) != 1:
        raise ValueError("Specify exactly one of n_epochs or n_iters in the experiment parameters.")
    key = limits[0]
    params[key] = positive_integer(params[key], key)
    return params


def positive_integer(value, name):
    """Accept JSON integers and integral floats such as the legacy 1e5 limit."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1
        or value != int(value)
    ):
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return int(value)


def trainer_limits(params):
    if "n_epochs" in params:
        return {"max_epochs": params["n_epochs"]}
    return {"max_steps": params["n_iters"]}
