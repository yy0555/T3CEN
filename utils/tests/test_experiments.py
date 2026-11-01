"""Release integration checks without downloads, GPU allocation, or training."""

import builtins
from contextlib import chdir
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

from exp.train import run
from utils import constants as c
from utils import train_utils


def experiment_settings():
    for path in sorted(c.EXPERIMENTS_PATH.glob("*.json")):
        for exp_name, settings in json.loads(path.read_text()).items():
            yield path.stem, exp_name, settings


class ExperimentTests(unittest.TestCase):
    def test_experiments_load_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            params = train_utils.get_experiment_params("cifar10", "h4s1l1")
            self.assertEqual(params["experiment_name"], "cifar10_h4s1l1")
            self.assertEqual(params["n_groups_hue"], 4)
            self.assertEqual(params["n_groups_luminance"], 1)
            self.assertTrue(c.DATA_DIR.is_absolute())
            self.assertTrue(c.LOG_DIR.is_relative_to(c.OUTPUT_DIR))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_release_experiments_have_one_seed_and_bind_to_their_loaders(self):
        from dataloader.dataloaders import parse_dataloader

        # Tables 1–7 define the main catalogue; Fig. 9 and Appendix C.1
        # additionally report the MNIST sweep (H1–H8 in the figure, H20 in prose).
        real_orders = {"h4s1l1", "h1s4l1", "h1s1l4", "h4s4l1", "h4s1l4", "h1s4l4"}
        expected = {
            f"{dataset}_{order}"
            for dataset in ("caltech101", "cifar10", "cifar100", "oxford_pets", "stanford_cars", "stl10")
            for order in real_orders
        }
        expected.update({
            "shapes3d_h3s1l1", "shapes3d_h4s1l1", "shapes3d_h1s3l1",
            "shapes3d_h3s3l3", "shapes3d_h4s4l4",
            "smallnorb_h1s1l3", "smallnorb_h1s1l4", "smallnorb_h1s1l8", "smallnorb_h1s1l16",
            "camelyon17_h4s1l1", "camelyon17_h1s4l1", "camelyon17_h1s1l4",
            "camelyon17_h4s4l1", "camelyon17_h1s4l4", "camelyon17_h4s4l4",
        })
        expected.update(f"hue_mnist_h{order}s1l1" for order in (*range(1, 9), 20))
        self.assertEqual(len(expected), 60)
        paths = sorted(c.EXPERIMENTS_PATH.glob("*.json"))
        expected_datasets = {
            "caltech101", "cifar10", "cifar100", "oxford_pets", "stanford_cars",
            "stl10", "shapes3d", "smallnorb", "camelyon17", "hue_mnist",
        }
        self.assertEqual({path.stem for path in paths}, expected_datasets)
        configurations = list(experiment_settings())
        self.assertEqual(len(configurations), 60)
        self.assertEqual({f"{dataset}_{name}" for dataset, name, _ in configurations}, expected)
        for exp_class, exp_name, settings in configurations:
            with self.subTest(dataset=exp_class, experiment=exp_name):
                params = train_utils.get_experiment_params(exp_class, exp_name)
                self.assertNotIn("ours", params)
                self.assertEqual(params["manual_seed"], 1999)
                self.assertEqual(settings["manual_seed"], 1999)
                self.assertEqual(params["experiment_name"], f"{exp_class}_{exp_name}")
                dataset = {"cifar": "cifar10", "STL10": "stl10"}.get(
                    params["dataset_name"], params["dataset_name"]
                )
                h, s, l = (settings[key] for key in (
                    "n_groups_hue", "n_groups_saturation", "n_groups_luminance"
                ))
                self.assertEqual(exp_class, dataset)
                self.assertEqual(exp_name, f"h{h}s{s}l{l}")
                loader = parse_dataloader(params["dataset_name"])
                self.assertNotIn("ours", inspect.signature(loader).parameters)
                inspect.signature(loader).bind(
                    n_groups_hue=h, n_groups_saturation=s, n_groups_luminance=l,
                    batch_size=params["batch_size"], **params["dataset_kwargs"],
                )
                limits = train_utils.trainer_limits(params)
                self.assertEqual(len(limits), 1)
                self.assertIsInstance(next(iter(limits.values())), int)

    def test_training_hyperparameters_match_paper_appendix_d(self):
        # Explicitly check the paper's values rather than accepting inherited
        # legacy settings just because they form a runnable configuration.
        expected = {
            # classes, backbone, optimizer, lr, batch, limit, duration, scheduler
            "caltech101": (101, "resnet18", "adam", 0.01, 16, "n_epochs", 300, "none"),
            "cifar": (10, "resnet44", "sgd", 0.1, 128, "n_epochs", 300, "cosine"),
            "cifar100": (100, "resnet44", "sgd", 0.1, 128, "n_epochs", 300, "cosine"),
            "stanford_cars": (196, "resnet18", "adam", 0.01, 16, "n_epochs", 300, "none"),
            "oxford_pets": (37, "resnet18", "adam", 0.01, 16, "n_epochs", 300, "none"),
            "STL10": (10, "resnet18", "adam", 0.01, 16, "n_epochs", 300, "none"),
            "shapes3d": (4, "resnet44", "adam", 0.0001, 128, "n_iters", 10000, "none"),
            "smallnorb": (5, "resnet18", "adam", 0.01, 16, "n_epochs", 300, "none"),
            "camelyon17": (2, "resnet50", "adam", 0.01, 32, "n_iters", 10000, "none"),
            "hue_mnist": (10, "z2cnn", "sgd", 0.001, 128, "n_epochs", 5, "none"),
        }
        real_datasets = {"caltech101", "cifar", "cifar100", "stanford_cars", "oxford_pets", "STL10"}
        for exp_class, exp_name, settings in experiment_settings():
            with self.subTest(dataset=exp_class, experiment=exp_name):
                values = expected[settings["dataset_name"]]
                keys = ("num_classes", "network_name", "optimizer", "lr", "batch_size")
                self.assertEqual(tuple(settings[key] for key in keys), values[:5])
                limit, duration, scheduler = values[5:]
                self.assertEqual(settings[limit], duration)
                self.assertNotIn("n_iters" if limit == "n_epochs" else "n_epochs", settings)
                self.assertEqual(settings["scheduler"], scheduler)
                self.assertEqual(settings["loss_name"], "cross_entropy")
                if settings["dataset_name"] in real_datasets:
                    order = tuple(settings[key] for key in (
                        "n_groups_hue", "n_groups_saturation", "n_groups_luminance"
                    ))
                    # Table 6 reports H4S4 only for saturation, H4L4 only for
                    # luminance; Table 7 includes H4S4 but excludes H4L4.
                    shifts = {
                        (4, 4, 1): ["original", "saturation"],
                        (4, 1, 4): ["luminance"],
                    }.get(order, ["original", "saturation", "luminance"])
                    self.assertEqual(settings["dataset_kwargs"], {
                        "test_shifts": shifts,
                    })
                elif settings["dataset_name"] == "shapes3d":
                    # Holdout fractions are explicit release choices; Appendix D
                    # specifies the shift protocols, not these split fractions.
                    h, s, l = (settings[key] for key in (
                        "n_groups_hue", "n_groups_saturation", "n_groups_luminance"
                    ))
                    protocol = "hsl" if l > 1 else "saturation" if s > 1 else "hue"
                    self.assertEqual(settings["dataset_kwargs"], {
                        "protocol": protocol, "train_fraction": 0.8, "split_seed": 1999,
                    })
                elif settings["dataset_name"] == "hue_mnist":
                    self.assertEqual(settings["dataset_kwargs"], {
                        "validation_fraction": 0.1, "split_seed": 1999,
                    })

    def test_camelyon_name_matches_its_parameters(self):
        params = train_utils.get_experiment_params("camelyon17", "h1s4l4")
        self.assertEqual(params["dataset_name"], "camelyon17")
        self.assertEqual(params["n_groups_hue"], 1)
        self.assertEqual(params["n_groups_saturation"], 4)
        self.assertEqual(params["n_groups_luminance"], 4)
        self.assertEqual(params["manual_seed"], 1999)

    def test_removed_implementation_switch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            with patch.object(c, "EXPERIMENTS_PATH", config_dir):
                for settings in [
                    {"ours": False}, {"ours": True}, {"dataset_kwargs": {"ours": False}},
                ]:
                    with self.subTest(settings=settings):
                        settings["n_epochs"] = 1
                        (config_dir / "obsolete.json").write_text(json.dumps({"h1s1l1": settings}))
                        with self.assertRaisesRegex(ValueError, "option was removed"):
                            train_utils.get_experiment_params("obsolete", "h1s1l1")

    def test_invalid_cli_inputs_do_not_import_training_dependencies(self):
        original_import = builtins.__import__

        def reject_training_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "lightning", "dataloader", "modules"}:
                raise AssertionError(f"Training dependency imported before validation: {name}")
            return original_import(name, *args, **kwargs)

        runner = CliRunner()
        selection = ["--exp_class", "cifar10", "--exp_name", "h4s1l1"]
        cases = [
            ([], 2, "Missing option"),
            (["--exp_class", "cifar10"], 2, "Missing option"),
            (["--exp_name", "h4s1l1"], 2, "Missing option"),
            (["--exp_class", "missing", "--exp_name", "h4s1l1"], 1, "Unknown dataset"),
            (["--exp_class", "cifar10", "--exp_name", "h99s1l1"], 1, "Unknown experiment"),
            (["--exp", "cifar10_h4s1l1"], 2, "No such option"),
            ([*selection, "--gpus", "-1"], 2, "Invalid value"),
            ([*selection, "--gpus", "1.5"], 2, "Invalid value"),
            ([*selection, "--devices", "1"], 2, "No such option"),
            (["--exp_class", "../cifar10", "--exp_name", "h4s1l1"], 1, "filename stem"),
            (["--exp_class", "cifar10", "--exp_name", "../h4s1l1"], 1, "filename stem"),
        ]
        with patch("builtins.__import__", side_effect=reject_training_import):
            self.assertEqual(runner.invoke(run, ["--help"]).exit_code, 0)
            for arguments, code, message in cases:
                with self.subTest(arguments=arguments):
                    result = runner.invoke(run, arguments)
                    self.assertEqual(result.exit_code, code, result.output)
                    self.assertIn(message, result.output)

    def test_training_limits_reject_ambiguous_or_nonintegral_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            cases = [
                {},
                {"n_epochs": 1, "n_iters": 2},
                {"n_iters": 1.5},
                {"n_iters": 0},
                {"n_epochs": True},
            ]
            with patch.object(c, "EXPERIMENTS_PATH", config_dir):
                for settings in cases:
                    with self.subTest(settings=settings):
                        (config_dir / "invalid.json").write_text(json.dumps({"h1s1l1": settings}))
                        with self.assertRaises(ValueError):
                            train_utils.get_experiment_params("invalid", "h1s1l1")
        params = train_utils.get_experiment_params("shapes3d", "h3s1l1")
        self.assertEqual(train_utils.trainer_limits(params), {"max_steps": 10000})
        self.assertIs(type(params["n_iters"]), int)

    def test_malformed_dataset_or_experiment_objects_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            with patch.object(c, "EXPERIMENTS_PATH", config_dir):
                for settings in [[], None, "not an object"]:
                    with self.subTest(settings=settings):
                        (config_dir / "invalid.json").write_text(json.dumps(settings))
                        with self.assertRaisesRegex(ValueError, "must map experiment names to configuration objects"):
                            train_utils.get_experiment_params("invalid", "h1s1l1")
                for settings in [[], None, "not an experiment"]:
                    with self.subTest(selected=settings):
                        (config_dir / "invalid.json").write_text(json.dumps({"h1s1l1": settings}))
                        with self.assertRaisesRegex(ValueError, "must contain an experiment object"):
                            train_utils.get_experiment_params("invalid", "h1s1l1")
                (config_dir / "invalid.json").write_text(
                    json.dumps({"h1s1l1": {"n_epochs": 1, "dataset_kwargs": []}})
                )
                with self.assertRaisesRegex(ValueError, "dataset_kwargs must be an object"):
                    train_utils.get_experiment_params("invalid", "h1s1l1")

    def test_selection_uses_named_preset_and_lists_available_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "example.json").write_text(json.dumps({
                "h1s1l1": {"n_epochs": 1, "n_groups_hue": 1},
                "h4s1l1": {"n_epochs": 2, "n_groups_hue": 4},
            }))
            with patch.object(c, "EXPERIMENTS_PATH", config_dir):
                params = train_utils.get_experiment_params("example", "h4s1l1")
                self.assertEqual(params["n_groups_hue"], 4)
                self.assertEqual(params["n_epochs"], 2)
                self.assertEqual(params["experiment_name"], "example_h4s1l1")
                with self.assertRaisesRegex(ValueError, "Unknown experiment") as context:
                    train_utils.get_experiment_params("example", "h9s1l1")
                self.assertIn("h1s1l1", str(context.exception))
                self.assertIn("h4s1l1", str(context.exception))
                (config_dir / "empty.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Unknown experiment"):
                    train_utils.get_experiment_params("empty", "h1s1l1")

    def test_both_selectors_require_plain_filename_stems(self):
        for invalid in ("", ".", "..", "../name", "dir/name", "dir\\name"):
            for exp_class, exp_name in ((invalid, "h1s1l1"), ("cifar10", invalid)):
                with self.subTest(dataset=exp_class, experiment=exp_name):
                    with self.assertRaisesRegex(ValueError, "filename stem"):
                        train_utils.get_experiment_params(exp_class, exp_name)

    def test_gpu_modes_pass_loader_keywords_and_contain_artifacts(self):
        # Exercise the real command and config merging without allocating GPUs,
        # downloading data, creating a model, or starting training.
        cases = [
            ([], "gpu", 1, "auto"),
            (["--gpus", "0"], "cpu", 1, "auto"),
            (["--gpus", "1"], "gpu", 1, "auto"),
            (["--gpus", "3"], "gpu", 3, "ddp"),
        ]
        exp_class, exp_name = "stanford_cars", "h4s1l1"
        params = train_utils.get_experiment_params(exp_class, exp_name)
        for arguments, accelerator, devices, strategy in cases:
            with self.subTest(arguments=arguments):
                loader = Mock(return_value=("train", "validation", "test"))
                logger = SimpleNamespace(log_dir=str(c.LOG_DIR / "synthetic" / "version_0"))
                with (
                    patch("torch.cuda.device_count", return_value=4),
                    patch("dataloader.dataloaders.parse_dataloader", return_value=loader),
                    patch("modules.lightning_model.LitHSGCNN") as model,
                    patch("lightning.pytorch.loggers.TensorBoardLogger", return_value=logger) as logging,
                    patch("lightning.pytorch.callbacks.ModelCheckpoint") as checkpoint,
                    patch("lightning.pytorch.Trainer") as trainer,
                ):
                    checkpoint.return_value.best_model_path = ""
                    result = CliRunner().invoke(run, ["--exp_class", exp_class, "--exp_name", exp_name, *arguments])
                    self.assertEqual(result.exit_code, 0, str(result.exception))
                    loader.assert_called_once_with(
                        n_groups_hue=params["n_groups_hue"],
                        n_groups_saturation=params["n_groups_saturation"],
                        n_groups_luminance=params["n_groups_luminance"],
                        batch_size=params["batch_size"],
                        **params["dataset_kwargs"],
                    )
                    logging.assert_called_once_with(save_dir=c.LOG_DIR, name=params["experiment_name"])
                    self.assertEqual(trainer.call_args.kwargs["default_root_dir"], c.OUTPUT_DIR)
                    self.assertEqual(trainer.call_args.kwargs["accelerator"], accelerator)
                    self.assertEqual(trainer.call_args.kwargs["devices"], devices)
                    self.assertEqual(trainer.call_args.kwargs["strategy"], strategy)
                    trainer.return_value.fit.assert_called_once_with(
                        model.return_value, train_dataloaders="train", val_dataloaders="validation"
                    )
                    trainer.return_value.test.assert_called_once_with(
                        model.return_value, dataloaders="test"
                    )
                    self.assertTrue(checkpoint.call_args.kwargs["dirpath"].is_relative_to(c.OUTPUT_DIR))

    def test_insufficient_gpus_fail_before_model_or_loader_creation(self):
        for arguments, available in [([], 0), (["--gpus", "2"], 1)]:
            with self.subTest(arguments=arguments, available=available):
                with (
                    patch("torch.cuda.device_count", return_value=available),
                    patch("dataloader.dataloaders.parse_dataloader") as loader,
                    patch("modules.lightning_model.LitHSGCNN") as model,
                    patch("lightning.pytorch.loggers.TensorBoardLogger") as logging,
                    patch("lightning.pytorch.Trainer") as trainer,
                ):
                    result = CliRunner().invoke(
                        run, ["--exp_class", "cifar10", "--exp_name", "h4s1l1", *arguments]
                    )
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIn("GPU", result.output)
                    self.assertIn(str(available), result.output)
                    loader.assert_not_called()
                    model.assert_not_called()
                    logging.assert_not_called()
                    trainer.assert_not_called()

    def test_all_modes_test_best_checkpoint_and_fall_back_if_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "best.ckpt"
            existing.touch()
            missing = Path(directory) / "missing.ckpt"
            for gpus in (0, 1, 3):
                for checkpoint_path in (existing, missing):
                    with self.subTest(gpus=gpus, checkpoint=checkpoint_path.name):
                        logger = SimpleNamespace(log_dir=str(c.LOG_DIR / "synthetic"))
                        with (
                            patch("torch.cuda.device_count", return_value=4),
                            patch("dataloader.dataloaders.parse_dataloader", return_value=Mock(
                                return_value=("train", "validation", "test")
                            )),
                            patch("modules.lightning_model.LitHSGCNN") as model,
                            patch("lightning.pytorch.loggers.TensorBoardLogger", return_value=logger),
                            patch("lightning.pytorch.callbacks.ModelCheckpoint") as checkpoint,
                            patch("lightning.pytorch.Trainer") as trainer,
                        ):
                            checkpoint.return_value.best_model_path = str(checkpoint_path)
                            result = CliRunner().invoke(run, [
                                "--exp_class", "cifar10", "--exp_name", "h4s1l1", "--gpus", str(gpus)
                            ])
                            self.assertEqual(result.exit_code, 0, str(result.exception))
                            if checkpoint_path == existing:
                                trainer.return_value.test.assert_called_once_with(
                                    ckpt_path=str(existing), dataloaders="test"
                                )
                            else:
                                trainer.return_value.test.assert_called_once_with(
                                    model.return_value, dataloaders="test"
                                )


if __name__ == "__main__":
    unittest.main()
