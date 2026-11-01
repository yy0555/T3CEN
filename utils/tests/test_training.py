"""Run the public training CLI on synthetic data with a real CPU Trainer."""

from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner
import lightning.pytorch as L
from lightning.pytorch.callbacks import ProgressBar
import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.train import run
from modules.networks.resnet import BasicBlock, ResNet
from utils import constants as c
from utils.train_utils import get_experiment_params


class TrainingSmokeTests(unittest.TestCase):
    def test_cpu_cli_trains_and_tests_best_checkpoint(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        params = get_experiment_params("cifar10", "h4s1l1")
        params.update(
            n_groups_hue=1, n_groups_saturation=1, n_groups_luminance=1,
            n_epochs=1, batch_size=2,
        )
        small_resnet = partial(ResNet, BasicBlock, [1, 1], shapes=[4, 8])
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])),
            batch_size=2,
        )
        trainers = []
        trainer_arguments = []
        test_results = []
        test_case = self

        class BoundedTrainer(L.Trainer):
            def __init__(self, **kwargs):
                trainer_arguments.append(kwargs.copy())
                # Keep the actual CLI logger and checkpoint callbacks. Disable
                # only progress rendering and bound work to one synthetic batch.
                kwargs["callbacks"] = [
                    callback for callback in kwargs.get("callbacks", [])
                    if not isinstance(callback, ProgressBar)
                ]
                kwargs.update(
                    limit_train_batches=1, limit_val_batches=1,
                    limit_test_batches=1, num_sanity_val_steps=0,
                    enable_progress_bar=False, enable_model_summary=False,
                    log_every_n_steps=1,
                )
                super().__init__(**kwargs)
                trainers.append(self)

            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                parameter = next(self.lightning_module.net.parameters())
                self.saved_parameter = parameter.detach().clone()
                # The CLI must restore its best checkpoint before testing,
                # rather than silently evaluate the current in-memory weights.
                with torch.no_grad():
                    parameter.add_(1)
                return result

            def test(self, *args, **kwargs):
                test_case.assertTrue(kwargs.get("ckpt_path"))
                test_case.assertFalse(torch.equal(
                    next(self.lightning_module.net.parameters()), self.saved_parameter
                ))
                kwargs["verbose"] = False
                result = super().test(*args, **kwargs)
                torch.testing.assert_close(
                    next(self.lightning_module.net.parameters()), self.saved_parameter,
                    rtol=0, atol=0,
                )
                test_results.extend(result)
                return result

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                patch.object(c, "LOG_DIR", output_dir / "logs"),
                patch.object(c, "OUTPUT_DIR", output_dir),
                patch("utils.train_utils.get_experiment_params", return_value=params),
                patch("modules.networks.networks.parse_name", return_value=small_resnet),
                patch("dataloader.dataloaders.parse_dataloader") as data_factory,
                patch("lightning.pytorch.Trainer", BoundedTrainer),
            ):
                data_factory.return_value.return_value = (loader, loader, loader)
                result = CliRunner().invoke(
                    run, ["--exp_class", "cifar10", "--exp_name", "h4s1l1", "--gpus", "0"]
                )
            self.assertEqual(result.exit_code, 0, msg=f"{result.output}\n{result.exception!r}")
            self.assertEqual(len(trainers), 1)
            self.assertEqual(trainer_arguments[0]["accelerator"], "cpu")
            self.assertEqual(trainer_arguments[0]["devices"], 1)
            self.assertNotEqual(trainer_arguments[0].get("strategy"), "ddp")
            trainer = trainers[0]
            self.assertEqual(trainer.global_step, 1)
            self.assertIn("test_accuracy", test_results[0])
            checkpoint_path = Path(trainer.checkpoint_callback.best_model_path)
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(checkpoint_path.is_relative_to(output_dir / "logs"))
            self.assertTrue(list(output_dir.rglob("events.out.tfevents.*")))
            self.assertTrue(Path(trainer.default_root_dir).is_relative_to(output_dir))
            data_factory.assert_called_once_with(params["dataset_name"])


if __name__ == "__main__":
    unittest.main()
