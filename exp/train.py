"""Train a named experiment on a CPU or a chosen number of GPUs."""

from pathlib import Path

import click

from utils import constants as c
from utils import train_utils


@click.command()
@click.option("--exp_class", required=True, help="Dataset JSON filename stem under exp/configs/.")
@click.option("--exp_name", required=True, help="Lifting preset inside the dataset JSON, e.g. h4s1l1.")
@click.option(
    "--gpus", type=click.IntRange(min=0), default=1, show_default=True,
    help="Number of CUDA GPUs to use; 0 runs on CPU. Uses DDP with more than one GPU.",
)
def run(exp_class, exp_name, gpus):
    try:
        params = train_utils.get_experiment_params(exp_class, exp_name)
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error

    if gpus:
        import torch

        available = torch.cuda.device_count()
        if gpus > available:
            raise click.ClickException(
                f"Requested {gpus} GPU(s), but only {available} CUDA GPU(s) are available. "
                "Use --gpus 0 to run on CPU."
            )

    import lightning.pytorch as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    from dataloader.dataloaders import parse_dataloader
    from modules import lightning_model

    logger = TensorBoardLogger(save_dir=c.LOG_DIR, name=params["experiment_name"])
    resume = params["resume"]
    if resume:
        model = lightning_model.LitHSGCNN.load_from_checkpoint(resume, params=params)
    else:
        model = lightning_model.LitHSGCNN(params)

    get_data_loader = parse_dataloader(params["dataset_name"])
    train_loader, val_loader, test_loader = get_data_loader(
        n_groups_hue=params["n_groups_hue"],
        n_groups_saturation=params["n_groups_saturation"],
        n_groups_luminance=params["n_groups_luminance"],
        batch_size=params["batch_size"],
        **params["dataset_kwargs"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / "checkpoints",
        filename=params["experiment_name"] + "-best",
        monitor=params.get("monitor_metric", "val_accuracy"),
        mode=params.get("monitor_mode", "max"),
        save_top_k=1,
        save_last=False,
    )
    trainer = L.Trainer(
        **train_utils.trainer_limits(params),
        accelerator="gpu" if gpus else "cpu",
        devices=gpus if gpus else 1,
        strategy="ddp" if gpus > 1 else "auto",
        logger=logger,
        default_root_dir=c.OUTPUT_DIR,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).is_file():
        click.echo(f"Testing with best checkpoint: {best_ckpt}")
        trainer.test(ckpt_path=best_ckpt, dataloaders=test_loader)
    else:
        click.echo("No best checkpoint found. Testing the current model.")
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    run()
