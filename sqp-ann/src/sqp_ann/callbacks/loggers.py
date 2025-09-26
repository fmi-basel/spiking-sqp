import os
import logging
import hydra
import wandb
import torchaudio
from flatdict import FlatDict
from sqp_ann.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CliLogger(BaseCallback):
    def on_training_epoch_end(self, i, avg_epoch_loss, **kw_args):
        if "stage" in kw_args:
            j = kw_args["stage"]
            logger.info(f"Train stage {j} epoch {i}/{self.trainer.epochs}, loss = {avg_epoch_loss:.3f}")
        else:
            logger.info(f"Train {i}/{self.trainer.epochs}, loss = {avg_epoch_loss:.3f}")

    def on_validation_end(self, i, avg_valid_metrics, **kw_args):
        metrics_str = ", ".join([f"{k} = {v:.3f}" for k, v in avg_valid_metrics.items()])
        if "stage" in kw_args:
            j = kw_args["stage"]
            logger.info(f"Validation at stage {j} epoch {i}: {metrics_str}")
        else:
            logger.info(f"Validation at epoch {i}: {metrics_str}")

    def on_testing_end(self, avg_test_metrics, **kw_args):
        metrics_str = ", ".join([f"{k} = {v:.3f}" for k, v in avg_test_metrics.items()])
        logger.info(f"Test results: {metrics_str}")


class WandbLogger(BaseCallback):
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
        
    def on_training_start(self, model, dataloader_train, dataloader_valid):
        super().on_training_start(model, dataloader_train, dataloader_valid)
        # setup run
        #wandb.init(project=self.experiment_name, name=self.run_name, config=self.trainer.all_config)
        kwargs = dict(project=self.experiment_name, config=self.trainer.all_config)
        if self.run_name is not None:
            kwargs["name"] = self.run_name
        wandb.init(**kwargs)

    def on_training_end(self, **kw_args):
        pass

    def on_training_epoch_end(self, i, avg_epoch_loss, **kw_args):
        if "stage" in kw_args:
            j = kw_args["stage"]
            wandb.log({ f"stage_{j}/train/loss" : avg_epoch_loss }, step=i)
        else:
            wandb.log( { 'train/loss' : avg_epoch_loss}, step=i)

    def on_validation_end(self, i, avg_valid_metrics, **kw_args):
        if "stage" in kw_args:
            j = kw_args["stage"]
            metrics = {f"stage_{j}/valid/{k}": v for k, v in avg_valid_metrics.items()}
        else:
            metrics = {f"valid/{k}": v for k, v in avg_valid_metrics.items()}
        wandb.log(metrics, step=i)

    def on_testing_start(self, model, dataloader_test, **kw_args):
        super().on_testing_start(model, dataloader_test)
    
    def on_testing_end(self, avg_test_metrics, **kw_args):
        metrics = {f"test/{k}": v for k, v in avg_test_metrics.items()}
        wandb.log(metrics)
        wandb.finish()
