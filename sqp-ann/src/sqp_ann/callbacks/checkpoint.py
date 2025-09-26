import os
import shutil
import logging
from sqp_ann.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class Checkpoint(BaseCallback):
    def __init__(self, store_best, store_latest, verbose):
        self.store_best = store_best
        self.store_latest = store_latest
        self.prev_latest_path = None
        self.best_loss = None
        self.best_path = None
        logging.getLogger("accelerate.checkpointing").setLevel(logging.WARNING)
        logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def on_training_epoch_end(self, i, **kw_args):
        if self.store_latest:
            if "stage" in kw_args:
                j = kw_args["stage"]
                latest_path = os.path.join(os.getcwd(), 'checkpoints', f'stage_{j:02d}', f'epoch_{i:04d}')
            else:
                latest_path = os.path.join(os.getcwd(), 'checkpoints', f'epoch_{i:04d}')
            if self.prev_latest_path:
                shutil.rmtree(self.prev_latest_path)
            self.trainer.accelerator.save_state(latest_path)
            logger.info(f"Latest checkpoint stored at {latest_path}")
            self.prev_latest_path = latest_path

    def on_validation_end(self, i, avg_valid_metrics, **kw_args):
        loss = avg_valid_metrics["loss"]
        if self.store_best:
            if self.best_loss is None or loss < self.best_loss:
                if "stage" in kw_args:
                    j = kw_args["stage"]
                    self.best_path = os.path.join(os.getcwd(), 'checkpoints', f'stage_{j:02d}', 'best')
                else:
                    self.best_path = os.path.join(os.getcwd(), 'checkpoints', 'best')
                self.trainer.accelerator.save_state(self.best_path)
                logger.info(f"New best checkpoint stored at {self.best_path}")
                self.best_loss = loss

    def reset(self):
        self.prev_latest_path = None
        self.best_loss = None
        self.best_path = None
