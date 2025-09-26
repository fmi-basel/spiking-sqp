import logging
from sqp_ann.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class LrOnPlateau(BaseCallback):
    def __init__(self, patience, delta, factor, run_every, verbose):
        self.patience = patience
        self.delta = delta
        self.factor = factor
        self.run_every = run_every
        self.best_loss = None
        self.patience_counter = patience
        assert self.run_every in ['epoch', 'valid'], "'run_every' must be either 'epoch' or 'valid'"
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def on_training_epoch_end(self, avg_epoch_loss, **kw_args):
        if self.run_every == 'epoch':
            self._check(avg_epoch_loss)

    def on_validation_end(self, avg_valid_metrics, **kw_args):
        loss = avg_valid_metrics["loss"]
        if self.run_every == 'valid':
            self._check(loss)

    def _check(self, curr_loss):
        if self.best_loss is None or self.best_loss - curr_loss > self.delta:
            self.patience_counter = self.patience
            self.best_loss = curr_loss
        else:
            self.patience_counter -= 1
            logger.info(f"Loss delta: {(self.best_loss - curr_loss):.3f} < {self.delta}, new patience = {self.patience_counter}")

        if self.patience_counter <= 0:
            for param_group in self.trainer.optimizer.param_groups:
                old_lr = float(param_group['lr'])
                new_lr = old_lr * self.factor
                param_group['lr'] = new_lr
            logger.info(f"No more patience, decreasing learning rate to {new_lr:.2e}")
            self.patience_counter = self.patience

    def reset(self):
        self.best_loss = None
        self.patience_counter = self.patience
