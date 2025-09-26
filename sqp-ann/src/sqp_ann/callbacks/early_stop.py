import logging
from sqp_ann.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EarlyStop(BaseCallback):
    def __init__(self, metric, patience, delta, run_every, verbose):
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.run_every = run_every
        self.best_score = None
        self.patience_counter = patience
        assert self.run_every in ['epoch', 'valid'], "'run_every' must be either 'epoch' or 'valid'"
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def on_training_epoch_end(self, avg_epoch_loss, **kw_args):
        if self.run_every == 'epoch':
            self._check(avg_epoch_loss)

    def on_validation_end(self, avg_valid_metrics, **kw_args):
        if self.metric == 'pcc':
            metric_value = -avg_valid_metrics["pcc"]
        else:
            metric_value = avg_valid_metrics["loss"]
            
        if self.run_every == 'valid':
            self._check(metric_value)
    
    def _check(self, curr_score):
        if self.best_score is None or self.best_score - curr_score > self.delta:
            self.patience_counter = self.patience
            self.best_score = curr_score
        else:
            self.patience_counter -= 1
            logger.info(f"Loss delta: {(self.best_score - curr_score):.3f} < {self.delta}, new patience = {self.patience_counter}")

        if self.patience_counter <= 0:
            logger.info("No more patience, stopping")
            self.trainer._trigger_stop()


    def reset(self):
        self.best_score = None
        self.patience_counter = self.patience
