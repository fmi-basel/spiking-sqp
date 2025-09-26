import numpy as np
import time
from tqdm import tqdm

from stork.models import RecurrentSpikingModel


class CustomRecurrentSpikingModel(RecurrentSpikingModel):
    def fit_validate(
        self,
        dataset,
        valid_dataset,
        nb_epochs=10,
        verbose=True,
        wandb=None,
        early_stopper=None,
        early_stop_metric="loss",
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []

        if early_stopper is not None:
            metric_names = self.get_metric_names()
            if early_stop_metric not in metric_names:
                print(f"Warning: {early_stop_metric} not found in metrics. Using 'loss' instead.")
                early_stop_metric = "loss"
            metric_idx = metric_names.index(early_stop_metric)
            early_stopper.reset()

        for ep in tqdm(range(nb_epochs)):
            t_start = time.time()
            
            self.train()
            ret_train = self.train_epoch(dataset)

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            if self.wandb is not None:
                self.wandb.log({key: value for (key, value) in zip(
                    self.get_metric_names() + self.get_metric_names(prefix="val_"),
                    ret_train.tolist() + ret_valid.tolist())})

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print("%02i %s --%s t_iter=%.2f"% (
                    ep,
                    self.get_metrics_string(ret_train),
                    self.get_metrics_string(ret_valid, prefix="val_"),
                    t_iter,
                ))

            if early_stopper is not None:
                if early_stopper(ret_valid[metric_idx], self):
                    print(f"\nEarly stopping triggered after epoch {ep}")
                    early_stopper.restore_best_state(self)
                    break

        self.hist = np.concatenate((np.array(self.hist_train),
                                    np.array(self.hist_valid)))
        self.fit_runs.append(self.hist)
        
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        
        history = {**dict1, **dict2}
        return history
