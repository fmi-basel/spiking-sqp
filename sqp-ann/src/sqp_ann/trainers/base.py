import os
import logging
import torch
import numpy as np
from torch import optim
from omegaconf import OmegaConf
from tqdm import tqdm
from sqp_ann.utils import compute_metrics

logger = logging.getLogger(__name__)


class BaseTrainer():
    def __init__(self, all_config, accelerator, epochs, valid_every, lr, debug_mode, callbacks=[]):
        self.all_config = OmegaConf.to_container(all_config, resolve=True)
        self.accelerator = accelerator
        self.epochs = epochs
        self.valid_every = valid_every
        self.lr = lr
        self.optimizer = None
        self.callbacks = callbacks
        self._dispatch_callbacks('on_init', trainer=self)
        self.stop_triggered = False
        if debug_mode:
            torch.autograd.set_detect_anomaly(True)


    def _train_epoch(self, model, dataloader_train):
        losses = []
        # loop over data
        for data in dataloader_train:
            self.optimizer.zero_grad()
            loss = model.train_step(data[0], data[1])
            if loss.isnan().any():
                raise FloatingPointError
            # backprop
            self.accelerator.backward(loss)
            self.optimizer.step()
            # store loss
            curr_loss = loss.item()
            losses.append(curr_loss)
            # aggregate and return
            avg_loss = np.mean(losses)
        return avg_loss


    def _validate(self, model, dataloader_valid):
        pred_list = []
        true_list = []
        losses = []
        # loop over data
        for data in dataloader_valid:
            # move data to cuda device
            data = [d.to(self.accelerator.device) for d in data]
            # process minibatch
            mos_pred, mos_true, loss = model.valid_step(data[0], data[1])
            # store predictions and targets
            pred_list.append(mos_pred.cpu())
            true_list.append(mos_true.cpu())
            # store loss
            curr_loss = loss.item()
            losses.append(curr_loss)
        # reshape predictions and target
        pred_tensor = torch.hstack(pred_list)
        true_tensor = torch.hstack(true_list)
        # compute metrics
        avg_metrics = compute_metrics(pred_tensor, true_tensor, colour="blue")
        # aggregate loss and return
        avg_metrics["loss"] = np.mean(losses)
        return avg_metrics


    def train(self, model, dataloader_train, dataloader_valid, checkpoint_path=None):
        # init optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # move stuff to cuda device(s)
        dataloader_train, model, self.optimizer = self.accelerator.prepare(
            dataloader_train, model, optimizer)
        # load previous state (if specified)
        if checkpoint_path:
            logger.info(f"Loading state from {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
        # callbacks
        self._dispatch_callbacks('on_training_start', model=model,
            dataloader_train=dataloader_train, dataloader_valid=dataloader_valid)
        # training loop (check for ctrl+c)
        logger.info(f"Training for {self.epochs} epochs...")
        try:
            with tqdm(range(1, self.epochs + 1)) as epoch_pbar:
                for epoch_idx in epoch_pbar:
                    # train 1 epoch
                    model.train()
                    avg_loss = self._train_epoch(model, dataloader_train)
                    # callbacks
                    self._dispatch_callbacks('on_training_epoch_end', i=epoch_idx, avg_epoch_loss=avg_loss)
                    # stop if a callback requested that
                    if self.stop_triggered:
                        break
                    # run validation on 1st, last, and every N epochs
                    if epoch_idx % self.valid_every == 0 or epoch_idx == 1 or epoch_idx == self.epochs:
                        model.eval()
                        # callbacks
                        self._dispatch_callbacks('on_validation_start', i=epoch_idx)
                        # actual validation function
                        avg_metrics = self._validate(model, dataloader_valid)
                        # callbacks
                        self._dispatch_callbacks('on_validation_end', i=epoch_idx, avg_valid_metrics=avg_metrics)
                        # stop if a callback requested that
                        if self.stop_triggered:
                            break
        except KeyboardInterrupt:
            logger.warning("User interrupted training, wrapping things up...")
        except FloatingPointError:
            logger.warning("Loss is NaN, wrapping things up...")
        finally:
            # run callbacks even if user terminated 
            self._dispatch_callbacks('on_training_end')
            logger.info(f"Training is over, find hydra logs at {os.getcwd()}")


    def test(self, model, dataloader_test, checkpoint_path=None):
        pred_list = []
        true_list = []
        # move model to cuda device(s)
        self.accelerator.clear()
        model = self.accelerator.prepare(model)
        # load previous state (if specified)
        if checkpoint_path:
            logger.info(f"Loading state from {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
        logger.info(f"Testing...")
        # callbacks
        self._dispatch_callbacks(
            'on_testing_start', 
            model=model, 
            dataloader_test=dataloader_test)
        # loop over data (check for ctrl+c)
        try:
            for data in dataloader_test:
                # move data to cuda device
                data = [d.to(self.accelerator.device) for d in data]
                # process minibatch
                mos_pred, mos_true = model.test_step(data[0], data[1])
                # store predictions and targets
                pred_list.append(mos_pred.cpu())
                true_list.append(mos_true.cpu())
        except KeyboardInterrupt:
            logger.warning("User interrupted testing, returning")
            return {}
        # reshape predictions and target
        pred_tensor = torch.hstack(pred_list)
        true_tensor = torch.hstack(true_list)
        # compute metrics
        avg_metrics = compute_metrics(pred_tensor, true_tensor, colour="green")
        # callbacks
        self._dispatch_callbacks('on_testing_end', avg_test_metrics=avg_metrics)
        logger.info(f"Testing is over.")
        return avg_metrics


    def _dispatch_callbacks(self, hook_name, **kwargs):
        for cb in self.callbacks:
            getattr(cb, hook_name)(**kwargs)


    def _trigger_stop(self):
        self.stop_triggered = True
