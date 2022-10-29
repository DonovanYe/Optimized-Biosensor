import wandb, torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from codesign.utils.wandb_imshow import wandb_imshow


class LitCoDesign(pl.LightningModule):
    def __init__(self, cfg, sampler, predictor, train_loss, val_test_loss):
        super().__init__()
        self.cfg = cfg
        self.sampler = sampler
        self.predictor = predictor
        self.train_loss = train_loss
        self.val_test_loss = val_test_loss
        self.save_hyperparameters(ignore=['sampler', 'predictor', 'train_loss', 'val_test_loss'])

    @property
    def name(self):
        return self.cfg.model.name

    @property
    def task(self):
        return self.cfg.task

    def _calc_loss_by_task(self, pred, value, label):
        args_dict = {
            'reg': (pred, value),
            'cls': (pred, label)
        }
        train_args = args_dict[self.train_loss.task]
        val_test_args = args_dict[self.val_test_loss.task]
        train_loss = self.train_loss(*train_args)
        val_or_test_loss = self.val_test_loss(*val_test_args)
        return train_loss, val_or_test_loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        measurements, value, label = batch
        measurements_selected = self.sampler(measurements)
        pred = self.predictor(measurements_selected)
        pred_train_loss, pred_val_test_loss = self._calc_loss_by_task(pred, value, label)
        return {
            'loss': pred_train_loss,
            'val_test_loss': pred_val_test_loss
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # log and visualize
        pred_train_loss, pred_val_test_loss = outputs['loss'], outputs['val_test_loss']
        accuracy_dict = {
            f'train_{self.train_loss.name}': pred_train_loss.item(),
            f'train_{self.val_test_loss.name}': pred_val_test_loss.item()
        }
        self.log_dict(accuracy_dict, on_step=False, on_epoch=True) # change the default on_step, on_epoch options for "on_train_batch_end" hook

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        measurements, value, label = batch
        measurements_selected = self.sampler(measurements)
        pred = self.predictor(measurements_selected)
        pred_train_loss, pred_val_loss = self._calc_loss_by_task(pred, value, label)
        return pred_train_loss, pred_val_loss, measurements_selected, pred, label

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log
        pred_train_loss, pred_val_loss, _, _, _ = outputs
        accuracy_dict = {
            f'val_{self.train_loss.name}': pred_train_loss.item(),
            f'val_{self.val_test_loss.name}': pred_val_loss.item()
        }
        self.log_dict(accuracy_dict)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        measurements, value, label = batch
        measurements_selected = self.sampler(measurements)
        pred = self.predictor(measurements_selected)
        pred_train_loss, pred_test_loss = self._calc_loss_by_task(pred, value, label)
        return pred_train_loss, pred_test_loss, measurements_selected, pred, label

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log
        pred_train_loss, pred_test_loss, _, _, _ = outputs
        accuracy_dict = {
            f'test_{self.train_loss.name}': pred_train_loss.item(),
            f'test_{self.val_test_loss.name}': pred_test_loss.item()
        }
        self.log_dict(accuracy_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer
