import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from codesign.engines.callbacks import SilentValidationProgressBar, ValidationCallback, TestCallback


class TrainValTest:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, model, data_module):
        # checkpoint saving config
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.exp_dir,
            filename=f'{{epoch}}-{{val_{model.val_test_loss.name}:.2f}}',
            monitor=f'val_{model.val_test_loss.name}',
            mode=model.val_test_loss.mode, 
            verbose=False
        )
        early_stop_callback = EarlyStopping(
            monitor=f'val_{model.val_test_loss.name}',
            patience=10, 
            mode=model.val_test_loss.mode, 
            verbose=False
        )

        # train model
        trainer = pl.Trainer(
            accelerator='auto', 
            devices=1, 
            logger=WandbLogger(**dict(self.cfg.logger)),
            callbacks=[
                SilentValidationProgressBar(),
                checkpoint_callback, 
                early_stop_callback,
                ValidationCallback(self.cfg.exp_dir, self.cfg.val_epoch_vis_freq),
                TestCallback(self.cfg.exp_dir, self.cfg.test_batch_vis_freq)
            ],
            **dict(self.cfg.trainer),
        )

        # train/val model
        trainer.fit(model, data_module)

        # test the model
        # trainer.test(model, data_module, ckpt_path='best')