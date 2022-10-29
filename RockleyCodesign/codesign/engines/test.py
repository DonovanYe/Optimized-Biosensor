import pytorch_lightning as pl
from codesign.engines.callbacks import TestCallback
from pytorch_lightning.loggers import WandbLogger

class Test:
    def __init__(self, cfg, data_cfg) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg

    def test_run(self, args, model, data_module) -> None:
        logger = WandbLogger(id=args.id, **dict(self.cfg.logger)) if args.id else False
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            logger=logger,
            callbacks=[TestCallback(self.cfg.exp_dir, self.data_cfg.vis_freq)]
        )
        trainer.test(model, data_module)