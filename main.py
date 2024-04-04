from omegaconf import DictConfig
import hydra
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from dataset.dataset import load_dataloader
from models.models import TokenClassificationModel
from transformers import AutoTokenizer, AutoModelForTokenClassification


@hydra.main(version_base=None, config_path="./confs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)

    # datasets
    train_dl, val_dl, test_dl, id2label = load_dataloader(cfg.datasets, cfg.tokenizer, tokenizer)

    # model
    model = AutoModelForTokenClassification.from_pretrained(cfg.models.name, num_labels=len(list(id2label.keys())))
    model = TokenClassificationModel(model, id2label, cfg.optim.lr)
    
    # wandb logger for monitoring
    wandb_logger = WandbLogger(**cfg.loggers)

    # callbacks
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    early_stop = EarlyStopping(**cfg.earlystopping)
    rich = RichProgressBar()
    lr_monitor = LearningRateMonitor(**cfg.lr_monitor)
    callbacks = [checkpoint_callback, early_stop, lr_monitor, rich]

    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)

    if cfg.do_train:
        trainer.fit(model, train_dl, val_dl)
    
    if cfg.do_test:
        if cfg.do_train:
            model = TokenClassificationModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=model.model, id2label=id2label, lr=cfg.optim.lr)
        elif hasattr(cfg, "checkpoint_file"):
            model = TokenClassificationModel.load_from_checkpoint(cfg.checkpoint_file, model=model.model, id2label=id2label, lr=cfg.optim.lr)
        trainer = Trainer(logger=wandb_logger, accelerator="gpu", callbacks=[rich], devices=1)
        trainer.test(model, test_dl)

if __name__ == "__main__":
    main()
