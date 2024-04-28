from omegaconf import DictConfig
import hydra
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from dataset.dataset import load_dataloader
from models.models import TokenClassificationModel, Seq2SeqModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM


def load_model_checkpoint(checkpoint_path: str, model, lr, task_type, id2label=None):
    if task_type == 'seq2seq':
        Seq2SeqModel.load_from_checkpoint(checkpoint_path, model=model, lr=lr)
    else:
        TokenClassificationModel.load_from_checkpoint(checkpoint_path, model=model, id2label=id2label, lr=lr) 

@hydra.main(version_base=None, config_path="./confs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(42)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)
    task_type = cfg.task_type

    # datasets
    train_dl, val_dl, test_dl, id2label = load_dataloader(cfg.datasets, cfg.tokenizer, tokenizer, task_type)

    # model
    if task_type == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.models.name)
        model = Seq2SeqModel(model, cfg.optim.lr)
    else:
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
            model = load_model_checkpoint(checkpoint_callback.best_model_path, model, cfg.optim.lr, task_type, id2label)
        elif hasattr(cfg, "checkpoint_file"):
            model = load_model_checkpoint(cfg.checkpoint_file, model, cfg.optim.lr, task_type, id2label)
        trainer = Trainer(logger=wandb_logger, accelerator="gpu", callbacks=[rich], devices=1)
        trainer.test(model, test_dl)

if __name__ == "__main__":
    main()
