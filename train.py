import torch
import argparse

from modules.trainmodule import STSTrainer
from SentenceTransformer.models import SentenceTransformer
from modules.dataloader import DataModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


WEIGHT = 'kykim/bert-kor-base'
ACCELERATOR = 'ddp' if torch.cuda.is_available() else 'cpu'

logger = TensorBoardLogger(
                           save_dir="SBert Logs",
                           name="regression",
                           default_hp_metric=False,

                           )

checkpoint_callback = ModelCheckpoint(
                                     monitor='validation_loss',
                                     dirpath='CKPT_DIR',
                                     filename='KoSBERT_regression',
                                     mode='min'
                                     )

early_stop_callback = EarlyStopping(
                                    monitor='validation_loss',
                                    min_delta=1e-4,
                                    patience=5,
                                    verbose=True,
                                    mode='min'
                                    )

trainer = Trainer(max_epochs=50,
                  gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                  accelerator=ACCELERATOR,
                  callbacks=[early_stop_callback, checkpoint_callback],
                  plugins=DDPPlugin(find_unused_parameters=True),
                  logger=logger,
                  log_every_n_steps=1
                  )


if __name__ == '__main__':
    sentence_bert = SentenceTransformer(is_cls=False,
                                        weight=WEIGHT)
    train_module = STSTrainer(sentence_bert=sentence_bert,
                              loss_function='mse')
    data_module = DataModule(dataset='sts', weight=WEIGHT)
    trainer.fit(model=train_module, datamodule=data_module)
    trainer.test(model=train_module, datamodule=data_module)
