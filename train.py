from modules.trainmodule import STSTrainer
from SentenceTransformer.models import SentenceTransformer
from modules.dataloader import DataModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

WEIGHT = 'kykim/bert-kor-base'

logger = TensorBoardLogger(
                           save_dir="SBert Logs",
                           name="regression",
                           default_hp_metric=False,
                           )

checkpoint_callback = ModelCheckpoint(
                                     monitor='val_loss',
                                     dirpath='CKPT_DIR',
                                     filename='KoSBERT_regression',
                                     mode='min'
                                     )

early_stop_callback = EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=1e-4,
                                    patience=20,
                                    verbose=True,
                                    mode='min'
                                    )

trainer = Trainer(max_epochs=50,
                  gpus=2,
                  accelerator='ddp',
                  callbacks=[early_stop_callback,
                             checkpoint_callback],
                  plugins=DDPPlugin(find_unused_parameters=True),
                  logger=logger
                  )


if __name__ == '__main__':
    sentence_bert = SentenceTransformer(is_cls=False,
                                        weight=WEIGHT)
    train_module = STSTrainer(sentence_bert=sentence_bert,
                              loss_function='cross_entropy')
    data_module = DataModule(dataset='sts', weight=WEIGHT)
    trainer.fit(model=train_module, datamodule=data_module)
    trainer.test(model=train_module, datamodule=data_module)
