import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_lightning import LightningModule


class BaseTrainer(LightningModule):
    def __init__(self,
                 sentence_bert: nn.Module,
                 loss_function: str = None,
                 learning_rate: float = 1e-4) -> None:
        super(BaseTrainer, self).__init__()
        self.sentence_bert = sentence_bert
        self.learning_rate = learning_rate
        if loss_function == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_function == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("I know what you forgot. But i wouldn't tell you this shit👻.")

    def configure_optimizers(self) -> list:
        optimizer = optim.AdamW(lr=self.learning_rate,
                                params=self.sentence_bert.head.parameters())
        return [optimizer]

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> dict:
        return self.sentence_bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)


class STSTrainer(BaseTrainer):
    def one_step(self, sent1, sent2) -> torch.Tensor:
        sent1_vector = self(input_ids=sent1['input_ids'],
                            attention_mask=sent1['attention_mask'],
                            token_type_ids=sent1['token_type_ids'])
        sent2_vector = self(input_ids=sent2['input_ids'],
                            attention_mask=sent2['attention_mask'],
                            token_type_ids=sent2['token_type_ids'])
        return F.cosine_similarity(sent1_vector, sent2_vector)

    def training_step(self, batch, batch_idx) -> dict:
        sent1, sent2, y = batch
        y_hat = self.one_step(sent1=sent1, sent2=sent2)
        loss = self.loss_fn(y, y_hat)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> dict:
        sent1, sent2, y = batch
        y_hat = self.one_step(sent1=sent1, sent2=sent2)
        loss = self.loss_fn(y, y_hat)
        self.log('validation_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx) -> None:
        sent1, sent2, y = batch
        y_hat = self.one_step(sent1=sent1, sent2=sent2)
        loss = self.loss_fn(y, y_hat)
        self.log('test_loss', loss, on_step=False, on_epoch=True)


class NLITrainer(BaseTrainer):
    def training_step(self, batch, batch_idx) -> dict:
        embeddings = [b for b in batch]
        loss = self.loss_fn(embeddings)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> dict:
        embeddings = [b for b in batch]
        loss = self.loss_fn(embeddings)
        self.log('validation_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx) -> None:
        embeddings = [b for b in batch]
        loss = self.loss_fn(embeddings)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
