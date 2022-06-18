import torch
from torch.cuda import device_count
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from transformers import AutoTokenizer
from datasets import load_dataset


class BaseDataset(Dataset):
    def __init__(self, dataset: str, phase: str = None, weight: str = None) -> None:
        super(BaseDataset, self).__init__()
        if phase is None:
            raise NotImplementedError('phase must be train, validation or test')

        if weight is None:
            raise NotImplementedError('weight must be declared.')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(weight)

        if dataset not in ['nli', 'sts']:
            raise NotImplementedError('dataset must be declared (sts or nli)')

        if phase == 'train':
            self.dataset = load_dataset("klue", dataset, split='train[:90%]')
        elif phase == 'validation':
            self.dataset = load_dataset("klue", dataset, split='train[-10%:]')
        elif phase == 'test':
            self.dataset = load_dataset("klue", dataset, split='validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def tokenize(self, text: str, padding: str = 'max_length', max_length: int = 128) -> dict:
        tokens = self.tokenizer.encode_plus(text=text, padding=padding, max_length=max_length)
        return {'input_ids': torch.tensor(tokens.get('input_ids')),
                'token_type_ids': torch.tensor(tokens.get('token_type_ids')),
                'attention_mask': torch.tensor(tokens.get('attention_mask'))}


class NLIDataset(BaseDataset):
    def __getitem__(self, idx) -> tuple:
        sentence1, sentence2 = self.dataset[idx]
        sentence1 = self.tokenize(sentence1['premise'])
        sentence2 = self.tokenize(sentence2['hypothesis'])
        label = self.dataset['label']
        return sentence1, sentence2, label


class STSDataset(BaseDataset):
    def __getitem__(self, idx) -> tuple:
        sentences = self.dataset[idx]
        sentence1 = self.tokenize(sentences['sentence1'])
        sentence2 = self.tokenize(sentences['sentence2'])
        label = sentences.get('labels').get('real-label')
        return sentence1, sentence2, torch.tensor(label).float()


class DataModule(LightningDataModule):
    def __init__(self,
                 dataset: str,
                 batch_size=512,
                 num_workers=None,
                 weight=None) -> None:
        super(DataModule, self).__init__()
        if dataset == 'sts':
            self.train_set = STSDataset(dataset=dataset, phase='train', weight=weight)
            self.val_set = STSDataset(dataset=dataset, phase='validation', weight=weight)
            self.test_set = STSDataset(dataset=dataset, phase='test', weight=weight)
        elif dataset == 'nli':
            self.train_set = NLIDataset(dataset=dataset, phase='train', weight=weight)
            self.val_set = NLIDataset(dataset=dataset, phase='validation', weight=weight)
            self.test_set = NLIDataset(dataset=dataset, phase='test', weight=weight)
        self.batch_size = batch_size

        if device_count() == 0 and num_workers is None:
            self.num_workers = 1
        elif device_count() != 0 and num_workers is None:
            self.num_workers = device_count() * 4
        else:
            self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_set,
                          batch_size=self.batch_size * 2,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_set,
                          batch_size=self.batch_size * 2,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=False)

