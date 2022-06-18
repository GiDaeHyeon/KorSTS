import torch
import torch.nn as nn
from layers import MeanPooling, LanguageModel


class SentenceTransformer(nn.Module):
    def __init__(self, is_cls: bool, weight: str = None, freeze: bool = True):
        super(SentenceTransformer, self).__init__()
        self.language_model = LanguageModel(weight=weight)

        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(p=.1, inplace=False),
            nn.Linear(in_features=768, out_features=768)
        )
        self.pooling = MeanPooling()
        self.is_cls = is_cls

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        if self.is_cls:
            output = self.language_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids).pooler_output
        else:
            output = self.language_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids).last_hidden_state
        output = self.head(output)
        features = {'token_embeddings': output, 'attention_mask': attention_mask}
        return self.pooling(features)
