import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel


class LanguageModel(nn.Module):
    def __init__(self, weight: str = None) -> None:
        super(LanguageModel, self).__init__()
        self.model = AutoModel.from_pretrained(weight)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask, token_type_ids)


class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super(MeanPooling, self).__init__()

    def __repr__(self):
        return "MeanPooling()"

    def forward(self,
                features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vector = sum_embeddings / sum_mask
        features['sentence_embedding'] = output_vector
        return features
