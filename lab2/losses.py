import torch
from torch import nn, Tensor
from typing import Iterable, Dict


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss used in STS regression.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding']
                      for sentence_feature in sentence_features]
        emb1, emb2 = embeddings
        sim = torch.cosine_similarity(emb1, emb2)
        return (sim - labels.view(-1)) ** 2


class SoftmaxLoss(nn.Module):
    """
    Softmax loss used in NLI classification.
    """

    def __init__(self,
                 model: nn.Module,
                 sentence_embedding_dimension: int,
                 num_labels: int):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(3 * sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding']
                for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        
        features = torch.cat(
            [rep_a, rep_b, torch.abs(rep_a - rep_b)],
            dim = 1)

        return nn.functional.cross_entropy(
            self.classifier(features),
            labels.view(-1))
