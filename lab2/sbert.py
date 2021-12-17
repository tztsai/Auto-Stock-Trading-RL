# %%
import os
import logging
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from sentence_transformers import models, SentenceTransformer

train_batch_size = 128
max_seq_length = 75
num_epochs = 1


class SBert(SentenceTransformer):
    """ Sentence BERT

    A SentenceTransformer model based on BERT, that can be used to map sentences / text to embeddings.
    """
    tokenizer_name = 'bert-base-uncased'
    model_name = 'bert-base-uncased'
    _cache_dir = 'model_cache'

    def __init__(self,
                 model_path=None,
                 device: Optional[str] = None,
                 max_seq_length: Optional[int] = None,
                 do_lower_case: bool = False,
                 cache_dir: Optional[str] = _cache_dir):
        if model_path is not None:
            return super().__init__(model_path)
        
        transformer = models.Transformer(
            model_name_or_path=self.model_name,
            tokenizer_name_or_path=self.tokenizer_name,
            do_lower_case=do_lower_case,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir
        )
        pooling = models.Pooling(transformer.get_word_embedding_dimension(),
                                 pooling_mode='mean')
        
        super().__init__(modules=[transformer, pooling],
                         device=device)
        