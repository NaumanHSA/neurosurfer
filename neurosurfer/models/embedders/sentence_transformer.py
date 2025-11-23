from typing import List, Union, Optional
import logging
import numpy as np
from abc import ABC, abstractmethod
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, models
from .base import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None, quantized: bool = True):
        super().__init__(model_name, logger)
        if quantized:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="auto")
            transformer_model = models.Transformer(
                model_name, model_args={"quantization_config": bnb_config}
            )
            pooling_model = models.Pooling(
                word_embedding_dimension=transformer_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            self.model = SentenceTransformer(modules=[transformer_model, pooling_model])
        else:
            self.model = SentenceTransformer(model_name)
        self.logger.info("SentenceTransformer embedding model initialized.")

    def embed(
        self,
        query: Union[str, List[str]],
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
    ) -> Union[List[float], List[List[float]]]:
        return self.model.encode(
            query,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings
        ).tolist()
