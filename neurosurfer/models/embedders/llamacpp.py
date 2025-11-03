from typing import List, Union, Optional
import logging
import numpy as np
from llama_cpp import Llama
from .base import BaseEmbedder


class LlamaCppEmbedder(BaseEmbedder):
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        
        self.model = Llama(
            model_path=config['model_path'],
            n_threads=config.get('n_threads', 4),
            verbose=config.get('verbose', False),
            n_gpu_layers=config.get('n_gpu_layers', 0),
            main_gpu=config.get('main_gpu', 0),
            embedding=True,
        )
        self.logger.info("Llama.cpp embedding model initialized.")

    def embed(
        self, query: Union[str, List[str]], normalize_embeddings: bool = True
    ) -> Union[List[float], List[List[float]]]:
        embedding = self.model.embed(query)
        if normalize_embeddings:
            embedding = np.array(embedding)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embedding = embedding.tolist()
        return embedding
