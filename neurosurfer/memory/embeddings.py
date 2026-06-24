"""Shim — embedders promoted to neurosurfer.embeddings (F1).

Memory still imports Embedder/get_embedder/NullEmbedder from here; all
symbols now live in the shared neurosurfer.embeddings module.
"""

from neurosurfer.embeddings import Embedder, NullEmbedder, _LocalEmbedder, get_embedder

__all__ = ["Embedder", "NullEmbedder", "_LocalEmbedder", "get_embedder"]
