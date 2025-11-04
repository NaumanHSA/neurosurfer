from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple
from neurosurfer.vectorstores.base import Doc, BaseVectorDB
from neurosurfer.models.embedders.base import BaseEmbedder


def pick_files_by_grouped_chunk_hits(
    embedder: BaseEmbedder,
    vector_db: BaseVectorDB,
    section_query: str,
    candidate_pool_size: int = 200,
    n_files: int = 10,
    file_key: str = "filename",
) -> List[str]:
    """
    Broad similarity search -> aggregate by file -> top-N.
    Useful for large codebases to decide focus files.
    """
    qemb = embedder.embed(section_query)
    hits: List[Tuple[Doc, float]] = vector_db.similarity_search(
        query_embedding=qemb, top_k=candidate_pool_size
    )
    by_file = defaultdict(float)
    for doc, sim in hits:
        fp = (doc.metadata or {}).get(file_key)
        if not fp:
            continue
        by_file[fp] += sim
    return [
        fp for fp, _ in sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)[:n_files]
    ]
