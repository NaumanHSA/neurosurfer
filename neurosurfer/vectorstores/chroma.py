import os
import uuid
import gc
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseVectorDB, Doc

def DefaultEmbeddingFunction():
    return None  # Replace with your actual embedding function if needed

class ChromaVectorStore(BaseVectorDB):
    def __init__(self, collection_name: str, clear_collection: bool = False, persist_directory: str = "chroma_storage"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        if clear_collection:
            self.clear_collection()
        print(f"[Init] ChromaVectorStore initialized with collection: {collection_name}")

    def add_documents(self, docs: List[Doc]):
        if not docs: 
            return
        ids, texts, embeddings, metadatas = [], [], [],  []
        for d in docs:
            ids.append(getattr(d, "id", None) or self._stable_id(d))
            texts.append(d.text)
            embeddings.append(d.embedding)
            metadatas.append(d.metadata)

        if hasattr(self.collection, "upsert"):
            self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        else:
            self.collection.delete(ids=ids)
            self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[Doc, float]]:
        where = {}
        if metadata_filter:
            for k, v in metadata_filter.items():
                where[k] = {"$in": v} if isinstance(v, list) else v

        args = dict(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # fetch extra then dedupe
            include=["documents", "metadatas", "distances"],
        )
        if where: args["where"] = where

        res = self.collection.query(**args)
        ids   = res.get("ids", [[]])[0]
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        max_dist = 1.0 - similarity_threshold if similarity_threshold else None  # distance = 1 - cosine_sim

        out: List[Tuple[Doc, float]] = []
        seen = set()
        for rid, txt, meta, dist in zip(ids, docs, metas, dists):
            if max_dist is not None and dist > max_dist:
                continue
            if rid in seen:
                # print(f"\n\nrid: {rid} already seen - {txt}\n\n")
                continue
            seen.add(rid)
            out.append((Doc(id=rid, text=txt, metadata=meta, embedding=None), 1.0 - dist))  # return cosine sim
            if len(out) >= top_k:
                break
        return out

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )

    def delete_documents(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = None

    def count(self) -> int:
        return self.collection.count()

    def list_all_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Doc]:
        results = self.collection.get(where=metadata_filter, include=["documents", "metadatas", "embeddings"])
        return [
            Doc(id=str(uuid.uuid4()), text=d, metadata=m, embedding=e)
            for d, m, e in zip(
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("embeddings", [])
            )
        ]
