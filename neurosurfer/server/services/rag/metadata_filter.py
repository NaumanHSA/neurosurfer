# neurosurfer/services/rag/metadata_filter.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from neurosurfer.server.db.models import NSFile


def build_metadata_filter_from_related_files(
    db: Session,
    *,
    user_id: int,
    thread_id: int,
    collection: str,
    related_files: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Map gate LLM's related_files (filenames) to NSFile IDs,
    then build a metadata_filter suitable for the vectorstore.
    """
    if not related_files:
        return None

    rows = (
        db.query(NSFile)
        .filter(
            NSFile.user_id == user_id,
            NSFile.thread_id == thread_id,
            NSFile.collection == collection,
            NSFile.filename.in_(related_files),
        )
        .all()
    )
    file_ids = [r.filename for r in rows]
    if not file_ids:
        return None

    # Adjust shape to match your vectorstore implementation
    return {"filename": related_files}
