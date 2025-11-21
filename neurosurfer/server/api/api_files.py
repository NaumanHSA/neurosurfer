# server/api/files_api.py
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..security import get_db, get_current_user
from ..db.models import User, NSFile

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id}")
def download_file(
    file_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Download a file that belongs to the current user.

    - Ensures the file exists
    - Ensures it belongs to this user
    - Streams the file with correct filename + mime type
    """
    ns = (
        db.query(NSFile)
        .filter(
            NSFile.id == file_id,
            NSFile.user_id == user.id,
        )
        .first()
    )
    if not ns:
        raise HTTPException(status_code=404, detail="File not found")

    if not ns.stored_path or not os.path.exists(ns.stored_path):
        raise HTTPException(status_code=404, detail="File content missing")

    return FileResponse(
        path=ns.stored_path,
        media_type=ns.mime or "application/octet-stream",
        filename=ns.filename,
    )
