# server/api/files_api.py
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..security import get_db, get_current_user
from ..db.models import User, NMFile

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/{file_id}")
def download_file(
    file_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    f = (
        db.query(NMFile)
        .filter(
            NMFile.id == file_id,
            NMFile.user_id == user.id,
        )
        .first()
    )
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    if not os.path.exists(f.stored_path):
        raise HTTPException(status_code=404, detail="File content missing on server")

    return FileResponse(
        path=f.stored_path,
        media_type=f.mime or "application/octet-stream",
        filename=f.filename,
    )
