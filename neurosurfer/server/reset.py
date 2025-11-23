import os 
import logging
from .config import APP_DATA_PATH
from neurosurfer.server.db.db import SessionLocal
from neurosurfer.server.db.models import NSFile, Message, ChatThread, User

logger = logging.getLogger(__name__)

def reset_db():
    """Clear all chats and threads from the database. Only leave Users Information."""
    db = SessionLocal()

    # Drop tables
    NSFile.__table__.drop(db.bind, checkfirst=True)
    Message.__table__.drop(db.bind, checkfirst=True)
    ChatThread.__table__.drop(db.bind, checkfirst=True)

    # Recreate tables
    NSFile.__table__.create(db.bind, checkfirst=True)
    Message.__table__.create(db.bind, checkfirst=True)
    ChatThread.__table__.create(db.bind, checkfirst=True)
    
    db.query(ChatThread).delete()
    db.query(Message).delete()
    db.query(NSFile).delete()
    db.commit()

    # reset vector db
    RAG_STORAGE_PATH = os.path.join(APP_DATA_PATH, "rag-storage")
    if os.path.exists(RAG_STORAGE_PATH):
        import shutil
        shutil.rmtree(RAG_STORAGE_PATH)

    # check if there are any users left
    logger.info(f"Number of users left: {db.query(User).count()}")
    logger.info(f"Number of threads left: {db.query(ChatThread).count()}")
    logger.info(f"Number of messages left: {db.query(Message).count()}")
    logger.info(f"Number of files left: {db.query(NSFile).count()}")
    db.close()
