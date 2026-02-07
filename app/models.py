from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    papers = relationship("Paper", back_populates="owner")


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)

    # Auth owner (keep it, but allow null if you upload without login)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Your route uses these:
    paper_id = Column(String(64), unique=True, index=True, nullable=False)  # like "a1b2c3d4e5f6"
    title = Column(String(512), nullable=False)  # filename or title
    source = Column(String(64), nullable=False, default="upload")  # upload / arxiv / etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="papers")
    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id = Column(Integer, primary_key=True, index=True)

    # papers.py uses paper_id_fk = paper.id
    paper_id_fk = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)

    chunk_id = Column(String(128), index=True, nullable=False)   # like "{paper_id}_0001"
    section = Column(String(128), nullable=False, default="unknown")

    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)

    lang = Column(String(16), nullable=False, default="en")

    text_original = Column(Text, nullable=False)
    text_en = Column(Text, nullable=True)

    embedding_id = Column(String(128), index=True, nullable=True)

    paper = relationship("Paper", back_populates="chunks")
