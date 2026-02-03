from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String, unique=True, index=True, nullable=False)  # stable string id
    title = Column(String, nullable=True)
    authors = Column(String, nullable=True)
    year = Column(String, nullable=True)
    source = Column(String, nullable=True)  # uploaded/local/arxiv/etc
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("PaperChunk", back_populates="paper", cascade="all, delete-orphan")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id = Column(Integer, primary_key=True, index=True)

    paper_id_fk = Column(Integer, ForeignKey("papers.id"), index=True, nullable=False)
    chunk_id = Column(String, index=True, nullable=False)  # e.g. paperid_0001
    section = Column(String, default="unknown", index=True)

    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)

    lang = Column(String, default="en", index=True)

    # store both versions
    text_original = Column(Text, nullable=False)
    text_en = Column(Text, nullable=False)

    # optional: store embedding_id (if your vector store uses ids)
    embedding_id = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="chunks")

# Helpful indexes for retrieval + filtering
Index("ix_chunks_paper_chunk", PaperChunk.paper_id_fk, PaperChunk.chunk_id)
Index("ix_chunks_section_page", PaperChunk.section, PaperChunk.page_start)
