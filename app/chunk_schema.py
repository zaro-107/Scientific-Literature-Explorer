# chunk_schema.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PaperChunk:
    paper_id: str
    chunk_id: str
    section: str
    page_start: Optional[int]
    page_end: Optional[int]
    lang: str
    text_original: str
    text_en: str  # if lang != en, translated text; else same as original
