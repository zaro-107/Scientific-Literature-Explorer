# language_utils.py
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic

def detect_language(text: str, default: str = "en") -> str:
    """
    Returns ISO 639-1 like 'en', 'hi', 'fr'.
    Uses langdetect; if text too short/invalid, returns default.
    """
    text = (text or "").strip()
    if len(text) < 30:
        return default
    try:
        return detect(text)
    except Exception:
        return default
