# translator.py
from typing import Dict, Optional
from transformers import MarianMTModel, MarianTokenizer

# Add more languages as you need
# NOTE: Marian models exist for many pairs; we map only a few common ones.
MARIAN_MODELS: Dict[str, str] = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
}

class TranslatorToEnglish:
    def __init__(self):
        self._cache = {}  # lang -> (tokenizer, model)

    def _load(self, lang: str):
        model_name = MARIAN_MODELS.get(lang)
        if not model_name:
            return None

        if lang in self._cache:
            return self._cache[lang]

        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        self._cache[lang] = (tok, model)
        return self._cache[lang]

    def translate(self, text: str, lang: str) -> str:
        """
        Translate `text` from `lang` to English. If lang not supported, returns original text.
        """
        if not text or not lang or lang == "en":
            return text

        pack = self._load(lang)
        if pack is None:
            return text  # fallback

        tok, model = pack
        # Keep it safe for long text by chunking
        return self._translate_long(text, tok, model)

    def _translate_long(self, text: str, tok, model, max_tokens: int = 450) -> str:
        words = text.split()
        if len(words) < 5:
            return text

        out = []
        buf = []
        cur = 0

        for w in words:
            buf.append(w)
            cur += 1
            if cur >= max_tokens:
                out.append(self._translate_once(" ".join(buf), tok, model))
                buf, cur = [], 0

        if buf:
            out.append(self._translate_once(" ".join(buf), tok, model))

        return "\n".join(out)

    def _translate_once(self, text: str, tok, model) -> str:
        batch = tok([text], return_tensors="pt", padding=True, truncation=True)
        gen = model.generate(**batch, max_new_tokens=512)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]
