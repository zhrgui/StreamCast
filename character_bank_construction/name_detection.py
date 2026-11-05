"""
Name detection in subtitles using spaCy POS tagging (proper nouns).
Adapted from naming/preprocessing/detect_name.py for streaming use.
"""

from typing import List

# Lazy load spacy to avoid import cost when not needed
_nlp = None


def _get_nlp(lang: str = "en"):
    global _nlp
    if _nlp is None:
        try:
            if lang == "en":
                import spacy
                _nlp = spacy.load("en_core_web_trf")
            elif lang == "zh":
                import spacy
                _nlp = spacy.load("zh_core_web_trf")
            else:
                import spacy
                _nlp = spacy.load("en_core_web_trf")
        except OSError:
            try:
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                _nlp = None
    return _nlp


def detect_names(text: str, lang: str = "en", **kwargs) -> List[str]:
    """
    Extract proper noun names from text using spaCy POS tagging.

    Consecutive PROPN tokens are joined into a single name span
    (e.g. "Harry Potter" from two adjacent PROPN tokens).

    Args:
        text: Subtitle or dialogue text.
        lang: "en" or "zh" for language model.

    Returns:
        List of unique names found in text.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp(lang)
    if nlp is None:
        return []

    doc = nlp(text.strip())
    names = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
    return names


