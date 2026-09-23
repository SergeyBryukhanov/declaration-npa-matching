"""
Нормализация и токенизация текста.

Тексты деклараций написаны ЗАГЛАВНЫМИ буквами и часто содержат смесь
русского/английского, артикулы, серийные номера. Тексты НПА и ТН ВЭД -
формальный русский текст в нижнем регистре. Приводим всё к единому виду
перед лексическим сопоставлением (BM25) и эмбеддингами.

Лемматизация через pymorphy2 - опционально: если библиотека недоступна
в окружении, тихо откатываемся на нормализацию без лемматизации (это
снижает recall BM25 на словоформах, но не ломает пайплайн). Причина
такого fallback - pymorphy2 не всегда предустановлен, а сеть на этапе
run.py запрещена, значит рассчитывать на pip install во время запуска
нельзя.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+", re.UNICODE)

# Небольшой список стоп-слов (предлоги/союзы), которые не несут смысла
# для сопоставления декларация <-> НПА, но часто встречаются и раздувают
# BM25-статистику. Сознательно короткий и консервативный список, чтобы
# не выкинуть значимые токены (например "и" может быть частью маркировки,
# поэтому используем только однозначно служебные слова).
_STOPWORDS = {
    "и", "в", "во", "не", "на", "с", "со", "к", "ко", "по", "из", "у",
    "о", "об", "от", "до", "для", "за", "при", "или", "а", "но", "же",
    "то", "это", "также", "также", "как", "что", "чтобы",
}

_translit_yo = str.maketrans({"ё": "е", "Ё": "Е"})


def normalize_text(text: str) -> str:
    """Базовая нормализация: приведение регистра и ё->е, схлопывание пробелов."""
    if not text:
        return ""
    text = text.translate(_translit_yo)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_morph():
    """Ленивая инициализация pymorphy2, если он установлен."""
    try:
        import pymorphy2  # type: ignore

        return pymorphy2.MorphAnalyzer()
    except Exception:
        return None


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    morph = _get_morph()
    if morph is None:
        return word
    try:
        return morph.parse(word)[0].normal_form
    except Exception:
        return word


def tokenize(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
    """
    Токенизация для BM25/лексического сопоставления.

    Возвращает список токенов в нижнем регистре, опционально лемматизированных
    и без стоп-слов. Числа и смешанные алфавитно-цифровые токены (артикулы,
    единицы измерения вида "12мм") сохраняются как есть - они часто несут
    ключевую информацию (пороговые значения в НПА).
    """
    norm = normalize_text(text)
    tokens = _WORD_RE.findall(norm)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if lemmatize and _get_morph() is not None:
        tokens = [_lemma(t) if t.isalpha() else t for t in tokens]
    return tokens


def truncate(text: str, max_chars: int) -> str:
    """Аккуратная обрезка текста для промпта LLM (по границе слова)."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.6:
        cut = cut[:last_space]
    return cut.rstrip() + "…"
