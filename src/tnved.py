"""
Парсинг и использование справочника ТН ВЭД (tnved_knowledge.txt).

Роль в пайплайне: НЕ обязательный источник истины, а мост нормализации
словаря. Декларанты часто дословно (или почти дословно) копируют
официальную формулировку товарной позиции ТН ВЭД в описание товара -
эта формулировка по регистру языка ближе к юридическому тексту НПА, чем
свободный текст декларации (бренды, артикулы, вставки на английском).

Мы НЕ используем ТН ВЭД как жёсткий фильтр по разделам/группам:
прямого соответствия "код ТН ВЭД -> decree_number" в данных нет
(проверено - в regulations.jsonl нет упоминаний ТН ВЭД, а в самом
справочнике нет ссылок на номера постановлений из нашего набора НПА).
Поэтому анкоринг используется только чтобы (а) обогатить запрос к
ретриверу НПА каноничной формулировкой и (b) дать LLM дополнительный
контекст для рассуждения. Раздел/группа НЕ используются как отсекающий
фильтр - это осознанное допущение, снижающее риск потери релевантных
НПА из-за неполного/ошибочного маппинга раздел<->НПА.

Формат исходного файла (см. пример):
    8482101009 | – – – прочие [подшипники шариковые, наибольший наружный
    диаметр которых не более 30 мм, прочие]

Парсим только "листовые" строки вида `<код><пробелы>|...[<полный текст>]`.
Текст в квадратных скобках уже включает контекст родительских уровней
иерархии (это "развёрнутое" описание) - именно его используем как
каноничную формулировку.

Блоки "ПРИМЕЧАНИЯ" (легальные пояснения по разделам/группам) в файле
тоже присутствуют, но в v1 решения не используются для экономии
времени разработки и бюджета запуска - см. README, раздел
"Рассмотренные альтернативы / что можно добавить далее".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .bm25 import BM25
from .text_normalize import normalize_text, tokenize

_LEAF_RE = re.compile(r"^\s*(\d{4,10})\s*\|.*\[(.+)\]\s*$")
_MIN_LEAF_CODE_LEN = 4  # отсекаем короткие "коды", которые на самом деле разделы/группы


@dataclass(frozen=True)
class TnvedEntry:
    code: str
    text: str            # полное развёрнутое описание, нормализованное (нижний регистр)


@dataclass(frozen=True)
class TnvedMatch:
    code: str
    text: str
    method: str   # "exact_substring" | "bm25"
    score: float


def parse_tnved_file(path: str) -> List[TnvedEntry]:
    entries: List[TnvedEntry] = []
    seen_codes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _LEAF_RE.match(line)
            if not m:
                continue
            code, desc = m.groups()
            if len(code) < _MIN_LEAF_CODE_LEN:
                continue
            norm_desc = normalize_text(desc)
            if not norm_desc:
                continue
            # В справочнике один и тот же лист изредка не повторяется, но на
            # всякий случай защищаемся от дублей кода с разным текстом -
            # оставляем первое вхождение (порядок в файле = порядок иерархии).
            if code in seen_codes:
                continue
            seen_codes.add(code)
            entries.append(TnvedEntry(code=code, text=norm_desc))
    return entries


class TnvedIndex:
    """BM25-индекс + быстрый substring-поиск по листовым описаниям ТН ВЭД."""

    def __init__(self, entries: List[TnvedEntry], bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.entries = entries
        self._tokens = [tokenize(e.text) for e in entries]
        self.bm25 = BM25(self._tokens, k1=bm25_k1, b=bm25_b)
        # Для быстрого substring-поиска держим только достаточно длинные
        # тексты (короткие фразы дают слишком много случайных совпадений).
        self._substr_candidates = [
            (i, e.text) for i, e in enumerate(entries) if len(e.text) >= 20
        ]

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "TnvedIndex":
        return cls(parse_tnved_file(path), **kwargs)

    def _exact_matches(self, declaration_text_norm: str) -> List[TnvedMatch]:
        matches = []
        for i, text in self._substr_candidates:
            if text in declaration_text_norm:
                matches.append(
                    TnvedMatch(
                        code=self.entries[i].code,
                        text=self.entries[i].text,
                        method="exact_substring",
                        # длина совпадения как прокси уверенности: длиннее
                        # совпавший официальный текст - выше уверенность.
                        score=float(len(text)),
                    )
                )
        matches.sort(key=lambda m: -m.score)
        return matches

    def anchor(self, declaration_text: str, top_k: int = 5) -> List[TnvedMatch]:
        """Top-k наиболее вероятных кодов ТН ВЭД для текста декларации."""
        norm = normalize_text(declaration_text)
        exact = self._exact_matches(norm)

        remaining = top_k - len(exact)
        bm25_matches: List[TnvedMatch] = []
        if remaining > 0:
            query_tokens = tokenize(declaration_text)
            exact_codes = {m.code for m in exact}
            for idx, score in self.bm25.top_k(query_tokens, k=remaining + len(exact_codes)):
                if score <= 0:
                    break
                entry = self.entries[idx]
                if entry.code in exact_codes:
                    continue
                bm25_matches.append(
                    TnvedMatch(code=entry.code, text=entry.text, method="bm25", score=score)
                )
                if len(bm25_matches) >= remaining:
                    break

        return (exact + bm25_matches)[:top_k]


def build_enriched_query(declaration_text: str, matches: List[TnvedMatch], max_matches: int = 2) -> str:
    """
    Обогащённый текст запроса для ретривера НПА: исходный текст декларации
    + каноничные формулировки ТН ВЭД (не более max_matches, чтобы не
    "размыть" запрос менее уверенными совпадениями).
    """
    parts = [declaration_text]
    for m in matches[:max_matches]:
        parts.append(m.text)
    return " . ".join(parts)
