"""
Правило "однозначный случай - LLM-реранк можно пропустить".

Идея: LLM-вызов - это ~11-13 секунд на декларацию, всё остальное почти
бесплатно. Если retrieval уже уверенно выделил лидера, LLM, скорее всего,
ничего не изменит в верхушке ранжирования - и вызов можно сэкономить,
отдав время тем декларациям, где он реально нужен.

ЧЕГО ЭТО ПРАВИЛО НЕ ДЕЛАЕТ - и это важно понимать:
  * оно не определяет, "правильный" ли лидер. Мы меряем только, насколько
    два НЕЗАВИСИМЫХ ретривера (лексический BM25 и семантический dense)
    согласны и насколько лидер оторвался от второго места;
  * "однозначность" по retrieval - НЕ то же самое, что "LLM согласится".
    Пример из реального прогона: декларация про подшипники лексически
    совпадает с прецизионными НПА почти дословно, а LLM осознанно занижает
    их score (категориальное совпадение без подтверждённого критерия).
    Поэтому пороги здесь не выбираются "на глаз": их подбирает
    scripts/calibrate_skip.py по логу прогона, где LLM отработала на ВСЕХ
    декларациях (режим --skip-llm shadow), - по доле случаев, где верхушка
    ранжирования retrieval и LLM совпала.

Признаки (все считаются из сырых сигналов, а не из RRF-скора: тот зависит
только от рангов и почти не отличает явный случай от спорного):
  signals_agree - лидеры BM25 и dense совпали (два независимых сигнала);
  dense_gap     - отрыв косинусной близости лидера от второго места;
  bm25_ratio    - во сколько раз BM25-скор лидера больше второго места;
  tnved_exact   - сработал точный substring-якорь ТН ВЭД (редкий, ~3%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .retrieval import RetrievalSignals

# BM25-отношение может быть бесконечным (у второго места нулевой скор) -
# ограничиваем, чтобы в CSV и порогах жили конечные числа.
_BM25_RATIO_CAP = 1e6


@dataclass(frozen=True)
class ConfidenceFeatures:
    signals_agree: Optional[bool]  # None, если dense-сигнал отключён
    dense_gap: Optional[float]
    bm25_ratio: float
    tnved_exact: bool


def compute_features(signals: RetrievalSignals, tnved_exact: bool) -> ConfidenceFeatures:
    if signals.dense_top1_id is None or signals.dense_top1 is None:
        signals_agree: Optional[bool] = None
        dense_gap: Optional[float] = None
    else:
        signals_agree = signals.bm25_top1_id == signals.dense_top1_id
        dense_gap = (
            signals.dense_top1 - signals.dense_top2
            if signals.dense_top2 is not None
            else None
        )

    denom = max(signals.bm25_top2, 1e-9)
    bm25_ratio = min(signals.bm25_top1 / denom, _BM25_RATIO_CAP)

    return ConfidenceFeatures(
        signals_agree=signals_agree,
        dense_gap=dense_gap,
        bm25_ratio=bm25_ratio,
        tnved_exact=tnved_exact,
    )


def is_confident(
    f: ConfidenceFeatures,
    min_dense_gap: float,
    min_bm25_ratio: float,
    require_tnved_exact: bool = False,
) -> bool:
    """True, если retrieval однозначен по ВСЕМ условиям сразу (консервативно:
    любое недостающее/отключённое условие - это "не уверены", а не "уверены")."""
    if f.signals_agree is not True:
        return False  # нет dense-сигнала или лидеры разошлись
    if f.dense_gap is None or f.dense_gap < min_dense_gap:
        return False
    if f.bm25_ratio < min_bm25_ratio:
        return False
    if require_tnved_exact and not f.tnved_exact:
        return False
    return True
