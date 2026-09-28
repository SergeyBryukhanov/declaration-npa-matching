"""
Гибридное ранжирование: BM25 (лексика) + dense-эмбеддинги (семантика),
объединённые через Reciprocal Rank Fusion (RRF).

RRF выбран вместо взвешенной суммы сырых скоров по практической причине:
BM25-скор и косинусное сходство живут в несопоставимых шкалах и их
относительный масштаб сильно зависит от длины/частотности текста конкретного
документа. RRF использует только ранги (позиции в топе), а не абсолютные
значения, поэтому не требует калибровки весов и устойчив к выбросам
любого из двух сигналов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25 import BM25
from .embeddings import EmbeddingBackend, cosine_sim_matrix
from .text_normalize import tokenize


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> Dict[str, float]:
    """
    rankings: список ранжирований, каждое - список id в порядке убывания
              релевантности по одному сигналу.
    Возвращает: id -> суммарный RRF-скор (больше = релевантнее).
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass(frozen=True)
class RetrievalSignals:
    """Сырые (до слияния рангов) сигналы двух ретриверов для одного запроса.

    Нужны, чтобы оценить, насколько retrieval "уверен" в лидере, - RRF-скор
    для этого не годится: он зависит только от рангов и почти не меняется
    между "явным" и "спорным" случаем (1/(k+1) против 1/(k+2)). Используется
    в src/skip_rule.py для решения, можно ли пропустить дорогой LLM-реранк.
    """

    bm25_top1_id: str
    bm25_top1: float
    bm25_top2: float
    dense_top1_id: Optional[str]  # None, если dense-сигнал отключён
    dense_top1: Optional[float]
    dense_top2: Optional[float]


@dataclass(frozen=True)
class SearchResult:
    ranked: List[Tuple[str, float]]  # [(doc_id, fused_score), ...] по убыванию
    signals: RetrievalSignals


class HybridCorpusIndex:
    """
    Гибридный индекс поверх фиксированного корпуса документов (используется
    и для регуляций, и для ТН ВЭД - логика идентична).
    """

    def __init__(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        embedding_backend: EmbeddingBackend | None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ):
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.rrf_k = rrf_k

        tokenized = [tokenize(t) for t in doc_texts]
        self.bm25 = BM25(tokenized, k1=bm25_k1, b=bm25_b)

        self.embedding_backend = embedding_backend
        self.doc_vecs = None
        if embedding_backend is not None:
            self.doc_vecs = embedding_backend.encode_passages(doc_texts)

    def search(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        """Возвращает [(doc_id, fused_score), ...] длиной top_k, по убыванию."""
        return self.search_with_signals(query_text, top_k).ranked

    def search_with_signals(self, query_text: str, top_k: int) -> SearchResult:
        """То же, что search(), плюс сырые BM25/dense-сигналы для оценки
        уверенности retrieval (см. RetrievalSignals)."""
        n = len(self.doc_ids)
        k_bm25 = min(max(top_k * 4, 50), n)  # берём с запасом перед слиянием рангов

        query_tokens = tokenize(query_text)
        bm25_top = self.bm25.top_k(query_tokens, k=k_bm25)
        bm25_ranking = [self.doc_ids[i] for i, _ in bm25_top]
        bm25_top1_id = bm25_ranking[0]
        bm25_top1 = bm25_top[0][1]
        bm25_top2 = bm25_top[1][1] if len(bm25_top) > 1 else 0.0

        rankings = [bm25_ranking]

        dense_top1_id: Optional[str] = None
        dense_top1: Optional[float] = None
        dense_top2: Optional[float] = None

        if self.embedding_backend is not None and self.doc_vecs is not None and n > 0:
            q_vec = self.embedding_backend.encode_queries([query_text])
            sims = cosine_sim_matrix(q_vec, self.doc_vecs)[0]
            k_dense = min(max(top_k * 4, 50), n)
            if k_dense >= n:
                order = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, k_dense)[:k_dense]
                order = part[np.argsort(-sims[part])]
            dense_ranking = [self.doc_ids[i] for i in order]
            rankings.append(dense_ranking)
            dense_top1_id = dense_ranking[0]
            dense_top1 = float(sims[order[0]])
            dense_top2 = float(sims[order[1]]) if len(order) > 1 else None

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        ranked_ids = sorted(fused.keys(), key=lambda i: -fused[i])[:top_k]
        ranked = [(doc_id, fused[doc_id]) for doc_id in ranked_ids]

        signals = RetrievalSignals(
            bm25_top1_id=bm25_top1_id,
            bm25_top1=float(bm25_top1),
            bm25_top2=float(bm25_top2),
            dense_top1_id=dense_top1_id,
            dense_top1=dense_top1,
            dense_top2=dense_top2,
        )
        return SearchResult(ranked=ranked, signals=signals)
