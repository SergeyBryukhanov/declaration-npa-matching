"""
Реализация Okapi BM25 "с нуля" на numpy.

Осознанное решение не тянуть внешний пакет (rank_bm25 и т.п.) ради лишней
универсальности: алгоритм простой, занимает ~60 строк, а собственная
реализация снимает риск несовместимости версий на машине проверяющего
и делает решение более самодостаточным (меньше внешних зависимостей ->
меньше что может сломаться при офлайн-запуске).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence

import numpy as np


class BM25:
    def __init__(self, corpus_tokens: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        corpus_tokens: список документов, каждый - список токенов
                       (см. text_normalize.tokenize).
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.n_docs = len(corpus_tokens)

        df: Counter = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.df = df

        # idf по классической формуле BM25 (Robertson-Sparck Jones + сглаживание)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        # Предвычисленные term-frequency векторы по терминам словаря, чтобы
        # get_scores() был векторизован по документам (а не по python-циклу).
        self._vocab = {term: i for i, term in enumerate(df.keys())}
        self._tf_matrix = np.zeros((self.n_docs, len(self._vocab)), dtype=np.float32)
        for doc_i, counts in enumerate(self.doc_freqs):
            for term, cnt in counts.items():
                self._tf_matrix[doc_i, self._vocab[term]] = cnt

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """BM25-скор запроса против каждого документа корпуса."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return scores
        for term in query_tokens:
            if term not in self._vocab:
                continue
            idf = self.idf[term]
            tf = self._tf_matrix[:, self._vocab[term]]
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens / (self.avgdl or 1.0))
            scores += idf * (tf * (self.k1 + 1)) / np.where(denom == 0, 1, denom)
        return scores

    def top_k(self, query_tokens: List[str], k: int) -> List[tuple]:
        """Возвращает [(doc_index, score), ...] топ-k по убыванию скора."""
        scores = self.get_scores(query_tokens)
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in order]
