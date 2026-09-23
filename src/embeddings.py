"""
Плотные эмбеддинги для семантического слоя гибридного поиска.

Продакшн-бэкенд грузит sentence-transformers модель ТОЛЬКО из локального
пути (models/multilingual-e5-small - готовится заранее prepare.py, сеть
во время run.py не используется). Модель: intfloat/multilingual-e5-small
(~118M параметров) - выбрана намеренно небольшой, т.к. её нужно держать
в памяти одновременно с BM25-индексами и (на этапе реранка) с 7B LLM,
а бюджет - 8 ГБ RAM без учёта видеопамяти GPU.

DummyHashEmbeddingBackend - НЕ для оценки решения, только для локальной
разработки/дымового теста пайплайна на машине без GPU/интернета (см.
run.py --dry-run). Даёт детерминированные, но семантически бессмысленные
векторы (мешок хэшированных токенов) - этого достаточно, чтобы проверить
форматы, склейку модулей и обработку ошибок, но не для оценки качества
ранжирования.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .text_normalize import tokenize


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode_passages(self, texts: List[str]) -> np.ndarray:
        """Эмбеддинги для "документов" (текстов НПА/ТН ВЭД). L2-нормализованы."""

    @abstractmethod
    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Эмбеддинги для запросов (текстов деклараций). L2-нормализованы."""


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerBackend(EmbeddingBackend):
    """Продакшн-бэкенд. Требует пакет sentence-transformers и локально
    сохранённые веса модели (см. prepare.py и config.DEFAULT_EMBEDDING_MODEL_DIR)."""

    def __init__(self, model_dir: str, batch_size: int = 64,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: "):
        from sentence_transformers import SentenceTransformer  # локальный импорт:
        # не тянем тяжёлую зависимость там, где используется только BM25/дамми-бэкенд
        # (например, в --dry-run режиме или юнит-тестах).

        self.model = SentenceTransformer(model_dir, device=None)  # device auto-detect (CPU/GPU)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _encode(self, texts: List[str], prefix: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


class DummyHashEmbeddingBackend(EmbeddingBackend):
    """ТОЛЬКО для dry-run/тестов. Мешок хэшированных токенов -> плотный вектор.

    Даёт слабый, но не нулевой сигнал лексического пересечения (два текста
    с общими токенами получат ненулевую косинусную близость), что достаточно
    для проверки, что пайплайн вообще работает end-to-end без реальной модели.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _encode_many(self, texts: List[str]) -> np.ndarray:
        mat = np.stack([self._encode_one(t) for t in texts]) if texts else np.zeros((0, self.dim))
        return _l2_normalize(mat).astype(np.float32)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return self._encode_many(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode_many(texts)


def cosine_sim_matrix(query_vecs: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
    """query_vecs: (n_q, d), passage_vecs: (n_p, d), оба L2-нормализованы -> (n_q, n_p)."""
    return query_vecs @ passage_vecs.T
