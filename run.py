#!/usr/bin/env python3
"""
Точка входа решения.

    python run.py --out ./out

Ожидает по умолчанию declarations.jsonl и regulations.jsonl в корне
проекта (рядом с run.py) и tnved_knowledge.txt там же - как и
предполагает задание. Модели должны быть уже скачаны локально
(см. prepare.py) - во время работы run.py сеть не используется.

Флаг --dry-run включает лёгкие заглушки вместо реальных Qwen2.5-7B и
эмбеддинг-модели (см. src/embeddings.py, src/llm_rerank.py) - это НЕ
режим для оценки решения, а способ за секунды проверить, что весь
пайплайн и формат вывода работают корректно на машине без GPU/интернета
(например, сразу после git clone, до того как отработает prepare.py).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src import config
from src.embeddings import DummyHashEmbeddingBackend, EmbeddingBackend, SentenceTransformerBackend
from src.io_utils import load_declarations, load_regulations, write_predictions_csv
from src.llm_rerank import LLMReranker, QwenLlamaCppReranker, StubReranker
from src.pipeline import Pipeline
from src.tnved import TnvedIndex
from src.validate import validate_predictions_file


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=config.DEFAULT_OUT_DIR, help="Директория для out/predictions.csv")
    ap.add_argument("--declarations", default=config.DEFAULT_DECLARATIONS_PATH)
    ap.add_argument("--regulations", default=config.DEFAULT_REGULATIONS_PATH)
    ap.add_argument("--tnved", default=config.DEFAULT_TNVED_PATH)
    ap.add_argument("--llm-model-path", default=config.llm_cfg.model_path)
    ap.add_argument("--embedding-model-dir", default=config.embedding_cfg.model_dir)
    ap.add_argument("--npa-candidate-k", type=int, default=config.retrieval_cfg.npa_candidate_k)
    ap.add_argument("--tnved-top-k", type=int, default=config.retrieval_cfg.tnved_top_k)
    ap.add_argument("--time-budget-min", type=float, default=config.runtime_cfg.time_budget_seconds / 60)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Заглушки вместо реальных моделей (для быстрой проверки пайплайна, не для оценки)",
    )
    ap.add_argument("--no-embeddings", action="store_true", help="Отключить dense-сигнал, только BM25")
    ap.add_argument("--log-level", default="INFO")
    return ap


def build_embedding_backend(args) -> EmbeddingBackend | None:
    if args.no_embeddings:
        return None
    if args.dry_run:
        return DummyHashEmbeddingBackend()
    return SentenceTransformerBackend(
        model_dir=args.embedding_model_dir,
        batch_size=config.embedding_cfg.batch_size,
        query_prefix=config.embedding_cfg.query_prefix,
        passage_prefix=config.embedding_cfg.passage_prefix,
    )


def build_llm_reranker(args) -> LLMReranker:
    if args.dry_run:
        return StubReranker()
    return QwenLlamaCppReranker(
        model_path=args.llm_model_path,
        n_ctx=config.llm_cfg.n_ctx,
        n_gpu_layers=config.llm_cfg.n_gpu_layers,
        n_threads=config.llm_cfg.n_threads,
        temperature=config.llm_cfg.temperature,
        max_new_tokens=config.llm_cfg.max_new_tokens,
        candidate_text_max_chars=config.llm_cfg.candidate_text_max_chars,
    )


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run")
    t_start = time.monotonic()

    # Фиксация seed для воспроизводимости. Основные компоненты пайплайна
    # детерминированы и без этого (BM25, косинусное сходство, RRF, LLM с
    # temperature=0.0), но seed фиксируется на случай тай-брейков в
    # np.argpartition/argsort и для любых будущих стохастических частей.
    random.seed(config.runtime_cfg.random_seed)
    np.random.seed(config.runtime_cfg.random_seed)

    if args.dry_run:
        logger.warning(
            "Запуск в режиме --dry-run: используются ЗАГЛУШКИ вместо Qwen2.5-7B и "
            "эмбеддинг-модели. Результат НЕ пригоден для оценки качества, только "
            "для проверки формата и работоспособности пайплайна."
        )

    logger.info("Загрузка данных...")
    declarations = load_declarations(args.declarations)
    regulations = load_regulations(args.regulations)
    logger.info("Деклараций: %d, регуляций: %d", len(declarations), len(regulations))

    logger.info("Построение индекса ТН ВЭД (%s)...", args.tnved)
    tnved_index = TnvedIndex.from_file(
        args.tnved,
        bm25_k1=config.retrieval_cfg.bm25_k1,
        bm25_b=config.retrieval_cfg.bm25_b,
    )
    logger.info("Листовых записей ТН ВЭД: %d", len(tnved_index.entries))

    logger.info("Инициализация эмбеддинг-бэкенда...")
    embedding_backend = build_embedding_backend(args)

    logger.info("Инициализация LLM-реранкера...")
    llm_reranker = build_llm_reranker(args)

    retrieval_cfg = config.RetrievalConfig(
        tnved_top_k=args.tnved_top_k,
        npa_candidate_k=args.npa_candidate_k,
        bm25_k1=config.retrieval_cfg.bm25_k1,
        bm25_b=config.retrieval_cfg.bm25_b,
        rrf_k=config.retrieval_cfg.rrf_k,
    )

    pipeline = Pipeline(
        declarations=declarations,
        regulations=regulations,
        tnved_index=tnved_index,
        embedding_backend=embedding_backend,
        llm_reranker=llm_reranker,
        retrieval_cfg=retrieval_cfg,
        llm_cfg=config.llm_cfg,
        time_budget_seconds=args.time_budget_min * 60,
        llm_cutoff_fraction=config.runtime_cfg.llm_cutoff_fraction,
    )

    logger.info("Запуск пайплайна на %d декларациях...", len(declarations))
    predictions = pipeline.run()

    out_csv = os.path.join(args.out, "predictions.csv")
    write_predictions_csv(out_csv, predictions)
    logger.info("Записано: %s", out_csv)

    logger.info("Валидация формата вывода...")
    validate_predictions_file(
        out_csv,
        declaration_ids={d.declaration_id for d in declarations},
        regulation_ids={r.regulation_id for r in regulations},
    )
    logger.info("Формат корректен.")

    elapsed = time.monotonic() - t_start
    logger.info("Готово за %.1f сек (%.1f мин).", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
