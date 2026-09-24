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

# --- Проверка версии Python (Проблемы №1 и №9 из истории отладки) ---------
# numpy/torch/llama-cpp-python публикуют готовые wheel только под Python
# 3.10-3.12. На более новых/старых версиях pip уходит в сборку из исходников,
# требующую компилятор C/C++, которого обычно нет - ошибка выглядит как
# "Preparing metadata (pyproject.toml) ... error" на numpy и ничего не
# говорит про настоящую причину (версию Python). Проверяем явно и сразу,
# до любых тяжёлых импортов, чтобы дать понятную инструкцию вместо стектрейса.
_SUPPORTED_PY = ((3, 10), (3, 13))  # [min, max) - поддерживаются 3.10, 3.11, 3.12
if not (_SUPPORTED_PY[0] <= sys.version_info[:2] < _SUPPORTED_PY[1]):
    sys.stderr.write(
        f"\nОШИБКА: обнаружен Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"а numpy/torch/llama-cpp-python из requirements.txt публикуют готовые сборки "
        f"только под Python 3.10-3.12.\n"
        f"Похоже, активировано не то виртуальное окружение (проверьте, что вы в venv, "
        f"а не в системном Python: 'python --version' должен показать 3.10.x-3.12.x).\n"
        f"См. README.md, раздел 'Установка'.\n\n"
    )
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np
from src import config
from src.embeddings import DummyHashEmbeddingBackend, EmbeddingBackend, SentenceTransformerBackend
from src.io_utils import PredictionsWriter, load_declarations, load_regulations
from src.llm_rerank import LLMReranker, QwenLlamaCppReranker, StubReranker
from src.pipeline import Pipeline
from src.tnved import TnvedIndex
from src.validate import validate_predictions_file


def check_models_present(args) -> None:
    """
    Проблема №6/№7: раньше отсутствие скачанных моделей проявлялось как
    ValueError с сырым traceback из sentence_transformers, глубоко внутри
    стека вызовов. Проверяем явно и заранее, с понятной инструкцией.
    """
    missing = []
    if not args.no_embeddings and not os.path.isdir(args.embedding_model_dir):
        missing.append(f"  - эмбеддинг-модель не найдена: {args.embedding_model_dir}")
    if not os.path.isfile(args.llm_model_path):
        missing.append(f"  - LLM не найдена: {args.llm_model_path}")
    if missing:
        sys.stderr.write(
            "\nОШИБКА: не найдены файлы моделей:\n" + "\n".join(missing) +
            "\n\nСкорее всего, вы ещё не запускали (или не полностью запустили) "
            "разовую подготовку окружения:\n    python prepare.py\n"
            "См. README.md, раздел 'Установка'.\n\n"
        )
        sys.exit(1)


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
    ap.add_argument("--verbose-llm", action="store_true",
                    help="Подробный нативный лог llama.cpp (для отладки)")
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
        verbose=args.verbose_llm,
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
    else:
        check_models_present(args)

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

    out_csv = os.path.join(args.out, "predictions.csv")
    # Инкрементальная запись (Проблема №10): каждая декларация дописывается
    # в CSV и сразу сбрасывается на диск, как только посчитана - если прогон
    # прервётся на середине, уже обработанные декларации не потеряются.
    with PredictionsWriter(out_csv) as writer:
        pipeline.run(on_result=writer.write_declaration)
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
