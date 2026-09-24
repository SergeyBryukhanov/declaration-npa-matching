"""
Центральная конфигурация решения.

Все "магические числа" пайплайна собраны здесь, чтобы:
  - их было легко подкрутить и обосновать в README одним местом;
  - run.py мог переопределять часть параметров через CLI без правки кода.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Пути по умолчанию (совпадают с корнем проекта, как того требует задание:
# `python run.py --out ./out` запускается из корня, где лежат
# declarations.jsonl и regulations.jsonl).
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DECLARATIONS_PATH = os.path.join(PROJECT_ROOT, "declarations.jsonl")
DEFAULT_REGULATIONS_PATH = os.path.join(PROJECT_ROOT, "regulations.jsonl")
DEFAULT_TNVED_PATH = os.path.join(PROJECT_ROOT, "tnved_knowledge.txt")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "out")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Локальные пути к весам моделей (заполняются prepare.py один раз, ДО запуска
# run.py; во время run.py сеть запрещена, поэтому модели должны быть уже
# на диске по этим путям).
DEFAULT_LLM_GGUF_PATH = os.path.join(DEFAULT_MODELS_DIR, "qwen2.5-7b-instruct-q4_k_m.gguf")
DEFAULT_EMBEDDING_MODEL_DIR = os.path.join(DEFAULT_MODELS_DIR, "multilingual-e5-small")

# Идентификаторы моделей "на бумаге" (используются только в prepare.py для
# скачивания и в README для фиксации версий; во время run.py не используются,
# т.к. вся загрузка идёт из DEFAULT_*_PATH).
# Репозиторий bartowski - известный, проверенный community-мирror официальных
# весов Qwen2.5-7B-Instruct в GGUF (тот же официальный конвертер llama.cpp).
# Официальный репозиторий Qwen/Qwen2.5-7B-Instruct-GGUF даёт на некоторых
# ревизиях 404 на ожидаемое имя файла - выбран более предсказуемый источник.
LLM_HF_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
LLM_HF_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
EMBEDDING_HF_REPO = "intfloat/multilingual-e5-small"


@dataclass
class RetrievalConfig:
    # Сколько кандидатов ТН ВЭД держим на декларацию (для обогащения запроса).
    tnved_top_k: int = 5
    # Минимальная длина фрагмента (в символах) для быстрого substring-якоря.
    tnved_exact_min_len: int = 20

    # Сколько кандидатов-НПА выходит из гибридного ретривера на LLM-реранк.
    npa_candidate_k: int = 25

    # Параметры BM25 (Okapi, стандартные значения).
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Константа сглаживания для Reciprocal Rank Fusion.
    rrf_k: int = 60

    # Вес плотного (эмбеддингового) сигнала относительно BM25 при слиянии
    # альтернативным способом (не используется при RRF, оставлено для
    # взвешенной суммы как fallback/для экспериментов).
    dense_weight: float = 0.5


@dataclass
class LLMConfig:
    model_path: str = DEFAULT_LLM_GGUF_PATH
    n_ctx: int = 8192          # с запасом под ~25 кандидатов + декларацию
    n_gpu_layers: int = -1     # -1 = выгрузить все возможные слои на GPU, если она есть
    n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    temperature: float = 0.0   # детерминированность важнее "креативности"
    max_new_tokens: int = 900  # JSON-список из ~25 объектов с полями id/score
    candidate_text_max_chars: int = 380  # обрезка текста НПА в промпте


@dataclass
class EmbeddingConfig:
    model_dir: str = DEFAULT_EMBEDDING_MODEL_DIR
    batch_size: int = 64
    # intfloat/e5-* модели требуют префиксов "query: "/"passage: " для
    # корректного качества (это задокументированное требование модели).
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass
class RuntimeConfig:
    # Общий бюджет по времени на run.py (лимит задания - 30 минут).
    # Оставляем запас на чтение/запись файлов и импорт библиотек.
    time_budget_seconds: int = 27 * 60
    # После превышения этой доли бюджета новые декларации обрабатываются
    # без LLM-реранка (только гибридный retrieval) - защитный механизм,
    # чтобы гарантированно уложиться в лимит на медленном железе.
    llm_cutoff_fraction: float = 0.85
    random_seed: int = 42


FINAL_TOP_N = 10  # ровно столько НПА нужно вернуть на декларацию (формат задания)

retrieval_cfg = RetrievalConfig()
llm_cfg = LLMConfig()
embedding_cfg = EmbeddingConfig()
runtime_cfg = RuntimeConfig()
