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
    # Снижено с 25 - см. историю правок в README: при "score all candidates"
    # (текущая формулировка промпта) объём генерации пропорционален этому
    # числу, а это и есть основной драйвер времени на декларацию (не размер
    # промпта - GPU быстро обрабатывает вход, но генерирует последовательно).
    npa_candidate_k: int = 13

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
    max_new_tokens: int = 350  # промпт снова просит оценить ВСЕ кандидаты
    # (не топ-10 - см. историю правок: формулировка "выбери топ-10" ухудшала
    # калибровку score, модель начинала завышать оценки заведомо нерелевантным
    # НПА). Экономия времени теперь через npa_candidate_k=13 (меньше объектов
    # для оценки), а не через смену формулировки задачи. 13 объектов JSON
    # ~230-260 токенов, оставлен запас.
    candidate_text_max_chars: int = 380  # обрезка текста НПА в промпте
    verbose: bool = False  # см. src/llm_rerank.py::QwenLlamaCppReranker


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
    # 28.5 мин, не 27: с адаптивным cutoff (см. pipeline.py::run) запас нужен
    # только на реальный оверхед вне цикла (загрузка модели/индексов - по
    # факту ~30-60с на Colab T4) и на запись/валидацию (доли секунды), а не
    # "про запас на всякий случай" - такой запас на быстром железе просто
    # без нужды отправлял бы декларации в конце списка на fallback (было
    # обнаружено на реальном прогоне: 27мин*0.92 отсекало бы 14 деклараций
    # из 151, хотя весь прогон физически укладывался в 27.5 мин из 30).
    time_budget_seconds: int = int(28.5 * 60)
    # После превышения этой доли бюджета новые декларации обрабатываются
    # без LLM-реранка (только гибридный retrieval) - защитный механизм.
    # Сама доля теперь работает вместе с адаптивным прогнозом в
    # pipeline.py::run (учитывает реальную скорость последних вызовов, а не
    # только текущий elapsed) - это позволяет держать долю близко к 1.0,
    # не рискуя проскочить бюджет: адаптивный прогноз сам тормозит ПЕРЕД
    # вызовом, который не влезет, вместо того чтобы полагаться на большой
    # статичный запас.
    llm_cutoff_fraction: float = 0.97
    random_seed: int = 42


FINAL_TOP_N = 10  # ровно столько НПА нужно вернуть на декларацию (формат задания)

retrieval_cfg = RetrievalConfig()
llm_cfg = LLMConfig()
embedding_cfg = EmbeddingConfig()
runtime_cfg = RuntimeConfig()
