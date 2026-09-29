"""
Центральная конфигурация решения.

Все "магические числа" пайплайна собраны здесь, чтобы:
  - их было легко подкрутить и обосновать в README одним местом;
  - run.py мог переопределять часть параметров через CLI без правки кода.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


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
    # Снижено 25 -> 13 -> 10: пропускная способность бесплатной T4 в Colab
    # заметно колеблется от прогона к прогону (наблюдалось 10.9с/декл и
    # 13.4с/декл на идентичном коде и железе - вероятно, разделяемый GPU),
    # поэтому берём запас по стоимости каждого вызова, а не только полагаемся
    # на адаптивный cutoff (см. RuntimeConfig ниже) - так в "медленный день"
    # больше деклараций физически успевает получить LLM-оценку, а не фолбэк.
    npa_candidate_k: int = 10

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
    max_new_tokens: int = 400  # промпт просит оценить ВСЕ кандидаты (сейчас
    # их 10 - см. RetrievalConfig.npa_candidate_k): ~180-200 токенов JSON в
    # типичном случае. 280 оказалось мало для длинных технических описаний
    # (одна и та же декларация стабильно получала оборванный ответ). Лимит -
    # потолок, а не цель: на времени обычных вызовов он не сказывается.
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
class SkipLLMConfig:
    """Пропуск LLM-реранка для "однозначных" случаев (см. src/skip_rule.py).

    mode:
      "off"    - LLM на всех (поведение по умолчанию; признаки уверенности всё
                 равно пишутся в timing_debug.csv для калибровки);
      "shadow" - LLM на всех, но дополнительно считается, какие декларации
                 БЫ пропустились, и насколько LLM совпала с retrieval на них.
                 Даёт полноценный predictions.csv И данные для калибровки
                 порогов за один прогон;
      "on"     - декларации, прошедшие правило, реально пропускают LLM.

    Пороги для режима "on" намеренно БЕЗ значений по умолчанию: их нужно
    подобрать по логу прогона (scripts/calibrate_skip.py), а не угадывать -
    иначе пропуск может молча ухудшить ранжирование там, где LLM исправляла
    retrieval (см. docstring skip_rule.py).
    """

    mode: str = "off"
    min_dense_gap: Optional[float] = None
    min_bm25_ratio: Optional[float] = None
    require_tnved_exact: bool = False

    def validate(self) -> None:
        if self.mode not in ("off", "shadow", "on"):
            raise ValueError(f"skip-llm mode должен быть off/shadow/on, получено: {self.mode!r}")
        if self.mode == "on" and (self.min_dense_gap is None or self.min_bm25_ratio is None):
            raise ValueError(
                "Режим --skip-llm on требует явных порогов --skip-min-dense-gap и "
                "--skip-min-bm25-ratio. Подберите их: сначала прогон с "
                "--skip-llm shadow, затем python scripts/calibrate_skip.py "
                "out/timing_debug.csv (см. README, раздел про пропуск LLM)."
            )


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
skip_llm_cfg = SkipLLMConfig()
