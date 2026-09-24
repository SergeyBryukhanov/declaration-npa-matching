"""
LLM-реранк кандидатов-НПА локальной моделью (Qwen2.5-7B-Instruct).

Почему llama.cpp / GGUF, а не transformers+bitsandbytes:
  - предсказуемый и небольшой footprint по RAM (важно: лимит 8 ГБ RAM не
    включает видеопамять GPU, значит именно ОЗУ - узкое место);
  - одинаково работает на CPU-only и на GPU (n_gpu_layers=-1 offload'ит,
    что может, остальное считает на CPU) - надёжнее для воспроизводимости
    на неизвестном железе проверяющего, чем CUDA-only 4-bit через
    bitsandbytes;
  - квантованный Q4_K_M чекпоинт Qwen2.5-7B-Instruct занимает ~4.7 ГБ,
    что оставляет запас под BM25-индексы/эмбеддинги/интерпретатор в
    пределах 8 ГБ. Рассмотренная альтернатива - transformers+bitsandbytes
    - даёт чуть более высокое качество генерации и удобнее батчить, но
    требует CUDA и не даёт таких же гарантий по памяти на CPU-фолбэке.

Промпт - listwise-реранк одним вызовом на декларацию (а не pointwise на
каждую пару декларация-НПА): при N≈25 кандидатах на декларацию и 151
декларации это 151 обращение к модели вместо 151*25=3775 - укладывается
в лимит времени.

Формат вывода - строгий JSON, распарсенный и провалидированный: модель
не может "изобрести" regulation_id вне поданного списка кандидатов
(невалидные id отбрасываются, а не передаются дальше).
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .text_normalize import truncate

logger = logging.getLogger("llm_rerank")


@dataclass(frozen=True)
class Candidate:
    regulation_id: str
    text: str
    decree_number: str
    retrieval_score: float


SYSTEM_PROMPT = """Ты - эксперт по таможенному регулированию и экспортному контролю РФ/ЕАЭС.
Тебе даётся описание товара из таможенной декларации и список НПА-кандидатов
(нормативно-правовые акты, регулирующие товары двойного назначения, ядерные,
ракетные, химические, биологические технологии и нетарифное регулирование ЕАЭС).

Твоя задача - оценить релевантность КАЖДОГО кандидата для данного товара.

КРИТИЧЕСКИ ВАЖНО - различай два уровня совпадения:
1. Категориальное совпадение: товар и НПА относятся к одной товарной категории
   (например, оба упоминают "подшипники" или "лазеры"), но у НПА есть
   конкретный количественный/качественный критерий (класс точности, диаметр,
   мощность, чистота вещества, частота и т.п.), который текст декларации
   НЕ подтверждает и не опровергает.
2. Подтверждённое совпадение: текст декларации содержит достаточно данных,
   чтобы заключить, что критерий НПА действительно выполняется.

Если совпадение только категориальное (п.1) - давай НИЗКИЙ score (0.05-0.25),
а не высокий, даже при сильном лексическом пересечении. Высокий score (>0.6)
ставь только когда конкретные числовые/технические критерии НПА либо явно
подтверждаются текстом, либо когда НПА не содержит уточняющих критериев вовсе
(и категориального совпадения достаточно).

Отвечай СТРОГО в формате JSON-массива, без пояснений до или после, без
markdown-разметки:
[{"id": "<regulation_id>", "score": <число от 0 до 1>}, ...]

В массиве должны быть ВСЕ id из списка кандидатов, отсортированные по
убыванию score. Не придумывай id, которых нет в списке кандидатов."""


def build_user_prompt(
    declaration_text: str,
    tnved_context: str,
    candidates: List[Candidate],
    candidate_text_max_chars: int = 380,
) -> str:
    lines = [
        f"ОПИСАНИЕ ТОВАРА ИЗ ДЕКЛАРАЦИИ:\n{declaration_text.strip()}",
    ]
    if tnved_context:
        lines.append(
            f"\nБЛИЖАЙШАЯ ОФИЦИАЛЬНАЯ ТОВАРНАЯ КЛАССИФИКАЦИЯ (ТН ВЭД, справочно, "
            f"НЕ является подтверждением применимости конкретного НПА):\n{tnved_context.strip()}"
        )
    lines.append("\nКАНДИДАТЫ НПА:")
    for i, c in enumerate(candidates, 1):
        snippet = truncate(c.text, candidate_text_max_chars)
        lines.append(f'{i}. id="{c.regulation_id}" (decree {c.decree_number}): {snippet}')
    lines.append(
        f"\nВерни JSON-массив ровно из {len(candidates)} объектов "
        f"(все id из списка выше), отсортированный по убыванию score."
    )
    return "\n".join(lines)


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_llm_json(raw_output: str, valid_ids: set) -> List[Tuple[str, float]]:
    """
    Робастный парсинг ответа LLM. Возвращает [(regulation_id, score), ...],
    отсортированный по убыванию score, только для валидных id.
    При полном сбое парсинга возвращает [] (пайплайн должен фолбэкнуться
    на retrieval-скор - см. pipeline.py).
    """
    match = _JSON_ARRAY_RE.search(raw_output)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    out = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        reg_id = item.get("id")
        score = item.get("score")
        if reg_id not in valid_ids or reg_id in seen:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not (score == score):  # NaN check
            continue
        seen.add(reg_id)
        out.append((reg_id, score))

    out.sort(key=lambda x: -x[1])
    return out


class LLMReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        declaration_text: str,
        tnved_context: str,
        candidates: List[Candidate],
    ) -> List[Tuple[str, float]]:
        """Возвращает [(regulation_id, score), ...] по убыванию score.

        Может вернуть МЕНЬШЕ элементов, чем len(candidates), если часть
        ответа модели не прошла валидацию - вызывающий код обязан
        досчитать недостающее (см. pipeline.fill_to_top_n)."""


class QwenLlamaCppReranker(LLMReranker):
    """Продакшн-реализация поверх llama-cpp-python + локальный GGUF Qwen2.5-7B-Instruct."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 900,
        candidate_text_max_chars: int = 380,
        verbose: bool = False,
    ):
        import llama_cpp  # локальный импорт: тяжёлая зависимость нужна
        from llama_cpp import Llama  # только в реальном режиме, не в --dry-run.

        # --- Диагностика GPU (Проблема №11) ------------------------------
        # n_gpu_layers=-1 в конфиге ничего не гарантирует: если установленная
        # сборка llama-cpp-python скомпилирована БЕЗ поддержки CUDA (обычный
        # CPU-wheel), офлоад молча не произойдёт и всё уйдёт на CPU без
        # единой ошибки. На Windows встречаются и обратные случаи: сборка
        # заявлена как CUDA, но офлоад всё равно не срабатывает (см.
        # https://github.com/abetlen/llama-cpp-python/issues/2079).
        # Поэтому явно логируем факт поддержки GPU этой сборкой ДО загрузки
        # модели, чтобы не гадать по одной лишь скорости, использует ли GPU.
        gpu_build_supported = None
        try:
            gpu_build_supported = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            pass  # старые версии llama-cpp-python могут не иметь этой функции

        if n_gpu_layers != 0:
            if gpu_build_supported is True:
                logger.info(
                    "llama-cpp-python собран с поддержкой GPU-офлоада, "
                    "запрошено n_gpu_layers=%s.", n_gpu_layers,
                )
            elif gpu_build_supported is False:
                logger.warning(
                    "Установленная сборка llama-cpp-python БЕЗ поддержки GPU-офлоада "
                    "(обычный CPU-wheel) - n_gpu_layers=%s будет проигнорирован, "
                    "вся генерация пойдёт на CPU. Чтобы задействовать GPU, "
                    "переустановите пакет с CUDA-сборкой (см. README, раздел "
                    "'Использование GPU').", n_gpu_layers,
                )
            else:
                logger.info(
                    "Не удалось определить наличие GPU-поддержки в этой версии "
                    "llama-cpp-python (нет llama_supports_gpu_offload). "
                    "Смотрите строки ниже при загрузке модели: там llama.cpp "
                    "печатает, сколько слоёв реально ушло на GPU."
                )

        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,  # намеренно True: именно verbose-вывод llama.cpp
            # показывает построчно, сколько слоёв ушло на GPU/CPU при загрузке -
            # самый надёжный способ проверить офлоад на практике, а не по API.
        )
        if n_threads:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.candidate_text_max_chars = candidate_text_max_chars

    def rerank(
        self,
        declaration_text: str,
        tnved_context: str,
        candidates: List[Candidate],
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        user_prompt = build_user_prompt(
            declaration_text, tnved_context, candidates, self.candidate_text_max_chars
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            raw = completion["choices"][0]["message"]["content"]
        except Exception:
            # Сбой генерации (например, переполнение контекста на длинном
            # промпте) - откатываемся на пустой результат, pipeline.py
            # досчитает по retrieval-скору.
            return []

        valid_ids = {c.regulation_id for c in candidates}
        return parse_llm_json(raw, valid_ids)


class StubReranker(LLMReranker):
    """ТОЛЬКО для dry-run/тестов: просто пропускает retrieval-порядок без изменений.

    Не используется при реальной оценке - активируется флагом run.py --dry-run."""

    def rerank(
        self,
        declaration_text: str,
        tnved_context: str,
        candidates: List[Candidate],
    ) -> List[Tuple[str, float]]:
        return [(c.regulation_id, c.retrieval_score) for c in candidates]
