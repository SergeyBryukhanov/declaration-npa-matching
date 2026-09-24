"""
Оркестрация полного пайплайна: декларация -> (анкоринг ТН ВЭД) ->
(гибридный retrieval по НПА) -> (LLM-реранк) -> ровно 10 НПА с score.

Ключевой инвариант, за который отвечает этот модуль: НЕЗАВИСИМО от того,
насколько "удачно" отработали ретривер и LLM, на выходе для каждой
декларации гарантированно 10 УНИКАЛЬНЫХ regulation_id с корректными
рангами 1..10 и монотонно убывающим score. Формат обязан быть верным
для 100% деклараций (так в задании), поэтому вся защитная логика
(fallback на retrieval-скор при неполном LLM-ответе, добор кандидатов)
живёт здесь, а не полагается на "идеальное" поведение модели.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from .config import FINAL_TOP_N, LLMConfig, RetrievalConfig
from .embeddings import EmbeddingBackend
from .io_utils import Declaration, Regulation
from .llm_rerank import Candidate, LLMReranker
from .retrieval import HybridCorpusIndex
from .tnved import TnvedIndex, build_enriched_query

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm всегда должен быть в requirements.txt,
    # но пайплайн не должен падать целиком только из-за отсутствия прогресс-бара.
    tqdm = None

logger = logging.getLogger("pipeline")

# Тип колбэка, вызываемого сразу после обработки каждой декларации - используется
# run.py для инкрементальной записи predictions.csv (см. "Проблема №10" в истории
# правок: раньше файл писался только в самом конце, и Ctrl+C на 11-й минуте
# уничтожал весь прогресс).
OnResultCallback = Callable[[str, List[Tuple[str, float]]], None]


def fill_to_top_n(
    llm_ranked: List[Tuple[str, float]],
    retrieval_ranked: List[Tuple[str, float]],
    top_n: int = FINAL_TOP_N,
) -> List[Tuple[str, float]]:
    """
    Сборка финального списка ровно из top_n уникальных (regulation_id, score).

    Приоритет: сначала валидные результаты LLM (score из LLM, диапазон
    [0,1]), затем добор оставшимися кандидатами по retrieval-скору -
    но их score ПОНИЖАЕТСЯ так, чтобы они гарантированно шли ниже любого
    LLM-скора (иначе "большее значение = более релевантно" может нарушиться,
    если retrieval-скор случайно окажется выше LLM-скора).

    Если и retrieval-кандидатов не хватило до top_n (не должно происходить
    при разумном npa_candidate_k >= 10, но защищаемся) - паникуем явно,
    вызывающий код (pipeline.run) обязан подать кандидатов с запасом.
    """
    used_ids = set()
    result: List[Tuple[str, float]] = []

    llm_min_score = min((s for _, s in llm_ranked), default=0.0)
    # Все fallback-очки строго ниже минимального LLM-скора, чтобы сортировка
    # "score убывает = релевантность убывает" не нарушалась на стыке.
    fallback_ceiling = min(llm_min_score, 0.0) - 1e-6 if llm_ranked else 0.0

    for reg_id, score in llm_ranked:
        if reg_id in used_ids:
            continue
        used_ids.add(reg_id)
        result.append((reg_id, score))
        if len(result) == top_n:
            return result

    # Добор оставшимися retrieval-кандидатами, с понижающимся синтетическим
    # score, чтобы сохранить строгий порядок и не завысить уверенность.
    offset = 1
    for reg_id, _retrieval_score in retrieval_ranked:
        if reg_id in used_ids:
            continue
        used_ids.add(reg_id)
        synthetic_score = fallback_ceiling - offset * 1e-6
        result.append((reg_id, synthetic_score))
        offset += 1
        if len(result) == top_n:
            return result

    raise RuntimeError(
        f"Не удалось набрать {top_n} уникальных кандидатов "
        f"(retrieval вернул слишком мало кандидатов - увеличьте npa_candidate_k)."
    )


class Pipeline:
    def __init__(
        self,
        declarations: List[Declaration],
        regulations: List[Regulation],
        tnved_index: TnvedIndex,
        embedding_backend: EmbeddingBackend | None,
        llm_reranker: LLMReranker,
        retrieval_cfg: RetrievalConfig,
        llm_cfg: LLMConfig,
        time_budget_seconds: float | None = None,
        llm_cutoff_fraction: float = 0.85,
    ):
        self.declarations = declarations
        self.regulations = regulations
        self.reg_by_id = {r.regulation_id: r for r in regulations}
        self.tnved_index = tnved_index
        self.embedding_backend = embedding_backend
        self.llm_reranker = llm_reranker
        self.cfg = retrieval_cfg
        self.llm_cfg = llm_cfg
        self.time_budget_seconds = time_budget_seconds
        self.llm_cutoff_fraction = llm_cutoff_fraction

        logger.info("Building hybrid NPA index over %d regulations...", len(regulations))
        self.npa_index = HybridCorpusIndex(
            doc_ids=[r.regulation_id for r in regulations],
            doc_texts=[r.text for r in regulations],
            embedding_backend=embedding_backend,
            bm25_k1=retrieval_cfg.bm25_k1,
            bm25_b=retrieval_cfg.bm25_b,
            rrf_k=retrieval_cfg.rrf_k,
        )

    def _process_one(self, decl: Declaration, use_llm: bool) -> List[Tuple[str, float]]:
        tnved_matches = self.tnved_index.anchor(decl.text, top_k=self.cfg.tnved_top_k)
        enriched_query = build_enriched_query(decl.text, tnved_matches, max_matches=2)

        retrieval_ranked = self.npa_index.search(enriched_query, top_k=self.cfg.npa_candidate_k)

        if not use_llm:
            return fill_to_top_n([], retrieval_ranked, top_n=FINAL_TOP_N)

        candidates = [
            Candidate(
                regulation_id=reg_id,
                text=self.reg_by_id[reg_id].text,
                decree_number=self.reg_by_id[reg_id].decree_number,
                retrieval_score=score,
            )
            for reg_id, score in retrieval_ranked
        ]
        tnved_context = " ".join(m.text for m in tnved_matches[:2])

        try:
            llm_ranked = self.llm_reranker.rerank(decl.text, tnved_context, candidates)
        except Exception:
            logger.exception("LLM-реранк упал для declaration_id=%s, фолбэк на retrieval", decl.declaration_id)
            llm_ranked = []

        return fill_to_top_n(llm_ranked, retrieval_ranked, top_n=FINAL_TOP_N)

    def run(self, on_result: Optional[OnResultCallback] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        on_result(declaration_id, ranked_10): вызывается сразу после того, как
        для декларации собраны итоговые 10 строк - до перехода к следующей.
        run.py использует это, чтобы дописывать predictions.csv построчно
        (см. io_utils.PredictionsWriter), а не одним файлом в конце: так
        прерванный на середине запуск не теряет уже посчитанный результат.
        """
        start = time.monotonic()
        cutoff_seconds = (
            self.time_budget_seconds * self.llm_cutoff_fraction
            if self.time_budget_seconds
            else None
        )

        predictions: Dict[str, List[Tuple[str, float]]] = {}
        llm_skipped_count = 0

        iterator = self.declarations
        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                total=len(self.declarations),
                desc="Ранжирование НПА",
                unit="декл",
                dynamic_ncols=True,
            )

        try:
            for i, decl in enumerate(self.declarations, 1):
                elapsed = time.monotonic() - start
                use_llm = cutoff_seconds is None or elapsed < cutoff_seconds
                if not use_llm:
                    llm_skipped_count += 1

                t_item_start = time.monotonic()
                ranked = self._process_one(decl, use_llm=use_llm)
                item_seconds = time.monotonic() - t_item_start

                predictions[decl.declaration_id] = ranked
                if on_result is not None:
                    on_result(decl.declaration_id, ranked)

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "сек/декл": f"{item_seconds:.1f}",
                            "LLM": "нет" if not use_llm else "да",
                        }
                    )
                    progress_bar.update(1)
                elif i % 10 == 0 or i == len(self.declarations):
                    # Фолбэк без tqdm: печатаем реже, но не раз в 25 (проблема
                    # №10 - на медленном железе первая строка могла не
                    # появляться очень долго).
                    logger.info(
                        "Обработано %d/%d деклараций (%.1fs, последняя заняла %.1fs)%s",
                        i, len(self.declarations), elapsed, item_seconds,
                        " [бюджет времени исчерпан, остаток без LLM]" if not use_llm else "",
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if llm_skipped_count:
            logger.warning(
                "%d деклараций обработаны БЕЗ LLM-реранка из-за лимита времени "
                "(только гибридный retrieval-скор).",
                llm_skipped_count,
            )

        return predictions
