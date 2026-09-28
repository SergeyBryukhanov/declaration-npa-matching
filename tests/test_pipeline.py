"""
Тесты. Не требуют GPU/сети/весов моделей - покрывают детерминированную
логику (парсинг, BM25, ТН ВЭД-якоринг, слияние, парсинг JSON от LLM,
гарантии формата вывода) плюс один интеграционный dry-run всего run.py
на реальных данных проекта.

Запуск:
    pytest tests/            # если pytest установлен
    python tests/test_pipeline.py   # без pytest, тем же файлом
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25
from src.io_utils import load_declarations, load_regulations
from src.llm_rerank import parse_llm_json
from src.pipeline import fill_to_top_n
from src.retrieval import HybridCorpusIndex, reciprocal_rank_fusion
from src.text_normalize import normalize_text, tokenize
from src.tnved import TnvedIndex, parse_tnved_file
from src.validate import ValidationError, validate_predictions_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECLARATIONS_PATH = os.path.join(PROJECT_ROOT, "declarations.jsonl")
REGULATIONS_PATH = os.path.join(PROJECT_ROOT, "regulations.jsonl")
TNVED_PATH = os.path.join(PROJECT_ROOT, "tnved_knowledge.txt")


# --------------------------------------------------------------------------
# text_normalize
# --------------------------------------------------------------------------

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_text("  ПРИВЕТ   МИР  ") == "привет мир"


def test_tokenize_keeps_numbers_and_units():
    toks = tokenize("диаметр 30 мм")
    assert "30" in toks and "мм" in toks


# --------------------------------------------------------------------------
# bm25
# --------------------------------------------------------------------------

def test_bm25_ranks_matching_document_first():
    corpus = [
        tokenize("подшипники шариковые диаметр 30 мм"),
        tokenize("нептуний обогащенный любая форма"),
        tokenize("зеркала для лазера"),
    ]
    bm = BM25(corpus)
    top = bm.top_k(tokenize("подшипники шариковые диаметр"), k=3)
    assert top[0][0] == 0
    assert top[0][1] > 0


def test_bm25_empty_query_gives_zero_scores():
    bm = BM25([tokenize("текст один"), tokenize("текст два")])
    scores = bm.get_scores([])
    assert (scores == 0).all()


# --------------------------------------------------------------------------
# retrieval fusion
# --------------------------------------------------------------------------

def test_reciprocal_rank_fusion_prefers_doc_ranked_high_in_both():
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "C"]], k=60)
    assert fused["A"] > fused["C"]
    assert fused["B"] > fused["C"]


# --------------------------------------------------------------------------
# tnved anchoring (реальный справочник)
# --------------------------------------------------------------------------

def test_tnved_parses_real_file_and_finds_known_bearing_code():
    entries = parse_tnved_file(TNVED_PATH)
    assert len(entries) > 10_000
    codes = {e.code for e in entries}
    assert "8482101009" in codes  # подшипники шариковые ≤30мм, прочие


def test_tnved_anchor_returns_plausible_match_for_bearing_declaration():
    idx = TnvedIndex.from_file(TNVED_PATH)
    text = "ПОДШИПНИКИ ШАРИКОВЫЕ, НАИБОЛЬШИЙ НАРУЖНЫЙ ДИАМЕТР КОТОРЫХ НЕ БОЛЕЕ 30 ММ, РАДИАЛЬНЫЕ"
    matches = idx.anchor(text, top_k=5)
    assert len(matches) > 0
    assert any("подшипник" in m.text for m in matches)


# --------------------------------------------------------------------------
# hybrid regulation retrieval (реальный корпус НПА)
# --------------------------------------------------------------------------

def test_hybrid_retrieval_finds_known_bearing_regulations():
    from src.embeddings import DummyHashEmbeddingBackend

    regs = load_regulations(REGULATIONS_PATH)
    idx = HybridCorpusIndex(
        doc_ids=[r.regulation_id for r in regs],
        doc_texts=[r.text for r in regs],
        embedding_backend=DummyHashEmbeddingBackend(),
    )
    results = idx.search("подшипники шариковые радиальные диаметр 30 мм", top_k=10)
    result_ids = {rid for rid, _ in results}
    # NPA0553/NPA0556 - известные регуляции про прецизионные подшипники,
    # должны быть среди топ-10 по лексическому пересечению (независимо
    # от того, подтвердит ли их LLM на следующем шаге).
    assert "NPA0553" in result_ids or "NPA0556" in result_ids


# --------------------------------------------------------------------------
# LLM JSON parsing robustness
# --------------------------------------------------------------------------

def test_parse_llm_json_clean():
    out = parse_llm_json('[{"id":"A","score":0.9},{"id":"B","score":0.1}]', {"A", "B"})
    assert out == [("A", 0.9), ("B", 0.1)]


def test_parse_llm_json_drops_hallucinated_ids():
    out = parse_llm_json('[{"id":"A","score":0.9},{"id":"GHOST","score":0.99}]', {"A", "B"})
    assert out == [("A", 0.9)]


def test_parse_llm_json_handles_markdown_and_preamble():
    raw = 'Ответ:\n```json\n[{"id": "A", "score": 0.5}]\n```\n'
    out = parse_llm_json(raw, {"A"})
    assert out == [("A", 0.5)]


def test_parse_llm_json_garbage_returns_empty():
    assert parse_llm_json("не могу помочь", {"A"}) == []


def test_parse_llm_json_dedupes_keeping_first():
    out = parse_llm_json('[{"id":"A","score":0.1},{"id":"A","score":0.9}]', {"A"})
    assert out == [("A", 0.1)]


# --------------------------------------------------------------------------
# fill_to_top_n - главный контракт корректности формата
# --------------------------------------------------------------------------

def test_fill_to_top_n_full_llm_result():
    llm = [(f"N{i}", 1.0 - i * 0.05) for i in range(10)]
    retr = [(f"N{i}", 0.5) for i in range(20)]
    out = fill_to_top_n(llm, retr, top_n=10)
    assert len(out) == 10
    assert len({rid for rid, _ in out}) == 10
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_fill_to_top_n_partial_llm_falls_back_and_stays_sorted():
    llm = [("N0", 0.9), ("N2", 0.5), ("N0", 0.99)]  # дубль N0, только 2 уникальных
    retr = [(f"N{i}", 1.0 - i * 0.01) for i in range(15)]
    out = fill_to_top_n(llm, retr, top_n=10)
    assert len(out) == 10
    assert len({rid for rid, _ in out}) == 10
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    # fallback-элементы должны идти строго ниже llm-элементов
    llm_ids = {"N0", "N2"}
    llm_scores = [s for rid, s in out if rid in llm_ids]
    fallback_scores = [s for rid, s in out if rid not in llm_ids]
    assert min(llm_scores) > max(fallback_scores)


def test_fill_to_top_n_raises_when_not_enough_candidates():
    try:
        fill_to_top_n([], [("N0", 1.0)], top_n=10)
        assert False, "должно было упасть"
    except RuntimeError:
        pass


# --------------------------------------------------------------------------
# validate.py
# --------------------------------------------------------------------------

def test_validate_accepts_correct_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "predictions.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["declaration_id", "rank", "regulation_id", "score"])
            for r in range(1, 11):
                w.writerow(["D1", r, f"R{r}", 1.0 - r * 0.01])
        validate_predictions_file(path, {"D1"}, {f"R{r}" for r in range(1, 11)})  # не должно бросить


def test_validate_rejects_duplicate_rank():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "predictions.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["declaration_id", "rank", "regulation_id", "score"])
            w.writerow(["D1", 1, "R1", 0.9])
            w.writerow(["D1", 1, "R2", 0.8])  # дублирующийся ранг
            for r in range(3, 11):
                w.writerow(["D1", r, f"R{r}", 1.0 - r * 0.01])
        try:
            validate_predictions_file(path, {"D1"}, {f"R{r}" for r in range(1, 11)})
            assert False, "должно было упасть"
        except ValidationError:
            pass


# --------------------------------------------------------------------------
# Интеграционный dry-run: полный run.py на реальных данных проекта
# --------------------------------------------------------------------------

def test_end_to_end_dry_run_produces_valid_output():
    decls = load_declarations(DECLARATIONS_PATH)
    regs = load_regulations(REGULATIONS_PATH)

    with tempfile.TemporaryDirectory() as tmp_out:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(PROJECT_ROOT, "run.py"),
                "--out", tmp_out,
                "--dry-run",
                "--log-level", "WARNING",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        out_csv = os.path.join(tmp_out, "predictions.csv")
        assert os.path.exists(out_csv)
        validate_predictions_file(
            out_csv,
            {d.declaration_id for d in decls},
            {r.regulation_id for r in regs},
        )


# --------------------------------------------------------------------------
# Пропуск LLM для однозначных случаев (src/skip_rule.py + pipeline)
# --------------------------------------------------------------------------

def _features(**kw):
    from src.skip_rule import ConfidenceFeatures

    base = dict(signals_agree=True, dense_gap=0.10, bm25_ratio=3.0, tnved_exact=False)
    base.update(kw)
    return ConfidenceFeatures(**base)


def test_is_confident_requires_all_conditions():
    from src.skip_rule import is_confident

    ok = dict(min_dense_gap=0.05, min_bm25_ratio=2.0)
    assert is_confident(_features(), **ok)
    assert not is_confident(_features(signals_agree=False), **ok)   # лидеры разошлись
    assert not is_confident(_features(signals_agree=None), **ok)    # нет dense-сигнала
    assert not is_confident(_features(dense_gap=0.01), **ok)        # мал отрыв
    assert not is_confident(_features(dense_gap=None), **ok)
    assert not is_confident(_features(bm25_ratio=1.1), **ok)        # BM25 не выделил лидера
    assert not is_confident(_features(), require_tnved_exact=True, **ok)
    assert is_confident(_features(tnved_exact=True), require_tnved_exact=True, **ok)


def test_skip_config_validation():
    from src.config import SkipLLMConfig

    SkipLLMConfig(mode="off").validate()
    SkipLLMConfig(mode="shadow").validate()
    SkipLLMConfig(mode="on", min_dense_gap=0.05, min_bm25_ratio=2.0).validate()
    for bad in (SkipLLMConfig(mode="on"),
                SkipLLMConfig(mode="on", min_dense_gap=0.05),
                SkipLLMConfig(mode="maybe")):
        try:
            bad.validate()
            assert False, f"должно было упасть: {bad}"
        except ValueError:
            pass


def test_search_with_signals_matches_search():
    from src.embeddings import DummyHashEmbeddingBackend

    regs = load_regulations(REGULATIONS_PATH)
    idx = HybridCorpusIndex([r.regulation_id for r in regs], [r.text for r in regs],
                            DummyHashEmbeddingBackend())
    q = "подшипники шариковые радиальные диаметр 30 мм"
    res = idx.search_with_signals(q, top_k=10)
    assert res.ranked == idx.search(q, top_k=10)
    sig = res.signals
    assert sig.bm25_top1 >= sig.bm25_top2 >= 0
    assert sig.dense_top1 >= sig.dense_top2
    assert sig.dense_top1_id is not None


_TNVED_INDEX_CACHE = {}


def _tiny_pipeline(skip_cfg, reranker):
    """12 регуляций с непересекающимся словарём; D1 - однозначная декларация
    (дословно текст R3), D2 - неоднозначная (по одному токену от четырёх НПА)."""
    from src.config import RetrievalConfig, LLMConfig
    from src.embeddings import DummyHashEmbeddingBackend
    from src.io_utils import Declaration, Regulation
    from src.pipeline import Pipeline

    if "idx" not in _TNVED_INDEX_CACHE:
        _TNVED_INDEX_CACHE["idx"] = TnvedIndex.from_file(TNVED_PATH)
    regs = [Regulation(f"R{i}", "1", f"слово{i}a слово{i}b слово{i}c") for i in range(1, 13)]
    decls = [
        Declaration("D1", "слово3a слово3b слово3c"),
        Declaration("D2", "слово1a слово5b слово9c слово11a"),
    ]
    return Pipeline(
        declarations=decls, regulations=regs,
        tnved_index=_TNVED_INDEX_CACHE["idx"],
        embedding_backend=DummyHashEmbeddingBackend(),
        llm_reranker=reranker,
        retrieval_cfg=RetrievalConfig(), llm_cfg=LLMConfig(),
        skip_cfg=skip_cfg,
    )


class _CountingReverseReranker:
    """Тестовый реранкер: считает вызовы и ПЕРЕВОРАЧИВАЕТ порядок кандидатов
    (лидер LLM заведомо не совпадает с лидером retrieval)."""

    def __init__(self):
        self.calls = []

    def rerank(self, declaration_text, tnved_context, candidates):
        self.calls.append(declaration_text)
        rev = list(reversed(candidates))
        return [(c.regulation_id, 1.0 - 0.05 * i) for i, c in enumerate(rev)]


def _assert_valid_output(predictions):
    for decl_id, ranked in predictions.items():
        assert len(ranked) == 10, decl_id
        assert len({rid for rid, _ in ranked}) == 10, decl_id
        scores = [sc for _, sc in ranked]
        assert scores == sorted(scores, reverse=True), decl_id


def test_skip_on_skips_only_confident_declaration():
    from src.config import SkipLLMConfig

    rr = _CountingReverseReranker()
    pipe = _tiny_pipeline(SkipLLMConfig(mode="on", min_dense_gap=0.05, min_bm25_ratio=2.0), rr)
    preds, timings = pipe.run()
    by_id = dict(timings)

    assert by_id["D1"].skip_reason == "confident" and not by_id["D1"].used_llm
    assert by_id["D2"].skip_reason == "" and by_id["D2"].used_llm
    assert rr.calls == ["слово1a слово5b слово9c слово11a"]  # LLM вызвана ТОЛЬКО для D2
    assert preds["D1"][0][0] == "R3"  # однозначный лидер сохранён
    _assert_valid_output(preds)


def test_skip_shadow_calls_llm_everywhere_and_records_agreement():
    from src.config import SkipLLMConfig

    rr = _CountingReverseReranker()
    pipe = _tiny_pipeline(SkipLLMConfig(mode="shadow", min_dense_gap=0.05, min_bm25_ratio=2.0), rr)
    preds, timings = pipe.run()
    by_id = dict(timings)

    assert len(rr.calls) == 2                       # shadow НИКОГДА не пропускает LLM
    assert by_id["D1"].would_skip is True and by_id["D2"].would_skip is False
    assert by_id["D1"].skip_reason == ""
    assert by_id["D1"].top1_agree is False          # реранкер перевернул порядок
    assert by_id["D1"].top3_overlap is not None
    _assert_valid_output(preds)


def test_skip_off_never_skips_and_has_no_would_skip():
    from src.config import SkipLLMConfig

    rr = _CountingReverseReranker()
    pipe = _tiny_pipeline(SkipLLMConfig(mode="off"), rr)
    _, timings = pipe.run()
    assert len(rr.calls) == 2
    assert all(t.would_skip is None for _, t in timings)   # пороги не заданы
    assert all(t.features is not None for _, t in timings)  # но признаки пишутся всегда


def test_confident_takes_precedence_over_budget_reason():
    from src.config import SkipLLMConfig
    from src.io_utils import Declaration

    pipe = _tiny_pipeline(SkipLLMConfig(mode="on", min_dense_gap=0.05, min_bm25_ratio=2.0),
                          _CountingReverseReranker())
    d1, d2 = pipe.declarations
    _, t1 = pipe._process_one(d1, allow_llm=False)
    _, t2 = pipe._process_one(d2, allow_llm=False)
    assert t1.skip_reason == "confident"   # не считается деградацией из-за бюджета
    assert t2.skip_reason == "budget"


def test_agreement_not_recorded_when_llm_gives_no_answer():
    """Пустой ответ LLM -> в итог подставляется retrieval; 'согласие' при этом
    было бы тривиально 100% и исказило бы калибровку - его не должно быть."""
    from src.config import SkipLLMConfig

    class _Silent:
        def rerank(self, *a, **k):
            return []

    pipe = _tiny_pipeline(SkipLLMConfig(mode="off"), _Silent())
    preds, timings = pipe.run()
    assert all(t.top1_agree is None for _, t in timings)
    _assert_valid_output(preds)


def test_timing_csv_and_calibration_script_roundtrip():
    from src.config import SkipLLMConfig
    from src.io_utils import write_timing_debug_csv

    pipe = _tiny_pipeline(SkipLLMConfig(mode="shadow", min_dense_gap=0.05, min_bm25_ratio=2.0),
                          _CountingReverseReranker())
    _, timings = pipe.run()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "timing_debug.csv")
        write_timing_debug_csv(path, timings)
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for col in ("skip_reason", "would_skip", "signals_agree", "dense_gap",
                    "bm25_ratio", "tnved_exact", "top1_agree", "top3_overlap"):
            assert col in rows[0], col
        r = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "calibrate_skip.py"),
             path, "--min-support", "1"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, r.stderr
        assert "Деклараций с ответом LLM: 2" in r.stdout


def test_cli_rejects_skip_on_without_thresholds():
    r = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "run.py"), "--out", tempfile.gettempdir() + "/x",
         "--dry-run", "--skip-llm", "on"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
    )
    assert r.returncode != 0
    assert "--skip-min-dense-gap" in r.stderr


def test_end_to_end_dry_run_shadow_mode_valid_output():
    decls = load_declarations(DECLARATIONS_PATH)
    regs = load_regulations(REGULATIONS_PATH)
    with tempfile.TemporaryDirectory() as tmp_out:
        r = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "run.py"), "--out", tmp_out, "--dry-run",
             "--skip-llm", "shadow", "--skip-min-dense-gap", "0.05", "--skip-min-bm25-ratio", "1.5",
             "--log-level", "WARNING"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        validate_predictions_file(os.path.join(tmp_out, "predictions.csv"),
                                  {d.declaration_id for d in decls},
                                  {x.regulation_id for x in regs})


if __name__ == "__main__":
    # Простой раннер без pytest: собирает все test_* функции модуля и
    # выполняет по очереди, печатая PASS/FAIL по каждой.
    tests = {name: obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)}
    failed = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошли.")
    sys.exit(1 if failed else 0)
