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
