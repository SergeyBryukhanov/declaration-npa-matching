"""
Загрузка входных данных и запись результата.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Declaration:
    declaration_id: str
    text: str  # нормализованное значение поля G31_1


@dataclass(frozen=True)
class Regulation:
    regulation_id: str
    decree_number: str
    text: str


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Некорректный JSON в {path}, строка {line_no}: {e}") from e
    return records


def load_declarations(path: str) -> List[Declaration]:
    raw = load_jsonl(path)
    out = []
    for r in raw:
        decl_id = r.get("declaration_id")
        text = r.get("G31_1") or ""
        if not decl_id:
            raise ValueError(f"Запись без declaration_id: {r}")
        out.append(Declaration(declaration_id=str(decl_id), text=str(text)))
    if len(out) != len({d.declaration_id for d in out}):
        raise ValueError("Обнаружены дублирующиеся declaration_id во входном файле")
    return out


def load_regulations(path: str) -> List[Regulation]:
    raw = load_jsonl(path)
    out = []
    for r in raw:
        reg_id = r.get("regulation_id")
        if not reg_id:
            raise ValueError(f"Запись без regulation_id: {r}")
        out.append(
            Regulation(
                regulation_id=str(reg_id),
                decree_number=str(r.get("decree_number") or ""),
                text=str(r.get("npa") or ""),
            )
        )
    if len(out) != len({r.regulation_id for r in out}):
        raise ValueError("Обнаружены дублирующиеся regulation_id во входном файле")
    return out


def write_predictions_csv(
    path: str,
    predictions: Dict[str, List[Tuple[str, float]]],
) -> None:
    """
    predictions: declaration_id -> [(regulation_id, score), ...] длиной ровно 10,
                 уже отсортированный по убыванию score (rank 1 = первый элемент).

    Пишет файл целиком за один раз. Для длинного прогона предпочтительнее
    PredictionsWriter ниже - он дописывает строки по мере готовности, чтобы
    прерванный запуск не терял уже посчитанный результат.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["declaration_id", "rank", "regulation_id", "score"])
        for decl_id, ranked in predictions.items():
            for rank, (reg_id, score) in enumerate(ranked, start=1):
                writer.writerow([decl_id, rank, reg_id, f"{score:.6f}"])


def write_timing_debug_csv(path: str, timings: List[Tuple[str, "StepTiming"]]) -> None:
    """
    Диагностический файл (НЕ часть обязательного формата задания) - по
    строке на декларацию: время по этапам + признаки уверенности retrieval
    + (если LLM вызывалась) согласие её ответа с retrieval. Открывается
    pandas'ом для построения распределений, и читается
    scripts/calibrate_skip.py для подбора порогов пропуска LLM.

        import pandas as pd
        df = pd.read_csv('out/timing_debug.csv')
        df['llm_s'].hist(bins=30)

    Пустая ячейка = значение неприменимо (например, top1_agree пуст, если LLM
    для декларации не вызывалась).
    """

    def fmt(v, digits=4):
        if v is None:
            return ""
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return v

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "declaration_id", "anchor_s", "retrieval_s", "llm_s", "total_s", "used_llm",
            "skip_reason", "would_skip", "signals_agree", "dense_gap", "bm25_ratio",
            "tnved_exact", "top1_agree", "top3_overlap",
        ])
        for decl_id, t in timings:
            feat = t.features
            writer.writerow([
                decl_id, fmt(t.anchor_s), fmt(t.retrieval_s), fmt(t.llm_s), fmt(t.total_s),
                int(t.used_llm), t.skip_reason, fmt(t.would_skip),
                fmt(feat.signals_agree if feat else None),
                fmt(feat.dense_gap if feat else None, 6),
                fmt(feat.bm25_ratio if feat else None, 4),
                fmt(feat.tnved_exact if feat else None),
                fmt(t.top1_agree), fmt(t.top3_overlap),
            ])


class PredictionsWriter:
    """
    Инкрементальная запись predictions.csv - строки декларации дописываются
    и сбрасываются на диск (flush) сразу после того, как она посчитана,
    а не в самом конце всего прогона.

    Зачем: при 151 декларации и LLM-реранке на CPU полный прогон может
    занимать десятки минут; без инкрементальной записи прерывание (Ctrl+C,
    сбой питания, случайное закрытие терминала) на середине уничтожало бы
    весь уже посчитанный результат. С этим классом на диске в любой момент
    лежит корректный CSV по уже обработанным декларациям (не по всем 151 -
    это не финальный валидный файл, пока не обработаны все, но и не пусто).

    Использование:
        with PredictionsWriter(path) as w:
            for decl in declarations:
                ranked = ...
                w.write_declaration(decl.declaration_id, ranked)
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["declaration_id", "rank", "regulation_id", "score"])
        self._file.flush()

    def write_declaration(self, declaration_id: str, ranked: List[Tuple[str, float]]) -> None:
        for rank, (reg_id, score) in enumerate(ranked, start=1):
            self._writer.writerow([declaration_id, rank, reg_id, f"{score:.6f}"])
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "PredictionsWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
