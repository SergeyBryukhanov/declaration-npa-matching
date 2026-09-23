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
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["declaration_id", "rank", "regulation_id", "score"])
        for decl_id, ranked in predictions.items():
            for rank, (reg_id, score) in enumerate(ranked, start=1):
                writer.writerow([decl_id, rank, reg_id, f"{score:.6f}"])
