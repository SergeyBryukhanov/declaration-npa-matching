"""
Жёсткая проверка формата out/predictions.csv по требованиям задания:
  - для каждой декларации ровно 10 строк;
  - ровно 10 УНИКАЛЬНЫХ regulation_id на декларацию;
  - ранги 1..10 без повторов;
  - regulation_id входит в множество известных id из regulations.jsonl;
  - score - конечное число;
  - нет деклараций из declarations.jsonl, отсутствующих в выводе.

Можно запускать отдельно как самопроверку:
    python -m src.validate --out ./out/predictions.csv \
        --declarations ./declarations.jsonl --regulations ./regulations.jsonl
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Set


class ValidationError(Exception):
    pass


def validate_predictions_file(
    csv_path: str,
    declaration_ids: Set[str],
    regulation_ids: Set[str],
    expected_n: int = 10,
) -> None:
    rows_by_decl: Dict[str, List[dict]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"declaration_id", "rank", "regulation_id", "score"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValidationError(
                f"Отсутствуют обязательные колонки. Есть: {reader.fieldnames}, "
                f"нужны: {sorted(required_cols)}"
            )
        for row in reader:
            rows_by_decl[row["declaration_id"]].append(row)

    missing = declaration_ids - set(rows_by_decl.keys())
    if missing:
        raise ValidationError(f"Отсутствуют строки для {len(missing)} деклараций, напр.: {list(missing)[:5]}")

    extra = set(rows_by_decl.keys()) - declaration_ids
    if extra:
        raise ValidationError(f"Найдены строки для неизвестных declaration_id: {list(extra)[:5]}")

    for decl_id, rows in rows_by_decl.items():
        if len(rows) != expected_n:
            raise ValidationError(f"{decl_id}: {len(rows)} строк вместо {expected_n}")

        reg_ids_seen = [r["regulation_id"] for r in rows]
        if len(set(reg_ids_seen)) != expected_n:
            raise ValidationError(f"{decl_id}: повторяющиеся regulation_id: {reg_ids_seen}")

        unknown = set(reg_ids_seen) - regulation_ids
        if unknown:
            raise ValidationError(f"{decl_id}: неизвестные regulation_id: {unknown}")

        ranks = sorted(int(r["rank"]) for r in rows)
        if ranks != list(range(1, expected_n + 1)):
            raise ValidationError(f"{decl_id}: ранги должны быть 1..{expected_n} без повторов, получено: {ranks}")

        for r in rows:
            try:
                score = float(r["score"])
            except ValueError:
                raise ValidationError(f"{decl_id}: score не число: {r['score']!r}")
            if not math.isfinite(score):
                raise ValidationError(f"{decl_id}: score не конечное число: {score}")

        # rank 1 должен соответствовать максимальному score (порядок согласован)
        by_rank = {int(r["rank"]): float(r["score"]) for r in rows}
        scores_in_rank_order = [by_rank[i] for i in range(1, expected_n + 1)]
        if scores_in_rank_order != sorted(scores_in_rank_order, reverse=True):
            raise ValidationError(f"{decl_id}: score не монотонно убывает по рангу: {scores_in_rank_order}")


def _main():
    ap = argparse.ArgumentParser(description="Самопроверка формата predictions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--declarations", required=True)
    ap.add_argument("--regulations", required=True)
    args = ap.parse_args()

    from .io_utils import load_declarations, load_regulations

    decl_ids = {d.declaration_id for d in load_declarations(args.declarations)}
    reg_ids = {r.regulation_id for r in load_regulations(args.regulations)}

    validate_predictions_file(args.out, decl_ids, reg_ids)
    print(f"OK: {args.out} прошёл все проверки формата "
          f"({len(decl_ids)} деклараций x 10 регуляций).")


if __name__ == "__main__":
    _main()
