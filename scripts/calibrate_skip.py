#!/usr/bin/env python3
"""
Подбор порогов пропуска LLM по логу прогона, где LLM отработала на ВСЕХ
декларациях (режим `python run.py --skip-llm shadow` или обычный прогон с off).

    python scripts/calibrate_skip.py out/timing_debug.csv
    python scripts/calibrate_skip.py out/timing_debug.csv --min-top1-agree 0.98 --min-support 15

Что именно измеряется - и чего НЕТ:
  Для каждой декларации в логе есть признаки уверенности retrieval и факт,
  совпала ли верхушка ранжирования retrieval с ответом LLM (лидер и топ-3).
  Скрипт перебирает пороги и для каждой комбинации показывает: сколько
  деклараций прошло бы правило, и как часто на них LLM ПОДТВЕРДИЛА лидера
  retrieval. Пропускать LLM безопасно там, где она почти всегда соглашается.

  Это согласие с ТЕКУЩИМ пайплайном, а не с эталонной разметкой (её у нас
  нет): если LLM ошибается систематически, скрипт этого не увидит.
  Кроме того, пороги подбираются на тех же 151 декларациях, на которых
  оцениваются, - при малой выборке возможен оптимизм; поэтому рядом с долей
  согласия выводится нижняя граница доверительного интервала Уилсона (95%).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from typing import List, Optional

BM25_RATIO_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) / denom


def _f(x: str) -> Optional[float]:
    return float(x) if x not in ("", None) else None


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("used_llm") != "1" or r.get("top1_agree", "") == "":
                continue  # нужны только декларации, где LLM реально ответила
            rows.append({
                "signals_agree": r["signals_agree"] == "1",
                "dense_gap": _f(r["dense_gap"]),
                "bm25_ratio": _f(r["bm25_ratio"]),
                "tnved_exact": r["tnved_exact"] == "1",
                "top1_agree": r["top1_agree"] == "1",
                "top3_overlap": int(r["top3_overlap"]),
                "llm_s": float(r["llm_s"]),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timing_csv")
    ap.add_argument("--min-top1-agree", type=float, default=0.97,
                    help="Требуемая доля совпадения лидера retrieval и LLM (по умолчанию 0.97)")
    ap.add_argument("--min-top3-overlap", type=float, default=2.7,
                    help="Требуемое среднее пересечение топ-3 из 3 (по умолчанию 2.7)")
    ap.add_argument("--min-support", type=int, default=15,
                    help="Минимум деклараций, проходящих правило (по умолчанию 15)")
    args = ap.parse_args()

    rows = load_rows(args.timing_csv)
    if not rows:
        print("В логе нет деклараций с ответом LLM и данными согласия (top1_agree). Нужен прогон, "
              "где LLM работала на декларациях: python run.py --skip-llm shadow --time-budget-min 90")
        return 1

    n = len(rows)
    base_agree = sum(r["top1_agree"] for r in rows)
    mean_llm = statistics.mean(r["llm_s"] for r in rows)
    print(f"Деклараций с ответом LLM: {n}")
    print(f"Лидер retrieval == лидер LLM: {100*base_agree/n:.1f}% ({base_agree}/{n}) - это потолок "
          f"согласия без всякого отбора")
    print(f"Средняя стоимость LLM-вызова: {mean_llm:.1f}с\n")

    agree_rows = [r for r in rows if r["signals_agree"] and r["dense_gap"] is not None]
    print(f"Из них лидеры BM25 и dense совпали: {len(agree_rows)} ({100*len(agree_rows)/n:.0f}%) - "
          f"только они вообще могут пройти правило\n")
    if not agree_rows:
        print("Нет кандидатов на пропуск: сигналы никогда не совпадали.")
        return 1

    gaps = sorted(r["dense_gap"] for r in agree_rows)
    gap_grid = sorted({0.0} | {round(gaps[min(int(len(gaps) * q), len(gaps) - 1)], 4)
                               for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})

    results = []
    seen = set()
    for g in gap_grid:
        for b in BM25_RATIO_GRID:
            subset = [r for r in agree_rows if r["dense_gap"] >= g and r["bm25_ratio"] >= b]
            key = tuple(id(r) for r in subset)
            if not subset or key in seen:
                continue
            seen.add(key)
            k = len(subset)
            a1 = sum(r["top1_agree"] for r in subset)
            results.append({
                "gap": g, "ratio": b, "n": k,
                "agree": a1 / k, "lb": wilson_lower(a1, k),
                "top3": statistics.mean(r["top3_overlap"] for r in subset),
                "saved_min": k * mean_llm / 60,
            })

    results.sort(key=lambda x: (-x["n"], -x["agree"]))
    print(f"{'dense_gap>=':>11} {'bm25_ratio>=':>12} {'пропущено':>10} {'top1 совп.':>11} "
          f"{'(нижн.гр.95%)':>14} {'top3/3':>7} {'экономия':>9}")
    for r in results[:25]:
        print(f"{r['gap']:>11.4f} {r['ratio']:>12.2f} {r['n']:>10d} {100*r['agree']:>10.1f}% "
              f"{100*r['lb']:>13.1f}% {r['top3']:>7.2f} {r['saved_min']:>7.1f}мин")

    ok = [r for r in results
          if r["n"] >= args.min_support
          and r["agree"] >= args.min_top1_agree
          and r["top3"] >= args.min_top3_overlap]
    print()
    if not ok:
        print(f"Ни одна комбинация не удовлетворяет требованиям (top1 >= {args.min_top1_agree:.0%}, "
              f"top3 >= {args.min_top3_overlap}, минимум {args.min_support} деклараций). Это тоже результат: "
              f"на этих данных правило не даёт безопасной экономии - пропуск LLM включать не стоит. "
              f"Можно ослабить требования флагами, но тогда осознанно принимая потери.")
        return 0

    best = ok[0]
    print("РЕКОМЕНДАЦИЯ (максимум пропусков при выполненных требованиях):")
    print(f"  пропустится {best['n']} из {n} деклараций, согласие лидера {100*best['agree']:.1f}% "
          f"(нижняя граница 95%: {100*best['lb']:.1f}%), экономия ≈ {best['saved_min']:.1f} мин")
    print(f"\n  python run.py --out ./out --skip-llm on "
          f"--skip-min-dense-gap {best['gap']} --skip-min-bm25-ratio {best['ratio']}\n")
    print("Оговорки: пороги подобраны на тех же данных, на которых работает решение; согласие "
          "измерено относительно ответов LLM, а не эталонной разметки. Смотрите на нижнюю границу "
          "интервала, а не только на долю.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
