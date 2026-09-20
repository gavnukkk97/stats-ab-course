#!/usr/bin/env python3
"""Regenerate course diagrams; keep logs and timings separate from lesson files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = {
    "M1": ("make_visuals_m1.py",),
    "M2": ("make_visuals_m2.py", "make_visuals_m2b.py"),
    "M3": ("make_visuals_m3.py",),
    "M4": ("make_visuals_m4.py",),
    "M5": ("make_visuals_m5.py",),
    "M6": ("make_visuals_m6.py",),
    "M7": ("make_visuals_m7.py",),
    "nav": ("make_visuals_nav.py",),
    "cases": ("make_visuals_cases.py",),
    "roadmap": ("make_roadmap.py",),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Пересоздать схемы курса. Без --module запускаются все генераторы. "
            "Рисунки сохраняются в папках images соответствующих материалов."
        ),
        epilog=(
            "Пример: python scripts/generate_figures.py --module M2 --module nav. "
            "Графики из практик пересоздаются запуском самих практик. "
            "После генерации обновите зеркало: python scripts/sync_course.py."
        ),
    )
    parser.add_argument(
        "--module",
        action="append",
        choices=GENERATORS,
        help="Группа схем; параметр можно повторить. По умолчанию — все группы.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "build" / "figures",
        help="Папка журналов (по умолчанию: build/figures в репозитории).",
    )
    args = parser.parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(log_dir / "matplotlib-cache"))
    groups = list(dict.fromkeys(args.module or GENERATORS))
    results = []
    started_at = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()

    for group in groups:
        for name in GENERATORS[group]:
            script = ROOT / "scripts" / "figures" / name
            log_path = log_dir / f"{script.stem}.log"
            start = time.perf_counter()
            print(f"{group}: {name}…", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"Interpreter: {sys.executable}\nScript: {script}\n\n")
                log.flush()
                try:
                    result = subprocess.run(
                        [sys.executable, str(script)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                    returncode = result.returncode
                except OSError as exc:
                    log.write(f"Не удалось запустить генератор: {exc}\n")
                    returncode = 1
                elapsed = time.perf_counter() - start
                log.write(f"\nExit code: {returncode}\nDuration: {elapsed:.3f} seconds\n")
            results.append(
                {
                    "module": group,
                    "script": str(script.relative_to(ROOT)),
                    "returncode": returncode,
                    "duration_seconds": round(elapsed, 3),
                    "log": str(log_path),
                }
            )
            status = "готово" if returncode == 0 else f"ошибка ({returncode})"
            print(f"  {status}, {elapsed:.1f} с; журнал: {log_path}", flush=True)

    summary = log_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "started_at": started_at,
                "duration_seconds": round(time.perf_counter() - run_start, 3),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failed = sum(item["returncode"] != 0 for item in results)
    print(f"Завершено: {len(results) - failed}/{len(results)}. Сводка: {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
