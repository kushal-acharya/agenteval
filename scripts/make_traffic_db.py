"""Generate the synthetic traffic database for S2 (execution-based grading).

Deterministic (seeded): everyone who runs this gets byte-identical data, so
gold SQL answers in datasets/transit_v1.jsonl stay stable.

    python scripts/make_traffic_db.py          # writes datasets/traffic.db

Schema
------
segments(id, corridor, functional_class, district, length_mi)
aadt(segment_id, year, aadt)                       -- yearly
volumes(segment_id, year, month, avg_weekday_volume)
speeds(segment_id, year, month, avg_speed_mph)
crashes(id, segment_id, intersection, crash_date, severity, contributing_factor)
detectors(id, segment_id, year, month, uptime_pct)

Planted signals (so eval questions have real answers, not noise):
- I-66 has the highest AADT corridor-wide in 2024          (tr-09)
- speeding is over-represented in US-29 crashes            (tr-13)
- S-12 average speeds drop ~4 mph from 2024-03 onward      (tr-14)
- crashes peak on Fri/Sat                                  (tr-15)
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 29
YEARS = [2021, 2022, 2023, 2024]

CORRIDORS = {
    # corridor: (functional_class, n_segments, base_aadt)
    "I-66": ("freeway", 8, 95_000),
    "US-29": ("arterial", 10, 28_000),
    "US-50": ("arterial", 8, 24_000),
    "SR-7": ("arterial", 8, 18_000),
    "Elm St": ("collector", 6, 6_000),
}

INTERSECTIONS = [
    "Main St & 5th Ave", "US-29 & Gallows Rd", "SR-7 & Broad St",
    "Elm St & Park Ave", "US-50 & Annandale Rd", "US-29 & Lee Hwy",
    "SR-7 & Maple Ave", "Main St & 2nd St",
]

SEVERITIES = ["K", "A", "B", "C", "O"]
SEVERITY_WEIGHTS = [0.01, 0.05, 0.14, 0.30, 0.50]
FACTORS = ["speeding", "alcohol", "distraction", "weather", "failure to yield", "other"]


def build(db_path: Path) -> None:
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE segments (
            id TEXT PRIMARY KEY, corridor TEXT NOT NULL,
            functional_class TEXT NOT NULL, district INTEGER NOT NULL,
            length_mi REAL NOT NULL);
        CREATE TABLE aadt (
            segment_id TEXT REFERENCES segments(id), year INTEGER, aadt INTEGER,
            PRIMARY KEY (segment_id, year));
        CREATE TABLE volumes (
            segment_id TEXT REFERENCES segments(id), year INTEGER, month INTEGER,
            avg_weekday_volume INTEGER, PRIMARY KEY (segment_id, year, month));
        CREATE TABLE speeds (
            segment_id TEXT REFERENCES segments(id), year INTEGER, month INTEGER,
            avg_speed_mph REAL, PRIMARY KEY (segment_id, year, month));
        CREATE TABLE crashes (
            id INTEGER PRIMARY KEY, segment_id TEXT REFERENCES segments(id),
            intersection TEXT, crash_date TEXT NOT NULL,
            severity TEXT NOT NULL, contributing_factor TEXT NOT NULL);
        CREATE TABLE detectors (
            id TEXT, segment_id TEXT REFERENCES segments(id),
            year INTEGER, month INTEGER, uptime_pct REAL,
            PRIMARY KEY (id, year, month));
    """)

    # --- segments -----------------------------------------------------------
    seg_rows, seg_meta = [], {}
    n = 0
    for corridor, (fclass, count, base) in CORRIDORS.items():
        for _ in range(count):
            n += 1
            sid = f"S-{n:02d}"
            seg_rows.append((sid, corridor, fclass, rng.randint(1, 3),
                             round(rng.uniform(0.4, 3.5), 2)))
            seg_meta[sid] = (corridor, base)
    cur.executemany("INSERT INTO segments VALUES (?,?,?,?,?)", seg_rows)

    # --- aadt / volumes / speeds / detectors --------------------------------
    for sid, (corridor, base) in seg_meta.items():
        seg_base = base * rng.uniform(0.75, 1.25)
        for year in YEARS:
            growth = 1 + 0.02 * (year - 2021)            # gentle growth
            if corridor == "I-66" and year == 2024:      # planted: tr-09
                growth *= 1.06
            aadt = int(seg_base * growth * rng.uniform(0.97, 1.03))
            cur.execute("INSERT INTO aadt VALUES (?,?,?)", (sid, year, aadt))
            base_speed = {"freeway": 62, "arterial": 41, "collector": 29}[
                CORRIDORS[corridor][0]]
            for month in range(1, 13):
                season = 1 + 0.06 * (month in (5, 6, 7, 8, 9)) - 0.05 * (month in (1, 2))
                vol = int(aadt * 1.07 * season * rng.uniform(0.96, 1.04))
                cur.execute("INSERT INTO volumes VALUES (?,?,?,?)",
                            (sid, year, month, vol))
                speed = base_speed + rng.uniform(-2.5, 2.5)
                if sid == "S-12" and (year, month) >= (2024, 3):  # planted: tr-14
                    speed -= 4.0
                cur.execute("INSERT INTO speeds VALUES (?,?,?,?)",
                            (sid, year, month, round(speed, 1)))
                cur.execute("INSERT INTO detectors VALUES (?,?,?,?,?)",
                            (f"D-{sid}", sid, year, month,
                             round(min(100.0, rng.gauss(96.5, 3.0)), 1)))

    # --- crashes -------------------------------------------------------------
    crash_id = 0
    day0, day1 = date(2021, 1, 1), date(2024, 12, 31)
    span = (day1 - day0).days
    for sid, (corridor, base) in seg_meta.items():
        yearly = max(2, int(base / 2500))                 # exposure ~ volume
        for _ in range(yearly * len(YEARS)):
            crash_id += 1
            d = day0 + timedelta(days=rng.randrange(span))
            if rng.random() < 0.35:                       # planted: tr-15 (Fri/Sat peak)
                shift = (4 - d.weekday()) % 7 if rng.random() < 0.5 else (5 - d.weekday()) % 7
                d = min(d + timedelta(days=shift), day1)
            factors = FACTORS.copy()
            weights = [0.18, 0.10, 0.22, 0.12, 0.18, 0.20]
            if corridor == "US-29":                       # planted: tr-13
                weights = [0.34, 0.08, 0.18, 0.10, 0.14, 0.16]
            cur.execute("INSERT INTO crashes VALUES (?,?,?,?,?,?)", (
                crash_id, sid,
                rng.choice(INTERSECTIONS) if rng.random() < 0.30 else None,
                d.isoformat(),
                rng.choices(SEVERITIES, SEVERITY_WEIGHTS)[0],
                rng.choices(factors, weights)[0],
            ))

    con.commit()
    counts = {t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("segments", "aadt", "volumes", "speeds", "crashes", "detectors")}
    con.close()
    print(f"wrote {db_path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent / "datasets" / "traffic.db")
