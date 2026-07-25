"""Load real NHTSA FARS data (fatal crashes) into datasets/fars.db.

    python scripts/load_fars.py                # downloads FARS 2023 (final), ~50 MB
    python scripts/load_fars.py --year 2024    # Annual Report File (subject to change)
    python scripts/load_fars.py --zip path.zip # use an already-downloaded zip

Source: NHTSA static downloads (https://www.nhtsa.gov/file-downloads),
pattern: .../FARS/{year}/National/FARS{year}NationalCSV.zip
If the URL pattern changes, download manually and pass --zip.

Loads a slim, analysis-ready subset of three files:
  accidents  (one row per fatal crash)
  vehicles   (one row per involved vehicle)
  persons    (one row per involved person)

Columns are lowercased; only the allowlisted ones present in the file are
kept (FARS adds/drops columns across years — we stay defensive). The *NAME
columns are FARS's human-readable decodes; prefer them in queries.
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"

WANTED = {
    "accidents": ["ST_CASE", "YEAR", "STATE", "STATENAME", "COUNTYNAME", "CITYNAME",
                  "MONTH", "MONTHNAME", "DAY", "DAY_WEEK", "DAY_WEEKNAME", "HOUR",
                  "TWAY_ID", "ROUTENAME", "FUNC_SYSNAME", "RUR_URBNAME",
                  "LGT_CONDNAME", "WEATHERNAME", "HARM_EVNAME", "MAN_COLLNAME",
                  "LATITUDE", "LONGITUD", "FATALS", "PERSONS", "VE_TOTAL"],
    "vehicles": ["ST_CASE", "VEH_NO", "NUMOCCS", "MAKENAME", "BODY_TYPNAME",
                 "MOD_YEAR", "TRAV_SP", "SPEEDRELNAME", "HIT_RUNNAME"],
    "persons": ["ST_CASE", "VEH_NO", "PER_NO", "AGE", "SEXNAME", "PER_TYPNAME",
                "INJ_SEVNAME", "REST_USENAME", "DRINKINGNAME"],
}
FILE_KEYS = {"accidents": "accident", "vehicles": "vehicle", "persons": "person"}


def _coerce(v: str):
    v = v.strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _find_member(zf: zipfile.ZipFile, key: str) -> str:
    for name in zf.namelist():
        stem = Path(name).stem.lower()
        if stem == key:
            return name
    raise SystemExit(f"could not find {key}.csv in the zip — contents: {zf.namelist()[:10]}...")


def load(zip_path: Path, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    with zipfile.ZipFile(zip_path) as zf:
        for table, wanted in WANTED.items():
            member = _find_member(zf, FILE_KEYS[table])
            with zf.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"))
                cols = [c for c in wanted if c in (reader.fieldnames or [])]
                if "ST_CASE" not in cols:
                    raise SystemExit(f"{member}: no ST_CASE column — wrong file?")
                col_sql = ", ".join(c.lower() for c in cols)
                con.execute(f"CREATE TABLE {table} ({col_sql})")
                rows = ([_coerce(r[c] or "") for c in cols] for r in reader)
                con.executemany(
                    f"INSERT INTO {table} VALUES ({','.join('?' * len(cols))})", rows)
            con.execute(f"CREATE INDEX idx_{table}_st_case ON {table}(st_case)")
    con.commit()
    for table in WANTED:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows")
    total = con.execute("SELECT SUM(fatals) FROM accidents").fetchone()[0]
    print(f"  total fatalities: {total:,}")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023,
                    help="2023 = final data; 2024 = Annual Report File")
    ap.add_argument("--db", default="datasets/fars.db")
    ap.add_argument("--zip", default=None, help="path to an already-downloaded zip")
    args = ap.parse_args()

    if args.zip:
        zip_path = Path(args.zip)
    else:
        url = URL.format(year=args.year)
        print(f"downloading {url} (~50 MB)...")
        tmp = Path(tempfile.gettempdir()) / f"fars{args.year}.zip"
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as e:
            sys.exit(f"download failed ({e}).\nGrab the zip manually from "
                     f"https://www.nhtsa.gov/file-downloads and rerun with --zip <path>")
        zip_path = tmp

    print(f"loading {zip_path} -> {args.db}")
    load(zip_path, Path(args.db))
    print("done. try: SELECT statename, COUNT(*) FROM accidents GROUP BY 1 ORDER BY 2 DESC LIMIT 5;")


if __name__ == "__main__":
    main()
