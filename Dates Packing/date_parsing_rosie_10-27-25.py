import pandas as pd
from pathlib import Path
import re
import json
from tqdm import tqdm
from typing import List, Optional, Dict, Union

# ───────────── Paths and constants ──────────────

BASE_DIR = Path(
    "/local/scratch/group/guldigroup/climate_change/wiki_history_rosie/date_tagging_pipeline/"
    "tagged_output_2025-09-20_deduped"
)
SUBFOLDERS = [
    "historical_objects_tagged",
    "history_of_ideologies_tagged",
    "history_of_sports_tagged",
]
OUT_DIR = Path.home() / "Desktop"

MIN_YEAR = -9999
MAX_YEAR = 2050

# Load dynasty year mapping
DYNASTY_FILE = Path(__file__).parent / "dynasty_years.json"
with open(DYNASTY_FILE, "r", encoding="utf-8") as f:
    DYNASTY_MAP = json.load(f)

# ───────────── Regex patterns ──────────────

DATE_TAG_RE = re.compile(r"<\s*date\s*>(.*?)<\s*/\s*date\s*>", re.IGNORECASE)
YEAR_BC_RE = re.compile(r"\b(\d{1,4})\s*(BC|BCE)\b", re.IGNORECASE)
YEAR_AD_RE = re.compile(r"\b(\d{1,4})(?:\s*(AD|CE))?\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_FULL_RE = re.compile(r"\b(\d{1,2})\s+[A-Za-z]+\s+(\d{2,4})\b")

# Year or era spans
YEAR_SPAN_ANY = re.compile(
    r"(?P<y1>\d{1,4})\s*(?P<era1>BC|BCE|AD|CE)?\s*(?:-|–|—|to)\s*(?P<y2>\d{1,4})\s*(?P<era2>BC|BCE|AD|CE)?",
    re.IGNORECASE,
)
DECADE_RE = re.compile(r"\b(\d{3})(\d)0s\b|\b'(\d{2})s\b", re.IGNORECASE)
CENTURY_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)?\s+centur(?:y|ies)\b", re.IGNORECASE)
DYNASTY_RE = re.compile(r"\b(" + "|".join(map(re.escape, DYNASTY_MAP.keys())) + r")\b", re.IGNORECASE)

# ───────────── Helpers ──────────────

def normalize_year(y: int, era: Optional[str]) -> int:
    """Convert BCE to negative years."""
    if era and era.upper() in ("BC", "BCE"):
        return -y
    return y

def clamp_year(y: int) -> Optional[int]:
    """Keep years within defined range."""
    return y if MIN_YEAR <= y <= MAX_YEAR else None

# ───────────── Year extraction logic ──────────────

def extract_year_or_span(s: str, stats: dict) -> List[Dict[str, Union[int, str]]]:
    """Extract both single years and spans from a <date> tag string."""
    out = []
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)

    # ISO → single
    for y, m, d in ISO_DATE_RE.findall(s):
        y_int = int(y)
        if clamp_year(y_int):
            out.append({"type": "single", "year": y_int})
            stats["single_years"] += 1

    # Year spans
    for m in YEAR_SPAN_ANY.finditer(s):
        y1 = normalize_year(int(m.group("y1")), m.group("era1"))
        y2 = normalize_year(int(m.group("y2")), m.group("era2"))
        if clamp_year(y1) and clamp_year(y2):
            out.append({"type": "span", "start": y1, "end": y2})
            stats["span_kept"] += 1

    # Single BC/AD years
    for y, era in YEAR_BC_RE.findall(s):
        yv = normalize_year(int(y), era)
        if clamp_year(yv):
            out.append({"type": "single", "year": yv})
            stats["single_years"] += 1
    for y, era in YEAR_AD_RE.findall(s):
        yv = normalize_year(int(y), era)
        if clamp_year(yv):
            out.append({"type": "single", "year": yv})
            stats["single_years"] += 1

    # Decades
    for m in DECADE_RE.finditer(s):
        if m.group(1):  # 1990s
            start = int(m.group(1) + m.group(2) + "0")
        elif m.group(3):  # '90s → 1900s assumption
            start = 1900 + int(m.group(3))
        else:
            continue
        out.append({"type": "span", "start": start, "end": start + 9})
        stats["span_kept"] += 1

    # Centuries
    for m in CENTURY_RE.finditer(s):
        century = int(m.group(1))
        start = (century - 1) * 100 + 1
        end = century * 100
        out.append({"type": "span", "start": start, "end": end})
        stats["span_kept"] += 1

    # Dynasties
    for m in DYNASTY_RE.finditer(s):
        key = m.group(1).lower()
        start, end = DYNASTY_MAP[key]
        out.append({"type": "span", "start": start, "end": end, "label": key})
        stats["span_kept"] += 1

    # Month–month spans with year → year–year
    m = re.search(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
        r"\s*(?:–|-|to)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})",
        s, re.IGNORECASE)
    if m:
        year = int(m.group(1))
        out.append({"type": "span", "start": year, "end": year})
        stats["span_kept"] += 1

    return out

def extract_dates(cell: Optional[str], stats: dict) -> List[Dict]:
    """Extract all <date> tags in a cell."""
    if not isinstance(cell, str) or not cell:
        return []
    tags = DATE_TAG_RE.findall(cell)
    all_items = []
    for tag in tags:
        all_items.extend(extract_year_or_span(tag, stats))
    return all_items

# ───────────── Processing folders ──────────────

def process_subfolder(subfolder: str):
    stats = {"single_years": 0, "span_kept": 0}
    folder_path = BASE_DIR / subfolder
    all_csvs = list(folder_path.rglob("*.csv"))
    results = {}

    for csv_file in tqdm(all_csvs, desc=f"Scanning {subfolder}", unit="file"):
        try:
            df = pd.read_csv(csv_file, low_memory=False, usecols=["filename", "date_tagged"])
        except Exception:
            continue
        if df.empty:
            continue
        df = df.drop_duplicates(subset=["filename"], keep="first")

        for _, row in df.iterrows():
            parsed = extract_dates(str(row["date_tagged"]), stats)
            if not parsed:
                continue
            fname = row["filename"]
            results.setdefault(fname, []).extend(parsed)

    final_rows = [{"filename": f, "parsed_years": json.dumps(yrs, ensure_ascii=False)} for f, yrs in results.items()]
    df_final = pd.DataFrame(final_rows)
    outfile = OUT_DIR / f"parsed_years_{subfolder}.csv"
    df_final.to_csv(outfile, index=False, encoding="utf-8")

    print(f"\n✅ {subfolder} done — {len(results)} files,"
          f" {stats['single_years']} singles, {stats['span_kept']} spans. Output: {outfile}")

def main():
    for sub in SUBFOLDERS:
        process_subfolder(sub)

if __name__ == "__main__":
    main()