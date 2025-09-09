#!/usr/bin/env python3
import argparse, glob, os, re, csv

ANSI_RE   = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')          # strip color codes
CSV_GRAB  = re.compile(r'(.+?\.csv)')                     # capture up to .csv (allows spaces, () etc.)

def parse_ok_line(line):
    """Return (input_csv, output_csv) from a single [OK] line, or None."""
    s = ANSI_RE.sub("", line).strip()
    if "[OK" not in s:
        return None

    # take text after the closing bracket
    payload = s.split("]", 1)[-1].strip()

    # normalize arrows
    if "->" in payload:
        left, right = payload.split("->", 1)
    elif "→" in payload:
        left, right = payload.split("→", 1)
    else:
        left, right = payload, ""

    # logs truncate long names with "..." – remove it so .csv is contiguous
    left  = left.replace("...", "").strip()
    right = right.replace("...", "").strip()

    mL = CSV_GRAB.search(left)
    mR = CSV_GRAB.search(right)
    if not mL:
        return None

    in_csv  = mL.group(1).strip()
    out_csv = mR.group(1).strip() if mR else ""
    return in_csv, out_csv

def main():
    ap = argparse.ArgumentParser(description="Count [OK] processed files from .out logs.")
    ap.add_argument("--dir", default=".", help="Directory with .out files.")
    ap.add_argument("--pattern", default="*.out", help="Glob for .out files (e.g., 'tag_dates_full_*.out').")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print(f"No files matched {args.pattern} in {args.dir}")
        return

    ok_pairs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parsed = parse_ok_line(line)
                if parsed:
                    ok_pairs.append(parsed)

    unique_inputs = sorted({a for a, _ in ok_pairs})

    # Write outputs
    txt_path = os.path.join(args.dir, "processed_files.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for a in unique_inputs:
            f.write(a + "\n")

    csv_path = os.path.join(args.dir, "processed_file_map.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_csv", "output_csv"])
        for a, b in sorted(set(ok_pairs)):
            w.writerow([a, b])

    print(f"Scanned {len(paths)} .out files; "
          f"Total [OK] lines: {len(ok_pairs)}; "
          f"Unique input files: {len(unique_inputs)}")
    print(f"Wrote {txt_path} and {csv_path}")

if __name__ == "__main__":
    main()
