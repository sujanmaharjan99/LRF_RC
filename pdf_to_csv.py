# extract_pdf_tables.py
# Usage:
#   python extract_pdf_tables.py "/path/to/input_pdfs" "/path/to/output_folder" --to_csv  # optional
#
# What it does:
#   - Detects rating-table grids with 0.00–0.09 columns
#   - Melts each row (e.g., 4.0..4.9) into (X, Y) pairs
#   - Guesses smart column names from page text:
#       * STGQ: "Gage height (ft)" and "Discharge (ft^3/s)"
#       * FLFC: "Fall ratio" and "Discharge factor"
#       * FALL: "Gage height (ft)" and "Fall rated (ft)"
#   - Writes one Excel (or CSV) per PDF

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except ImportError:
    sys.stderr.write(
        "Please install pdfplumber first:\n  pip install pdfplumber pandas openpyxl\n"
    )
    raise

INCREMENTS = [f"{i/100:.2f}" for i in range(10)]  # '0.00'..'0.09'

def guess_names(page_text: str, default_x="X", default_y="Y"):
    text = (page_text or "").lower()
    # Simple keyword checks
    if "expanded rating table: 18" in text or "gage height" in text and "discharge" in text:
        return "Gage height (ft)", "Discharge (ft^3/s)"
    if "fall ratio" in text and "discharge factor" in text:
        return "Fall ratio", "Discharge factor"
    if "fall rated" in text and "gage height" in text:
        return "Gage height (ft)", "Fall rated (ft)"
    # Fallbacks
    return default_x, default_y

_num_re = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\.\d+|[-+]?\d+")

def to_number(cell):
    if cell is None:
        return None
    # strip asterisks and spaces, keep digits, commas, dots, signs
    s = "".join(ch for ch in str(cell) if ch in "0123456789+-.," )
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        # last resort: find first numeric pattern
        m = _num_re.search(str(cell))
        return float(m.group(0).replace(",", "")) if m else None

def extract_widest_table(page):
    # Try line-based detection first, then lattice-agnostic
    table_settings_variants = [
        dict(vertical_strategy="lines", horizontal_strategy="lines"),
        dict(vertical_strategy="text", horizontal_strategy="text", snap_tolerance=3, join_tolerance=2, edge_min_length=3),
    ]
    best = None
    for ts in table_settings_variants:
        try:
            tables = page.extract_tables(table_settings=ts) or []
        except Exception:
            tables = []
        for t in tables:
            # choose a table with at least 11 columns (row label + 10 increments)
            width = max(len(row) for row in t if row)
            if width >= 11 and (best is None or width > best[0]):
                best = (width, t)
    return best[1] if best else None

def melt_rating_table(table_rows):
    """
    Expect rows like:
      [base, v00, v01, ..., v09, maybe trailing columns...]
    Returns list of (x, y) with x = base + increment
    """
    pairs = []
    for raw in table_rows:
        if not raw:
            continue
        # Normalize row length
        row = list(raw) + [None] * max(0, 11 - len(raw))
        base = to_number(row[0])
        # Guard: row might be header or junk
        if base is None:
            continue
        for j in range(1, 11):  # columns 0.00..0.09
            y = to_number(row[j])
            if y is None:
                continue
            x = round(base + (j - 1) * 0.01, 2)
            pairs.append((x, y))
    return pairs

def process_pdf(pdf_path: Path, out_dir: Path, to_csv: bool):
    with pdfplumber.open(str(pdf_path)) as pdf:
        all_pairs = []
        x_name, y_name = "X", "Y"
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            # grab names once from the first page that has them
            if (x_name, y_name) == ("X", "Y"):
                x_name, y_name = guess_names(page_text, "X", "Y")

            table = extract_widest_table(page)
            if not table:
                continue

            # Try to find a header row that contains many of the increments
            header_idx = None
            for idx, row in enumerate(table[:5]):  # look near top
                cells = [c.strip() if isinstance(c, str) else c for c in row if c]
                hits = sum(1 for inc in INCREMENTS if any(inc in (c or "") for c in cells))
                if hits >= 6:
                    header_idx = idx
                    break

            data_rows = table[(header_idx + 1):] if header_idx is not None else table
            pairs = melt_rating_table(data_rows)
            all_pairs.extend(pairs)

    if not all_pairs:
        raise RuntimeError(f"No tabular pairs detected in {pdf_path.name}. If this file is scanned, try OCR first.")

    df = pd.DataFrame(all_pairs, columns=[x_name, y_name]).sort_values(by=[x_name, y_name]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    if to_csv:
        out_file = out_dir / f"{stem}__pairs.csv"
        df.to_csv(out_file, index=False)
    else:
        out_file = out_dir / f"{stem}__pairs.xlsx"
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="pairs")
    print(f"Wrote {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder containing PDF files")
    parser.add_argument("output_dir", help="Folder to write Excel or CSV files")
    parser.add_argument("--to_csv", action="store_true", help="Write CSVs instead of Excel")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    pdfs = sorted(p for p in in_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {in_dir}")
        return
    for pdf in pdfs:
        try:
            process_pdf(pdf, out_dir, args.to_csv)
        except Exception as e:
            sys.stderr.write(f"[WARN] {pdf.name}: {e}\n")

if __name__ == "__main__":
    main()
