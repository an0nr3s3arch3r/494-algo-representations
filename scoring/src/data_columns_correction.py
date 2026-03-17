#!/usr/bin/env python3
"""
Preprocess / fix data files so main.py can run without column errors.

Run from scoring/src (or with correct paths):
  python data_columns_correction.py

Fixes:
- notes: if TSV has 23 columns (missing isCollaborativeNote), adds that column (value 0) and overwrites the file.
- Optionally pass custom paths via --notes, --ratings, etc.
"""

import argparse
import sys

# Ensure we can import scoring when run as script
if __name__ == "__main__":
    import os
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import pandas as pd

from scoring import constants as c


EXPECTED_NOTE_COLUMNS = 24
MISSING_COLUMN_NAME = "isCollaborativeNote"


def fix_notes(path: str, in_place: bool = True) -> bool:
    """If notes TSV has 23 columns, add isCollaborativeNote (0) and write back. Return True if file was fixed."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if len(df.columns) == EXPECTED_NOTE_COLUMNS:
        print(f"Notes already has {EXPECTED_NOTE_COLUMNS} columns: {path}")
        return False
    if len(df.columns) != EXPECTED_NOTE_COLUMNS - 1:
        raise ValueError(
            f"Notes has {len(df.columns)} columns; expected {EXPECTED_NOTE_COLUMNS} or {EXPECTED_NOTE_COLUMNS - 1}. "
            f"Cannot add '{MISSING_COLUMN_NAME}'."
        )
    # Add missing column
    df[MISSING_COLUMN_NAME] = "0"
    # Ensure column order matches expected (in case header order differs)
    if list(df.columns) != c.noteTSVColumns:
        df = df.reindex(columns=c.noteTSVColumns)
    out_path = path if in_place else path.replace(".tsv", "_fixed.tsv")
    df.to_csv(out_path, sep="\t", index=False, header=True)
    print(f"Fixed notes: added '{MISSING_COLUMN_NAME}', wrote {out_path}")
    return True


def fix_ratings(path: str, in_place: bool = True) -> bool:
    """Ensure ratings has exactly the 33 required columns in order. Scorer adds correlatedRater in memory."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) == c.ratingTSVColumns:
        print(f"Ratings already has {len(c.ratingTSVColumns)} columns: {path}")
        return False
    # Keep only the 33 columns the scorer expects (drop any extra e.g. correlatedRater)
    for col in c.ratingTSVColumns:
        if col not in df.columns:
            df[col] = ""
    df = df[c.ratingTSVColumns]
    out_path = path if in_place else path.replace(".tsv", "_fixed.tsv")
    df.to_csv(out_path, sep="\t", index=False, header=True)
    print(f"Fixed ratings: wrote exactly {len(c.ratingTSVColumns)} columns to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix data column schema for Community Notes scorer.")
    parser.add_argument("--notes", default="data/notes-00000.tsv", help="Path to notes TSV")
    parser.add_argument("--ratings", default="data/ratings-00000.tsv", help="Path to ratings TSV")
    parser.add_argument("--no-inplace", action="store_true", help="Write to _fixed.tsv instead of overwriting")
    parser.add_argument("--notes-only", action="store_true", help="Only fix notes")
    parser.add_argument("--ratings-only", action="store_true", help="Only fix ratings")
    args = parser.parse_args()
    in_place = not args.no_inplace

    if not args.ratings_only:
        fix_notes(args.notes, in_place=in_place)
    if not args.notes_only:
        try:
            fix_ratings(args.ratings, in_place=in_place)
        except FileNotFoundError:
            print(f"Ratings file not found: {args.ratings} (skipping)")


if __name__ == "__main__":
    main()
