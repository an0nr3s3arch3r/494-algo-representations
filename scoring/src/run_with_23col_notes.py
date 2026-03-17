#!/usr/bin/env python3
"""Run the scorer with notes that have 23 columns by adding the missing 24th column on the fly.

Use this instead of main.py when your notes TSV has 23 columns (missing isCollaborativeNote).
No changes to process_data.py or constants.py are required.

Example (same as main.py, but handles 23-column notes):
  python run_with_23col_notes.py ^
    --enrollment data/userEnrollment-00000.tsv ^
    --notes data/notes-00000.tsv ^
    --ratings data/ratings-00000.tsv ^
    --status data/noteStatusHistory-00000.compat.tsv ^
    --outdir data ^
    --scorers MFCoreScorer
"""

import argparse
import os
import sys
import tempfile

EXPECTED_NOTE_COLUMNS = 24
MISSING_COLUMN_NAME = "isCollaborativeNote"
MISSING_COLUMN_VALUE = "0"


def ensure_24_column_notes(notes_path: str) -> str:
    """If notes_path has 23 columns, write a 24-column version and return its path; else return notes_path."""
    with open(notes_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    num_cols = len(first_line.rstrip("\n").split("\t"))
    if num_cols == EXPECTED_NOTE_COLUMNS:
        return notes_path
    if num_cols != EXPECTED_NOTE_COLUMNS - 1:
        raise ValueError(
            f"Notes file has {num_cols} columns; expected {EXPECTED_NOTE_COLUMNS} or {EXPECTED_NOTE_COLUMNS - 1}. "
            f"Cannot add missing column '{MISSING_COLUMN_NAME}'."
        )
    # 23 columns: add 24th column (header "isCollaborativeNote", data rows "0")
    dir_name = os.path.dirname(notes_path)
    base = os.path.basename(notes_path)
    fd, out_path = tempfile.mkstemp(suffix=".tsv", prefix=base + ".", dir=dir_name or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            with open(notes_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.rstrip("\n")
                    if not line:
                        out.write("\n")
                        continue
                    # First line: if it looks like a header (first column is "noteId"), add column name; else add value
                    if i == 0 and line.split("\t")[0].strip() == "noteId":
                        suffix = MISSING_COLUMN_NAME
                    else:
                        suffix = MISSING_COLUMN_VALUE
                    out.write(line + "\t" + suffix + "\n")
        return out_path
    except Exception:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default=None, help="Path to notes TSV")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args, rest = parser.parse_known_args()
    # Rebuild argv: replace --notes <path> with --notes <new_path> if we created a 24-col file
    notes_path = args.notes
    new_argv = [sys.argv[0]]
    i = 0
    argv_rest = sys.argv[1:]
    while i < len(argv_rest):
        if argv_rest[i] == "--notes" and i + 1 < len(argv_rest):
            notes_path = argv_rest[i + 1]
            notes_path = ensure_24_column_notes(notes_path)
            new_argv.append("--notes")
            new_argv.append(notes_path)
            i += 2
            continue
        new_argv.append(argv_rest[i])
        i += 1
    sys.argv = new_argv
    from scoring.runner import main as runner_main
    runner_main()


if __name__ == "__main__":
    main()
