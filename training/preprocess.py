"""
Preprocessing script: converts training CSV to memory-mappable binary files.

Parses FEN strings and UCI moves directly (no python-chess dependency) and writes
compact numpy arrays that can be memory-mapped at training time.

Usage:
    python -m training.preprocess training/data/training_data.csv --output-dir training/data/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Piece character → token ID (matches chet/tokenizer.py)
PIECE_TO_TOKEN = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}

TOKEN_TURN_WHITE = 13
TOKEN_TURN_BLACK = 14
TOKEN_CLS = 15


def parse_fen_tokens(fen: str, out: np.ndarray) -> None:
    """Parse a FEN string into a pre-allocated 66-element uint8 array.

    Layout matches chet/tokenizer.py:
        [sq0 .. sq63] [turn_token] [CLS]
    Square indices use python-chess convention: a1=0, b1=1, ..., h8=63.
    """
    out[:64] = 0

    space_idx = fen.index(" ")
    turn_char = fen[space_idx + 1]

    # Parse piece placement (FEN ranks go 8→1, i.e. top to bottom)
    sq = 56  # a8
    for ch in fen[:space_idx]:
        if ch == "/":
            sq -= 16  # next rank down
        elif "1" <= ch <= "8":
            sq += ord(ch) - 48
        else:
            out[sq] = PIECE_TO_TOKEN[ch]
            sq += 1

    out[64] = TOKEN_TURN_WHITE if turn_char == "w" else TOKEN_TURN_BLACK
    out[65] = TOKEN_CLS


def parse_uci_target(uci: str) -> int:
    """Parse a UCI move string (e.g. 'e2e4') into a target index.

    Returns from_sq * 64 + to_sq (range 0–4095).
    """
    from_sq = (int(uci[1]) - 1) * 8 + (ord(uci[0]) - ord("a"))
    to_sq = (int(uci[3]) - 1) * 8 + (ord(uci[2]) - ord("a"))
    return from_sq * 64 + to_sq


def count_data_rows(csv_path: str) -> int:
    """Count non-header rows in a CSV file."""
    count = 0
    with open(csv_path, "r") as f:
        next(f)  # skip header
        for _ in f:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess training CSV into memory-mappable binary files"
    )
    parser.add_argument("csv_path", help="Path to training CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Skip pre-shuffling the data"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: count rows ---
    print("Counting rows...")
    N = count_data_rows(args.csv_path)
    print(f"Found {N:,} positions")

    # --- Step 2: create output arrays (memmap-backed .npy files) ---
    tokens_path = os.path.join(args.output_dir, "tokens.npy")
    targets_path = os.path.join(args.output_dir, "targets.npy")

    if not args.no_shuffle:
        # Write to temp files first; final files are created during shuffle
        write_tokens_path = tokens_path + ".tmp"
        write_targets_path = targets_path + ".tmp"
    else:
        write_tokens_path = tokens_path
        write_targets_path = targets_path

    tokens = np.lib.format.open_memmap(
        write_tokens_path, mode="w+", dtype=np.uint8, shape=(N, 66)
    )
    targets = np.lib.format.open_memmap(
        write_targets_path, mode="w+", dtype=np.uint16, shape=(N,)
    )

    # --- Step 3: parse CSV and fill arrays ---
    row_buf = np.zeros(66, dtype=np.uint8)

    with open(args.csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for i, row in enumerate(tqdm(reader, total=N, desc="Parsing")):
            parse_fen_tokens(row[0], row_buf)
            tokens[i] = row_buf
            targets[i] = parse_uci_target(row[1])

    tokens.flush()
    targets.flush()

    # --- Step 4: shuffle ---
    if not args.no_shuffle:
        print("Shuffling...")
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(N)

        shuffled_tokens = np.lib.format.open_memmap(
            tokens_path, mode="w+", dtype=np.uint8, shape=(N, 66)
        )
        shuffled_targets = np.lib.format.open_memmap(
            targets_path, mode="w+", dtype=np.uint16, shape=(N,)
        )

        CHUNK = 1_000_000
        for start in tqdm(range(0, N, CHUNK), desc="Shuffling"):
            end = min(start + CHUNK, N)
            idx = perm[start:end]
            shuffled_tokens[start:end] = tokens[idx]
            shuffled_targets[start:end] = targets[idx]

        shuffled_tokens.flush()
        shuffled_targets.flush()

        # Clean up temp files
        del tokens, targets
        os.remove(write_tokens_path)
        os.remove(write_targets_path)

    # --- Step 5: write metadata ---
    metadata = {
        "num_positions": N,
        "source_csv": os.path.basename(args.csv_path),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    tokens_size = os.path.getsize(tokens_path)
    targets_size = os.path.getsize(targets_path)
    print(f"\nDone! Wrote {N:,} positions to {args.output_dir}")
    print(f"  tokens.npy:  {tokens_size / 1e9:.2f} GB")
    print(f"  targets.npy: {targets_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
