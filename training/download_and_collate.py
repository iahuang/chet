#!/usr/bin/env python3
"""
Data pipeline to download and collate chess training data.

Downloads Lichess Elite games and puzzles, applies position-based sampling
to mitigate early-game bias, and outputs a CSV file for training.
"""

import sys
import csv
import random
import zipfile
import multiprocessing as mp
import requests
from pathlib import Path
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
import chess
import chess.pgn
import io

# Import data sources
from data_sources import LICHESS_ELITE_DATASETS, LICHESS_PUZZLE_DATSET

# Configuration
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_CSV = DATA_DIR / "training_data.csv"
CHUNK_SIZE = 8192


def ensure_dirs():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"⏭  {dest_path.name} already exists, skipping download")
        return

    print(f"⬇  Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file."""
    print(f"📦 Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def keep_probability(move_num: int) -> float:
    """Return the probability of keeping a position at this move number."""
    # Move 0: 1%, Move 10: 51%, Move 20+: 100%
    return min(1.0, 0.01 + 0.05 * move_num)


LOG_EVERY_N_GAMES = 10000


def process_pgn_file(pgn_path: Path) -> Path:
    """
    Parse a PGN file and write kept positions to a temporary CSV.

    Returns the path to the temporary CSV file.
    """
    label = pgn_path.name
    out_path = pgn_path.with_suffix('.csv')
    kept = 0
    total = 0
    games = 0

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f, \
         open(out_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games += 1
            if games % LOG_EVERY_N_GAMES == 0:
                print(f"  [{label}] {games:,} games, {kept:,} / {total:,} positions kept")

            headers = dict(game.headers)
            white_elo = headers.get('WhiteElo', 'Unknown')
            black_elo = headers.get('BlackElo', 'Unknown')
            result = headers.get('Result', '*')

            board = game.board()
            move_num = 0

            for move in game.mainline_moves():
                total += 1

                # Decide whether to keep BEFORE computing FEN.
                # board.fen() is expensive; skip it for rejected positions.
                if move_num >= 20 or random.random() < keep_probability(move_num):
                    fen = board.fen()
                    writer.writerow([
                        fen, move.uci(), move_num,
                        'lichess_elite', white_elo, black_elo, result,
                        '', '', '', ''
                    ])
                    kept += 1

                board.push(move)
                move_num += 1

    print(f"  [{label}] DONE — {games:,} games, {kept:,} / {total:,} positions kept")
    return out_path


def parse_puzzle_csv(csv_path: Path, max_puzzles: Optional[int] = None) -> Iterator[Tuple[str, str, int, dict]]:
    """
    Parse Lichess puzzle CSV.

    Note: Puzzle format is different - we extract the position after the opponent's
    move and the first solution move.

    Args:
        csv_path: Path to puzzle CSV
        max_puzzles: Maximum number of puzzles to include (None for all)

    Yields:
        Tuple of (fen, move_uci, move_number, metadata)
    """
    try:
        import zstandard as zstd
    except ImportError:
        print("⚠ Warning: zstandard not installed, skipping puzzle dataset")
        print("  Install with: pip install zstandard")
        return

    if not csv_path.exists():
        print(f"⚠ Warning: {csv_path} not found, skipping puzzles")
        return

    print(f"📖 Parsing puzzles from {csv_path.name}")

    with open(csv_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            csv_reader = csv.DictReader(text_stream)

            count = 0
            for row in csv_reader:
                if max_puzzles and count >= max_puzzles:
                    break

                puzzle_id = row['PuzzleId']
                fen = row['FEN']
                moves = row['Moves'].split()
                rating = row['Rating']
                popularity = row['Popularity']
                themes = row['Themes']

                # Apply first move (opponent's move) to get puzzle position
                board = chess.Board(fen)
                if len(moves) < 2:
                    continue

                try:
                    opponent_move = chess.Move.from_uci(moves[0])
                    board.push(opponent_move)

                    # The solution is the second move
                    solution_move = moves[1]
                    puzzle_fen = board.fen()

                    # Estimate move number from piece count (rough heuristic)
                    piece_count = len(board.piece_map())
                    estimated_move_num = max(0, (32 - piece_count) * 2)

                    # Apply sampling (puzzles are naturally more mid/late game)
                    if estimated_move_num >= 20 or random.random() < keep_probability(estimated_move_num):
                        metadata = {
                            'puzzle_id': puzzle_id,
                            'rating': rating,
                            'popularity': popularity,
                            'themes': themes,
                            'source': 'lichess_puzzle'
                        }
                        yield (puzzle_fen, solution_move, estimated_move_num, metadata)
                        count += 1

                except (ValueError, chess.IllegalMoveError):
                    continue


def download_all_data():
    """Download all data sources."""
    print("=" * 60)
    print("DOWNLOADING DATA")
    print("=" * 60)

    # Download Elite datasets
    for url in LICHESS_ELITE_DATASETS:
        filename = url.split('/')[-1]
        dest_path = RAW_DIR / filename

        # Skip if already extracted
        if filename.endswith('.zip'):
            extract_dir = RAW_DIR / filename.replace('.zip', '')
            if extract_dir.exists():
                print(f"⏭  {extract_dir.name}/ already exists, skipping download")
                continue

        download_file(url, dest_path)

        # Extract and delete zip
        if filename.endswith('.zip'):
            extract_zip(dest_path, extract_dir)
            print(f"🗑  Deleting {dest_path.name}")
            dest_path.unlink()

    # Download puzzle dataset
    puzzle_filename = LICHESS_PUZZLE_DATSET.split('/')[-1]
    puzzle_path = RAW_DIR / puzzle_filename
    download_file(LICHESS_PUZZLE_DATSET, puzzle_path)


CSV_COLUMNS = [
    'fen', 'move_uci', 'move_number', 'source',
    'white_elo', 'black_elo', 'result',
    'puzzle_id', 'rating', 'popularity', 'themes'
]


def collate_data():
    """Parse all data and write to CSV."""
    print("\n" + "=" * 60)
    print("COLLATING DATA")
    print("=" * 60)

    # Process PGN files in parallel
    pgn_files = sorted(RAW_DIR.rglob('*.pgn'))
    print(f"\n📁 Found {len(pgn_files)} PGN files — processing in parallel")

    num_workers = min(len(pgn_files), mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        temp_csvs = pool.map(process_pgn_file, pgn_files)

    # Merge temp CSVs + puzzles into final output
    print(f"\n📝 Merging into {OUTPUT_CSV.name}")
    total_positions = 0

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS)

        # Append each temp CSV
        for temp_csv in temp_csvs:
            with open(temp_csv, 'r', encoding='utf-8') as f:
                for line in f:
                    csvfile.write(line)
                    total_positions += 1
            temp_csv.unlink()

        # Process puzzle dataset
        puzzle_csv = RAW_DIR / 'lichess_db_puzzle.csv.zst'
        if puzzle_csv.exists():
            print(f"\n🧩 Processing puzzles")
            count = 0

            for fen, move, move_num, metadata in tqdm(
                parse_puzzle_csv(puzzle_csv),
                desc="Puzzles",
                total=100000
            ):
                writer.writerow([
                    fen, move, move_num,
                    metadata['source'], '', '', '',
                    metadata.get('puzzle_id', ''),
                    metadata.get('rating', ''),
                    metadata.get('popularity', ''),
                    metadata.get('themes', '')
                ])
                count += 1
                total_positions += 1

            print(f"  ✓ Extracted {count:,} puzzle positions")

    print("\n" + "=" * 60)
    print(f"✅ COMPLETE")
    print("=" * 60)
    print(f"Total positions: {total_positions:,}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"File size: {OUTPUT_CSV.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    """Main pipeline."""
    ensure_dirs()

    # Check dependencies
    try:
        import chess
        import chess.pgn
    except ImportError:
        print("❌ Error: python-chess not installed")
        print("  Install with: pip install chess")
        sys.exit(1)

    try:
        import tqdm
    except ImportError:
        print("❌ Error: tqdm not installed")
        print("  Install with: pip install tqdm")
        sys.exit(1)

    # Download data
    download_all_data()

    # Collate into CSV
    collate_data()

    print("\n🎉 Pipeline complete!")


if __name__ == '__main__':
    main()
