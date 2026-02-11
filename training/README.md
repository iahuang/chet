# Chess Training Data Pipeline

This directory contains scripts to download and process chess game data for training the Chet neural chess engine.

## Overview

The pipeline downloads high-level chess games from Lichess Elite databases and tactical puzzles, applies intelligent sampling to mitigate early-game bias, and outputs a training-ready CSV file.

## Problem: Early Game Bias

Chess game databases suffer from severe distribution imbalance:
- **Every game starts from the same position** (starting position appears in 100% of games)
- **Popular openings dominate** (e.g., Ruy Lopez, Sicilian Defense appear millions of times)
- **Endgames are underrepresented** (many games end before reaching endgame)

Without correction, the first 10 moves could represent 50%+ of all training positions.

## Solution: Probability-Based Sampling

We use a deterministic hash-based sampling strategy that:
- **Heavily downsamples early positions** (move 1: 1% kept, move 10: 51% kept)
- **Preserves late-game positions** (move 20+: 100% kept)
- **Is deterministic and reproducible** (same FEN always gets same decision)
- **Uses zero memory overhead** (stateless decision based on FEN hash)

The sampling probability is: `keep_prob = min(1.0, 0.01 + 0.05 * move_num)`

## Usage

### 1. Install Dependencies

```bash
# From project root, using the virtual environment
source ../.venv/bin/activate
pip install -r training/requirements.txt
```

### 2. Run the Pipeline

```bash
python training/download_and_collate.py
```

This will:
1. Download ~13 months of Lichess Elite games (2600+ Elo players)
2. Download Lichess puzzle database
3. Parse PGN files and extract positions
4. Apply sampling strategy
5. Output to `training/data/training_data.csv`

**Note:** This will download several GB of data and may take 1-2 hours depending on connection speed.

### 3. Analyze the Distribution

```bash
python training/analyze_distribution.py
```

This shows the distribution of positions by game phase, verifying that sampling worked correctly.

## Output Format

The output CSV has the following columns:

| Column | Description |
|--------|-------------|
| `fen` | Position in FEN notation |
| `move_uci` | Move played in UCI format (e.g., "e2e4") |
| `move_number` | 0-indexed move number |
| `source` | Data source (lichess_elite or lichess_puzzle) |
| `white_elo` | White player's Elo rating (games only) |
| `black_elo` | Black player's Elo rating (games only) |
| `result` | Game result: 1-0, 0-1, 1/2-1/2, or * (games only) |
| `puzzle_id` | Lichess puzzle ID (puzzles only) |
| `rating` | Puzzle difficulty rating (puzzles only) |
| `popularity` | Puzzle popularity score (puzzles only) |
| `themes` | Puzzle themes/tags (puzzles only) |

## Data Sources

- **Lichess Elite Database**: Games played by 2600+ Elo rated players
  - 13 months of data (Dec 2021 - Dec 2022)
  - Source: https://database.nikonoel.fr/

- **Lichess Puzzle Database**: Tactical puzzles with ratings and themes
  - Source: https://database.lichess.org/#puzzles

## File Structure

```
training/
├── data_sources.py           # URLs for data sources
├── download_and_collate.py   # Main pipeline script
├── analyze_distribution.py   # Distribution analysis tool
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── data/                    # Created by pipeline
    ├── raw/                 # Downloaded files
    └── training_data.csv    # Final output
```

## Customization

To adjust the sampling strategy, modify the `should_keep_position()` function in `download_and_collate.py`:

```python
# Current: 1% at move 0, 51% at move 10, 100% at move 20+
keep_prob = min(1.0, 0.01 + 0.05 * move_num)

# More aggressive: Keep even less from opening
keep_prob = min(1.0, 0.005 + 0.03 * move_num)

# More conservative: Keep more from opening
keep_prob = min(1.0, 0.05 + 0.05 * move_num)
```

## Expected Results

With proper sampling, you should see a more balanced distribution:
- Opening (moves 0-9): ~20-30% (vs 50%+ without sampling)
- Mid-game (moves 10-39): ~50-60%
- Endgame (moves 40+): ~10-20%

Run `analyze_distribution.py` to verify the actual distribution.
