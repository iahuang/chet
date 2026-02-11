#!/usr/bin/env python3
"""
Analyze the distribution of positions in the training dataset.

This script helps verify that the sampling strategy successfully mitigated
the early-game bias.
"""

import csv
from collections import Counter
from pathlib import Path
import sys


def analyze_distribution(csv_path: Path):
    """Analyze move number distribution in the dataset."""
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found")
        print("  Run download_and_collate.py first")
        sys.exit(1)

    print(f"📊 Analyzing {csv_path}")
    print("=" * 60)

    move_counts = Counter()
    source_counts = Counter()
    total_positions = 0

    # Read CSV and collect statistics
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            move_num = int(row['move_number'])
            source = row['source']

            # Bin move numbers into groups
            if move_num < 10:
                bin_key = f"0-9 (opening)"
            elif move_num < 20:
                bin_key = f"10-19 (early mid)"
            elif move_num < 30:
                bin_key = f"20-29 (mid)"
            elif move_num < 40:
                bin_key = f"30-39 (late mid)"
            else:
                bin_key = f"40+ (endgame)"

            move_counts[bin_key] += 1
            source_counts[source] += 1
            total_positions += 1

    # Print results
    print(f"\nTotal positions: {total_positions:,}\n")

    print("Distribution by Game Phase:")
    print("-" * 60)
    for phase in ["0-9 (opening)", "10-19 (early mid)", "20-29 (mid)",
                  "30-39 (late mid)", "40+ (endgame)"]:
        count = move_counts[phase]
        percentage = (count / total_positions) * 100
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        print(f"{phase:20s} {count:8,} ({percentage:5.1f}%) {bar}")

    print("\nDistribution by Source:")
    print("-" * 60)
    for source, count in source_counts.items():
        percentage = (count / total_positions) * 100
        print(f"{source:20s} {count:8,} ({percentage:5.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Analysis complete")
    print("\nExpected result: More balanced distribution compared to raw data")
    print("Without sampling, opening (0-9) would be 50%+ of dataset")


if __name__ == '__main__':
    csv_path = Path(__file__).parent / "data" / "training_data.csv"
    analyze_distribution(csv_path)
