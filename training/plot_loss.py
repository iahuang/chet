"""Plot training and validation loss from a training log CSV."""

import argparse
import csv
import sys

import matplotlib.pyplot as plt


def load_log(path: str):
    train_x, train_y = [], []
    val_x, val_y = [], []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples = int(row["examples_seen"])
            if row["train_loss"]:
                train_x.append(examples)
                train_y.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_x.append(examples)
                val_y.append(float(row["val_loss"]))

    return train_x, train_y, val_x, val_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", default="training_log.csv", help="Path to training_log.csv")
    args = parser.parse_args()

    train_x, train_y, val_x, val_y = load_log(args.log)
    if not train_x and not val_x:
        print("No data found in log file.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_x, train_y, linewidth=0.8, alpha=0.85, label="Train loss")
    if val_x:
        ax.plot(val_x, val_y, "o-", markersize=6, linewidth=1.5, label="Val loss")

    ax.set_xscale("log")
    ax.set_xlabel("Examples seen")
    ax.set_ylabel("Loss")
    ax.set_title("Training progress")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("training/training_loss.png", dpi=150)


if __name__ == "__main__":
    main()
