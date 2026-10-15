#!/usr/bin/env python3
"""ex04 - Min-max normalization: x' = (x - min) / (max - min), in [0, 1].
Usage: ./Normalization.py [Train_knight.csv] [Test_knight.csv]
Min and max are computed on the Train file and applied to both files."""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

TARGET = "knight"


def find_csv(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.join(here, "..")):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    sys.exit(f"Error: {name} not found")


def pick_mixed_pair(train):
    y = train[TARGET].map({"Jedi": 1, "Sith": 0})
    feats = train.drop(columns=[TARGET]).select_dtypes("number")
    corr = feats.corrwith(y).abs().sort_values(ascending=False)
    return list(corr.index[-2:])


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else find_csv("Train_knight.csv")
    test_path = sys.argv[2] if len(sys.argv) > 2 else find_csv("Test_knight.csv")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: cannot read CSV ({e})")

    features = [c for c in train.select_dtypes("number").columns if c != TARGET]
    lo = train[features].min()
    rng = (train[features].max() - lo).replace(0, 1)  # avoid division by zero

    train_n, test_n = train.copy(), test.copy()
    for name, raw, norm_df in (("Train", train, train_n), ("Test", test, test_n)):
        norm_df[features] = (raw[features] - lo) / rng
        print(f"--- {name}: raw data ---")
        print(raw[features].head().round(2).to_string(index=False, max_cols=7))
        print(f"--- {name}: normalized data ---")
        print(norm_df[features].head().round(2).to_string(index=False, max_cols=7))
        print()

    fx, fy = pick_mixed_pair(train)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color in (("Sith", "red"), ("Jedi", "blue")):
        sub = train_n[train_n[TARGET] == label]
        axes[0].scatter(sub[fx], sub[fy], c=color, s=12, alpha=0.3, label=label)
    axes[0].set_title("Train (normalized)")
    axes[1].scatter(test_n[fx], test_n[fy], c="green", s=12, alpha=0.3, label="Knight")
    axes[1].set_title("Test (normalized)")
    for ax in axes:
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend()
    fig.tight_layout()
    fig.savefig("normalization.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
