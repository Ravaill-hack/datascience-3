#!/usr/bin/env python3
"""ex03 - Standardization: z = (x - mean) / std.
Usage: ./standardization.py [Train_knight.csv] [Test_knight.csv]
Mean and std are computed on the Train file and applied to both files
(no data leakage from Test)."""
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


def pick_separating_pair(train):
    y = train[TARGET].map({"Jedi": 1, "Sith": 0})
    feats = train.drop(columns=[TARGET]).select_dtypes("number")
    corr = feats.corrwith(y).abs().sort_values(ascending=False)
    return list(corr.index[:2])


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else find_csv("Train_knight.csv")
    test_path = sys.argv[2] if len(sys.argv) > 2 else find_csv("Test_knight.csv")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: cannot read CSV ({e})")

    features = [c for c in train.select_dtypes("number").columns if c != TARGET]
    mean = train[features].mean()
    std = train[features].std(ddof=0)  # population std, same as sklearn StandardScaler

    train_std, test_std = train.copy(), test.copy()
    for name, raw, std_df in (("Train", train, train_std), ("Test", test, test_std)):
        std_df[features] = (raw[features] - mean) / std
        print(f"--- {name}: raw data ---")
        print(raw[features].head().round(2).to_string(index=False, max_cols=7))
        print(f"--- {name}: standardized data ---")
        print(std_df[features].head().round(2).to_string(index=False, max_cols=7))
        print()

    fx, fy = pick_separating_pair(train)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color in (("Sith", "red"), ("Jedi", "blue")):
        sub = train_std[train_std[TARGET] == label]
        axes[0].scatter(sub[fx], sub[fy], c=color, s=12, alpha=0.3, label=label)
    axes[0].set_title("Train (standardized)")
    axes[1].scatter(test_std[fx], test_std[fy], c="green", s=12, alpha=0.3, label="Knight")
    axes[1].set_title("Test (standardized)")
    for ax in axes:
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend()
    fig.tight_layout()
    fig.savefig("standardization.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
