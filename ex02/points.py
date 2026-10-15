import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

TARGET = "knight"


def pick_pairs(train):
    """Use the correlation with the target to choose the features:
    strongest two -> separated clusters, weakest two -> mixed clusters."""
    y = train[TARGET].map({"Jedi": 1, "Sith": 0})
    feats = train.drop(columns=[TARGET]).select_dtypes("number")
    corr = feats.corrwith(y).abs().sort_values(ascending=False)
    return list(corr.index[:2]), list(corr.index[-2:])


def scatter_train(ax, df, fx, fy):
    for label, color in (("Sith", "red"), ("Jedi", "blue")):
        sub = df[df[TARGET] == label]
        ax.scatter(sub[fx], sub[fy], c=color, s=12, alpha=0.3, label=label)
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.legend()


def scatter_test(ax, df, fx, fy):
    ax.scatter(df[fx], df[fy], c="green", s=12, alpha=0.3, label="Knight")
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.legend()


def main():
    try:
        train = pd.read_csv("Train_knight.csv")
        test = pd.read_csv("Test_knight.csv")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: invalid CSV ({e})")

    separated, mixed = pick_pairs(train)
    print(f"Separated clusters : {separated}")
    print(f"Mixed clusters     : {mixed}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, (fx, fy) in enumerate((separated, mixed)):
        scatter_train(axes[0][col], train, fx, fy)
        scatter_test(axes[1][col], test, fx, fy)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
