#!/usr/bin/env python3

import os
import sys

import pandas as pd

TARGET = "knight"


def find_csv(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.join(here, "..")):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    sys.exit(f"Error: {name} not found")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_csv("Train_knight.csv")
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: cannot read {path} ({e})")
    if TARGET not in df.columns:
        sys.exit(f"Error: no '{TARGET}' column in {path}")

    # Jedi -> 1, Sith -> 0 (matches the sign of the subject's example)
    df[TARGET] = df[TARGET].map({"Jedi": 1, "Sith": 0})
    corr = df.corr(numeric_only=True)[TARGET]
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print(corr.to_string())


if __name__ == "__main__":
    main()
