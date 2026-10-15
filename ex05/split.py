import argparse
import sys

import pandas as pd

TARGET = "knight"
TRAIN_RATIO = 0.8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: cannot read {args.csv} ({e})")

    if TARGET in df.columns:
        train = df.groupby(TARGET, group_keys=False).sample(
            frac=TRAIN_RATIO, random_state=args.seed)
    else:
        train = df.sample(frac=TRAIN_RATIO, random_state=args.seed)
    valid = df.drop(train.index)

    train.sample(frac=1, random_state=args.seed).to_csv("Training_knight.csv", index=False)
    valid.sample(frac=1, random_state=args.seed).to_csv("Validation_knight.csv", index=False)
    print(f"Training_knight.csv   : {len(train)} rows ({len(train) / len(df):.0%})")
    print(f"Validation_knight.csv : {len(valid)} rows ({len(valid) / len(df):.0%})")


if __name__ == "__main__":
    main()
