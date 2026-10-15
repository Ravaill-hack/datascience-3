import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET = "knight"
NCOLS = 5
TRAIN_FILE = "Train_knight.csv"
TEST_FILE = "Test_knight.csv"


def build_grid(n):
	"""
	Builds the grid depending on the number of plots
	"""
	nrows = -(-n // NCOLS)
	fig, axes = plt.subplots(nrows, NCOLS, figsize=(3 * NCOLS, 2.4 * nrows))
	axes = np.atleast_1d(axes).ravel()
	return fig, axes


def plot_test():
	"""
	Creates the plot for the test dataset
	"""
	df = pd.read_csv(TEST_FILE)
	features = [c for c in df.columns if c != TARGET]
	fig, axes = build_grid(len(features))
	for ax, col in zip(axes, features):
		ax.hist(df[col], bins=40, color="green", alpha=0.5, label="Knight")
		ax.set_title(col, fontsize=9)
		ax.legend(fontsize=6)
	fig.suptitle(TEST_FILE)
	fig.tight_layout()
	return fig


def plot_train():
	"""
	Creates the plot for the train dataset
	"""
	df = pd.read_csv(TRAIN_FILE)
	features = [c for c in df.columns if c != TARGET]
	fig, axes = build_grid(len(features))
	jedi = df[df[TARGET] == "Jedi"]
	sith = df[df[TARGET] == "Sith"]
	for ax, col in zip(axes, features):
		bins = np.histogram_bin_edges(df[col].dropna(), bins=40)
		ax.hist(jedi[col], bins=bins, color="blue", alpha=0.5, label="Jedi")
		ax.hist(sith[col], bins=bins, color="red", alpha=0.3, label="Sith")
		ax.set_title(col, fontsize=9)
		ax.legend(fontsize=6)
	fig.suptitle(TRAIN_FILE)
	fig.tight_layout()
	return fig


def main():
	"""
	Draws the two plots
	"""
	try:
		plot_test()
		plot_train()
	except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
		sys.exit(f"Error: invalid CSV ({e})")
	plt.show()


if __name__ == "__main__":
	main()
