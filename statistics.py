
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SourceAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data = np.frombuffer(
            self.file_path.read_bytes(),
            dtype=np.uint8
        )

        if self.data.size == 0:
            raise ValueError("The file is empty")

        self.symbol_counts = None
        self.symbol_probabilities = None
        self.pair_counts = None
        self.pair_probabilities = None

        self._analyze()

    @staticmethod
    def _entropy(probabilities):
        p = probabilities[probabilities > 0]
        return float(-np.sum(p * np.log2(p)))

    def _analyze(self):
        # Individual symbol frequencies
        self.symbol_counts = np.bincount(
            self.data, minlength=256
        )

        self.symbol_probabilities = (
            self.symbol_counts / self.data.size
        )

        # Consecutive overlapping symbol pairs
        self.pair_counts = np.zeros(
            (256, 256), dtype=np.int64
        )

        if self.data.size >= 2:
            np.add.at(
                self.pair_counts,
                (self.data[:-1], self.data[1:]),
                1
            )

        total_pairs = self.data.size - 1

        self.pair_probabilities = (
            self.pair_counts / total_pairs
            if total_pairs > 0
            else np.zeros((256, 256))
        )

    def entropy_symbol(self):
        return self._entropy(self.symbol_probabilities)

    def entropy_pair(self):
        return self._entropy(self.pair_probabilities)

    def entropy_conditional(self):
        if self.data.size < 2:
            return float("nan")

        # Marginal distribution of the first
        # symbol in each observed pair
        p_first = self.pair_probabilities.sum(axis=1)

        return (
            self.entropy_pair()
            - self._entropy(p_first)
        )

    def plot_symbols(self):
        plt.figure(figsize=(12, 4))

        plt.bar(
            np.arange(256),
            self.symbol_probabilities,
            width=1
        )

        plt.xlabel("Symbol (byte value)")
        plt.ylabel("Probability")
        plt.title("Symbol frequency distribution")
        plt.xlim(0, 255)
        plt.tight_layout()
        plt.show()

    def plot_pairs(self):
        plt.figure(figsize=(9, 7))

        plt.imshow(
            self.pair_probabilities,
            origin="lower",
            aspect="auto",
            interpolation="nearest"
        )

        plt.colorbar(label="Joint probability")
        plt.xlabel("Second symbol")
        plt.ylabel("First symbol")
        plt.title("Joint symbol-pair distribution")
        plt.tight_layout()
        plt.show()

    def print_results(self):
        print(f"File: {self.file_path}")
        print(f"Size: {self.data.size} bytes")
        print(
            "Distinct symbols:",
            np.count_nonzero(self.symbol_counts)
        )
        print(
            f"H(X): {self.entropy_symbol():.4f} bits/symbol"
        )

        if self.data.size >= 2:
            print(
                f"H(X,Y): {self.entropy_pair():.4f} bits/pair"
            )
            print(
                f"H(X,Y)/2: "
                f"{self.entropy_pair() / 2:.4f} bits/symbol"
            )
            print(
                f"H(Y|X): "
                f"{self.entropy_conditional():.4f} bits/symbol"
            )
        else:
            print("Not enough data for pair analysis")
