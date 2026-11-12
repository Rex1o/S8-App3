import argparse
from pathlib import Path
import arithmetic
import huffman
import lz77
import rcl
import statistics


def parse_args():
    parser = argparse.ArgumentParser(
        description="File compression using different algorithms."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the file to compress"
    )

    parser.add_argument(
        "-a", "--algorithm",
        type=str.upper,
        choices=["RCL", "LZ77", "ARITHMETIC", "HUFFMAN"],
        required=True,
        help="Compression algorithm to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(f"File not found: {args.file}")

    data = args.file.read_bytes()
    analyzer = statistics.SourceAnalyzer(args.file)
    analyzer.print_results()
    analyzer.plot_symbols()
    analyzer.plot_pairs()

    print(f"File: {args.file}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Original size: {len(data)} bytes")

    if args.algorithm == "RCL":
        print("Using RCL compression")
        compressed = rcl.compress(data)
        decompressed = rcl.decompress(compressed)

    elif args.algorithm == "LZ77":
        print("Using LZ77 compression")
        compressed = lz77.compress(data)
        decompressed = lz77.decompress(compressed)

    elif args.algorithm == "ARITHMETIC":
        print("Using Arithmetic compression")
        compressed = arithmetic.compress(data)
        decompressed = arithmetic.decompress(compressed)

    elif args.algorithm == "HUFFMAN":
        print("Using Huffman compression")
        compressed = huffman.compress(data)
        decompressed = huffman.decompress(compressed)




if __name__ == "__main__":
    main()