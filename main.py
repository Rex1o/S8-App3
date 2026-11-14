import argparse
from pathlib import Path
import arithmetic
import huffman
import lz77
import rcl
import statistics
import numpy as np

# This is only usefull for the huffman graph viz
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import struct

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def load_image(path: Path) -> tuple[NDArray[np.uint8], int, int]:
    with Image.open(path) as img:
        img = img.convert("L")  # Force 8-bit grayscale
        width, height = img.size
        pixels = np.asarray(img, dtype=np.uint8).flatten()

    return pixels, width, height


def save_image(data: NDArray[np.uint8], width: int, height: int, path: Path):
    pixels = data.reshape((height, width))
    Image.fromarray(pixels, mode="L").save(path)

def pack_image_header(compressed: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    header = np.frombuffer(struct.pack("<II", width, height), dtype=np.uint8)
    return np.concatenate((header, compressed))

def unpack_image_header(data: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], int, int]:
    if len(data) < 8:
        raise ValueError("Invalid compressed image header")
    width, height = struct.unpack("<II", data[:8].tobytes())
    return data[8:], width, height

def parse_args():
    parser = argparse.ArgumentParser(description="File compression using different algorithms.")
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

    is_image = args.file.suffix.lower() in IMAGE_EXTENSIONS

    # Load source
    if is_image:
        data, width, height = load_image(args.file)
    else:
        data = np.frombuffer(args.file.read_bytes(), dtype=np.uint8)

    analyzer = statistics.SourceAnalyzer(data)
    analyzer.print_results()
    analyzer.plot_symbols()
    analyzer.plot_pairs()

    print(f"File: {args.file}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Original size: {len(data)} bytes")

    # Compress raw data (pixels only for images)
    if args.algorithm == "RCL":
        compressed = rcl.compress(data)
        decompress_fn = rcl.decompress

    elif args.algorithm == "LZ77":
        compressed = lz77.compress(data)
        decompress_fn = lz77.decompress

    elif args.algorithm == "ARITHMETIC":
        compressed = arithmetic.compress(data)
        decompress_fn = arithmetic.decompress

    elif args.algorithm == "HUFFMAN":
        compressed = huffman.compress(data)
        decompress_fn = huffman.decompress

    # Add image header AFTER compression
    if is_image:
        compressed = pack_image_header(compressed, width, height)

    print(f"Compressed size: {len(compressed)} bytes")
    print(f"Compression ratio: {100 * len(compressed) / len(data):.2f}%")

    # Extract image header BEFORE decompression
    if is_image:
        compressed_data, width, height = unpack_image_header(compressed)
    else:
        compressed_data = compressed

    # Decompress
    decompressed = decompress_fn(compressed_data)

    assert np.array_equal(data, decompressed), "Decompression failed!"

    # Reconstruct image
    if is_image:
        output_path = args.file.with_name(
            args.file.stem + "_decompressed.png"
        )

        save_image(decompressed, width, height, output_path)
        print(f"Image saved to: {output_path}")

    print("Decompression successful!")

if __name__ == "__main__":
    main()