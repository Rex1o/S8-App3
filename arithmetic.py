
from numpy.typing import NDArray
import numpy as np
from bisect import bisect_right

# 32-bit arithmetic coding constants
BITS = 32
FULL = 1 << BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = 3 * QUARTER

# Keep cumulative frequencies small enough for integer coding.
MAX_TOTAL = 1 << 15


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.count = 0

    def write(self, bit: int):
        self.current = (self.current << 1) | bit
        self.count += 1

        if self.count == 8:
            self.data.append(self.current)
            self.current = 0
            self.count = 0

    def finish(self) -> bytes:
        if self.count:
            self.data.append(self.current << (8 - self.count))
        return bytes(self.data)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.index = 0

    def read(self) -> int:
        byte_index = self.index // 8
        bit_index = 7 - (self.index % 8)
        self.index += 1

        if byte_index >= len(self.data):
            return 0

        return (self.data[byte_index] >> bit_index) & 1


def build_frequencies(data: NDArray[np.uint8]) -> list[int]:
    counts = np.bincount(data, minlength=256).tolist()
    total = sum(counts)

    if total <= MAX_TOTAL:
        return counts

    # Scale frequencies while preserving every present symbol.
    divisor = (total + MAX_TOTAL - 1) // MAX_TOTAL

    while True:
        frequencies = [
            max(1, count // divisor) if count else 0
            for count in counts
        ]

        if sum(frequencies) <= MAX_TOTAL:
            return frequencies

        divisor += 1


def cumulative_frequencies(freq: list[int]) -> list[int]:
    cumulative = [0]

    for f in freq:
        cumulative.append(cumulative[-1] + f)

    return cumulative


def compress(data_to_compress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    data = np.asarray(data_to_compress, dtype=np.uint8).ravel()

    frequencies = build_frequencies(data)
    cumulative = cumulative_frequencies(frequencies)
    total = cumulative[-1]

    # Header:
    # 8 bytes: original number of symbols
    # 2 bytes: number of distinct symbols
    # For each distinct symbol:
    #   1 byte: symbol
    #   4 bytes: scaled frequency
    header = bytearray()
    header.extend(len(data).to_bytes(8, "big"))

    symbols = [s for s, f in enumerate(frequencies) if f > 0]
    header.extend(len(symbols).to_bytes(2, "big"))

    for symbol in symbols:
        header.append(symbol)
        header.extend(frequencies[symbol].to_bytes(4, "big"))

    if len(data) == 0:
        return np.frombuffer(bytes(header), dtype=np.uint8).copy()

    writer = BitWriter()

    low = 0
    high = FULL - 1
    pending = 0

    def emit(bit: int):
        nonlocal pending

        writer.write(bit)

        # Emit the opposite bit for each pending E3 condition.
        while pending:
            writer.write(1 - bit)
            pending -= 1

    for symbol in data:
        symbol = int(symbol)

        # Narrow the interval using integer arithmetic.
        width = high - low + 1

        new_low = low + (width * cumulative[symbol]) // total
        new_high = low + (width * cumulative[symbol + 1]) // total - 1

        low = new_low
        high = new_high

        # Renormalization
        while True:

            # E1: Interval entirely in lower half.
            if high < HALF:
                emit(0)

            # E2: Interval entirely in upper half.
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF

            # E3: Interval entirely in middle half.
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER

            else:
                break

            # Expand interval back to 32-bit precision.
            low = low * 2
            high = high * 2 + 1

    # Finalize the encoded bitstream.
    pending += 1

    if low < QUARTER:
        emit(0)
    else:
        emit(1)

    return np.frombuffer(
        bytes(header) + writer.finish(),
        dtype=np.uint8
    ).copy()


def decompress(data_to_uncompress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    raw = np.asarray(data_to_uncompress, dtype=np.uint8).tobytes()

    if len(raw) < 10:
        raise ValueError("Invalid arithmetic-coded data")

    original_size = int.from_bytes(raw[0:8], "big")
    symbol_count = int.from_bytes(raw[8:10], "big")

    if symbol_count > 256 or len(raw) < 10 + 5 * symbol_count:
        raise ValueError("Invalid frequency table")

    frequencies = [0] * 256
    offset = 10

    for _ in range(symbol_count):
        symbol = raw[offset]
        frequency = int.from_bytes(raw[offset + 1:offset + 5], "big")

        if frequency == 0 or frequencies[symbol] != 0:
            raise ValueError("Invalid frequency table")

        frequencies[symbol] = frequency
        offset += 5

    if original_size == 0:
        if symbol_count != 0:
            raise ValueError("Invalid empty frequency table")
        return np.empty(0, dtype=np.uint8)

    cumulative = cumulative_frequencies(frequencies)
    total = cumulative[-1]

    if not 0 < total <= MAX_TOTAL:
        raise ValueError("Invalid frequency total")

    reader = BitReader(raw[offset:])

    # Initialize the decoder with the first 32 bits.
    value = 0

    for _ in range(BITS):
        value = (value << 1) | reader.read()

    low = 0
    high = FULL - 1

    result = np.empty(original_size, dtype=np.uint8)

    for i in range(original_size):
        width = high - low + 1

        # Map the encoded value to a cumulative frequency.
        scaled = ((value - low + 1) * total - 1) // width

        # Find the symbol containing that frequency.
        symbol = bisect_right(cumulative, scaled) - 1

        if symbol < 0 or symbol >= 256 or frequencies[symbol] == 0:
            raise ValueError("Invalid encoded symbol")

        result[i] = symbol

        # Reconstruct the interval.
        new_low = low + (width * cumulative[symbol]) // total
        new_high = low + (width * cumulative[symbol + 1]) // total - 1

        low = new_low
        high = new_high

        # Same renormalization as the encoder.
        while True:

            if high < HALF:
                pass

            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF

            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER

            else:
                break

            low = low * 2
            high = high * 2 + 1

            # Shift in the next encoded bit.
            value = value * 2 + reader.read()

    return result
