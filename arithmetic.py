
from numpy.typing import NDArray
import numpy as np
from bisect import bisect_right

# 32-bit arithmetic coding constants
BITS = 32
FULL = 1 << BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = 3 * QUARTER

MAX_TOTAL = 1 << 15
MAX_IMAGE_SIZE = 256 * 256

SYMBOL_COUNT = 256
FREQUENCY_BYTES = 2
SIZE_BYTES = 3

SPARSE = 0
DENSE = 1

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
    counts = np.bincount(data, minlength=SYMBOL_COUNT).tolist()
    total = sum(counts)

    if total <= MAX_TOTAL:
        return counts

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


def pack_header(data_size: int, frequencies: list[int]) -> bytes:
    if not 0 <= data_size <= MAX_IMAGE_SIZE:
        raise ValueError("Invalid image size")

    if len(frequencies) != SYMBOL_COUNT:
        raise ValueError("Invalid frequency table size")

    if any(f < 0 or f > MAX_TOTAL for f in frequencies):
        raise ValueError("Invalid frequency")

    symbols = [
        symbol
        for symbol, frequency in enumerate(frequencies)
        if frequency > 0
    ]

    sparse_size = 2 + len(symbols) * (1 + FREQUENCY_BYTES)
    dense_size = SYMBOL_COUNT * FREQUENCY_BYTES

    header = bytearray(data_size.to_bytes(SIZE_BYTES, "big"))

    if sparse_size < dense_size:
        header.append(SPARSE)
        header.extend(len(symbols).to_bytes(2, "big"))

        for symbol in symbols:
            header.append(symbol)
            header.extend(
                frequencies[symbol].to_bytes(FREQUENCY_BYTES, "big")
            )

    else:
        header.append(DENSE)

        for frequency in frequencies:
            header.extend(frequency.to_bytes(FREQUENCY_BYTES, "big"))

    return bytes(header)


def unpack_header(raw: bytes) -> tuple[int, list[int], int]:
    if len(raw) < SIZE_BYTES + 1:
        raise ValueError("Invalid arithmetic-coded data")

    original_size = int.from_bytes(raw[:SIZE_BYTES], "big")
    format_type = raw[SIZE_BYTES]

    if original_size > MAX_IMAGE_SIZE:
        raise ValueError("Invalid original size")

    frequencies = [0] * SYMBOL_COUNT
    offset = SIZE_BYTES + 1

    if format_type == SPARSE:
        if len(raw) < offset + 2:
            raise ValueError("Invalid sparse header")

        symbol_count = int.from_bytes(raw[offset:offset + 2], "big")
        offset += 2

        if symbol_count > SYMBOL_COUNT:
            raise ValueError("Invalid symbol count")

        if len(raw) < offset + symbol_count * (1 + FREQUENCY_BYTES):
            raise ValueError("Invalid frequency table")

        for _ in range(symbol_count):
            symbol = raw[offset]
            offset += 1

            frequency = int.from_bytes(
                raw[offset:offset + FREQUENCY_BYTES], "big"
            )
            offset += FREQUENCY_BYTES

            if frequency == 0 or frequencies[symbol] != 0:
                raise ValueError("Invalid frequency table")

            frequencies[symbol] = frequency

    elif format_type == DENSE:
        required_size = SYMBOL_COUNT * FREQUENCY_BYTES

        if len(raw) < offset + required_size:
            raise ValueError("Invalid dense header")

        for symbol in range(SYMBOL_COUNT):
            frequencies[symbol] = int.from_bytes(
                raw[offset:offset + FREQUENCY_BYTES], "big"
            )
            offset += FREQUENCY_BYTES

    else:
        raise ValueError("Invalid header format")

    total = sum(frequencies)

    if original_size == 0:
        if total != 0:
            raise ValueError("Invalid empty frequency table")
    elif not 0 < total <= MAX_TOTAL:
        raise ValueError("Invalid frequency total")

    return original_size, frequencies, offset


def compress(data_to_compress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    data = np.asarray(data_to_compress, dtype=np.uint8).ravel()

    if len(data) > MAX_IMAGE_SIZE:
        raise ValueError("Image exceeds maximum size")

    frequencies = build_frequencies(data)
    cumulative = cumulative_frequencies(frequencies)
    total = cumulative[-1]

    header = pack_header(len(data), frequencies)

    if len(data) == 0:
        return np.frombuffer(header, dtype=np.uint8).copy()

    writer = BitWriter()

    low = 0
    high = FULL - 1
    pending = 0

    def emit(bit: int):
        nonlocal pending

        writer.write(bit)

        while pending:
            writer.write(1 - bit)
            pending -= 1

    for symbol in data:
        symbol = int(symbol)

        # Narrow the interval.
        width = high - low + 1

        new_low = low + (width * cumulative[symbol]) // total
        new_high = low + (width * cumulative[symbol + 1]) // total - 1

        low = new_low
        high = new_high

        # Renormalization.
        while True:

            # E1: Lower half.
            if high < HALF:
                emit(0)

            # E2: Upper half.
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF

            # E3: Middle half.
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER

            else:
                break

            low = low * 2
            high = high * 2 + 1

    # Finalize the bitstream.
    pending += 1

    if low < QUARTER:
        emit(0)
    else:
        emit(1)

    return np.frombuffer(
        header + writer.finish(),
        dtype=np.uint8
    ).copy()


def decompress(data_to_uncompress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    raw = np.asarray(data_to_uncompress, dtype=np.uint8).tobytes()

    original_size, frequencies, offset = unpack_header(raw)

    if original_size == 0:
        return np.empty(0, dtype=np.uint8)

    cumulative = cumulative_frequencies(frequencies)
    total = cumulative[-1]

    reader = BitReader(raw[offset:])

    # Initialize decoder with 32 bits.
    value = 0

    for _ in range(BITS):
        value = (value << 1) | reader.read()

    low = 0
    high = FULL - 1

    result = np.empty(original_size, dtype=np.uint8)

    for i in range(original_size):
        width = high - low + 1

        # Map encoded value to cumulative frequency.
        scaled = ((value - low + 1) * total - 1) // width

        symbol = bisect_right(cumulative, scaled) - 1

        if symbol < 0 or symbol >= SYMBOL_COUNT:
            raise ValueError("Invalid encoded symbol")

        if frequencies[symbol] == 0:
            raise ValueError("Invalid encoded symbol")

        result[i] = symbol

        # Reconstruct the interval.
        new_low = low + (width * cumulative[symbol]) // total
        new_high = low + (width * cumulative[symbol + 1]) // total - 1

        low = new_low
        high = new_high

        # Renormalization.
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

            value = value * 2 + reader.read()

    return result
