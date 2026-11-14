
from numpy.typing import NDArray
import numpy as np

# LZ77 configuration
TOKEN_BITS = 16
LENGTH_BITS = 4
MIN_MATCH = 3

# Derived constants
DISTANCE_BITS = TOKEN_BITS - LENGTH_BITS

WINDOW_SIZE = 1 << DISTANCE_BITS
MAX_MATCH = MIN_MATCH + (1 << LENGTH_BITS) - 1

LENGTH_MASK = (1 << LENGTH_BITS) - 1
TOKEN_BYTES = TOKEN_BITS // 8

# Validate configuration
assert TOKEN_BITS % 8 == 0
assert 0 < LENGTH_BITS < TOKEN_BITS
assert MIN_MATCH > 0


def compress(data_to_compress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    data = data_to_compress.tobytes()
    n = len(data)

    # Header: original uncompressed size (64 bits).
    output = bytearray(n.to_bytes(8, "big"))

    pos = 0

    while pos < n:
        flags = 0
        tokens = bytearray()

        # One flag bit per token, packed into a byte.
        for bit in range(8):
            if pos >= n:
                break

            best_length = 0
            best_distance = 0

            window_start = max(0, pos - WINDOW_SIZE)
            max_length = min(MAX_MATCH, n - pos)

            # Find the longest matching sequence in the window.
            if max_length >= MIN_MATCH:
                prefix = data[pos:pos + MIN_MATCH]
                candidate = data.rfind(prefix, window_start, pos)

                while candidate != -1:
                    length = MIN_MATCH

                    # Extend the match as far as possible.
                    while (
                        length < max_length
                        and data[candidate + length] == data[pos + length]
                    ):
                        length += 1

                    if length > best_length:
                        best_length = length
                        best_distance = pos - candidate

                    if best_length == max_length:
                        break

                    candidate = data.rfind(
                        prefix, window_start, candidate
                    )

            if best_length >= MIN_MATCH:
                # Flag 1 = back-reference.
                flags |= 1 << bit

                # Pack distance and length into one token.
                packed = (
                    ((best_distance - 1) << LENGTH_BITS)
                    | (best_length - MIN_MATCH)
                )

                tokens.extend(packed.to_bytes(TOKEN_BYTES, "big"))
                pos += best_length

            else:
                # Flag 0 = literal byte.
                tokens.append(data[pos])
                pos += 1

        output.append(flags)
        output.extend(tokens)

    return np.frombuffer(bytes(output), dtype=np.uint8)


def decompress(data_to_uncompress: NDArray[np.uint8]) -> NDArray[np.uint8]:
    data = data_to_uncompress.tobytes()

    HEADER_BYTES = 8

    if len(data) < HEADER_BYTES:
        raise ValueError("Invalid LZ77 header")

    original_size = int.from_bytes(data[:HEADER_BYTES], "big")

    output = bytearray()
    pos = HEADER_BYTES

    while len(output) < original_size:
        if pos >= len(data):
            raise ValueError("Truncated LZ77 data")

        flags = data[pos]
        pos += 1

        # Decode up to 8 tokens from the flag byte.
        for bit in range(8):
            if len(output) >= original_size:
                break

            is_match = (flags >> bit) & 1

            if is_match:
                if pos + TOKEN_BYTES > len(data):
                    raise ValueError("Truncated match")

                packed = int.from_bytes(
                    data[pos:pos + TOKEN_BYTES], "big"
                )
                pos += TOKEN_BYTES

                # Extract distance and length.
                distance = (packed >> LENGTH_BITS) + 1
                length = (packed & LENGTH_MASK) + MIN_MATCH

                if distance > len(output):
                    raise ValueError("Invalid match distance")

                if len(output) + length > original_size:
                    raise ValueError("Invalid match length")

                # Copy byte by byte to support overlapping matches.
                for _ in range(length):
                    output.append(output[-distance])

            else:
                if pos >= len(data):
                    raise ValueError("Truncated literal")

                output.append(data[pos])
                pos += 1

    if pos != len(data):
        raise ValueError("Unexpected trailing data")

    return np.frombuffer(bytes(output), dtype=np.uint8).copy()
