from numpy.typing import NDArray
import numpy as np
from statistics import SourceAnalyzer
import statistics

def compress(data_to_compress : NDArray[np.uint8]) -> NDArray[np.uint8]:
    if data_to_compress.size == 0:
        return np.array([], dtype=np.uint8)

    # Create a list instead of an array to ensure we don't copy the np array over and over
    result = list()
    count = 1
    previous = data_to_compress[0]

    for current in data_to_compress[1:]:
        if current == previous and count < 255:
            count += 1
        else:
            result.extend([count, previous])
            previous = current
            count = 1

    result.extend([count, previous])

    return np.array(result, dtype=np.uint8)

def decompress(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if len(data) % 2 != 0:
        raise ValueError("Invalid RCL compressed data")

    result = list()

    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i + 1]
        result.extend([value] * count)

    return np.array(result, dtype=np.uint8)