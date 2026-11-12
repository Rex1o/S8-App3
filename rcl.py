def compress(data_to_compress : bytes) -> bytes:
    if not data_to_compress:
        return b""

    result = bytearray()
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
    return bytes(result)

def decompress(data: bytes) -> bytes:
    if len(data) % 2 != 0:
        raise ValueError("Invalid RCL compressed data")

    result = bytearray()

    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i + 1]
        result.extend([value] * count)

    return bytes(result)