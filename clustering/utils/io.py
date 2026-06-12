import numpy as np
import struct


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    n, d = m.shape
    myimt = "i" * d
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            bin = struct.pack(myimt, *m[i])
            f.write(bin)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    m = m.astype("float32")
    write_ivecs(filename, m.view("int32"))


def read_ibin(filename):
    n, d = np.fromfile(filename, count=2, dtype="int32")
    a = np.fromfile(filename, dtype="int32")
    print(f"\t{filename} readed")
    return a[2:].reshape(n, d)


def read_fbin(filename):
    return read_ibin(filename).view("float32")


def read_raw_fbin(filename, shape=None):
    """
    Read raw float32 binary file without header.
    
    Args:
        filename: Path to the binary file
        shape: Tuple (n, d) for reshaping. If None, returns 1D array
    
    Returns:
        numpy array with float32 data
    """
    data = np.fromfile(filename, dtype="float32")
    if shape is not None:
        data = data.reshape(shape)
    return data


def write_raw_fbin(filename, data):
    """
    Write numpy array as raw float32 binary file without header.
    
    Args:
        filename: Path to the output binary file
        data: numpy array to save (will be converted to float32)
    """
    # Ensure data is float32
    data = data.astype("float32")
    
    # Write to file
    data.tofile(filename)
    print(f"\t{filename} wrote ({data.shape} -> {data.size * 4} bytes)")
