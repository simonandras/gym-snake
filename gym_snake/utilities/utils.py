
import numpy as np


def xor(a: bool, b: bool) -> bool:
    """
    Logical xor
    """

    return (a and not b) or (not a and b)


def array_in_collection(coll, x: np.ndarray) -> bool:
    """
    returns True if there is an x in coll
    """

    for i in coll:
        if np.array_equal(i, x):
            return True

    return False


def increase_resolution(a: np.array, k: int) -> np.ndarray:
    """
    a should have shape (m, n)
    """

    m, n = a.shape

    result = np.zeros(shape=(k * m, k * n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            result[k*i:k*(i+1), k*j:k*(j+1)] = a[i, j] * np.ones(shape=(k, k), dtype=np.float32)

    return result
