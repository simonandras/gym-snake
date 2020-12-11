
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


def rotate(a: np.ndarray) -> np.ndarray:
    """
    Rotates the array in the negative direction
    Should be used on square shaped arrays
    """

    result = np.zeros(a.shape, dtype=np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[j][a.shape[0] - i - 1] = a[i][j]

    return result


def rotate_times(a: np.ndarray, n: int) -> np.ndarray:
    """
    n times rotates the array in the negative direction
    Should be used on square shaped arrays
    """

    result = np.copy(a)

    for _ in range(n):
        result = rotate(result)

    return result


def mirror(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Should be used on square shaped arrays
    axis: 0 is the y axis; 1 is the x axis
    """

    result = np.zeros(a.shape, dtype=np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if axis == 0:
                result[a.shape[0] - i - 1][j] = a[i][j]
            elif axis == 1:
                result[i][a.shape[1] - j - 1] = a[i][j]

    return result
