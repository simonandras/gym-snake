
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
