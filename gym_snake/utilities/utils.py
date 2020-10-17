
import numpy as np


def array_in_collection(coll, x) -> bool:
    """
    x: np.array

    returns True if there is an x in coll
    """
    for i in coll:
        if np.array_equal(i, x):
            return True

    return False


def beside_each_other(x1, x2) -> bool:
    """
    Checks whether x1 and x2 are adjacent cells or not

    x1, x2: 2d np.array([y, x])
    The first coordinate corresponds to the vertical axes
    """

    if np.array_equal(x2, np.array([x1[0]+1, x[1]])) or \
            np.array_equal(x2, np.array([x1[0]-1, x[1]])) or \
            np.array_equal(x2, np.array([x1[0], x[1]+1])) or \
            np.array_equal(x2, np.array([x1[0], x[1]-1])):
        return True
    else:
        return False


def around(x) -> list:
    """
    Return the list of vector around x: [down, up, right, left]

    x: np.array([a, b])
    """

    return [np.array(x[0]+1, x[1]), np.array(x[0]-1, x[1]), np.array(x[0], x[1]+1), np.array(x[0], x[1]-1)]
