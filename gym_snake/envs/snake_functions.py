
import numpy as np


def beside_each_other(x1: np.ndarray, x2: np.ndarray) -> bool:
    """
    Checks whether x1 and x2 are adjacent cells or not

    x1, x2: 2d np.array([y, x])
    The first coordinate corresponds to the vertical axes
    """

    if np.array_equal(x2, np.array([x1[0]+1, x1[1]])) or \
            np.array_equal(x2, np.array([x1[0]-1, x1[1]])) or \
            np.array_equal(x2, np.array([x1[0], x1[1]+1])) or \
            np.array_equal(x2, np.array([x1[0], x1[1]-1])):
        return True
    else:
        return False


def around(x: np.ndarray) -> list:
    """
    Return the list of vector around x: [down, up, right, left]

    x: np.array([a, b])
    """

    return [np.array([x[0]+1, x[1]]),
            np.array([x[0]-1, x[1]]),
            np.array([x[0], x[1]+1]),
            np.array([x[0], x[1]-1])]


def where_is_compared(x: np.ndarray, y: np.ndarray) -> str:
    """
    returns where is y compared to x

         y
         |
    y - [x] - y
         |
         y
    """

    if x[0] == y[0]:
        if x[1] < y[1]:
            return "right"
        elif x[1] > y[1]:
            return "left"
        else:
            raise ValueError("x and y cant be equal")
    elif x[1] == y[1]:
        if x[0] < y[0]:
            return "down"
        elif x[0] > y[0]:
            return "up"
        else:
            raise ValueError("x and y cant be equal")
    else:
        raise ValueError("x and y should be adjacent")
