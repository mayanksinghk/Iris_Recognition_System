import cv2
import math
import numpy as np
import inspect
from typing import List, Tuple, Optional

def load_image(filepath: str, show: bool = False) -> np.ndarray:
    """
    Loads an image in grayscale format.
    """
    img = cv2.imread(filepath, 0)
    if show:
        cv2.imshow(filepath, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

def point_in_circle(c_col: int, c_row: int, c_radius: float, p_col: int, p_row: int) -> bool:
    """
    Evaluates whether a specific pixel coordinate falls within a defined circle boundary[cite: 2].
    """
    return distance(c_col, c_row, p_col, p_row) <= c_radius

def angle_v(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Computes the angle in degrees between two spatial coordinates[cite: 2].
    """
    return math.degrees(math.atan2(-(y2 - y1), (x2 - x1)))

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculates the standard Euclidean distance between two points[cite: 2].
    """
    dst = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dst

def mean(x: List[float]) -> float:
    """
    Calculates the arithmetic mean of a provided list of values[cite: 2].
    """
    sum_val = 0.0
    for i in range(len(x)):
        sum_val += x[i]
    return sum_val / len(x)

def median(x: List[float]) -> float:
    """
    Calculates the median of a list utilizing numpy arrays[cite: 2].
    """
    return float(np.median(np.array(x)))

def standard_dev(x: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes both the mean and standard deviation of a given list[cite: 2].
    Includes frame inspection for debugging empty list errors[cite: 2].
    """
    if not x:
        print('Error: empty list parameter in standard_dev() !')
        print(inspect.getouterframes(inspect.currentframe())[1])
        print()
        return None, None
        
    m = mean(x)
    sumsq = 0.0
    for i in range(len(x)):
        sumsq += (x[i] - m) ** 2
        
    return m, math.sqrt(sumsq / len(x))