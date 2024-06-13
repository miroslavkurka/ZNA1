import numpy as np
from numba import njit


@njit
def reduce_bar_table(bar_table):
    """
    Extract indices where elements are greater than 0 for each row in the BAR table .

    Parameters:
    bar_table (np.ndarray): Input 2D array.

    Returns:
    list of lists: List containing indices of non-zero elements for each row.
    """
    bar_table = np.asarray(bar_table)

    true_indices = [list(np.where(row > 0)[0]) for row in bar_table]

    return true_indices


@njit
def object_intersection(bar_table):
    """
    Find the intersection of objects in the bar table.

    Args:
        bar_table: 2D array containing the bar table. (i.e. matrix of objects and attributes)

    Returns:
        concepts: List of lists containing the intersection of objects.
    """

    reduced_bar_table = reduce_bar_table(bar_table)
    concepts = [[_ for _ in range(len(bar_table[0]))]]
    for row in reduced_bar_table:
        for c in concepts:
            intersection = np.intersect1d(row, c)
            if intersection not in concepts:
                concepts.append(intersection)
    return concepts
