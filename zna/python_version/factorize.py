import numpy as np


def set_cover(universe, subsets):
    """
    Find the minimum subset of subsets that covers the universe greedy algorithm.

    Args:
        universe:
        subsets:

    Returns: cover of the universe

    """
    universe_set = np.array(list(universe))
    subsets_array = np.array([np.array(list(s)) for s in subsets], dtype=object)

    elements = np.unique(np.concatenate(subsets_array))

    if not np.array_equal(np.sort(elements), np.sort(universe_set)):
        return None

    covered = np.array([], dtype=object)
    cover = []

    while len(covered) < len(elements):
        # Find the subset with the maximum uncovered elements
        subset = max(subsets, key=lambda s: len(set(s) - set(covered)))
        cover.append(subset)
        covered = np.unique(np.concatenate((covered, list(subset))))

    return cover


# Example usage
universe = {1, 2, 3, 4, 5}
subsets = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]

print(set_cover(universe, subsets))
