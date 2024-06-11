import numpy as np
from numba import njit


@njit
def set_cover(universe, subsets):
    # Convert universe to a sorted NumPy array
    universe_arr = np.array(sorted(universe))
    # Convert subsets to a list of NumPy arrays
    subsets_arr = [np.array(list(s)) for s in subsets]

    # Flatten subsets and find unique elements
    all_elements = np.concatenate(subsets_arr)
    unique_elements = np.unique(all_elements)

    # Check if unique elements match universe
    if not np.array_equal(np.sort(unique_elements), universe_arr):
        return None

    covered = np.array([], dtype=np.int64)
    cover = []

    # Track which elements are covered
    covered_set = set()

    while len(covered_set) < len(universe_arr):
        # Find the subset with the maximum number of uncovered elements
        max_coverage = -1
        best_subset = None
        for subset in subsets_arr:
            uncovered_elements = np.setdiff1d(subset, covered)
            num_uncovered = len(uncovered_elements)
            if num_uncovered > max_coverage:
                max_coverage = num_uncovered
                best_subset = subset

        if best_subset is not None:
            cover.append(best_subset)
            covered = np.unique(np.concatenate((covered, best_subset)))
            covered_set.update(best_subset)

    return cover


# Example usage
universe = {1, 2, 3, 4, 5}
subsets = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]

cover = set_cover(universe, subsets)
for subset in cover:
    print(set(subset))
