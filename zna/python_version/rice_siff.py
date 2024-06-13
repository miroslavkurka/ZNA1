import numpy as np
import object_intersection


def pseudometric(concept_one, concept_two):
    """
    Calculate the Rice-Siff pseudometric between two concepts.
    Args:
        concept_one:
        concept_two:

    Returns:
        float: Pseudometric between two concepts. Which is 1 - (|A ∩ B| / |A ∪ B|)

    """
    union_size=len(np.union1d(concept_one, concept_two))
    intersection_size=len(np.intersect1d(concept_one, concept_two))
    if union_size == 0:
        return 1
    return 1 - (intersection_size / union_size)


def find_minimum_similarity(reduced_context):
    """
    Find the pair of rows with the minimum similarity score in the matrix_of_indices.

    Parameters:
        reduced_context (np.ndarray): Input 2D array.
            arrays with indices of non-zero elements for each row.
    Returns:
    tuple: Minimum similarity score, and indices of the two rows with that score.
    """
    min_similarity = np.inf
    index1, index2 = -1, -1
    num_rows = len(reduced_context)

    # Iterate over all pairs of rows
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            similarity = pseudometric(reduced_context[i], reduced_context[j])
            if similarity < min_similarity:
                min_similarity = similarity
                index1, index2 = i, j

    return min_similarity, index1, index2


def rice_siff(context):
    """
    Rice-Siff algorithm for computing the similarity between objects in a context.
    Args:
        context: 2D array containing the context.

    Returns:
        list of lists: List containing the intersection of objects.
    """
    c = object_intersection.reduce_bar_table(context)
    d = object_intersection.reduce_bar_table(context)
    while len(d) > 1:
        min_similarity, index1, idx2 = find_minimum_similarity(d)
        row1 = d[index1]
        row2 = d[idx2]
        intersection = np.intersect1d(row1, row2).tolist()

        # Remove two most similar rows
        d = [d[i] for i in range(len(d)) if i != index1 and i != idx2]

        # Append unique intersections to c and d
        if intersection not in c:
            c.append(intersection)
        if intersection not in d:
            d.append(intersection)
    return c
