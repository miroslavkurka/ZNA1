import numpy as np
def pseudometric(concept_one, concept_two):
    """
    Calculate the Rice-Siff pseudometric between two concepts.
    Args:
        concept_one:
        concept_two:

    Returns:
        float: Pseudometric between two concepts. Which is 1 - (|A ∩ B| / |A ∪ B|)

    """
    return 1 - (len(np.intersect1d(concept_one, concept_two)) / len(np.union1d(concept_one, concept_two)))

def minimum_similarity():
    """
    Find the minimum similarity between two concepts.
    Args:

    Returns:
        float: Minimum similarity between two concepts.
        int: Index of the first concept.
        int: Index of the second concept.
    """


def rice_siff(context):
    """
    Rice-Siff algorithm for computing the similarity between objects in a context.
    Args:
        context: 2D array containing the context.

    Returns:
        list of lists: List containing the intersection of objects.
    """
    c = rice_siff.reduce_bar_table(context)
    d = rice_siff.reduce_bar_table(context)
    while len(d) > 1:
        min_similarity, index1, idx2 = rice_siff.find_minimum_similarity(d)
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