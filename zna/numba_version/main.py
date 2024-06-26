import numpy as np
import object_intersection as ob
import rice_siff as rs
import matplotlib.pyplot as plt
import factorize as fc
from tabulate import tabulate

def plot_data(num_of_matrices, lengths_objintr, lengths_ricesiff):
    x = np.arange(num_of_matrices)
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, lengths_objintr, width, label='ObjIntr')
    rects2 = ax.bar(x + width / 2, lengths_ricesiff, width, label='RiceSiff')

    ax.set_xlabel('Matrix Index')
    ax.set_ylabel('Number of Concepts')
    ax.set_title('Number of Concepts by Object Intersection and Rice-Siff Algorithms')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Matrix {i + 1}' for i in x])
    ax.legend()

    fig.tight_layout()

    plt.show()
def generate_matrix(n_of_rows, n_of_cols):
    return np.random.choice([0, 1], size=(n_of_rows, n_of_cols))

if __name__ == "__main__":
    num_of_matrices = 20
    num_of_rows = 4
    num_of_cols = 4

    lengths_objintr = []
    lengths_ricesiff = []

    counter = 0
    results = []
    for _ in range (num_of_matrices):
        counter+=1
        matrix = generate_matrix(num_of_rows,num_of_cols)
        concepts_from_objintr = ob.object_intersection(matrix)
        concepts_from_objintr = [set(el) for el in concepts_from_objintr]

        concepts_from_ricesiff = rs.rice_siff(matrix)
        concepts_from_ricesiff = [set(el) for el in concepts_from_ricesiff]

        cover_objintr = fc.set_cover(set(range(num_of_cols)), concepts_from_objintr)
        cover_ricesiff = fc.set_cover(set(range(num_of_cols)), concepts_from_ricesiff)


        lengths_objintr.append(len(concepts_from_objintr))
        lengths_ricesiff.append(len(concepts_from_ricesiff))
        results.append({
            "Round": counter,
            "Objintr Concepts": len(concepts_from_objintr),
            "Objintr Sets": concepts_from_objintr,
            "Ricesiff Concepts": len(concepts_from_ricesiff),
            "Ricesiff Sets": concepts_from_ricesiff,
            "Cover Objintr": cover_objintr,
            "Cover Ricesiff": cover_ricesiff
        })

    headers = ["Round", "Object Intersection Concepts", "Object Intersection Sets", "Rice-Siff Concepts", "Rice-Siff Sets", "Cover Object Intersection", "Cover Rice-Siff"]
    table = []

    for result in results:
        table.append([
            result["Round"],
            result["Objintr Concepts"],
            result["Objintr Sets"],
            result["Ricesiff Concepts"],
            result["Ricesiff Sets"],
            result["Cover Objintr"],
            result["Cover Ricesiff"]
        ])

    print(tabulate(table, headers, tablefmt="grid"))