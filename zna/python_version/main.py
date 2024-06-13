import numpy as np
import object_intersection as ob
import rice_siff as rs
import matplotlib.pyplot as plt
import factorize as fc
from tabulate import tabulate
import time


def plot_concepts_comparison(mat_size,rounds, lengths_objintr, lengths_ricesiff):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, lengths_objintr, label='Objintr Concepts', marker='o')
    plt.plot(rounds, lengths_ricesiff, label='Ricesiff Concepts', marker='x')
    plt.xlabel('Round Number')
    plt.ylabel('Number of Concepts')
    plt.title(f'Comparison of Concepts Detected by Objintr and Ricesiff for matrix size {mat_size}x{mat_size}')
    plt.legend()
    plt.grid(True)
    plt.show()


def benchmark_performance(num_of_matrices, num_of_rows, num_of_cols):
    times = {
        'generate_matrix': [],
        'object_intersection': [],
        'rice_siff': [],
        'set_cover': []
    }

    for _ in range(num_of_matrices):
        # Benchmark generate_matrix
        start_time = time.perf_counter()
        matrix = generate_matrix(num_of_rows, num_of_cols)
        times['generate_matrix'].append(time.perf_counter() - start_time)

        # Benchmark object_intersection
        start_time = time.perf_counter()
        concepts_from_objintr = ob.object_intersection(matrix)
        times['object_intersection'].append(time.perf_counter() - start_time)

        # Benchmark rice_siff
        start_time = time.perf_counter()
        concepts_from_ricesiff = rs.rice_siff(matrix)
        times['rice_siff'].append(time.perf_counter() - start_time)

        # Benchmark set_cover
        start_time = time.perf_counter()
        fc.set_cover(set(range(num_of_cols)), [set(el) for el in concepts_from_objintr])
        times['set_cover'].append(time.perf_counter() - start_time)

    # Plot the benchmarking results
    plt.figure(figsize=(12, 8))
    labels = times.keys()
    avg_times = [sum(times[label]) / num_of_matrices for label in labels]

    plt.bar(labels, avg_times, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Function')
    plt.ylabel('Average Time Taken (seconds)')
    plt.title('Performance Benchmarking of Functions')
    plt.show()
def generate_matrix(n_of_rows, n_of_cols):
    return np.random.choice([0, 1], size=(n_of_rows, n_of_cols))


if __name__ == "__main__":
    num_of_matrices = 20
    num_of_rows = 20
    num_of_cols = 20

    lengths_objintr = []
    lengths_ricesiff = []

    counter = 0
    results = []
    rounds = list(range(1, num_of_matrices + 1))
    for _ in rounds:
        counter += 1
        matrix = generate_matrix(num_of_rows, num_of_cols)
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

    headers = ["Round", "Object Intersection Concepts", "Object Intersection Sets", "Rice-Siff Concepts",
               "Rice-Siff Sets", "Cover Object Intersection", "Cover Rice-Siff"]
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

    plot_concepts_comparison(num_of_rows, rounds, lengths_objintr, lengths_ricesiff)
    benchmark_performance(num_of_matrices, num_of_rows, num_of_cols)
