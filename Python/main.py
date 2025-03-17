import numpy as np
from numba import njit, prange, config
import multiprocessing as mp
import os
from time import perf_counter
import shutil

config.PARALLEL_DIAGNOSTICS = 0
config.THREADING_LAYER = 'tbb'
config.FASTMATH = True


NUMBA_THREADS = int(os.environ.get('NUMBA_NUM_THREADS', os.cpu_count()))
print(f"Using {NUMBA_THREADS} threads for Numba parallelization")


@njit(fastmath=True, cache=True, nogil=True)
def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx*dx + dy*dy)


@njit(fastmath=True, parallel=True, nogil=True)
def pairwise_distances(points):
    n = points.shape[0]
    dist = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i, n):
            dist[i,j] = euclidean_distance(points[i], points[j])
            dist[j,i] = dist[i,j]
    return dist


@njit(fastmath=True)
def nearest_neighbor(points):
    n = points.shape[0]
    path = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    path[0] = 0
    visited[0] = True

    for i in range(1, n):
        last = path[i - 1]
        min_dist = np.inf
        best = -1

        for j in prange(n):
            if not visited[j]:
                dist = euclidean_distance(points[last], points[j])
                if dist < min_dist:
                    min_dist = dist
                    best = j

        path[i] = best
        visited[best] = True

    return path


@njit(fastmath=True)
def two_opt_optimize(points, path, dist_matrix):
    n = path.size
    improved = True
    total_improvements = 0

    while improved:
        improved = False
        best_delta = 0.0
        best_i = -1
        best_j = -1

        for i in prange(1, n - 1):
            for j in prange(i + 2, n):
                a, b = path[i - 1], path[i]
                c, d = path[j], path[(j + 1) % n]

                delta = (dist_matrix[a, b] + dist_matrix[c, d]) \
                        - (dist_matrix[a, c] + dist_matrix[b, d])

                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_delta > 1e-12:
            path[best_i:best_j + 1] = path[best_i:best_j + 1][::-1]
            improved = True
            total_improvements += 1

    return path


@njit(fastmath=True)
def three_opt_optimize(points, path, dist_matrix):
    n = path.size
    if n < 6:
        return path

    improved = True
    max_iterations = 30
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        best_delta = 0.0
        best_case = -1
        best_i, best_j, best_k = -1, -1, -1

        for i in prange(n - 5):
            for j in prange(i + 2, n - 3):
                for k in prange(j + 2, n - 1):
                    a, b = path[i], path[i + 1]
                    c, d = path[j], path[j + 1]
                    e, f = path[k], path[(k + 1) % n]

                    original = dist_matrix[a, b] + dist_matrix[c, d] + dist_matrix[e, f]

                    cases = np.array([
                        dist_matrix[a, c] + dist_matrix[b, e] + dist_matrix[d, f],
                        dist_matrix[a, d] + dist_matrix[e, b] + dist_matrix[c, f],
                        dist_matrix[a, e] + dist_matrix[d, c] + dist_matrix[b, f]
                    ])

                    best_case_local = np.argmin(cases)
                    delta = original - cases[best_case_local]

                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_j, best_k = i, j, k
                        best_case = best_case_local

        if best_delta > 1e-12:
            i, j, k = best_i, best_j, best_k
            segment = path[i + 1:k + 1].copy()

            if best_case == 0:
                path[i + 1:j + 1] = segment[:j - i][::-1]
            elif best_case == 1:
                path[i + 1:k + 1] = np.concatenate((
                    segment[j - i:],
                    segment[:j - i]
                ))
            elif best_case == 2:
                path[i + 1:j + 1] = segment[:j - i][::-1]
                path[j + 1:k + 1] = segment[j - i:][::-1]

            improved = True

    return path


def process_file(filename):
    points = read_tsp_file(filename)
    dist_matrix = pairwise_distances(points)

    start = perf_counter()
    path = nearest_neighbor(points)
    nn_time = perf_counter() - start

    start = perf_counter()
    path = two_opt_optimize(points, path, dist_matrix)
    two_opt_time = perf_counter() - start
    two_opt_dist = path_distance(points, path, dist_matrix)

    three_opt_dist = 0.0
    three_opt_time = 0.0
    if points.shape[0] < 1000:
        start = perf_counter()
        path = three_opt_optimize(points, path, dist_matrix)
        three_opt_time = perf_counter() - start
        three_opt_dist = path_distance(points, path, dist_matrix)

    print(f"{os.path.basename(filename)}\t"
          f"{two_opt_dist:.5f}\t{two_opt_time:.5f}\t"
          f"{three_opt_dist:.5f}\t{three_opt_time:.5f}")


@njit(fastmath=True)
def path_distance(points, path, dist_matrix):
    dist = 0.0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i], path[i + 1]]
    dist += dist_matrix[path[-1], path[0]]
    print(dist)
    return dist


def read_tsp_file(filename):
    with open(filename) as f:
        n = int(next(f).strip())
        points = np.loadtxt(f, dtype=np.float64, max_rows=n)
    return points


def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


if __name__ == "__main__":
    if os.path.exists('__pycache__'):
        shutil.rmtree('__pycache__')

    print("Enter 1 to process a single file, 2 to process a directory:")
    choice = input().strip()

    if choice == '1':
        print("Enter TSP file path:")
        filename = input().strip()
        if os.path.exists(filename):
            process_file(filename)
        else:
            print("Error: File not found.")
    elif choice == '2':
        print("Enter directory path:")
        directory = input().strip()
        if os.path.isdir(directory):
            files = sorted([os.path.join(directory, f)
                            for f in os.listdir(directory)], key=natural_sort_key)

            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.map(process_file, files)
        else:
            print("Error: Directory not found.")
    else:
        print("Invalid choice!")
