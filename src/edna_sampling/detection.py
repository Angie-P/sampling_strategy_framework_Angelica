"""The detection model, the station optimisers, and the geometry they need.

Pure array code: numpy at import time, plus `mesh.TriangleLocator` (shapely)
inside the two functions that need the model mesh. In particular no
oceantracker, no xarray and no numba at import time, so this module installs and
loads on a laptop that will never run the model.

That is deliberate and is the desktop/HPC split in miniature. Turning a run's raw
chunked output into an analysis product needs the modelling stack and belongs in
`edna_sampling.condense`; deciding where to sample given a concentration field
does not, and belongs here. `tests/test_imports.py` enforces the separation, and
`tests/test_imports.py` keeps it that way.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "greedy", "greedy_fast", "simulated_annealing", "genetic_algorithm",
    "optimize_stations", "valid_cell_mask", "mask_below_seafloor",
    "convert_geographic_to_stats_grid_indices",
]


def greedy(detectable, i):
    """
    Find the ideal subset of size i that maximizes the number of detectable sources.

    Parameters:
    detectable (numpy.ndarray): A 3D array of shape (n_releases, n_y, n_x) where the
                                last two dims index the stats grid in (y, x) order.
    i (int): The size of the subset to be selected.

    Returns:
    list: List of (iy, ix) tuples — grid indices of the selected locations.
          Note the y-then-x order, matching the array's spatial axes.
    set: Indices of release groups covered by the selected locations.
    """
    n, n_y, n_x = detectable.shape
    selected_locations = []
    covered_sources = set()

    for _ in range(i):
        best_location = None
        best_coverage = -1

        for iy in range(n_y):
            for ix in range(n_x):
                if (iy, ix) in selected_locations:
                    continue

                new_coverage = np.sum(detectable[:, iy, ix]) - len(covered_sources.intersection(np.where(detectable[:, iy, ix])[0]))

                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_location = (iy, ix)

        if best_location is not None:
            selected_locations.append(best_location)
            covered_sources.update(np.where(detectable[:, best_location[0], best_location[1]])[0])

    return selected_locations, covered_sources


def simulated_annealing(detectable, k, initial_temperature=None, cooling_rate=0.995,
                        max_iterations=5000, seed_with_greedy=False, random_seed=None):
    """
    Find the subset of size k that maximizes the number of detectable sources
    using Simulated Annealing.

    Objective: |{sources with any nonzero detectability across the chosen cells}|.

    Parameters:
    detectable (numpy.ndarray): shape (n_releases, n_y, n_x).
    k (int): subset size.
    initial_temperature (float | None): starting temperature. None auto-calibrates
        so ~80% of worsening moves are accepted at the start.
    cooling_rate (float): geometric cooling factor per iteration.
    max_iterations (int): number of swap proposals.
    seed_with_greedy (bool): start the chain from the greedy solution.
    random_seed (int | None): RNG seed for reproducibility.

    Returns:
    list[tuple[int, int]]: (iy, ix) indices of selected cells.
    set[int]: covered release-group indices.
    """
    rng = np.random.default_rng(random_seed)
    n_releases, n_y, n_x = detectable.shape
    n_cells = n_y * n_x
    # fill masked seafloor/out-of-domain cells with 0 before any indexing
    detectable_flat = (np.ma.filled(detectable, 0) > 0).reshape(n_releases, n_cells)

    def coverage_count(subset_flat):
        return int(detectable_flat[:, subset_flat].any(axis=1).sum())

    # Initial subset (flat indices)
    if seed_with_greedy:
        greedy_locs, _ = greedy(detectable, k)
        current = np.array([iy * n_x + ix for iy, ix in greedy_locs], dtype=np.int64)
    else:
        current = rng.choice(n_cells, size=k, replace=False)

    current_set = set(int(c) for c in current)
    current_fit = coverage_count(current)
    best = current.copy()
    best_fit = current_fit

    def propose_swap(subset, subset_set):
        j = int(rng.integers(0, k))
        while True:
            cand = int(rng.integers(0, n_cells))
            if cand not in subset_set:
                break
        trial = subset.copy()
        trial[j] = cand
        return trial, j, cand

    # Auto-calibrate initial temperature from a small sample of worsening moves
    if initial_temperature is None:
        neg_deltas = []
        for _ in range(50):
            trial, _, _ = propose_swap(current, current_set)
            d = coverage_count(trial) - current_fit
            if d < 0:
                neg_deltas.append(d)
        if neg_deltas:
            mean_neg = abs(float(np.mean(neg_deltas)))
            # P(accept) = exp(-|delta|/T) = 0.8  →  T = |delta| / -ln(0.8)
            initial_temperature = mean_neg / -math.log(0.8)
        else:
            initial_temperature = 1.0

    temperature = float(initial_temperature)

    for _ in range(max_iterations):
        trial, j, cand = propose_swap(current, current_set)
        trial_fit = coverage_count(trial)
        delta = trial_fit - current_fit

        if delta > 0 or rng.random() < math.exp(delta / temperature):
            old_val = int(current[j])
            current = trial
            current_set.discard(old_val)
            current_set.add(cand)
            current_fit = trial_fit
            if current_fit > best_fit:
                best = current.copy()
                best_fit = current_fit

        temperature *= cooling_rate

    best_locations = [(int(idx // n_x), int(idx % n_x)) for idx in best]
    covered = set(int(s) for s in np.where(detectable_flat[:, best].any(axis=1))[0])
    return best_locations, covered


def genetic_algorithm(detectable, k,
                      population_size=100, n_generations=200,
                      mutation_rate=0.1, crossover_rate=0.7,
                      tournament_size=3, elitism=2,
                      seed_with_greedy=False, random_seed=None):
    """
    Find the subset of size k that maximizes the number of detectable sources
    using a Genetic Algorithm.

    Representation: each chromosome is a length-k array of unique flat cell
    indices. Selection is by tournament; crossover draws the child from the
    union of two parents' cells; mutation replaces a gene with a random
    non-member; the top `elitism` individuals are carried unchanged.

    Objective: |{sources with any nonzero detectability across the chosen cells}|.

    Parameters:
    detectable (numpy.ndarray): shape (n_releases, n_y, n_x).
    k (int): subset size.
    population_size, n_generations: search budget (~pop * gens evals).
    mutation_rate (float): per-gene replacement probability.
    crossover_rate (float): probability of recombining two parents.
    tournament_size (int): tournament selection pressure.
    elitism (int): top-k carried unchanged each generation.
    seed_with_greedy (bool): include the greedy solution in the initial population.
    random_seed (int | None): RNG seed for reproducibility.

    Returns:
    list[tuple[int, int]]: (iy, ix) indices of selected cells.
    set[int]: covered release-group indices.
    """
    rng = np.random.default_rng(random_seed)
    n_releases, n_y, n_x = detectable.shape
    n_cells = n_y * n_x
    # fill masked seafloor/out-of-domain cells with 0 before any indexing
    detectable_flat = (np.ma.filled(detectable, 0) > 0).reshape(n_releases, n_cells)

    def fitness(subset):
        return int(detectable_flat[:, subset].any(axis=1).sum())

    def random_subset():
        return rng.choice(n_cells, size=k, replace=False)

    # Initial population
    population = [random_subset() for _ in range(population_size)]
    if seed_with_greedy:
        greedy_locs, _ = greedy(detectable, k)
        population[0] = np.array([iy * n_x + ix for iy, ix in greedy_locs], dtype=np.int64)

    fitnesses = np.array([fitness(p) for p in population])
    best_idx = int(np.argmax(fitnesses))
    best = population[best_idx].copy()
    best_fit = int(fitnesses[best_idx])

    def tournament_pick():
        idx = rng.integers(0, len(population), size=tournament_size)
        return population[idx[int(np.argmax(fitnesses[idx]))]]

    for _ in range(n_generations):
        order = np.argsort(-fitnesses)
        new_population = [population[i].copy() for i in order[:elitism]]

        while len(new_population) < population_size:
            p1 = tournament_pick()
            p2 = tournament_pick()

            # Set-union crossover: child drawn from cells appearing in either parent
            if rng.random() < crossover_rate:
                pool = np.unique(np.concatenate([p1, p2]))
                if len(pool) >= k:
                    child = rng.choice(pool, size=k, replace=False)
                else:
                    # parents fully overlap — pad with random extras
                    extra_pool = np.setdiff1d(np.arange(n_cells), pool, assume_unique=True)
                    extra = rng.choice(extra_pool, size=k - len(pool), replace=False)
                    child = np.concatenate([pool, extra])
            else:
                child = p1.copy()

            # Mutation: replace each gene with prob mutation_rate
            child_set = set(int(c) for c in child)
            for j in range(k):
                if rng.random() < mutation_rate:
                    while True:
                        cand = int(rng.integers(0, n_cells))
                        if cand not in child_set:
                            break
                    child_set.discard(int(child[j]))
                    child_set.add(cand)
                    child[j] = cand

            new_population.append(child)

        population = new_population
        fitnesses = np.array([fitness(p) for p in population])
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = int(fitnesses[gen_best_idx])
            best = population[gen_best_idx].copy()

    best_locations = [(int(idx // n_x), int(idx % n_x)) for idx in best]
    covered = set(int(s) for s in np.where(detectable_flat[:, best].any(axis=1))[0])
    return best_locations, covered


def mask_below_seafloor(c, x_stats, y_stats, z_stats, x_grid, y_grid, triangles, water_depth):
    """
    Mask 3D concentration array where depth layers intersect with or are below the seafloor.

    Parameters:
    -----------
    c : ndarray
        Concentration array with shape (time, release_group, depth_index, x_index, y_index)
    x_stats : ndarray
        X coordinates of the stats grid (201,)
    y_stats : ndarray
        Y coordinates of the stats grid (201,)
    x_grid : ndarray
        X coordinates of the original grid with bathymetry (126942,)
    y_grid : ndarray
        Y coordinates of the original grid with bathymetry (126942,)
    water_depth : ndarray
        Water depth at each grid point (126942,)
    height_of_depth_layer : float
        Height of each depth layer in meters

    Returns:
    --------
    c_masked : np.ma.MaskedArray
        Masked concentration array
    """

    # Create 2D meshgrid for stats grid
    X_stats, Y_stats = np.meshgrid(x_stats, y_stats, indexing='xy')

    # Locate each stats point on the model's own triangles and interpolate the
    # water depth there: nan off the mesh, and no matplotlib trifinder, which
    # refuses a mesh with overlapping triangles (see mesh.TriangleLocator)
    from edna_sampling.mesh import TriangleLocator
    locator = TriangleLocator(x_grid, y_grid, triangles)
    triangle_indices, weights = locator.locate(X_stats, Y_stats)
    water_depth_stats = locator.interpolate(water_depth, triangle_indices, weights)

    depth_layer_centers = z_stats
    depth_layer_top = depth_layer_centers - np.diff(z_stats)[0]/2
    depth_layer_bottom = depth_layer_centers + np.diff(z_stats)[0]/2

    number_of_depth_layers = len(depth_layer_centers)


    # Create mask: True where we want to mask out (depth >= water_depth).
    # water_depth_stats has shape (n_y, n_x) from meshgrid(..., indexing='xy').
    # mask_3d shape: (n_y, n_x, n_z) to match c shape (time, group, y, x, z).
    mask_3d = np.zeros((len(y_stats), len(x_stats), number_of_depth_layers), dtype=bool)

    for depth_idx in range(number_of_depth_layers):
        layer_depth = depth_layer_bottom[depth_idx]
        # Mask where layer depth >= water depth (intersects or below seafloor),
        # and where the depth is unknown
        mask_3d[:, :, depth_idx] = ~(layer_depth < water_depth_stats)

    # Points outside the mesh have triangle_index == -1
    outside_domain = triangle_indices == -1

    # Combine: broadcast outside_domain (n_y, n_x) across z by adding a trailing axis
    mask_3d = mask_3d | outside_domain[:, :, np.newaxis]

    # Broadcast mask to full array shape (time, release_group, y, x, z)
    full_mask = np.broadcast_to(
        mask_3d[np.newaxis, np.newaxis, :, :, :],
        c.shape
    ) 

    # Create masked array
    c_masked = np.ma.masked_array(c, mask=full_mask, dtype=np.float32)

    return c_masked, water_depth_stats


def convert_geographic_to_stats_grid_indices(sampling_locations, longitude_grid, latitude_grid):
    """
    Convert geographic coordinates to grid indices.

    Args:
        sampling_locations (numpy.ndarray): Array of shape (n_locations, 2) containing 
            (lon,lat) coordinates of sampling locations
        longitude_grid (numpy.ndarray): 1D array of longitude values
        latitude_grid (numpy.ndarray): 1D array of latitude values

    Returns:
        numpy.ndarray: Array of shape (n_locations, 2) containing (x,y) grid indices
    """
    # Get grid boundaries
    x_min = longitude_grid.min()
    x_max = longitude_grid.max() 
    y_min = latitude_grid.min()
    y_max = latitude_grid.max()

    # Calculate grid cell sizes
    x_cell_size = (x_max - x_min) / (len(longitude_grid) - 1)
    y_cell_size = (y_max - y_min) / (len(latitude_grid) - 1)

    # Convert sampling locations to grid indices using nearest neighbor approach
    sampling_indices = []
    for loc in sampling_locations:
        # Skip points outside grid boundaries
        if (loc[0] < x_min or loc[0] > x_max or 
            loc[1] < y_min or loc[1] > y_max):
            print(f'Skipping sampling location outside grid boundaries: {loc}')
            continue

        # Calculate grid indices
        x_idx = int(round((loc[0] - x_min) / x_cell_size))
        y_idx = int(round((loc[1] - y_min) / y_cell_size))

        # Add to valid indices list
        sampling_indices.append((x_idx, y_idx))

    return np.array(sampling_indices)


def greedy_fast(detectable, k):
    """`greedy` with the same contract, O(k) vectorised instead of an O(k·n_y·n_x)
    Python triple loop.

    Originally written in 2026_07_22_paper_figures.ipynb because the coverage
    sweeps call the optimiser thousands of times. It is the implementation the
    paper figures actually use; `tests/test_stats.py` pins it against `greedy`.
    """
    n_rel, n_y, n_x = detectable.shape
    det = detectable.reshape(n_rel, n_y * n_x)
    covered = np.zeros(n_rel, dtype=bool)
    chosen: list[int] = []
    for _ in range(k):
        gain = (det & ~covered[:, None]).sum(axis=0)
        if chosen:
            gain[chosen] = -1
        best = int(np.argmax(gain))
        covered |= det[:, best]
        chosen.append(best)
    locations = [(int(i // n_x), int(i % n_x)) for i in chosen]
    return locations, {int(s) for s in np.where(covered)[0]}


def optimize_stations(detectable, k, method="greedy", *, random_seed=0):
    """The one switch point for which optimiser makes the headline selection.

    `detectable` is a bool array (n_releases, n_y, n_x). Every method returns the
    same `(locations, covered_set)` contract, locations as (iy, ix) grid indices.
    """
    if method in ("greedy", "greedy_fast"):
        return greedy_fast(detectable, k)
    if method == "greedy_reference":
        return greedy(detectable, k)
    if method == "simulated_annealing":
        return simulated_annealing(detectable, k, random_seed=random_seed)
    if method == "genetic":
        return genetic_algorithm(detectable, k, population_size=200,
                                 n_generations=400, random_seed=random_seed)
    raise ValueError(
        f"unknown optimizer {method!r}; expected one of greedy, greedy_reference, "
        f"simulated_annealing, genetic"
    )


def valid_cell_mask(x_grid, y_grid, triangles, x_stats, y_stats):
    """Boolean (n_y, n_x) mask of stats-grid cells inside the model domain.

    Excludes land and outside-mesh cells by triangle membership - the same
    test `mask_below_seafloor` applies, exposed on its own for callers that want
    only the horizontal footprint.
    """
    from edna_sampling.mesh import TriangleLocator

    X, Y = np.meshgrid(x_stats, y_stats, indexing="xy")
    triangle_indices, _ = TriangleLocator(x_grid, y_grid, triangles).locate(X, Y)
    return triangle_indices != -1
