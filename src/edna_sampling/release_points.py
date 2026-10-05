"""Release-point generation: where the eDNA sources are placed in a site.

`generate_release_locations_using_lloyd_relax` builds the admissible region as

    depth-band polygon  n  model-area polygon  [n  always-flooded polygon]

and then spreads `n_points` over it by Lloyd relaxation, so sources are evenly
spaced rather than clustered. "Evenly" means uniform per unit seabed area: the
relaxation weights Voronoi cells by area, never by depth or water volume. The
depth distribution of the result is therefore the region's own hypsometry.

The mesh arrives as a `Mesh` (see `edna_sampling.mesh`), not as a path: the
caller has already decided where it comes from, and this module should not care
whether it was derived from the hindcast or named by a profile.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def _depth_band_polygon(triang, node_depth, depth_range, bbox_poly):
    # Smooth depth band via tricontourf — the filled region between the
    # d_min and d_max isobaths, intersected with the user's bbox.
    # tricontourf returns outer rings (CCW, signed area > 0) and holes
    # (CW, signed area < 0) mixed together; treat them separately so islands
    # inside the band remain as holes rather than being filled in.
    d_min, d_max = depth_range
    fig, ax = plt.subplots()
    try:
        cs = ax.tricontourf(triang, node_depth, levels=[d_min, d_max])
        outer_polys, hole_polys = [], []
        for path in cs.get_paths():
            for ring in path.to_polygons():
                if len(ring) < 4:
                    continue
                area = 0.5 * np.sum(
                    ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]
                )
                p = Polygon(ring)
                if not p.is_valid or p.is_empty:
                    continue
                (outer_polys if area > 0 else hole_polys).append(p)
    finally:
        plt.close(fig)
    band = unary_union(outer_polys)
    if hole_polys:
        band = band.difference(unary_union(hole_polys))
    return band.intersection(bbox_poly)


#: variables `_flooded_fraction_polygon` needs, and which only SCHISM has
_SCHISM_VARS = ('SCHISM_hgrid_node_x', 'SCHISM_hgrid_node_y',
                'SCHISM_hgrid_face_nodes', 'wetdry_node')

_SHYFEM_VARS = ("water_level", "total_depth")

def detect_flooded_fraction_reader(paths, reader_type):
    if reader_type is not None:
        return reader_type

    with xr.open_dataset(paths[0]) as ds:
        if all(v in ds for v in _SHYFEM_VARS):
            return "shyfem"

    return "schism"

def _require_schism(ds, path):
    """Fail with the reason, not with a KeyError three frames down.

    Everything else in this package is hindcast-format agnostic, because
    OceanTracker's reader does the parsing. This one filter is not: wet/dry
    state has no equivalent OceanTracker exposes, so it reads SCHISM's own
    `wetdry_node` flag directly. A ROMS or FVCOM hindcast reaches here only if
    the config set `min_flooded_fraction`, so name that key in the message.
    """
    missing = [v for v in _SCHISM_VARS if v not in ds]
    if missing:
        raise ValueError(
            f"release_points.min_flooded_fraction needs SCHISM wet/dry output, and "
            f"{path} has no {', '.join(missing)}.\n"
            f"  This is the one filter that is not hindcast-format agnostic: no other "
            f"format exposes an equivalent flag through OceanTracker.\n"
            f"  Remove `min_flooded_fraction` from the config to use the depth band "
            f"and model-area polygon alone."
        )

def _require_shyfem(ds, path):
    missing = [v for v in _SHYFEM_VARS if v not in ds]
    if missing:
        raise ValueError(
            f"release_points.min_flooded_fraction needs SHYFEM wet/dry output, "
            f"and {path} has no {', '.join(missing)}."
        )

def _flooded_fraction_polygon_schism(schism_output_paths, min_flooded_fraction, bbox_poly):
    """
    Build a polygon covering only the parts of the domain that are wet at
    least `min_flooded_fraction` of the time, based on SCHISM's per-node
    `wetdry_node` flag (0 = wet, 1 = dry) across the given output files.

    Unlike `_depth_band_polygon`, this does NOT use tricontourf: a large
    fraction of nodes typically sit at exactly frac_wet == 1.0 (a flat
    plateau right at the threshold we filter on), and tricontourf's
    marching-triangles level-crossing algorithm silently drops flat regions
    pinned at the boundary level -- it only fills the thin transitional
    fringe, undercounting the true always-flooded area by ~100x. Instead,
    build the region directly as the union of triangles whose vertices are
    ALL above the threshold, which is exact for this kind of near-binary
    field and has no such degeneracy.
    """
    with xr.open_dataset(schism_output_paths[0]) as ds0:
        _require_schism(ds0, schism_output_paths[0])
        node_x = ds0['SCHISM_hgrid_node_x'].values
        node_y = ds0['SCHISM_hgrid_node_y'].values
        face_nodes = ds0['SCHISM_hgrid_face_nodes'].values[:, :3].astype(int) - 1

    wetdry_sum, n_time_total = None, 0
    for p in schism_output_paths:
        with xr.open_dataset(p) as ds:
            wd = ds['wetdry_node'].values          # (time, n_node), 1 = dry, 0 = wet
        wetdry_sum = wd.sum(axis=0) if wetdry_sum is None else wetdry_sum + wd.sum(axis=0)
        n_time_total += wd.shape[0]
    frac_wet_node = 1 - wetdry_sum / n_time_total

    node_ok = frac_wet_node >= min_flooded_fraction
    tri_ok = node_ok[face_nodes].all(axis=1)
    flooded_polys = [
        Polygon(np.column_stack([node_x[t], node_y[t]])) for t in face_nodes[tri_ok]
    ]
    region = unary_union(flooded_polys)
    return region.intersection(bbox_poly)

def _flooded_fraction_polygon_shyfem(
    mesh,
    shyfem_output_paths,
    min_flooded_fraction,
    bbox_poly,
    minimum_total_water_depth=0.25,
):
    """Build a polygon from SHYFEM cells that are wet for enough of the hindcast.

    A node is considered wet when:

        water_level + water_depth >= minimum_total_water_depth

    A triangle is considered wet at a given time only if all three of
    its nodes are wet. The flooded fraction of each triangle is then
    the fraction of hindcast time steps for which the triangle is wet.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing node coordinates and triangle connectivity.
    shyfem_output_paths : sequence of str
        Paths to SHYFEM hindcast NetCDF files.
    min_flooded_fraction : float
        Minimum fraction of the hindcast during which a triangle must be
        wet to be included.
    bbox_poly : shapely geometry
        Bounding polygon used to clip the resulting flooded region.
    minimum_total_water_depth : float, default=0.25
        Minimum total water depth, in metres, used to distinguish wet
        from dry nodes, following OceanTracker's SHYFEM handling.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Polygon representing the part of the mesh satisfying the
        flooded-fraction criterion.
    """
    import xarray as xr
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # Check that the first file contains the variables required for the
    # SHYFEM flooded-fraction calculation.

    with xr.open_dataset(shyfem_output_paths[0]) as ds:
        _require_shyfem(ds, shyfem_output_paths[0])

    # Accumulate wet/dry information over all hindcast files.

    wet_count = np.zeros(mesh.triangles.shape[0], dtype=np.int64)
    n_time_total = 0

    for path in shyfem_output_paths:

        with xr.open_dataset(path) as ds:
            _require_shyfem(ds, path)

            water_level = np.asarray(ds["water_level"].values)
            water_depth = np.asarray(ds["total_depth"].values)

            # OceanTracker's SHYFEM reader treats NaN water levels as
            # shallow water above the bed by 0.05 m.
            bed = -water_depth

            water_level = water_level.copy()
            is_nan = np.isnan(water_level)

            bed_2d = np.broadcast_to(bed, water_level.shape)
            water_level[is_nan] = bed_2d[is_nan] + 0.05

            # Node wet/dry status
            # wet if:
            #     water_level + water_depth >= minimum_total_water_depth

            node_wet = (
                water_level
                + water_depth[np.newaxis, :]
                >= minimum_total_water_depth
            )

            # Triangle wet/dry status
            # A triangle is wet only if ALL three nodes are wet.
            #
            # node_wet:
            #     (time, node)
            # mesh.triangles:
            #     (triangle, 3)
            # result:
            #     (time, triangle)

            triangle_wet = node_wet[:, mesh.triangles].all(axis=2)

            # Count the number of wet time steps for each triangle.
            wet_count += triangle_wet.sum(axis=0)

            n_time_total += triangle_wet.shape[0]

    if n_time_total == 0:
        raise ValueError(
            "No time steps found in SHYFEM output files."
        )

    # Fraction of time during which each triangle is wet.
    fraction_wet = wet_count / n_time_total

    # Keep triangles satisfying the requested minimum flooded fraction.
    triangle_ok = fraction_wet >= min_flooded_fraction

    # Convert accepted triangles into polygons.
    polygons = []

    for triangle in mesh.triangles[triangle_ok]:
        coords = [
            (
                mesh.node_x[node],
                mesh.node_y[node],
            )
            for node in triangle
        ]
        polygons.append(Polygon(coords))

    if not polygons:
        return bbox_poly.intersection(
            Polygon()
        )

    flooded_region = unary_union(polygons)

    # Limit the flooded region to the bounding polygon used by the
    # release-point generation.
    return flooded_region.intersection(bbox_poly)


def _random_points_in_polygon(region, n_points, rng):
    minx, miny, maxx, maxy = region.bounds
    accepted = []
    while len(accepted) < n_points:
        need = n_points - len(accepted)
        batch = max(4 * need, 128)
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        for x, y in zip(xs, ys):
            if region.contains(Point(x, y)):
                accepted.append((x, y))
                if len(accepted) == n_points:
                    break
    return np.asarray(accepted)


def _far_field_frame(region, factor=10.0):
    """Four sentinel generators far outside `region`.

    Their only job is to bound the Voronoi diagram so every real point gets a
    finite cell. They must sit well outside the region: a sentinel *inside* or
    *on* the boundary is a Voronoi generator like any other and claims the
    territory around itself, which is exactly the bias this avoids (see
    `_clipped_voronoi_cells`).
    """
    minx, miny, maxx, maxy = region.bounds
    w = (maxx - minx) * factor or 1.0
    h = (maxy - miny) * factor or 1.0
    return np.array([
        [minx - w, miny - h], [maxx + w, miny - h],
        [maxx + w, maxy + h], [minx - w, maxy + h],
    ])


def _clipped_voronoi_cells(points, region, frame_pts):
    """Voronoi cells of `points`, clipped to `region`.

    `frame_pts` are far-field sentinels (see `_far_field_frame`) that make every
    real cell finite without competing for area, so each clipped cell is the
    true region cell of its point and Lloyd relaxation converges to a genuine
    centroidal Voronoi tessellation - i.e. uniform point density per unit area.

    This used to seed the Voronoi input with the region's own boundary vertices
    instead. That also bounds the cells, but a boundary vertex is a generator
    too: a point at distance h from a densely sampled boundary then owns only
    the locations closer to it than to that boundary - a parabolic sliver
    opening inward, whose centroid lies inward of the point. Every iteration
    therefore walked points away from the boundary. The effect scaled with how
    finely the boundary was sampled, and the coastline (a union of mesh
    triangles) is far more convoluted than the offshore isobath, so sources
    drained out of shallow water: in the Cape Rodney example the 0-10 m band
    held 20% of the eligible area but ended up with 5% of the sources.
    """
    all_points = np.vstack([points, frame_pts])
    vor = Voronoi(all_points)
    cells = []
    for i in range(len(points)):
        region_index = vor.point_region[i]
        vertex_indices = vor.regions[region_index]
        if len(vertex_indices) == 0 or -1 in vertex_indices:
            cells.append(None)
            continue
        cell_poly = Polygon(vor.vertices[vertex_indices])
        if not cell_poly.is_valid:
            cell_poly = cell_poly.buffer(0)
        clipped = cell_poly.intersection(region)
        cells.append(clipped if not clipped.is_empty and clipped.area > 0 else None)
    return cells


def _lloyd_relax(points, region, cv_target=0.05, max_iter=100, plateau_tol=0.01,
                 plateau_patience=15):
    """Spread `points` evenly over `region` by Lloyd relaxation.

    Stops on whichever comes first:

      * the cell-area CV drops below `cv_target`, or
      * the CV plateaus - the best CV so far has not improved by more than
        `plateau_tol` (relative) over the last `plateau_patience` iterations, or
      * `max_iter` is reached.

    The plateau test matters because `cv_target` is not reachable for a real
    site. Equal-area cells need a region a tessellation can actually partition
    evenly; a coastal depth band is ragged, pierced by islands and often split
    into pieces, so the CV bottoms out well above 0.05 (~0.14 at Cape Rodney)
    no matter how long it runs. Without this test the loop silently spent every
    one of `max_iter` iterations and the `cv_target` check never once fired.

    The test tracks the best CV rather than the last one: past the first dozen
    iterations the CV wobbles by a few tenths of a percent from step to step, so
    a run of small *consecutive* gains is noise, not convergence. What the
    release points actually need is settled long before either bound - density
    per unit area is uniform to within sampling noise after ~10 iterations, and
    the long tail only polishes cell areas.

    Returns (points, info); `info['converged']` says whether `cv_target` was
    actually met, and `info['stop']` why the loop ended.
    """
    frame_pts = _far_field_frame(region)
    cv_history = []
    n_iter = 0
    best_cv = np.inf
    best_iter = 0
    stop = 'max_iter'
    for n_iter in range(1, max_iter + 1):
        cells = _clipped_voronoi_cells(points, region, frame_pts)
        new_points = points.copy()
        areas = []
        for i, cell in enumerate(cells):
            if cell is None:
                continue
            c = cell.centroid
            # With holes (islands) or split cells, the mass-weighted centroid
            # can land in a hole or between pieces; fall back to a point
            # guaranteed to be inside the geometry.
            if not cell.covers(c):
                c = cell.representative_point()
            new_points[i] = [c.x, c.y]
            areas.append(cell.area)
        points = new_points
        areas = np.asarray(areas)
        cv = areas.std() / areas.mean() if areas.size and areas.mean() > 0 else np.inf
        cv_history.append(cv)
        if cv < cv_target:
            stop = 'cv_target'
            break
        if np.isfinite(cv) and cv < best_cv * (1.0 - plateau_tol):
            best_cv, best_iter = cv, n_iter
        elif n_iter - best_iter >= plateau_patience:
            stop = 'plateau'
            break
    return points, {'n_iter': n_iter, 'cv_history': cv_history,
                    'final_cv': cv_history[-1], 'stop': stop,
                    'converged': stop == 'cv_target'}


def generate_release_locations_using_lloyd_relax(
    mesh,
    path_to_polygon,
    depth_range=(0.0, 50.0),
    n_points=100,
    cv_target=0.05,
    max_iter=100,
    seed=None,
    hindcast_output_paths=None,
    hindcast_type=None,
    min_flooded_fraction=None,
    echo=None,
):
    """
    hindcast_output_paths (list[str] or None): paths to `*.nc`
        files to compute flooded-fraction from. Required if `min_flooded_fraction` is set.
    hindcast_type (str): name of the hydrodinamic output type given in input.
        It is not mandatory: it is a shortcut to decide which function to use in order to
        use min_flooded_fraction when specified, and it is used only in relation with this parameter
    min_flooded_fraction (float or None): minimum fraction of the SCHISM
        record (0-1) a location must be wet (`wetdry_node == 0`) to be
        eligible for a release point, in addition to `depth_range`. E.g.
        1.0 restricts releases to locations that are never dry across the
        given files — keeps release points off intertidal flats that would
        otherwise sit right at the edge of `depth_range` but spend part of
        the time out of the water. None (default) disables this filter,
        matching the original behaviour.
    echo (callable or None): where to report how the relaxation settled. None
        (default) is silent.

    Points come out spread uniformly per unit *seabed area* over the admissible
    region: the relaxation weights cells by area alone, with no depth or volume
    term. A design that samples by water volume, or that stratifies by depth,
    would need a weighted centroid here.
    """
    triang = mesh.triangulation()

    bbox = np.loadtxt(path_to_polygon, delimiter=',')
    bbox_poly = Polygon(bbox)

    node_minx, node_miny, node_maxx, node_maxy = mesh.bounds()
    bx0, by0, bx1, by1 = bbox_poly.bounds
    if bx1 < node_minx or bx0 > node_maxx or by1 < node_miny or by0 > node_maxy:
        raise ValueError(
            f"Bounding box {bbox_poly.bounds} does not overlap grid extent "
            f"({node_minx}, {node_miny}, {node_maxx}, {node_maxy}). "
            "Check that both are in the same coordinate reference system."
        )

    region = _depth_band_polygon(triang, mesh.water_depth, depth_range, bbox_poly)
    if region.is_empty:
        raise ValueError(
            f"No area satisfies depth_range={depth_range} inside bounding box."
        )

    if min_flooded_fraction is not None:
        if not hindcast_output_paths:
            raise ValueError(
                "hindcast_output_paths is required when min_flooded_fraction is set."
            )

        if hindcast_type == "shyfem":
            flooded_region = _flooded_fraction_polygon_shyfem(
                mesh,
                hindcast_output_paths,
                min_flooded_fraction,
                bbox_poly,
            )
        elif hindcast_type == "schism":
            flooded_region = _flooded_fraction_polygon_schism(
                hindcast_output_paths,
                min_flooded_fraction,
                bbox_poly,
            )
        else:
            raise ValueError(
                f"Unknown flooded-fraction reader type: {hindcast_type!r}"
            )

        region = region.intersection(flooded_region)

        if region.is_empty:
            raise ValueError(
                f"No area satisfies depth_range={depth_range} AND "
                f"min_flooded_fraction={min_flooded_fraction} inside bounding box."
            )

    rng = np.random.default_rng(seed)
    points = _random_points_in_polygon(region, n_points, rng)
    points, info = _lloyd_relax(points, region, cv_target=cv_target, max_iter=max_iter)
    if echo is not None:
        why = {'cv_target': f"reached cv_target {cv_target:g}",
               'plateau': "cell-area spread stopped improving",
               'max_iter': f"hit max_iter {max_iter}"}[info['stop']]
        echo(f"  Lloyd relaxation: {info['n_iter']} iterations, "
             f"final cell-area cv {info['final_cv']:.3f} ({why})")
    return points


def plot_release_points(
    mesh,
    points,
    depth_range,
    bbox_poly_coords,
    stats_grid_center=None,
    stats_grid_span=None,
    save_path=None,
):
    triang = mesh.triangulation()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.tricontourf(triang, mesh.water_depth, levels=[depth_range[0], depth_range[1]], alpha=0.3)
    ax.triplot(triang, color='grey', linewidth=0.2, alpha=0.5)

    closed = np.vstack([bbox_poly_coords, bbox_poly_coords[0]])
    ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=1.5, label='Sampling region')

    if stats_grid_center is not None and stats_grid_span is not None:
        cx, cy = stats_grid_center
        hx, hy = stats_grid_span[0] / 2, stats_grid_span[1] / 2
        stats_box = np.array([
            [cx - hx, cy - hy], [cx + hx, cy - hy],
            [cx + hx, cy + hy], [cx - hx, cy + hy],
            [cx - hx, cy - hy],
        ])
        ax.plot(stats_box[:, 0], stats_box[:, 1], 'b--', linewidth=1.5, label='Stats grid')

    ax.scatter(points[:, 0], points[:, 1], s=20, c='red', zorder=5)
    for i, (px, py) in enumerate(points):
        ax.annotate(str(i), (px, py), textcoords='offset points', xytext=(3, 3), fontsize=6)

    all_coords = [bbox_poly_coords]
    if stats_grid_center is not None and stats_grid_span is not None:
        all_coords.append(stats_box)
    all_coords = np.vstack(all_coords)
    ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
