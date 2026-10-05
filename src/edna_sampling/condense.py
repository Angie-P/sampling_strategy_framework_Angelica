"""Chunked model output -> one portable analysis product.

A production run leaves 400 chunk directories and ~6-150 GB behind, on the machine
that has the hindcast. The analysis wants a single file it can open with plain
`xarray` on a laptop. This is the stage in between, and it is the desktop/HPC
boundary: everything upstream needs the modelling stack, everything downstream
does not.

What it writes
--------------
`conc_vert_avg(time, release_group, y, x)` - the tow-window-averaged eDNA
concentration, which is the only sizeable quantity the analysis actually reads -
plus the grid geometry, effective volumes, release points and provenance.

The field is ~99% zeros (each source's plume touches ~1% of the grid), so it is
stored zlib-compressed: ~50 MB for 400 groups against 2.5 GB dense. Level 4;
level 9 measurably buys nothing.

2D and 3D condense to the *same* quantity
-----------------------------------------
A `gridded_2d` statistic already counts particles in the tow window. A
`gridded_3d` one is summed over the window's layers first. Both are then divided
by the same effective volume, so a v33 (3D) and v34 (2D) run of the same
experiment produce directly comparable products - which is the point of keeping
both. See docs/2d-stats-assessment.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import math

import numpy as np

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig

FORMAT_VERSION = 1
"""Bumped whenever the layout changes incompatibly. `open_condensed` refuses a
version it does not know, rather than failing obscurely three functions later."""

_ZLIB = {"zlib": True, "complevel": 4}


# --------------------------------------------------------------------------
# effective volume
# --------------------------------------------------------------------------

def effective_volume(x_stats, y_stats, cell_area, x_grid, y_grid, triangles,
                     water_depth, tow_depth, *, subsample=5, policy="weight"):
    """Water volume of the tow window in each stats cell, in m^3.

    The 3D path historically used `dz * n_wet` - a 1 m staircase on bathymetry
    sampled at the cell centre, which is exact where the water is deeper than the
    tow window and wrong by a median 3% (p95 36%) in the sixth of the domain that
    is shallower. That sixth is the near-shore band the analysis cares about.

    Here the bathymetry is sampled at `subsample` x `subsample` points inside each
    cell and the wet thickness averaged, so a sloping cell gets its true mean
    thickness and a part-land cell is down-weighted rather than kept whole:

        V_eff = A_cell * mean_over_subpoints( clip(water_depth, 0, tow_depth) )

    Each sub-point is located on the model's own triangles and the depth
    interpolated linearly there (`mesh.TriangleLocator`), so a sub-point is wet
    exactly where the model has water. That also keeps working on a mesh with
    overlapping triangles, which matplotlib's trifinder rejects outright.

    `policy="drop"` instead zeroes any cell with a dry sub-point, which is the
    conservative variant worth a sensitivity check; `"weight"` is the default
    because dropping discards real coastal detections and makes the footprint
    boundary depend on grid alignment.

    Tide is not included: that needs SCHISM `elev` and would make this a
    (time, y, x) field. See docs/2d-stats-assessment.md section 3.
    """
    from edna_sampling.mesh import TriangleLocator

    if policy not in ("weight", "drop"):
        raise ConfigError(f"partial-cell policy must be 'weight' or 'drop', got {policy!r}")
    if subsample < 1:
        raise ConfigError(f"subsample must be at least 1, got {subsample}")

    x_stats = np.asarray(x_stats, dtype=float)
    y_stats = np.asarray(y_stats, dtype=float)
    dx = float(np.diff(x_stats)[0]) if x_stats.size > 1 else 0.0
    dy = float(np.diff(y_stats)[0]) if y_stats.size > 1 else 0.0

    # sub-point offsets at cell centres of an s x s division
    frac = (np.arange(subsample) + 0.5) / subsample - 0.5
    ox, oy = np.meshgrid(frac * dx, frac * dy, indexing="xy")

    locator = TriangleLocator(x_grid, y_grid, triangles)

    X, Y = np.meshgrid(x_stats, y_stats, indexing="xy")
    thickness = np.zeros(X.shape, dtype=float)
    in_domain = np.zeros(X.shape, dtype=bool)

    for k in range(ox.size):
        tri, weights = locator.locate(X + ox.flat[k], Y + oy.flat[k])
        inside = tri != -1
        depth = locator.interpolate(water_depth, tri, weights)
        wet = np.where(np.isnan(depth), 0.0, np.clip(depth, 0.0, tow_depth))
        thickness += wet
        in_domain |= inside
    thickness /= ox.size

    if policy == "drop" and np.isfinite(tow_depth):
        thickness = np.where(np.isclose(thickness, tow_depth), thickness, 0.0)
    # With an infinite window every wet cell is "fully covered" by definition, so
    # `drop` has nothing to drop and coincides with `weight`.

    volume = np.asarray(cell_area, dtype=float) * thickness
    return volume, in_domain


# --------------------------------------------------------------------------
# reading one chunk
# --------------------------------------------------------------------------

@dataclass
class _Chunk:
    counts: np.ndarray          # (time, y, x) already reduced over the tow window
    times: np.ndarray
    group_name: str


def _tow_depth(spec) -> float:
    """How thick the tow window is, in metres, for either kind of statistic.

    `inf` means the tow spans the whole water column, which is what a `gridded_2d`
    statistic with no vertical selection counts. `effective_volume` clips the
    bathymetry at this value, so an infinite window simply gives the full column
    depth - no special case needed there.
    """
    if spec.kind == "gridded_2d":
        if spec.near_seasurface is not None:
            return float(spec.near_seasurface)
        if spec.near_seabed is not None:
            return float(spec.near_seabed)
        return math.inf          # no vertical selection: the whole column
    if spec.z_max is None or spec.z_min is None or not spec.layers:
        raise ConfigError(
            f"statistic {spec.name!r}: gridded_3d needs layers, z_min and z_max.")
    # the analysis window is layers 0..layers-2, i.e. one layer short of the full
    # range - see DEPTH_INDEX_RANGE in 2026_07_22_paper_figures.ipynb
    dz = (float(spec.z_max) - float(spec.z_min)) / int(spec.layers)
    return dz * (int(spec.layers) - 1)


def _read_chunk(path: Path, spec) -> _Chunk:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        counts = ds["count"].values
        times = ds["time"].values
        names = ds["release_group_names"].values if "release_group_names" in ds else None
    if counts.shape[1] != 1:
        raise ConfigError(
            f"{path.name}: expected one release group per chunk, found {counts.shape[1]}. "
            f"Condensing several groups per chunk is not implemented.")
    counts = counts[:, 0]
    if spec.kind == "gridded_3d":
        counts = counts[..., :int(spec.layers) - 1].sum(axis=-1)
    name = str(names[0]) if names is not None and len(names) else path.parent.name
    return _Chunk(counts=counts.astype(np.float64), times=times, group_name=name)


# --------------------------------------------------------------------------
# condensing a run
# --------------------------------------------------------------------------

def condensed_path(site: SiteConfig, profile: MachineProfile) -> Path:
    from edna_sampling.params import root_output_dir
    return root_output_dir(site, profile) / f"{site.run_name}_condensed.nc"


def condense(site: SiteConfig, profile: MachineProfile, *, out: Path | None = None,
             subsample: int = 5, policy: str = "weight", echo=print) -> Path:
    """Stitch a run's chunks into one self-describing netCDF file.

    Reads `grid000.nc` once rather than once per chunk: it is 27 MB copied
    identically into all 400 chunk directories, and re-reading it is most of the
    hour the notebook's loader spends.
    """
    import xarray as xr

    from edna_sampling.backends import COMPLETE, chunk_dir, run_status

    spec = site.statistic()
    status = run_status(site, profile)
    done = sorted(status[COMPLETE])
    if not done:
        raise ConfigError(
            f"{site.run_name}: no complete chunks to condense. Run the model first, "
            f"and check `edna status`.")
    if len(done) < site.n_chunks:
        echo(f"  warning: condensing {len(done)} of {site.n_chunks} chunks; "
             f"missing {site.n_chunks - len(done)}")

    first = chunk_dir(site, profile, done[0])
    grid_file = first / "grid000.nc"
    if not grid_file.is_file():
        raise ConfigError(f"no grid file at {grid_file}")
    with xr.open_dataset(grid_file) as g:
        x_grid = g["x"].values[:, 0]
        y_grid = g["x"].values[:, 1]
        triangles = g["triangles"].values
        water_depth = g["water_depth"].values

    stats_files = sorted(first.glob(f"stats_*{spec.name}.nc"))
    if not stats_files:
        raise ConfigError(
            f"{first.name} has no statistics file for {spec.name!r}. "
            f"Found: {', '.join(p.name for p in first.glob('stats_*.nc')) or '(none)'}")
    with xr.open_dataset(stats_files[0]) as ds:
        x_stats = ds["x"].values[0]
        y_stats = ds["y"].values[0]
        cell_area = ds["cell_area"].values[0]

    tow = _tow_depth(spec)
    tow_label = "whole column" if not math.isfinite(tow) else f"{tow:g} m"
    echo(f"  tow window {tow_label}, effective volume at {subsample}x{subsample} "
         f"sub-points, policy {policy}")
    volume, in_domain = effective_volume(
        x_stats, y_stats, cell_area, x_grid, y_grid, triangles, water_depth, tow,
        subsample=subsample, policy=policy)

    # Mean bathymetric depth of each stats cell, carried so that a depth filter can
    # be applied - and re-applied differently - in post-processing. Filtering in
    # the model (`water_depth_min` on the statistic) bakes the choice into 400
    # chunks of output and cannot be undone without re-running.
    from edna_sampling.mesh import TriangleLocator
    locator = TriangleLocator(x_grid, y_grid, triangles)
    _X, _Y = np.meshgrid(np.asarray(x_stats, float), np.asarray(y_stats, float), indexing="xy")
    cell_depth = locator.interpolate(water_depth, *locator.locate(_X, _Y))

    conc = None
    group_names, times = [], None
    for i, n in enumerate(done):
        d = chunk_dir(site, profile, n)
        matches = sorted(d.glob(f"stats_*{spec.name}.nc"))
        if not matches:
            raise ConfigError(f"chunk {n}: no statistics file for {spec.name!r}")
        chunk = _read_chunk(matches[0], spec)
        if conc is None:
            times = chunk.times
            conc = np.zeros((len(chunk.times), len(done)) + chunk.counts.shape[1:],
                            dtype=np.float32)
        elif chunk.counts.shape[0] != conc.shape[0]:
            raise ConfigError(
                f"chunk {n} has {chunk.counts.shape[0]} output times but chunk "
                f"{done[0]} has {conc.shape[0]}; these are not one run.")
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(volume > 0, chunk.counts / volume, 0.0)
        conc[:, i] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        group_names.append(chunk.group_name)
        if echo and (i + 1) % 50 == 0:
            echo(f"  {i + 1}/{len(done)} chunks")

    ds = xr.Dataset(
        data_vars={
            "conc_vert_avg": (("time", "release_group", "y", "x"), conc,
                              {"units": "copies m-3",
                               "long_name": "tow-window-averaged eDNA concentration"}),
            "effective_volume": (("y", "x"), volume.astype(np.float32),
                                 {"units": "m3", "long_name":
                                  "water volume of the tow window per stats cell"}),
            "cell_area": (("y", "x"), np.asarray(cell_area, dtype=np.float32),
                          {"units": "m2"}),
            "in_domain": (("y", "x"), in_domain,
                          {"long_name": "stats cell intersects the model mesh"}),
            "cell_depth": (("y", "x"), cell_depth.astype(np.float32),
                           {"units": "m", "long_name":
                            "still-water bathymetric depth at the stats cell centre, "
                            "nan off the mesh"}),
        },
        coords={"time": times, "release_group": np.array(group_names, dtype=object),
                "x": x_stats, "y": y_stats},
        attrs={
            "format": "edna-condensed",
            "format_version": FORMAT_VERSION,
            "run_name": site.run_name,
            "config_fingerprint": site.fingerprint(),
            "statistic": spec.name,
            "statistic_kind": spec.kind,
            "tow_depth_m": tow,
            "partial_cell_policy": policy,
            "volume_subsample": subsample,
            "copies_per_particle": site.source.copies_per_particle,
            "n_chunks_condensed": len(done),
            "n_chunks_total": site.n_chunks,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )

    target = Path(out) if out is not None else condensed_path(site, profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(target, encoding={"conc_vert_avg": _ZLIB, "effective_volume": _ZLIB})
    return target


def open_condensed(path):
    """Open a condensed product, refusing a format version this code cannot read."""
    import xarray as xr

    ds = xr.open_dataset(path)
    if ds.attrs.get("format") != "edna-condensed":
        ds.close()
        raise ConfigError(f"{path}: not an edna condensed product")
    version = int(ds.attrs.get("format_version", -1))
    if version != FORMAT_VERSION:
        ds.close()
        raise ConfigError(
            f"{path}: condensed format v{version}, but this code reads v{FORMAT_VERSION}. "
            f"Re-run `edna condense`.")
    return ds
