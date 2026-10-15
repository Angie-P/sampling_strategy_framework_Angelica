"""One site's figures, and the derived numbers behind them.

Reads a condensed product (`edna condense`) and nothing else - no oceantracker, no
hindcast, no chunk directories - so this runs on a laptop.

One site at a time. A multi-site comparison like the paper's is written by calling
`site_results` twice and plotting both, which is the caller's business rather than
something this module models.

`summary.json` is the point of the whole exercise as much as the figures are: it is
the derived numbers - chosen stations, sources covered, footprint areas - whose
survival under the 2D switch is the open question in
docs/2d-stats-assessment.md section 7. Two runs are compared by diffing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edna_sampling.analysis import (coverage_curve, detectability_map, detectable,
                                    footprint_area_km2, representative_source,
                                    sources_detected_by)
from edna_sampling.detection import optimize_stations


def site_results(ds, *, filtered_volume, threshold_copies, n_stations,
                 optimizer="greedy", random_seed=0, k_max=10, time_index=None,
                 ):
    """Everything one site's figures need, computed once.

    `ds` is an open condensed dataset. `time_index` selects the snapshot the
    station choice is made on; the default is the time with the largest total
    detectable area, which is the most favourable moment to sample and so the one
    the headline station count refers to.
    """
    conc = ds["conc_vert_avg"].values
    cell_area = ds["cell_area"].values
    det = detectable(conc, filtered_volume, threshold_copies)

    footprint = footprint_area_km2(det, cell_area)
    if time_index is None:
        time_index = int(np.argmax(footprint.mean(axis=1)))

    stations, covered = optimize_stations(
        det[time_index], int(n_stations), method=optimizer, random_seed=random_seed)

    # How many sources each station adds over those before it. The optimisers
    # always return exactly k stations, even once no station can add anything -
    # the surplus then lands on whatever cell argmax happens to hit, typically
    # grid corner (0, 0). Reporting those as recommended sampling locations would
    # send someone to a cell where nothing is detectable, so the gains are
    # published alongside and `useful_stations` counts the ones that earn a visit.
    gains, running = [], set()
    for iy, ix in stations:
        adds = sources_detected_by(det[time_index], [(iy, ix)]) - running
        gains.append(len(adds))
        running |= adds
    n_useful = sum(1 for g in gains if g > 0)
    curve = coverage_curve(det[time_index], k_max,
                           method=optimizer, random_seed=random_seed)

    n_sources = det.shape[1]
    xs, ys = ds["x"].values, ds["y"].values
    return {
        "n_sources": n_sources,
        "n_times": det.shape[0],
        "time_index": time_index,
        "stations_grid": [(int(iy), int(ix)) for iy, ix in stations],
        "stations_lonlat": [(float(xs[ix]), float(ys[iy])) for iy, ix in stations],
        "station_gains": gains,
        "useful_stations": n_useful,
        "sources_covered": len(covered),
        "coverage_fraction": len(covered) / n_sources if n_sources else 0.0,
        "coverage_curve": curve.tolist(),
        "footprint_km2_mean": float(footprint[time_index].mean()),
        "footprint_km2_median": float(np.median(footprint[time_index])),
        "representative_source": representative_source(footprint),
        "_det": det,
        "_footprint": footprint,
        "_hot": detectability_map(det),
    }


def _public(results):
    return {k: v for k, v in results.items() if not k.startswith("_")}


def write_figures(ds, results, out_dir: Path, *, dpi=150, land_shapefile = None) -> list[Path]:
    """Render the single-site figures. Returns what was written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    t = results["time_index"]
    xs, ys = ds["x"].values, ds["y"].values

    # FIGURE 1. HOW MANY SOURCES EACH CELL CAN DETECT, AT THE CHOSEN TIME
    fig, ax = plt.subplots(figsize=(6, 5))
    hot = results["_hot"][t]

    # Maximum number of detectable sources
    max_sources = int(np.nanmax(hot))
    # Avoid an invalid colormap if all cells have zero detectability
    max_sources = max(1, max_sources)
    # Discrete colormap: one colour for each integer count
    cmap = plt.get_cmap("viridis", max_sources)
    # Boundaries halfway between consecutive integers
    bounds = np.arange(0.5, max_sources + 1.5, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # Hide cells where no source is detectable
    hot_plot = np.where(hot > 0, hot, np.nan)

    im = ax.pcolormesh(xs, ys, hot_plot, shading="nearest", cmap=cmap, norm=norm, zorder=1)

    # Colourbar: one tick every 5 sources
    ticks = np.arange(5, max_sources + 1, 5)
    # Show the maximum value if it is not a multiple of 5
    if max_sources < 5:
        ticks = np.arange(1, max_sources + 1)
    elif max_sources % 5 != 0:
        ticks = np.append(ticks, max_sources)
    fig.colorbar(im, ax=ax, ticks=ticks, label="sources detectable")

    # Overlay land polygons, if a shapefile was supplied. 
    # Note that the .shp file needs to come with the other files that have been created at the same time by QGIS, so put them together in the same folder
    if land_shapefile is not None:
        import geopandas as gpd

        land = gpd.read_file(land_shapefile)

        if land.crs is None:
            raise ValueError(
                "The land shapefile has no CRS defined. "
                "Please specify its coordinate reference system."
            )
        # Convert land coordinates to geographic lon/lat
        if land.crs.to_epsg() != 4326:
            land = land.to_crs(epsg=4326)

        land.plot(ax=ax, color="lightgrey", edgecolor="grey", linewidth=0.4, zorder=2)
        # Restore the extent of the statistical grid
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        pad_x = (xmax - xmin) * 0.03
        pad_y = (ymax - ymin) * 0.03
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # Only the stations that add coverage. The optimiser returns exactly k, and
    # any surplus beyond saturation sits on an arbitrary cell - drawing those
    # would put a marker where nothing is detectable.
    useful = [(lon, lat) for (lon, lat), gain
              in zip(results["stations_lonlat"], results["station_gains"]) if gain > 0]
    if useful:
        sx, sy = zip(*useful)
        n = len(useful)
        ax.plot(sx, sy, "r*", markersize=9, markeredgecolor="k",
                label=f"{n} station{'s' if n > 1 else ''}"
                      + (f" ({len(results['stations_lonlat']) - n} more add nothing)"
                         if n < len(results["stations_lonlat"]) else ""))
        ax.legend(loc="upper right")

    # Geographic aspect correction for longitude/latitude coordinates
    lat0 = np.nanmean(ys)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat0)), adjustable="box")

    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"{ds.attrs.get('run_name','')}  detectability at t={t}")
    p = out_dir / "detectability_map.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # FIGURE 2. FOOTPRINT AREA OVER TIME: SPREAD ACROSS SOURCES, PLUS THE MEDIAN SOURCE
    fig, ax = plt.subplots(figsize=(6, 4))
    fp = results["_footprint"]
    hours = np.arange(fp.shape[0])
    ax.fill_between(hours, np.percentile(fp, 5, axis=1), np.percentile(fp, 95, axis=1),
                    alpha=0.25, label="5-95th percentile of sources")
    ax.plot(hours, np.median(fp, axis=1), lw=2, label="median source")
    ax.plot(hours, fp[:, results["representative_source"]], "--", lw=1,
            label=f"representative source #{results['representative_source']}")
    ax.axvline(t, color="k", ls=":", lw=1, label=f"chosen time t={t}")
    ax.set(xlabel="output time step", ylabel="detection footprint [km$^2$]",
           title="footprint area per source")
    ax.legend(fontsize=8)
    p = out_dir / "footprint_timeseries.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # FIGURE 3. DIMINISHING RETURN FROM ADDING STATIONS
    fig, ax = plt.subplots(figsize=(6, 4))
    curve = np.asarray(results["coverage_curve"])
    k = np.arange(1, len(curve) + 1)
    ax.plot(k, curve / results["n_sources"] * 100, "o-")
    ax.axvline(len(results["stations_grid"]), color="r", ls=":",
               label=f"{len(results['stations_grid'])} stations")
    ax.set(xlabel="number of stations", ylabel="sources detected [%]",
           title="coverage vs station count", ylim=(0, 100))
    ax.legend(fontsize=8)
    p = out_dir / "coverage_curve.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    summary = _public(results)
    summary["source"] = {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in ds.attrs.items()}
    p = out_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2, default=str))
    written.append(p)
    return written
