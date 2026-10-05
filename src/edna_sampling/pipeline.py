"""Pipeline stages: the glue between a config and the low-level modules.

Each stage is a plain function taking a `SiteConfig` and a `MachineProfile`, so
the CLI stays thin and the stages remain callable from a notebook or a test.
Stages are idempotent: running one twice does no extra work.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig
from edna_sampling.mesh import load_mesh
from edna_sampling.release_points import detect_flooded_fraction_reader

__all__ = [
    "release_points_path",
    "release_points_figure_path",
    "hindcast_files",
    "ensure_release_points",
    "run_chunk",
    "run_all",
]


def release_points_path(site: SiteConfig) -> Path:
    """Where this run's release points are cached.

    Keyed by `run_name`, so bumping `version` forks the cache rather than
    silently reusing an earlier version's source locations. Lives beside the
    site config, because the points are part of the experiment's definition
    rather than of its output.
    """
    if site.source_path is None:
        raise ConfigError("release_points_path needs a config loaded from a file")
    return site.source_path.parent / f"{site.run_name}_lloyd_points.csv"


def release_points_figure_path(site: SiteConfig, profile: MachineProfile) -> Path:
    """Diagnostic figure showing the generated source distribution."""
    return Path(profile.output_root) / f"{site.run_name}_release_points.png"


def _time_index(path: str) -> list[tuple[int, int, str]]:
    """Sort key putting hindcast files in time order.

    Natural ordering over the *whole path*, so runs of digits compare as numbers:
    `schout_2.nc` before `schout_10.nc`, and `2018/2/` before `2018/10/`.

    The directory part matters. SCHISM output is commonly split into per-month
    directories whose file numbering restarts at 1, so a hindcast spanning
    several months has a `schout_1.nc` in each. Sorting on the filename alone
    would interleave January, February and March - silently, and the
    flooded-fraction mask is computed over exactly this ordering.

    Each element is (is_text, number, text) so the key is totally ordered even
    when paths differ in shape.
    """
    parts = re.split(r"(\d+)", str(path))
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts]


def hindcast_files(site: SiteConfig, profile: MachineProfile) -> list[str]:
    """Hindcast files for this run, in time order."""
    entry = profile.hindcast(site.hindcast)
    if not entry.input_dir.is_dir():
        raise ConfigError(
            f"hindcast directory does not exist: {entry.input_dir}\n"
            f"  (profile {profile.name!r}, hindcast {site.hindcast!r})"
        )
    files = sorted(glob.glob(str(entry.input_dir / entry.file_mask)), key=_time_index)
    if not files:
        raise ConfigError(
            f"no files matching {entry.file_mask!r} in {entry.input_dir} "
            f"(profile {profile.name!r})"
        )
    return files


def ensure_release_points(
    site: SiteConfig,
    profile: MachineProfile,
    *,
    force: bool = False,
    make_figure: bool = True,
) -> tuple[np.ndarray, bool]:
    """Load this run's release points, generating and caching them if needed.

    Returns (points, generated), where `generated` says whether the relaxation
    actually ran. Every chunk of a run calls this and must get the same points,
    which is why the result is cached rather than recomputed per job.
    """
    site.validate_inputs(profile)
    cache = release_points_path(site)

    if cache.exists() and not force:
        points = np.loadtxt(cache, delimiter=",")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ConfigError(f"{cache}: expected an (n, 2) array of lon/lat, got {points.shape}")
        if len(points) != site.release_points.n:
            raise ConfigError(
                f"{cache} holds {len(points)} points but {site.run_name} declares "
                f"release_points.n = {site.release_points.n}.\n"
                f"  Either restore the matching cache, bump `version`, or regenerate with --force."
            )
        return points, False

    from edna_sampling.mesh import ensure_mesh

    rp = site.release_points
    hindcast_paths = None
    hindcast_type = None

    if rp.min_flooded_fraction is not None:
        # the flooded-fraction mask must be computed over the same record the
        # model itself reads, or "always wet" means something different to the
        # two of them
        hindcast_paths = hindcast_files(site, profile)
        entry = profile.hindcast(site.hindcast)
        hindcast_type = detect_flooded_fraction_reader(
            hindcast_paths,
            entry.reader_type,
        )

    mesh = load_mesh(ensure_mesh(profile, site.hindcast))
    points = _generate(site, mesh, hindcast_paths, hindcast_type)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(cache, points, delimiter=",")

    if make_figure:
        _write_figure(site, profile, points, mesh)
    return points, True


def _generate(site: SiteConfig, mesh, hindcast_paths: Sequence[str] | None, hindcast_type: str | None) -> np.ndarray:
    from edna_sampling.release_points import generate_release_locations_using_lloyd_relax

    rp = site.release_points
    points = generate_release_locations_using_lloyd_relax(
        mesh=mesh,
        path_to_polygon=str(site.domain),
        depth_range=rp.depth_range,
        n_points=rp.n,
        seed=rp.seed,
        hindcast_output_paths=list(hindcast_paths) if hindcast_paths else None,
        min_flooded_fraction=rp.min_flooded_fraction,
        hindcast_type=hindcast_type,
        echo=print,
    )
    points = np.asarray(points, dtype=float)
    if len(points) != rp.n:
        raise ConfigError(
            f"{site.run_name}: relaxation returned {len(points)} points, expected {rp.n}"
        )
    return points


def _write_figure(site: SiteConfig, profile: MachineProfile, points: np.ndarray,
                  mesh) -> Path:
    import matplotlib
    matplotlib.use("Agg")           # stages must work over ssh with no display
    import matplotlib.pyplot as plt

    from edna_sampling.release_points import plot_release_points

    # centre the diagnostic on the largest stats grid, so the figure shows the
    # sources in the context of the region they are actually counted over
    widest = max(site.stats, key=lambda s: s.span[0] * s.span[1])
    out = release_points_figure_path(site, profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_release_points(
        mesh=mesh,
        points=points,
        depth_range=site.release_points.depth_range,
        bbox_poly_coords=np.loadtxt(site.domain, delimiter=","),
        stats_grid_center=list(widest.grid_center),
        stats_grid_span=list(widest.span),
        save_path=str(out),
    )
    plt.close(fig)
    return out


def run_chunk(
    site: SiteConfig,
    profile: MachineProfile,
    chunk_number: int,
    *,
    ignore_manifest: bool = False,
    dry_run: bool = False,
    echo=print,
) -> Path:
    """Run one chunk of a site's release groups. Returns its output directory.

    This is what `edna run --chunk N` does, and the whole integration point for
    running on a cluster: one chunk is one independent process, so a job script
    calls this (or the CLI) once per array index.

    The two guards are the reason to call this rather than assembling
    `build_params` output yourself:

    * the release points must already be cached. Generating them here would have
      every chunk of a run racing to write the same file, and chunks could then
      disagree about where the sources are.
    * the config must not have changed since the run started, which
      `check_manifest` enforces. Pass ``ignore_manifest=True`` only when you mean
      to continue a run under an edited config.

    `dry_run` creates the output directory and returns without calling
    OceanTracker, which is enough to check paths and parameters.
    """
    from edna_sampling.params import build_params, chunk_run_name

    site.validate_inputs(profile)

    cache = release_points_path(site)
    if not cache.exists():
        config_ref = site.source_path if site.source_path is not None else "<config>"
        raise ConfigError(
            f"release points not generated yet: {cache}\n"
            f"  Run once, before submitting any chunk:  edna points {config_ref} "
            f"--profile {profile.name}"
        )
    if not ignore_manifest:
        from edna_sampling.manifest import check_manifest
        check_manifest(site, profile, cache)

    points = np.loadtxt(cache, delimiter=",")
    params = build_params(site, profile, chunk_number, points)

    out = Path(params["root_output_dir"]) / chunk_run_name(site, chunk_number)
    out.mkdir(parents=True, exist_ok=True)
    params["root_output_dir"] = str(Path(params["root_output_dir"]))

    if echo:
        echo(f"{site.run_name}: chunk {chunk_number}/{site.n_chunks}, "
             f"{len(params['release_groups'])} release group(s) -> {out}")
    if dry_run:
        if echo:
            echo("dry run; not calling oceantracker")
        return out

    # imported here, not at module scope: importing oceantracker takes seconds
    # and prints a banner, and nothing else in this module needs it
    from oceantracker import main as ot_main
    ot_main.run(params)
    return out


def run_all(
    site: SiteConfig,
    profile: MachineProfile,
    *,
    chunks: Sequence[int] | None = None,
    resume: bool = False,
    n_parallel: int | None = None,
    verbose: bool = False,
    ignore_manifest: bool = False,
    subsample: int = 5,
    policy: str = "weight",
    condensed_out: Path | None = None,
    figures_out: Path | None = None,
    time_index: int | None = None,
    echo=print,
) -> dict:
    """Everything downstream of the release points: submit, condense, figures.

    The three stages are separate commands because that is where the desktop/HPC
    round trip happens - but on one machine, running a site end to end is a
    single act, and the intermediate arguments are almost always the defaults.

    Release points are a precondition, not a step: they want looking at before
    committing to a few hundred model runs, and generating them here would let
    a typo in the config silently move every source. Run `edna points` first.

    Stops at the first stage that fails rather than carrying on. A condensed
    product built from a run with failed chunks is not obviously wrong when you
    open it - it just quietly holds fewer sources than the config declares - so
    the failure is worth more than the artifact. Re-run with `resume=True` once
    the cause is fixed.

    Returns {'chunks', 'condensed', 'figures', 'results'}; `figures` and
    `results` are what `write_figures` and `site_results` produced.
    """
    from edna_sampling.backends import incomplete_chunks, submit_local
    from edna_sampling.condense import condense, open_condensed
    from edna_sampling.figures import site_results, write_figures
    from edna_sampling.manifest import check_manifest, write_manifest

    site.validate_inputs(profile)
    cache = release_points_path(site)
    if not cache.exists():
        raise ConfigError(
            f"release points not generated yet: {cache}\n"
            f"  Run once before this:  edna points <config> --profile {profile.name}"
        )

    # ---- 1. model ------------------------------------------------------
    if chunks is not None:
        wanted = list(chunks)
    elif resume:
        wanted = incomplete_chunks(site, profile)
    else:
        wanted = list(range(1, site.n_chunks + 1))

    result = None
    if not wanted:
        echo(f"{site.run_name}: all {site.n_chunks} chunk(s) already complete")
    else:
        if not ignore_manifest:
            check_manifest(site, profile, cache)
        echo(f"  provenance: {write_manifest(site, profile, cache)}")
        echo(f"{site.run_name}: {len(wanted)} chunk(s) locally")
        result = submit_local(site, profile, wanted, n_parallel=n_parallel,
                              dry_run=False, verbose=verbose)
        if result.failed:
            raise ConfigError(
                f"{site.run_name}: {len(result.failed)} chunk(s) failed: "
                f"{result.failed}\n"
                f"  Nothing was condensed. See <output>/logs/chunk_NNN.log, then "
                f"re-run with resume=True."
            )

    # ---- 2. condense ---------------------------------------------------
    spec = site.statistic()
    echo(f"{site.run_name}: condensing {spec.name} ({spec.kind})")
    condensed = condense(site, profile, out=condensed_out, subsample=subsample,
                         policy=policy, echo=echo)
    echo(f"  wrote {condensed}  ({condensed.stat().st_size / 1e6:.1f} MB)")

    # ---- 3. figures ----------------------------------------------------
    a = site.analysis
    # beside whatever was actually written, so --condensed-out carries the
    # figures with it - the same default `edna figures` uses
    out_dir = Path(figures_out) if figures_out is not None else condensed.parent / "figures"
    ds = open_condensed(condensed)
    try:
        results = site_results(
            ds, filtered_volume=site.filtered_volume,
            threshold_copies=a.detection.threshold_copies,
            n_stations=a.optimizer.n_stations, optimizer=a.optimizer.kind,
            random_seed=a.optimizer.random_seed, k_max=a.sweeps.n_stations_max,
            time_index=time_index)
        written = write_figures(ds, results, out_dir)
    finally:
        ds.close()

    echo(f"  {results['sources_covered']}/{results['n_sources']} sources "
         f"({results['coverage_fraction']:.1%}) from "
         f"{results['useful_stations']} station(s) at t={results['time_index']}")
    for p in written:
        echo(f"  wrote {p}")

    return {"chunks": result, "condensed": condensed,
            "figures": written, "results": results}
