"""Turn a site config + machine profile into an OceanTracker parameter dict.

`build_params` is deliberately pure: it creates no directories, claims no chunk
numbers and touches no filesystem. Everything it needs is an argument, so the
whole translation from "what the experiment is" to "what OceanTracker is told"
is unit-testable without a hindcast, a cluster or a model run.

Chunk numbers are 1-based, matching the existing `<run_name>_chunk_NNN` output
directories that `edna_sampling.stats.StatsData` reads. Chunk *n* always covers
release points [(n-1)*k, n*k), so the mapping from chunk to source locations is
a property of the config rather than of the order in which jobs happened to
start - which is what made the previous fcntl-counter scheme irreproducible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from edna_sampling.config import ConfigError, MachineProfile, SiteConfig, StatsSpec

__all__ = ["build_params", "chunk_slice", "chunk_run_name", "root_output_dir", "n_chunks"]

_STATS_CLASS = {
    "gridded_2d": "oceantracker.particle_statistics.gridded_statistics2D.GriddedStats2D_timeBased",
    "gridded_3d": "oceantracker.particle_statistics.gridded_statistics3D.GriddedStats3D_timeBased",
}


def n_chunks(site: SiteConfig) -> int:
    """Number of chunks the run is split into."""
    return site.n_chunks


def chunk_slice(site: SiteConfig, chunk_number: int) -> slice:
    """Release-point indices belonging to a 1-based chunk number."""
    total = site.n_chunks
    if not 1 <= chunk_number <= total:
        raise ConfigError(
            f"chunk_number {chunk_number} out of range: this run has {total} chunk(s) "
            f"({site.release_points.n} release points at "
            f"{site.chunking.release_groups_per_chunk} per chunk)."
        )
    per = site.chunking.release_groups_per_chunk
    start = (chunk_number - 1) * per
    return slice(start, min(start + per, site.release_points.n))


def chunk_run_name(site: SiteConfig, chunk_number: int) -> str:
    """Output file base for one chunk, e.g. 'v33_cape_rodney_chunk_001'."""
    return f"{site.run_name}_chunk_{chunk_number:03d}"


def root_output_dir(site: SiteConfig, profile: MachineProfile) -> Path:
    """Directory holding every chunk of this run on this machine."""
    return Path(profile.output_root) / site.run_name


def _stats_params(spec: StatsSpec, update_interval: float) -> dict[str, Any]:
    """One entry of OceanTracker's `particle_statistics` list.

    Optional settings are only emitted when set, so the generated dict does not
    pin down parameters the config did not express an opinion about.
    """
    p: dict[str, Any] = {
        "name": spec.name,
        "class_name": _STATS_CLASS[spec.kind],
        "update_interval": update_interval,
        "status_list": list(spec.status_list),
        "grid_center": list(spec.grid_center),
        "rows": spec.rows,
        "cols": spec.cols,
        "span_x": spec.span[0],
        "span_y": spec.span[1],
        "write_connectivity": spec.write_connectivity,
        "release_group_centered_grids": spec.release_group_centered_grids,
    }
    if spec.start is not None:
        p["start"] = spec.start
    if spec.end is not None:
        p["end"] = spec.end
    if spec.water_depth_min is not None:
        p["water_depth_min"] = spec.water_depth_min
    if spec.water_depth_max is not None:
        p["water_depth_max"] = spec.water_depth_max

    if spec.kind == "gridded_3d":
        p["layers"] = spec.layers
        p["z_min"] = spec.z_min
        p["z_max"] = spec.z_max
        # always explicit; see docs/2d-stats-assessment.md and config.StatsSpec
        p["vertical_range_measured_relative_to"] = spec.vertical_range_measured_relative_to
    else:
        # a 2D stat expresses its tow window as a particle filter, evaluated
        # per particle per step against the instantaneous surface / true bed
        if spec.near_seasurface is not None:
            p["near_seasurface"] = spec.near_seasurface
        if spec.near_seabed is not None:
            p["near_seabed"] = spec.near_seabed
        if spec.z_range is not None:
            p["z_min"], p["z_max"] = spec.z_range

    p.update(spec.extra)
    return p


def _release_groups(site: SiteConfig, points: Sequence[Sequence[float]],
                    start_index: int) -> list[dict[str, Any]]:
    """One release group per source location in this chunk.

    Group names carry the *global* release-point index, so a group keeps its
    identity no matter how the run is chunked.
    """
    src = site.source
    pulse_size = src.particles_per_release(site.model.time_step)
    groups = []
    for offset, point in enumerate(points):
        group = {
            "name": f"node_{start_index + offset:04d}",
            "points": [[float(point[0]), float(point[1])]],
            "pulse_size": pulse_size,
            "release_interval": site.model.time_step,
            "release_at_bottom": src.release_at_bottom,
            "release_offset_from_surface_or_bottom": src.release_offset_from_surface_or_bottom,
            "max_cycles_to_find_release_points": src.max_cycles_to_find_release_points,
        }
        # There is no run-level start setting in oceantracker: the release
        # schedule is what decides when the model starts stepping, so `start`
        # belongs on every group. Left unset, releases begin at the start of
        # whatever hindcast files the profile points at.
        if site.model.start is not None:
            group["start"] = site.model.start
        groups.append(group)
    return groups


def build_params(
    site: SiteConfig,
    profile: MachineProfile,
    chunk_number: int,
    release_points: Sequence[Sequence[float]],
    *,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Parameters for one chunk of a run.

    Args:
        site: the experiment definition.
        profile: the machine this will run on (hindcast + output locations).
        chunk_number: 1-based; chunk *n* covers points [(n-1)*k, n*k).
        release_points: the full set of source locations for the run, as
            produced by the release-point stage. Sliced here, so every chunk
            sees the same numbering.
        root_dir: override the output directory; defaults to
            `<profile.output_root>/<run_name>`.

    Returns:
        A dict suitable for `oceantracker.main.run`.
    """
    if len(release_points) != site.release_points.n:
        raise ConfigError(
            f"{site.run_name}: got {len(release_points)} release points but the config "
            f"declares release_points.n = {site.release_points.n}. The cached point set is "
            "stale; bump `version` or regenerate it."
        )

    model = site.model
    sl = chunk_slice(site, chunk_number)
    chunk_points = list(release_points)[sl]
    out_dir = Path(root_dir) if root_dir is not None else root_output_dir(site, profile)

    reader = {
    "input_dir": str(profile.hindcast(site.hindcast).input_dir),
    "file_mask": profile.hindcast(site.hindcast).file_mask,
    }

    if profile.hindcast(site.hindcast).grd_file_name is not None:
        reader["grd_file_name"] = str(profile.hindcast(site.hindcast).grd_file_name)

    params: dict[str, Any] = {
        "root_output_dir": str(out_dir),
        "output_file_base": chunk_run_name(site, chunk_number),
        "processors": model.processors,
        "particle_buffer_initial_size": int(
            site.particle_buffer_size * site.chunking.release_groups_per_chunk),
        "max_run_duration": model.duration_hours * 3600.0,
        "time_step": model.time_step,
        "write_tracks": model.write_tracks,
        "screen_output_time_interval": model.screen_output_time_interval,
        "time_buffer_size": model.time_buffer_size,
        "reader": reader,
        "trajectory_modifiers": [{
            "name": "eDNA decay",
            "class_name": "CullRate",
            # exponential decay with the configured half-life
            "decay_rate": math.log(2) / (model.edna_half_life_hours * 3600.0),
        }],
        "resuspension": {
            "critical_friction_velocity": model.critical_friction_velocity,
        },
        "release_groups": _release_groups(site, chunk_points, sl.start),
        "particle_statistics": [_stats_params(s, model.output_time_interval) for s in site.stats],
    }

    if model.tracks_writer_skip_properties:
        params["tracks_writer"] = {
            "update_interval": model.output_time_interval,
            "turn_off_write_particle_properties_list": list(model.tracks_writer_skip_properties),
        }
    return params
