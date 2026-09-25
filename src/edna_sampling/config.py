"""Declarative experiment configuration and machine profiles.

Two files describe a run, and the split between them is the point:

  a *site config*      what the experiment is - geometry, sources, model
                       settings, statistics. Portable, version-controlled,
                       identical on every machine.

  a *machine profile*  where things live on one particular computer -
                       hindcast directories, output root, parallelism. Never
                       committed as part of an experiment's definition.

The same site config therefore runs on a desktop and on an HPC cluster by
swapping ``--profile``. Nothing in a site config may be an absolute path.

Paths inside a site config are resolved relative to the directory containing
that config, so a site and its geometry files move as a unit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = [
    "ConfigError",
    "ReleasePoints",
    "Source",
    "Model",
    "Chunking",
    "StatsSpec",
    "SiteConfig",
    "HindcastSource",
    "MachineProfile",
]

# Headroom on the derived particle buffer. The expectation is accurate to a few
# tenths of a percent, so this is slack for the Poisson scatter and for release
# schedules that do not divide evenly into the run.
_BUFFER_MARGIN = 1.2


class ConfigError(ValueError):
    """A configuration file is missing something, or says something impossible."""


# --------------------------------------------------------------------------
# small validation helpers - the error text is the user interface here
# --------------------------------------------------------------------------

def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        known = ", ".join(sorted(mapping)) or "(nothing)"
        raise ConfigError(f"{where}: missing required key {key!r}. Got: {known}")
    return mapping[key]


def _check_unknown(mapping: dict, allowed: Sequence[str], where: str) -> None:
    extra = set(mapping) - set(allowed)
    if extra:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def _pair(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{where}: expected a two-element list, got {value!r}")
    return (float(value[0]), float(value[1]))


def _positive(value: Any, where: str) -> float:
    v = float(value)
    if v <= 0:
        raise ConfigError(f"{where}: must be > 0, got {v}")
    return v


# --------------------------------------------------------------------------
# site config sections
# --------------------------------------------------------------------------

def _domain_path(raw: Any, base: Path) -> Path:
    """The site's `domain:` - the model-area polygon, as a path beside the config.

    It is a bare path rather than a section because it is the only thing left in
    it: the mesh moved to the hindcast, and everything else about where sources
    may go (`depth_range`, `min_flooded_fraction`) belongs to `release_points`,
    which is what actually applies it.
    """
    if isinstance(raw, dict):
        if "grid" in raw:
            raise ConfigError(
                "domain.grid is no longer part of a site config. The mesh is a property "
                "of the hindcast, not of the experiment: it is derived from the hindcast "
                "by `edna mesh` and cached per machine. Delete the key - `hindcast:` "
                "already says which mesh applies."
            )
        inner = raw.get("polygon", "<polygon>.csv")
        raise ConfigError(
            f"domain: expected the polygon path directly, got a mapping. The section "
            f"held nothing else once the mesh moved to the hindcast, so it collapsed to "
            f"one line. Write:\n  domain: {inner}"
        )
    return _resolve(raw, base, "domain")


def _isotime(value: Any, where: str) -> str:
    """An ISO-8601 datetime, validated at parse time rather than by OceanTracker
    hours into a run. Returned as text, which is what the params want."""
    from datetime import datetime

    text = str(value)
    try:
        datetime.fromisoformat(text)
    except ValueError:
        raise ConfigError(
            f"{where}: expected an ISO-8601 datetime such as '2018-01-02T00:00:00', "
            f"got {text!r}"
        ) from None
    return text


def _hindcast_name(raw: Any) -> str:
    """The site config's `hindcast:` - a logical name the profile resolves."""
    if isinstance(raw, dict):
        raise ConfigError(
            "hindcast: expected a logical hindcast name, got a mapping. `file_mask` moved "
            "to the machine profile, because how a hindcast's files are named is a "
            "property of the copy on this machine rather than of the experiment. Write:\n"
            f"  hindcast: {raw.get('name', '<name>')}\n"
            "and put input_dir / file_mask under `hindcasts:` in your profile."
        )
    name = str(raw)
    if "/" in name or name.endswith(".nc"):
        raise ConfigError(
            f"hindcast: expected a logical name the machine profile resolves "
            f"(e.g. `hauraki_gulf`), not a path - got {name!r}."
        )
    return name


@dataclass(frozen=True)
class ReleasePoints:
    """How the eDNA source locations are chosen (Lloyd relaxation)."""

    n: int
    depth_range: tuple[float, float]
    seed: int = 42
    min_flooded_fraction: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ReleasePoints":
        where = "release_points"
        _check_unknown(d, ["n", "depth_range", "seed", "min_flooded_fraction"], where)
        n = int(_require(d, "n", where))
        if n < 1:
            raise ConfigError(f"{where}.n: must be >= 1, got {n}")
        lo, hi = _pair(_require(d, "depth_range", where), f"{where}.depth_range")
        if lo >= hi:
            raise ConfigError(f"{where}.depth_range: need min < max, got [{lo}, {hi}]")
        mff = d.get("min_flooded_fraction")
        if mff is not None and not 0.0 < float(mff) <= 1.0:
            raise ConfigError(
                f"{where}.min_flooded_fraction: must be in (0, 1], got {mff}. "
                "1.0 keeps only cells that are never dry."
            )
        return cls(n=n, depth_range=(lo, hi), seed=int(d.get("seed", 42)),
                   min_flooded_fraction=None if mff is None else float(mff))


@dataclass(frozen=True)
class Source:
    """What one eDNA source is: shedding rate and how it is discretised."""

    shedding_rate_per_hour: float
    copies_per_particle: float
    release_at_bottom: bool = True
    release_offset_from_surface_or_bottom: float = 1.0
    max_cycles_to_find_release_points: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        where = "source"
        allowed = ["shedding_rate_per_hour", "copies_per_particle", "release_at_bottom",
                   "release_offset_from_surface_or_bottom", "max_cycles_to_find_release_points"]
        _check_unknown(d, allowed, where)
        return cls(
            shedding_rate_per_hour=_positive(_require(d, "shedding_rate_per_hour", where),
                                             f"{where}.shedding_rate_per_hour"),
            copies_per_particle=_positive(_require(d, "copies_per_particle", where),
                                          f"{where}.copies_per_particle"),
            release_at_bottom=bool(d.get("release_at_bottom", True)),
            release_offset_from_surface_or_bottom=float(
                d.get("release_offset_from_surface_or_bottom", 1.0)),
            max_cycles_to_find_release_points=int(d.get("max_cycles_to_find_release_points", 10)),
        )

    def particles_per_release(self, time_step: float) -> int:
        """Particles released per pulse, one pulse every `time_step` seconds.

        A particle stands for `copies_per_particle` eDNA copies, so the pulse
        size is the shedding rate expressed in particles over one time step.
        Raises if that rounds below one particle, which would silently model a
        source that sheds nothing.
        """
        per_second = (self.shedding_rate_per_hour / 3600.0) / self.copies_per_particle
        per_release = per_second * time_step
        if per_release < 1:
            raise ConfigError(
                f"source: particles_per_release = {per_release:.4g} < 1. "
                f"Lower copies_per_particle (now {self.copies_per_particle:g}), raise "
                f"shedding_rate_per_hour (now {self.shedding_rate_per_hour:g}), or raise "
                f"model.time_step (now {time_step:g} s)."
            )
        return int(per_release)


@dataclass(frozen=True)
class Model:
    """OceanTracker run settings that are not about statistics."""

    time_step: float
    duration_hours: float
    edna_half_life_hours: float
    critical_friction_velocity: float
    # when the releases begin; unset means the start of the hindcast files
    start: str | None = None
    output_time_interval: float = 3600.0
    processors: int = 20
    particle_buffer_per_release_group: int | None = None
    time_buffer_size: int = 25
    screen_output_time_interval: float = 3600.0
    write_tracks: bool = False
    tracks_writer_skip_properties: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        where = "model"
        allowed = ["time_step", "duration_hours", "edna_half_life_hours", "start",
                   "critical_friction_velocity", "output_time_interval", "processors",
                   "particle_buffer_per_release_group", "time_buffer_size",
                   "screen_output_time_interval", "write_tracks",
                   "tracks_writer_skip_properties"]
        _check_unknown(d, allowed, where)
        return cls(
            time_step=_positive(_require(d, "time_step", where), f"{where}.time_step"),
            duration_hours=_positive(_require(d, "duration_hours", where), f"{where}.duration_hours"),
            edna_half_life_hours=_positive(_require(d, "edna_half_life_hours", where),
                                           f"{where}.edna_half_life_hours"),
            start=(None if d.get("start") is None
                   else _isotime(d["start"], f"{where}.start")),
            critical_friction_velocity=float(_require(d, "critical_friction_velocity", where)),
            output_time_interval=float(d.get("output_time_interval", 3600.0)),
            processors=int(d.get("processors", 20)),
            particle_buffer_per_release_group=(
                None if d.get("particle_buffer_per_release_group") is None
                else _positive(d["particle_buffer_per_release_group"],
                               "model.particle_buffer_per_release_group")),
            time_buffer_size=int(d.get("time_buffer_size", 25)),
            screen_output_time_interval=float(d.get("screen_output_time_interval", 3600.0)),
            write_tracks=bool(d.get("write_tracks", False)),
            tracks_writer_skip_properties=tuple(d.get("tracks_writer_skip_properties", ()) or ()),
        )


@dataclass(frozen=True)
class Chunking:
    """How the release points are split across jobs."""

    release_groups_per_chunk: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Chunking":
        where = "chunking"
        _check_unknown(d, ["release_groups_per_chunk"], where)
        n = int(d.get("release_groups_per_chunk", 1))
        if n < 1:
            raise ConfigError(f"{where}.release_groups_per_chunk: must be >= 1, got {n}")
        return cls(release_groups_per_chunk=n)


@dataclass(frozen=True)
class StatsSpec:
    """One gridded-statistics product.

    `kind` is "gridded_2d" or "gridded_3d". The vertical settings are only
    meaningful for 3D; the particle filters (`near_seasurface`, `near_seabed`,
    `z_min`/`z_max`) are only accepted for 2D, which is how a 2D stat expresses
    a tow window - see docs/2d-stats-assessment.md.

    `extra` is an escape hatch passed straight through to OceanTracker for
    parameters this schema does not model.
    """

    name: str
    kind: str
    grid_center: tuple[float, float]
    rows: int
    cols: int
    span: tuple[float, float]
    start: str | None = None
    end: str | None = None
    status_list: tuple[str, ...] = ("moving",)
    water_depth_min: float | None = None
    water_depth_max: float | None = None
    write_connectivity: bool = False
    release_group_centered_grids: bool = False
    # 3D only
    layers: int | None = None
    z_min: float | None = None
    z_max: float | None = None
    vertical_range_measured_relative_to: str | None = None
    # 2D only - tow windows
    near_seasurface: float | None = None
    near_seabed: float | None = None
    z_range: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    KINDS = ("gridded_2d", "gridded_3d")
    VERTICAL_MODES = ("geoid", "surface", "bottom")

    @classmethod
    def from_dict(cls, d: dict, index: int) -> "StatsSpec":
        where = f"stats[{index}]"
        allowed = ["name", "kind", "grid_center", "rows", "cols", "span", "start", "end",
                   "status_list", "water_depth_min", "water_depth_max", "write_connectivity",
                   "release_group_centered_grids", "layers", "z_min", "z_max",
                   "vertical_range_measured_relative_to", "near_seasurface", "near_seabed",
                   "z_range", "extra"]
        _check_unknown(d, allowed, where)

        name = str(_require(d, "name", where))
        kind = str(_require(d, "kind", where))
        if kind not in cls.KINDS:
            raise ConfigError(f"{where}.kind: must be one of {', '.join(cls.KINDS)}, got {kind!r}")

        three_d_only = ["layers", "z_min", "z_max", "vertical_range_measured_relative_to"]
        two_d_only = ["near_seasurface", "near_seabed", "z_range"]
        wrong = [k for k in (two_d_only if kind == "gridded_3d" else three_d_only) if d.get(k) is not None]
        if wrong:
            other = "gridded_2d" if kind == "gridded_3d" else "gridded_3d"
            raise ConfigError(
                f"{where}: {', '.join(wrong)} only apply to {other}, but kind is {kind!r}."
            )

        if kind == "gridded_3d":
            for k in ("layers", "z_min", "z_max"):
                _require(d, k, where)
            mode = d.get("vertical_range_measured_relative_to")
            if mode is None:
                raise ConfigError(
                    f"{where}: gridded_3d must state vertical_range_measured_relative_to "
                    f"({', '.join(cls.VERTICAL_MODES)}). oceantracker changed this default "
                    "from 'surface' to 'geoid' after v0.5.3.6, so relying on the default "
                    "silently changes what the vertical bins mean."
                )
            if mode not in cls.VERTICAL_MODES:
                raise ConfigError(
                    f"{where}.vertical_range_measured_relative_to: must be one of "
                    f"{', '.join(cls.VERTICAL_MODES)}, got {mode!r}"
                )
            if float(d["z_min"]) >= float(d["z_max"]):
                raise ConfigError(f"{where}: need z_min < z_max, got {d['z_min']}, {d['z_max']}")
            if mode != "geoid" and float(d["z_min"]) < 0:
                raise ConfigError(
                    f"{where}: with vertical_range_measured_relative_to={mode!r}, z_min is a "
                    f"distance from the {'surface' if mode == 'surface' else 'sea bed'} and must "
                    f"be >= 0, got {d['z_min']}"
                )
        else:
            if d.get("near_seasurface") is not None and d.get("near_seabed") is not None:
                raise ConfigError(
                    f"{where}: set at most one of near_seasurface / near_seabed; they are "
                    "different tow references. Use two stats entries to compare them."
                )
            if d.get("z_range") is not None and (
                    d.get("near_seasurface") is not None or d.get("near_seabed") is not None):
                raise ConfigError(
                    f"{where}: z_range cannot be combined with near_seasurface / near_seabed "
                    "(oceantracker rejects fixed-z together with a surface/bed reference)."
                )

        zr = d.get("z_range")
        return cls(
            name=name,
            kind=kind,
            grid_center=_pair(_require(d, "grid_center", where), f"{where}.grid_center"),
            rows=int(_require(d, "rows", where)),
            cols=int(_require(d, "cols", where)),
            span=_pair(_require(d, "span", where), f"{where}.span"),
            start=d.get("start"),
            end=d.get("end"),
            status_list=tuple(d.get("status_list", ("moving",))),
            water_depth_min=_opt_float(d.get("water_depth_min")),
            water_depth_max=_opt_float(d.get("water_depth_max")),
            write_connectivity=bool(d.get("write_connectivity", False)),
            release_group_centered_grids=bool(d.get("release_group_centered_grids", False)),
            layers=None if d.get("layers") is None else int(d["layers"]),
            z_min=_opt_float(d.get("z_min")),
            z_max=_opt_float(d.get("z_max")),
            vertical_range_measured_relative_to=d.get("vertical_range_measured_relative_to"),
            near_seasurface=_opt_float(d.get("near_seasurface")),
            near_seabed=_opt_float(d.get("near_seabed")),
            z_range=None if zr is None else _pair(zr, f"{where}.z_range"),
            extra=dict(d.get("extra", {}) or {}),
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _resolve(value: Any, base: Path, where: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        raise ConfigError(
            f"{where}: absolute path {str(p)!r} is not allowed in a site config - it would "
            "bind the experiment to one machine. Use a path relative to the config file, or "
            "put the location in a machine profile."
        )
    return (base / p).resolve()


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """When a sample counts as a detection.

    A sample detects a source when the eDNA concentration at the sampling cell,
    multiplied by the volume actually filtered, reaches `threshold_copies`:

        concentration [copies/m^3] * filtered_volume [m^3] >= threshold_copies

    `filtered_volume` is not a free parameter - it is
    `individuals_per_source * copies_per_particle * sample_volume_m3`, where
    `copies_per_particle` comes from the site's `source:` block rather than being
    restated here. See `SiteConfig.filtered_volume`.
    """

    threshold_copies: float = 10.0
    sample_volume_m3: float = 1.0
    individuals_per_source: float = 100.0

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        where = "analysis.detection"
        allowed = ["threshold_copies", "sample_volume_m3", "individuals_per_source"]
        _check_unknown(d, allowed, where)
        return cls(
            threshold_copies=_positive(d.get("threshold_copies", 10.0),
                                       f"{where}.threshold_copies"),
            sample_volume_m3=_positive(d.get("sample_volume_m3", 1.0),
                                       f"{where}.sample_volume_m3"),
            individuals_per_source=_positive(d.get("individuals_per_source", 100.0),
                                             f"{where}.individuals_per_source"),
        )


@dataclass(frozen=True)
class Optimizer:
    """Which optimiser chooses the stations, and how many to choose."""

    kind: str = "greedy"
    n_stations: int = 3
    random_seed: int = 0

    KINDS = ("greedy", "greedy_reference", "simulated_annealing", "genetic")

    @classmethod
    def from_dict(cls, d: dict) -> "Optimizer":
        where = "analysis.optimizer"
        _check_unknown(d, ["kind", "n_stations", "random_seed"], where)
        kind = str(d.get("kind", "greedy"))
        if kind not in cls.KINDS:
            raise ConfigError(
                f"{where}.kind: must be one of {', '.join(cls.KINDS)}, got {kind!r}")
        n = int(d.get("n_stations", 3))
        if n < 1:
            raise ConfigError(f"{where}.n_stations: must be at least 1, got {n}")
        return cls(kind=kind, n_stations=n, random_seed=int(d.get("random_seed", 0)))


@dataclass(frozen=True)
class Sweeps:
    """Ranges the diagnostic figures sweep over.

    `volume_exponents_m3` is (start, stop, step) of log10 sample volume, so the
    default sweeps 1e-3 to 1e2 m^3 in quarter decades.
    """

    coverage_levels: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75)
    n_stations_max: int = 10
    volume_exponents_m3: tuple[float, float, float] = (-3.0, 2.0, 0.25)
    n_naive_draws: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "Sweeps":
        where = "analysis.sweeps"
        _check_unknown(d, ["coverage_levels", "n_stations_max", "volume_exponents_m3",
                           "n_naive_draws"], where)
        levels = tuple(float(v) for v in d.get("coverage_levels", (0.125, 0.25, 0.5, 0.75)))
        if not levels or any(not 0 < v <= 1 for v in levels):
            raise ConfigError(
                f"{where}.coverage_levels: expected fractions in (0, 1], got {list(levels)}")
        exps = tuple(float(v) for v in d.get("volume_exponents_m3", (-3.0, 2.0, 0.25)))
        if len(exps) != 3:
            raise ConfigError(
                f"{where}.volume_exponents_m3: expected [start, stop, step], got {list(exps)}")
        if exps[2] <= 0 or exps[1] <= exps[0]:
            raise ConfigError(
                f"{where}.volume_exponents_m3: need stop > start and step > 0, got {list(exps)}")
        return cls(
            coverage_levels=levels,
            n_stations_max=int(d.get("n_stations_max", 10)),
            volume_exponents_m3=exps,  # type: ignore[arg-type]
            n_naive_draws=int(d.get("n_naive_draws", 100)),
        )

    def volumes_m3(self):
        """The swept sample volumes, in m^3."""
        import numpy as np
        start, stop, step = self.volume_exponents_m3
        return 10.0 ** np.arange(start, stop + step / 2, step)


@dataclass(frozen=True)
class Analysis:
    """How a run's output is turned into figures.

    Every field has a default, so `analysis:` is optional and the defaults are the
    values the paper used. It is deliberately **excluded from the fingerprint**:
    these parameters do not change what the model computes, so re-deciding a
    detection threshold must not invalidate a finished 400-chunk run.
    """

    statistic: str | None = None
    detection: Detection = field(default_factory=Detection)
    optimizer: Optimizer = field(default_factory=Optimizer)
    sweeps: Sweeps = field(default_factory=Sweeps)

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        where = "analysis"
        _check_unknown(d, ["statistic", "detection", "optimizer", "sweeps"], where)
        statistic = d.get("statistic")
        return cls(
            statistic=None if statistic is None else str(statistic),
            detection=Detection.from_dict(d.get("detection", {}) or {}),
            optimizer=Optimizer.from_dict(d.get("optimizer", {}) or {}),
            sweeps=Sweeps.from_dict(d.get("sweeps", {}) or {}),
        )


@dataclass(frozen=True)
class SiteConfig:
    """A complete experiment definition, independent of any machine."""

    name: str
    version: str
    domain: Path
    hindcast: str
    release_points: ReleasePoints
    source: Source
    model: Model
    stats: tuple[StatsSpec, ...]
    chunking: Chunking = field(default_factory=Chunking)
    analysis: Analysis = field(default_factory=Analysis)
    source_path: Path | None = None

    SECTIONS = ["name", "version", "domain", "hindcast", "release_points",
                "source", "model", "stats", "chunking", "analysis"]

    @property
    def run_name(self) -> str:
        """Version-prefixed run identifier, e.g. 'v33_cape_rodney'.

        This keys the output directory and the release-point cache, so bumping
        `version` forks both rather than silently reusing an older run's points.
        """
        return f"{self.version}_{self.name}"

    @property
    def particles_alive_at_end(self) -> float:
        """Expected particles alive per release group when the run finishes.

        eDNA decay is a Poisson process, so a release group's particle count obeys

            dN/dt = R - alpha*N     ->     N(t) = (R/alpha) * (1 - exp(-alpha*t))

        where `R` is the release rate in particles per second and
        `alpha = ln(2) / half_life`. N rises monotonically toward the equilibrium
        `R/alpha`, so the largest value over a run is the one at its end.

        Checked against a real run: this predicts 86,096 for the 47 h v33
        cape_rodney configuration, which logged 86,114 alive at its peak.
        """
        import math

        per_release = self.source.particles_per_release(self.model.time_step)
        rate = per_release / self.model.time_step                  # particles / s
        duration = self.model.duration_hours * 3600.0
        half_life = self.model.edna_half_life_hours * 3600.0
        if half_life <= 0:
            raise ConfigError("model.edna_half_life_hours must be positive")
        alpha = math.log(2.0) / half_life
        return (rate / alpha) * (1.0 - math.exp(-alpha * duration))

    @property
    def particle_buffer_size(self) -> int:
        """Particles to allocate per release group.

        `model.particle_buffer_per_release_group` when set, otherwise derived from
        `particles_alive_at_end` with a margin.

        Deriving it matters because OceanTracker's own forecast ignores decay - it
        sizes the buffer from the *cumulative number released*
        (`forecasted_number_alive = cumulative_number_released`), which for these
        configurations is about 5.6x the number ever simultaneously alive. It then
        takes `min(forecast, this setting)`, so a smaller honest value binds.

        The margin covers the Poisson scatter about the expectation (sd is
        sqrt(N), i.e. ~0.3% here) with room to spare. Undersizing is recoverable
        in any case: the buffer grows in multiples of this value rather than
        overflowing.
        """
        import math

        if self.model.particle_buffer_per_release_group is not None:
            return int(self.model.particle_buffer_per_release_group)
        return int(math.ceil(self.particles_alive_at_end * _BUFFER_MARGIN))

    @property
    def filtered_volume(self) -> float:
        """Volume of eDNA-bearing water one sample effectively filters, in m^3.

        `individuals_per_source * copies_per_particle * sample_volume_m3`. Note
        `copies_per_particle` comes from the `source:` block, so the model's
        discretisation and the analysis cannot disagree about it.
        """
        d = self.analysis.detection
        return d.individuals_per_source * self.source.copies_per_particle * d.sample_volume_m3

    def statistic(self) -> StatsSpec:
        """The statistic the analysis should read.

        With one statistic defined, that one. With several, `analysis.statistic`
        must name which - guessing would silently analyse the wrong field, and
        auckland v3 deliberately carries a 2D and a 3D statistic at once.
        """
        if self.analysis.statistic is not None:
            for spec in self.stats:
                if spec.name == self.analysis.statistic:
                    return spec
            known = ", ".join(s.name for s in self.stats)
            raise ConfigError(
                f"analysis.statistic: {self.analysis.statistic!r} is not defined by this "
                f"config. Known statistics: {known}")
        if len(self.stats) == 1:
            return self.stats[0]
        known = ", ".join(s.name for s in self.stats)
        raise ConfigError(
            f"this config defines {len(self.stats)} statistics ({known}), so "
            f"`analysis.statistic:` must say which one the analysis reads.")

    @property
    def n_chunks(self) -> int:
        per = self.chunking.release_groups_per_chunk
        return (self.release_points.n + per - 1) // per

    @classmethod
    def load(cls, path: str | Path) -> "SiteConfig":
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"site config not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML - {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(raw, base=path.parent, source_path=path)

    @classmethod
    def from_dict(cls, raw: dict, base: Path, source_path: Path | None = None) -> "SiteConfig":
        where = "site config"
        _check_unknown(raw, cls.SECTIONS, where)
        stats_raw = _require(raw, "stats", where)
        if not isinstance(stats_raw, list) or not stats_raw:
            raise ConfigError(f"{where}.stats: expected a non-empty list of statistics")
        stats = tuple(StatsSpec.from_dict(s, i) for i, s in enumerate(stats_raw))
        names = [s.name for s in stats]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ConfigError(f"{where}.stats: duplicate name(s) {', '.join(sorted(dupes))}")

        cfg = cls(
            name=str(_require(raw, "name", where)),
            version=str(_require(raw, "version", where)),
            domain=_domain_path(_require(raw, "domain", where), base),
            hindcast=_hindcast_name(_require(raw, "hindcast", where)),
            release_points=ReleasePoints.from_dict(_require(raw, "release_points", where)),
            source=Source.from_dict(_require(raw, "source", where)),
            model=Model.from_dict(_require(raw, "model", where)),
            stats=stats,
            chunking=Chunking.from_dict(raw.get("chunking", {}) or {}),
            analysis=Analysis.from_dict(raw.get("analysis", {}) or {}),
            source_path=source_path,
        )
        # fail early rather than inside a scheduled job
        cfg.source.particles_per_release(cfg.model.time_step)
        return cfg

    def validate_inputs(self, profile: "MachineProfile | None" = None) -> None:
        """Check that the geometry this config points at actually exists.

        Kept out of `load` on purpose: parsing a config should work anywhere,
        including on a machine that has not fetched the site data yet. Call this
        at the point of use, where a missing file is genuinely an error.

        The polygon is a path beside the config and is always checked. The
        hindcast is a logical name, so it can only be checked once a profile is
        known - pass one, and its directory is resolved and checked too.

        The mesh is deliberately *not* checked: it is derived from the hindcast
        by `edna mesh` and is legitimately absent until then.
        """
        missing = []
        if not self.domain.is_file():
            missing.append(("domain", self.domain))
        if profile is not None:
            directory = profile.hindcast(self.hindcast).input_dir
            if not directory.is_dir():
                missing.append((f"hindcast {self.hindcast!r}", directory))
        if missing:
            lines = "\n".join(f"  {label}: {path}" for label, path in missing)
            raise ConfigError(f"{self.run_name}: input(s) not found:\n{lines}")

    def fingerprint(self) -> str:
        """Stable short hash of the experiment definition, for provenance.

        Covers everything that changes what is computed. Deliberately excludes
        `source_path` and the resolved absolute paths, so the same experiment
        fingerprints identically on a desktop and on a cluster.

        Also excludes `analysis`: those parameters describe how finished output is
        turned into figures, not what the model computes. Including them would
        mean re-deciding a detection threshold invalidates a 400-chunk run and
        trips the mid-run drift check in manifest.py for no reason.
        """
        payload = asdict(self)
        payload.pop("source_path", None)
        payload.pop("analysis", None)
        # the polygon is an absolute path once resolved; only its name is portable
        payload["domain"] = Path(payload["domain"]).name
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# machine profile
# --------------------------------------------------------------------------

def _machine_path(value: Any) -> Path:
    """A path in a profile. `~` is expanded - a profile is a per-user file, and a
    literal `~` directory is never what someone writing `~/edna` meant."""
    return Path(str(value)).expanduser()


@dataclass(frozen=True)
class HindcastSource:
    """One hindcast as this machine holds it.

    `input_dir` and `file_mask` are handed straight to OceanTracker's reader,
    which detects the format from the files themselves - SCHISM, ROMS, FVCOM,
    DELFT3D-FM - so nothing here is specific to any one of them.

    `mesh` is optional and is an *override*: the node/triangle/water-depth mesh
    is derivable from the hindcast (that is what `edna mesh` does) and is cached
    per machine, so it only needs naming when you already have one you would
    rather use.
    """

    name: str
    input_dir: Path
    file_mask: str = "*.nc"
    mesh: Path | None = None

    KEYS = ["input_dir", "file_mask", "grd_file_name", "mesh"]

    @classmethod
    def from_entry(cls, name: str, raw: Any, where: str, base: Path | None) -> "HindcastSource":
        # the shorthand: `name: /path/to/files` when the defaults will do
        if not isinstance(raw, dict):
            return cls(name=name, input_dir=_machine_path(raw))
        spot = f"{where}.hindcasts.{name}"
        _check_unknown(raw, cls.KEYS, spot)
        grd_file_name = raw.get("grd_file_name")
        mesh = raw.get("mesh")
        return cls(
            name=name,
            input_dir=_machine_path(_require(raw, "input_dir", spot)),
            file_mask=str(raw.get("file_mask", "*.nc")),
            grd_file_name=None if grd_file_name is None else _relative_to(_machine_path(grd_file_name), base),
            mesh=None if mesh is None else _relative_to(_machine_path(mesh), base),
        )


def _relative_to(p: Path, base: Path | None) -> Path:
    """Resolve a profile-relative path against the profile's own directory, so a
    profile can ship a file beside it without hard-coding anyone's home."""
    if not p.is_absolute() and base is not None:
        return (base / p).resolve()
    return p


@dataclass(frozen=True)
class MachineProfile:
    """Where things live on one machine, and how jobs are launched there."""

    name: str
    hindcasts: dict[str, HindcastSource]
    output_root: Path
    scratch: Path | None = None
    local: dict = field(default_factory=dict)
    python: str | None = None
    source_path: Path | None = None

    KEYS = ["hindcasts", "output_root", "scratch", "local", "python"]
    LOCAL_KEYS = ["n_parallel_jobs"]
    RETIRED = {
        "scheduler": "there is only one way to launch chunks now: `edna submit` runs them "
                     "locally, and a cluster wraps `edna run --chunk N` itself",
        "slurm": "this package no longer generates sbatch scripts",
        "grids": "the mesh belongs to the hindcast it came from, so it moved under "
                 "`hindcasts:`. Either drop it and let `edna mesh` derive one, or name "
                 "the file you have as `mesh:` inside the hindcast's own entry",
    }

    @classmethod
    def load(cls, path: str | Path) -> "MachineProfile":
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"machine profile not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML - {exc}") from exc
        where = f"profile {path.name}"
        retired = [k for k in cls.RETIRED if k in raw]
        if retired:
            detail = "; ".join(f"`{k}`: {cls.RETIRED[k]}" for k in retired)
            keys = ", ".join(repr(k) for k in retired)
            verb = "is" if len(retired) == 1 else "are"
            noun = "key" if len(retired) == 1 else "keys"
            raise ConfigError(f"{where}: {keys} {verb} no longer supported. {detail}. "
                              f"Delete the {noun}.")
        _check_unknown(raw, cls.KEYS, where)

        hindcasts = _require(raw, "hindcasts", where)
        if not isinstance(hindcasts, dict) or not hindcasts:
            raise ConfigError(
                f"{where}.hindcasts: expected a non-empty mapping of logical name -> "
                f"either a directory or a block with input_dir / file_mask / mesh")

        local = dict(raw.get("local", {}) or {})
        # every other section rejects unknown keys; without this a typo'd
        # `n_parallel_job:` is silently ignored and the run is serial
        _check_unknown(local, cls.LOCAL_KEYS, f"{where}.local")

        scratch = raw.get("scratch")
        return cls(
            name=path.stem,
            hindcasts={str(k): HindcastSource.from_entry(str(k), v, where, path.parent)
                       for k, v in hindcasts.items()},
            output_root=_machine_path(_require(raw, "output_root", where)),
            scratch=None if scratch is None else _machine_path(scratch),
            local=local,
            python=None if raw.get("python") is None else str(raw["python"]),
            source_path=path,
        )

    def hindcast(self, name: str) -> HindcastSource:
        """Resolve a logical hindcast name to what this machine holds."""
        if name not in self.hindcasts:
            known = ", ".join(sorted(self.hindcasts)) or "(none)"
            raise ConfigError(
                f"profile {self.name!r} defines no hindcast {name!r}. Known: {known}. "
                f"Add it under `hindcasts:` in the profile."
            )
        return self.hindcasts[name]

    @property
    def intermediate_root(self) -> Path:
        """Where derived products that are not a run's output belong - the mesh
        cache. `scratch` when the profile names one, else beside the output."""
        return self.scratch if self.scratch is not None else self.output_root

    def mesh_path(self, hindcast_name: str) -> Path:
        """The mesh file for a hindcast: the profile's override, or the cache
        location `edna mesh` writes and `edna points` reads."""
        entry = self.hindcast(hindcast_name)
        if entry.mesh is not None:
            return entry.mesh
        return self.intermediate_root / "meshes" / f"{hindcast_name}.nc"
