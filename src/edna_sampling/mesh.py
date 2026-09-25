"""The hydrodynamic mesh: node positions, triangles, and water depth.

Three arrays are all any stage of this package ever wants from a hindcast's
grid - `release_points` triangulates them to build the depth band, `condense`
interpolates bathymetry onto the statistics grid, `detection` tests which cells
are inside the domain. Nothing reads the rest.

**The mesh is derived, not supplied.** It is a property of the hindcast, so
`derive_mesh` obtains it by asking OceanTracker to open the hindcast and write
its grid, and `ensure_mesh` caches the result once per hindcast per machine.
That is what lets an experiment run against any format OceanTracker reads -
SCHISM, ROMS, FVCOM, DELFT3D-FM - rather than only the one whose mesh somebody
happened to copy into the repository.

`load_mesh` additionally accepts a mesh you already have, in any of the three
shapes these files come in (see `_READERS`), so a profile's `mesh:` override
can point at an OceanTracker `grid000.nc` or a raw SCHISM file directly.

Importing this module does not import oceantracker; only `derive_mesh` does, and
only when the cache is cold.
"""

from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edna_sampling.config import ConfigError, MachineProfile

__all__ = ["Mesh", "load_mesh", "write_mesh", "derive_mesh", "ensure_mesh"]

FORMAT = "edna-mesh"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Mesh:
    """An unstructured triangular mesh with a depth at every node.

    `node_x`/`node_y` are in the hindcast's own horizontal coordinates - degrees
    for the geographic hindcasts this package is used with, metres for a
    projected one. Nothing here converts between them; the site config's polygon
    and statistics grid must be in the same coordinates, which
    `generate_release_locations_using_lloyd_relax` checks by overlap.
    """

    node_x: np.ndarray
    node_y: np.ndarray
    triangles: np.ndarray
    water_depth: np.ndarray
    source: Path | None = None

    def __post_init__(self) -> None:
        n = len(self.node_x)
        if len(self.node_y) != n or len(self.water_depth) != n:
            raise ConfigError(
                f"mesh{f' {self.source}' if self.source else ''}: node_x, node_y and "
                f"water_depth must be the same length, got "
                f"{n}, {len(self.node_y)}, {len(self.water_depth)}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ConfigError(
                f"mesh{f' {self.source}' if self.source else ''}: triangles must be "
                f"(n, 3), got {self.triangles.shape}")

    @property
    def n_nodes(self) -> int:
        return len(self.node_x)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def triangulation(self):
        """A `matplotlib.tri.Triangulation` over these nodes."""
        from matplotlib.tri import Triangulation
        return Triangulation(self.node_x, self.node_y, self.triangles)

    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the nodes."""
        return (float(self.node_x.min()), float(self.node_y.min()),
                float(self.node_x.max()), float(self.node_y.max()))


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _from_edna(ds):
    return (ds["node_x"].values, ds["node_y"].values,
            ds["triangles"].values, ds["water_depth"].values)


def _from_oceantracker(ds):
    # grid000.nc, as written into every run's output directory
    x = ds["x"].values
    return x[:, 0], x[:, 1], ds["triangles"].values, ds["water_depth"].values


def _from_schism(ds):
    # a raw SCHISM output or hgrid file; face_nodes are 1-based, and a mesh with
    # quads carries 4 columns of which the first 3 are the triangle
    depth = ds["depth"] if "depth" in ds else ds["SCHISM_hgrid_node_depth"]
    return (ds["SCHISM_hgrid_node_x"].values, ds["SCHISM_hgrid_node_y"].values,
            ds["SCHISM_hgrid_face_nodes"].values[:, :3].astype(int) - 1, depth.values)


# probed in order; the first whose key variables are all present wins
_READERS = [
    ("edna mesh", ("node_x", "node_y", "triangles", "water_depth"), _from_edna),
    ("oceantracker grid", ("x", "triangles", "water_depth"), _from_oceantracker),
    ("SCHISM", ("SCHISM_hgrid_node_x", "SCHISM_hgrid_face_nodes"), _from_schism),
]


def load_mesh(path: str | Path) -> Mesh:
    """Read a mesh file, whichever of the known shapes it is in."""
    import xarray as xr

    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"no mesh at {path}\n"
            f"  Derive one from the hindcast:  edna mesh <config> --profile <profile>")
    with xr.open_dataset(path) as ds:
        for label, keys, read in _READERS:
            if all(k in ds for k in keys):
                node_x, node_y, triangles, water_depth = read(ds)
                return Mesh(
                    node_x=np.asarray(node_x, dtype=float).ravel(),
                    node_y=np.asarray(node_y, dtype=float).ravel(),
                    triangles=np.asarray(triangles, dtype=np.int32),
                    water_depth=np.asarray(water_depth, dtype=float).ravel(),
                    source=path,
                )
        known = ", ".join(label for label, _, _ in _READERS)
        raise ConfigError(
            f"{path}: not a mesh this package recognises (tried: {known}).\n"
            f"  Variables present: {', '.join(sorted(ds.variables)[:12])}...\n"
            f"  Drop `mesh:` from the profile and let `edna mesh` derive one instead.")


def write_mesh(mesh: Mesh, path: str | Path) -> Path:
    """Write a mesh in this package's own compact form.

    Four variables rather than OceanTracker's twenty: ~2.8 MB against 27 MB for
    the Hauraki Gulf mesh, and nothing in it that only means something inside a
    model run.
    """
    import xarray as xr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zlib = {"zlib": True, "complevel": 4}
    ds = xr.Dataset(
        data_vars={
            "node_x": ("node", mesh.node_x, {"long_name": "node x / longitude"}),
            "node_y": ("node", mesh.node_y, {"long_name": "node y / latitude"}),
            "triangles": (("triangle", "vertex"), mesh.triangles.astype(np.int32),
                          {"long_name": "0-based node indices of each triangle"}),
            "water_depth": ("node", mesh.water_depth.astype(np.float32),
                            {"units": "m", "long_name": "still-water depth at the node"}),
        },
        attrs={"format": FORMAT, "format_version": FORMAT_VERSION},
    )
    ds.to_netcdf(path, encoding={v: zlib for v in ds.data_vars})
    return path


# --------------------------------------------------------------------------
# deriving one from a hindcast
# --------------------------------------------------------------------------

def derive_mesh(input_dir: Path, file_mask: str, out: Path, *, grd_file_name: Path | None = None, echo=print) -> Path:
    """Build a mesh from a hindcast by having OceanTracker read it.

    OceanTracker writes `grid000.nc` while setting up a run, *before* it needs
    release groups - so a run configured with none writes the grid and then
    stops with "no release groups". That is the whole trick, and it is why this
    does not care whether the run reports success: what matters is whether a
    grid appeared, which is checked explicitly. A future version that wrote the
    grid later would produce a clear "no grid file" error here rather than
    anything silent.

    The alternative - calling `make_a_reader_from_params` - would be faster
    still, but it is wired into OceanTracker's global `shared_info` singleton
    and is not usable standalone. Going through the public `main.run` API costs
    about ten seconds and keeps this working across upstream refactors.

    Doing it this way is also what makes the format-independence real: the
    reader detects SCHISM / ROMS / FVCOM / DELFT3D-FM from the files themselves,
    so nothing in this package has to know which it is.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ConfigError(f"hindcast directory does not exist: {input_dir}")

    echo(f"  reading {input_dir}/{file_mask}")
    with tempfile.TemporaryDirectory(prefix="edna_mesh_") as tmp:
        reader_params = {
            "input_dir": str(input_dir),
            "file_mask": file_mask,
        }
        if grd_file_name is not None:
            reader_params["grd_file_name"] = str(grd_file_name)

        params = {
            "root_output_dir": tmp,
            "output_file_base": "mesh",
            "time_step": 1.0,
            "max_run_duration": 1.0,       # must be > 0; no particles ever move
            "write_tracks": False,
            "time_buffer_size": 2,         # one buffer's worth of hindcast, not 25
            "screen_output_time_interval": 3600.0,
            "reader": reader_params,
        }
        # The run is *expected* to end with "No particle release_groups found", so
        # its screen output is captured rather than shown: a page of OceanTracker
        # fatal-error banners under a command that succeeded reads as a failure.
        # It is kept and printed if no grid appears, where it is the diagnosis.
        transcript = io.StringIO()
        try:
            with redirect_stdout(transcript), redirect_stderr(transcript):
                # imported here too: oceantracker prints a banner on import
                from oceantracker import main as ot_main
                ot_main.run(params)
        except Exception as exc:                      # noqa: BLE001 - see docstring
            reason = exc
        else:
            reason = None

        produced = sorted(Path(tmp).rglob("grid*.nc"))
        if not produced:
            tail = "\n".join(transcript.getvalue().splitlines()[-25:])
            raise ConfigError(
                f"could not derive a mesh from {input_dir}.\n"
                f"  OceanTracker read the hindcast but wrote no grid file"
                + (f", and stopped with: {reason}" if reason is not None else ".")
                + f"\n  Check that {file_mask!r} matches files there, and that they are a "
                f"format OceanTracker reads.\n\n--- OceanTracker output ---\n{tail}"
            )
        mesh = load_mesh(produced[0])

    out = Path(out)
    write_mesh(mesh, out)
    echo(f"  {mesh.n_nodes:,} nodes, {mesh.n_triangles:,} triangles "
         f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def ensure_mesh(profile: MachineProfile, hindcast_name: str, *,
                force: bool = False, echo=print) -> Path:
    """The mesh for a hindcast on this machine, deriving and caching it if needed.

    A profile that names its own `mesh:` is taken at its word and never
    regenerated - that override exists precisely for "I already have one".
    """
    entry = profile.hindcast(hindcast_name)
    target = profile.mesh_path(hindcast_name)

    if entry.mesh is not None:
        if not target.is_file():
            raise ConfigError(
                f"profile {profile.name!r}: hindcast {hindcast_name!r} names "
                f"mesh: {target}, which does not exist.\n"
                f"  Fix the path, or drop `mesh:` and let `edna mesh` derive one.")
        return target

    if target.is_file() and not force:
        return target

    echo(f"deriving the mesh for {hindcast_name!r}")
    return derive_mesh(entry.input_dir, entry.file_mask, target, grd_file_name=entry.grd_file_name, echo=echo)
