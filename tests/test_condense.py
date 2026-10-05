"""The condense stage: effective volume, and the format contract.

Geometry is checked on hand-built meshes where the right answer is known by
inspection, rather than on model output.
"""

import math

import numpy as np
import pytest

from edna_sampling.condense import FORMAT_VERSION, _tow_depth, effective_volume
from edna_sampling.config import ConfigError, StatsSpec


def _flat_mesh(depth, half=10.0, n=6):
    """A square mesh spanning [-half, half]^2 with a constant or callable depth."""
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            tris += [[a, b, d], [a, d, c]]
    wd = np.full(x.shape, float(depth)) if not callable(depth) else depth(x, y)
    return x, y, np.array(tris), wd


def _one_cell(x_grid, y_grid, tris, wd, tow, **kw):
    """Effective volume of a single 2x2-unit stats cell at the origin."""
    area = np.array([[4.0]])
    vol, inside = effective_volume([0.0], [0.0], area, x_grid, y_grid, tris, wd, tow, **kw)
    return float(vol[0, 0]), bool(inside[0, 0])


def test_deep_water_gives_area_times_tow_depth():
    """Where the water is deeper than the tow window the answer is exact and the
    old staircase was already right."""
    x, y, tris, wd = _flat_mesh(100.0)
    vol, inside = _one_cell(x, y, tris, wd, tow=19.0)
    assert inside
    assert vol == pytest.approx(4.0 * 19.0, rel=1e-6)


def test_water_shallower_than_the_tow_window_limits_the_volume():
    x, y, tris, wd = _flat_mesh(5.0)
    vol, _ = _one_cell(x, y, tris, wd, tow=19.0)
    assert vol == pytest.approx(4.0 * 5.0, rel=1e-6)


def test_a_sloping_cell_gets_its_mean_thickness_not_a_staircase():
    """Depth ramps 0..20 m across the domain, so a cell centred at the origin
    averages the true wet thickness rather than snapping to whole metres."""
    x, y, tris, wd = _flat_mesh(lambda x, y: np.clip(10.0 + x, 0.0, None), half=10.0, n=21)
    vol, _ = _one_cell(x, y, tris, wd, tow=19.0, subsample=9)
    assert vol == pytest.approx(4.0 * 10.0, rel=0.02)   # mean depth at x=0 is 10 m


def test_drop_policy_zeroes_a_cell_that_is_not_fully_covered():
    """The conservative sensitivity check: anything not full-thickness is dropped."""
    x, y, tris, wd = _flat_mesh(5.0)
    weighted, _ = _one_cell(x, y, tris, wd, tow=19.0, policy="weight")
    dropped, _ = _one_cell(x, y, tris, wd, tow=19.0, policy="drop")
    assert weighted > 0
    assert dropped == 0.0


def test_drop_policy_keeps_a_fully_covered_cell():
    x, y, tris, wd = _flat_mesh(100.0)
    dropped, _ = _one_cell(x, y, tris, wd, tow=19.0, policy="drop")
    assert dropped == pytest.approx(4.0 * 19.0, rel=1e-6)


def test_cells_outside_the_mesh_get_no_volume():
    x, y, tris, wd = _flat_mesh(100.0, half=1.0)
    area = np.array([[4.0]])
    vol, inside = effective_volume([50.0], [50.0], area, x, y, tris, wd, 19.0)
    assert vol[0, 0] == 0.0
    assert not inside[0, 0]


def test_a_mesh_with_overlapping_triangles_still_gives_a_volume(folded_mesh):
    """The SHYFEM Venice grid has one element folded over its neighbours, which
    made matplotlib's trifinder refuse the whole mesh and stopped `condense`.
    These are those four elements, scaled, with the stats cell in the overlap."""
    x, y, tris = folded_mesh
    wd = np.full(6, 10.0)
    vol, inside = effective_volume([9.75], [-2.36], np.array([[1.0]]), x, y, tris, wd, 19.0)
    assert inside[0, 0]
    assert vol[0, 0] == pytest.approx(10.0)


@pytest.mark.parametrize("policy,match", [("nearest", "weight"), ("", "weight")])
def test_unknown_partial_cell_policy_is_rejected(policy, match):
    x, y, tris, wd = _flat_mesh(100.0)
    with pytest.raises(ConfigError, match=match):
        _one_cell(x, y, tris, wd, tow=19.0, policy=policy)


# --- the tow window is read from whichever statistic kind is configured ------

def _spec(**kw):
    base = dict(name="s", kind="gridded_2d", grid_center=(0.0, 0.0), rows=2, cols=2,
                span=(1.0, 1.0))
    base.update(kw)
    return StatsSpec(**base)


def test_tow_depth_of_a_2d_statistic_is_its_near_seasurface():
    assert _tow_depth(_spec(near_seasurface=19.0)) == 19.0


def test_tow_depth_of_a_3d_statistic_is_the_analysis_window():
    """20 layers over 0-20 m, window layers 0..18 -> 19 m, matching the 2D form.
    This equality is what lets a 3D run and its 2D twin be compared."""
    spec = _spec(kind="gridded_3d", layers=20, z_min=0.0, z_max=20.0,
                 vertical_range_measured_relative_to="surface")
    assert _tow_depth(spec) == 19.0


def test_a_2d_statistic_with_no_vertical_selection_is_a_whole_column_tow():
    """Both shipped examples are like this: a gridded_2d with no
    near_seasurface/near_seabed counts every particle in the column, so the
    window is unbounded and the effective volume is the full column depth."""
    assert _tow_depth(_spec()) == math.inf


def test_tow_depth_of_a_2d_statistic_is_its_near_seabed():
    assert _tow_depth(_spec(near_seabed=4.0)) == 4.0


def test_an_unbounded_window_gives_the_full_column_volume():
    x, y, tris, wd = _flat_mesh(12.0)
    vol, inside = _one_cell(x, y, tris, wd, tow=math.inf)
    assert inside
    assert vol == pytest.approx(4.0 * 12.0, rel=1e-6)


def test_drop_policy_keeps_everything_when_the_window_is_unbounded():
    """`drop` zeroes cells the window does not fully cover. Nothing fails that
    test when the window is the whole column, so it must not zero the grid -
    np.isclose(x, inf) is False for every finite x."""
    x, y, tris, wd = _flat_mesh(12.0)
    weighted, _ = _one_cell(x, y, tris, wd, tow=math.inf, policy="weight")
    dropped, _ = _one_cell(x, y, tris, wd, tow=math.inf, policy="drop")
    assert dropped == weighted == pytest.approx(4.0 * 12.0, rel=1e-6)


def test_format_version_is_checked_on_read(tmp_path):
    """A stale product should say so, not fail three functions later."""
    import xarray as xr
    from edna_sampling.condense import open_condensed

    p = tmp_path / "old.nc"
    xr.Dataset({"a": ("x", [1.0])},
               attrs={"format": "edna-condensed",
                      "format_version": FORMAT_VERSION + 1}).to_netcdf(p)
    with pytest.raises(ConfigError, match="Re-run `edna condense`"):
        open_condensed(p)


def test_a_foreign_netcdf_is_rejected(tmp_path):
    import xarray as xr
    from edna_sampling.condense import open_condensed

    p = tmp_path / "other.nc"
    xr.Dataset({"a": ("x", [1.0])}).to_netcdf(p)
    with pytest.raises(ConfigError, match="not an edna condensed product"):
        open_condensed(p)
