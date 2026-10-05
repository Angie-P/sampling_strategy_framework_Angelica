"""Reading and writing the hydrodynamic mesh.

`derive_mesh` is not tested here: it needs oceantracker and a real hindcast, so
it is exercised by running `edna mesh`, not by the suite - which must stay
seconds-fast on a laptop. What *is* tested is everything that happens to a mesh
once it exists, including the three on-disk shapes one can arrive in.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from edna_sampling.config import ConfigError
from edna_sampling.mesh import Mesh, TriangleLocator, load_mesh, write_mesh

# a unit square split into two triangles, with a depth at each corner
NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
DEPTH = np.array([5.0, 10.0, 15.0, 20.0])


def _mesh():
    return Mesh(node_x=NODE_X, node_y=NODE_Y, triangles=TRIANGLES, water_depth=DEPTH)


def _assert_matches(mesh):
    np.testing.assert_allclose(mesh.node_x, NODE_X)
    np.testing.assert_allclose(mesh.node_y, NODE_Y)
    np.testing.assert_array_equal(mesh.triangles, TRIANGLES)
    np.testing.assert_allclose(mesh.water_depth, DEPTH)


def test_round_trips_through_its_own_format(tmp_path):
    path = write_mesh(_mesh(), tmp_path / "m.nc")
    _assert_matches(load_mesh(path))


def test_reads_an_oceantracker_grid_file(tmp_path):
    """`grid000.nc`, which every run drops in its output directory - so a profile
    can point `mesh:` at one someone already has."""
    path = tmp_path / "grid000.nc"
    xr.Dataset({
        "x": (("node", "vector2D"), np.column_stack([NODE_X, NODE_Y])),
        "triangles": (("tri", "vertex"), TRIANGLES),
        "water_depth": ("node", DEPTH),
    }).to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_reads_a_raw_schism_file(tmp_path):
    """SCHISM face_nodes are 1-based, and a mixed mesh carries a fourth column
    that is not part of the triangle."""
    path = tmp_path / "schout_1.nc"
    quads = np.column_stack([TRIANGLES + 1, np.full(len(TRIANGLES), -99)])
    xr.Dataset({
        "SCHISM_hgrid_node_x": ("node", NODE_X),
        "SCHISM_hgrid_node_y": ("node", NODE_Y),
        "SCHISM_hgrid_face_nodes": (("face", "vertex"), quads),
        "depth": ("node", DEPTH),
    }).to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_its_own_format_wins_over_the_others(tmp_path):
    """A file carrying both shapes must be read as the edna one, or a mesh that
    happens to keep an `x` variable would silently be read the wrong way."""
    path = tmp_path / "both.nc"
    ds = xr.Dataset({
        "node_x": ("node", NODE_X), "node_y": ("node", NODE_Y),
        "triangles": (("tri", "vertex"), TRIANGLES),
        "water_depth": ("node", DEPTH),
        "x": (("node", "vector2D"), np.column_stack([NODE_Y, NODE_X])),  # transposed
    })
    ds.to_netcdf(path)
    _assert_matches(load_mesh(path))


def test_a_missing_mesh_says_how_to_make_one(tmp_path):
    with pytest.raises(ConfigError, match="edna mesh"):
        load_mesh(tmp_path / "nope.nc")


def test_an_unrecognised_file_lists_what_was_tried(tmp_path):
    path = tmp_path / "other.nc"
    xr.Dataset({"temperature": ("node", DEPTH)}).to_netcdf(path)
    with pytest.raises(ConfigError, match="not a mesh this package recognises"):
        load_mesh(path)


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ConfigError, match="same length"):
        Mesh(node_x=NODE_X, node_y=NODE_Y[:2], triangles=TRIANGLES, water_depth=DEPTH)


def test_non_triangular_connectivity_is_rejected():
    with pytest.raises(ConfigError, match=r"\(n, 3\)"):
        Mesh(node_x=NODE_X, node_y=NODE_Y, water_depth=DEPTH,
             triangles=np.zeros((2, 4), dtype=np.int32))


def test_bounds_and_locator_describe_the_same_mesh():
    mesh = _mesh()
    assert mesh.bounds() == (0.0, 0.0, 1.0, 1.0)
    assert mesh.n_nodes == 4 and mesh.n_triangles == 2
    # the locator is what condense and detection use to test membership
    tri, _ = mesh.locator().locate(np.array([0.5, 2.0]), np.array([0.4, 2.0]))
    assert tri[0] != -1                                          # inside
    assert tri[1] == -1                                          # outside


def test_the_written_form_is_much_smaller_than_an_oceantracker_grid(tmp_path):
    """The reason for having a format of our own: the Hauraki Gulf mesh is 2.8 MB
    here against 27 MB as a grid000.nc, and none of the difference is read."""
    n = 20_000
    rng = np.random.default_rng(0)
    big = Mesh(node_x=rng.random(n), node_y=rng.random(n),
               triangles=rng.integers(0, n, (2 * n, 3)).astype(np.int32),
               water_depth=rng.random(n) * 50)
    ours = write_mesh(big, tmp_path / "ours.nc")

    fat = tmp_path / "grid000.nc"
    xr.Dataset({
        "x": (("node", "vector2D"), np.column_stack([big.node_x, big.node_y])),
        "triangles": (("tri", "vertex"), big.triangles),
        "water_depth": ("node", big.water_depth),
        # the parts a grid000.nc carries and nothing here ever reads
        "adjacency": (("tri", "vertex"), big.triangles),
        "bc_transform": (("tri", "r", "c"), rng.random((2 * n, 3, 2))),
        "node_to_tri_map": (("node", "m"), rng.integers(0, n, (n, 10))),
    }).to_netcdf(fat)

    assert Path(ours).stat().st_size < fat.stat().st_size / 3


# --- locating points on the mesh ----------------------------------------------

def _linear(x, y):
    """Any containing triangle interpolates a linear field exactly, so it is the
    right answer wherever a point is found, whichever triangle wins."""
    return 1.0 + 3.0 * np.asarray(x) + 2.0 * np.asarray(y)


def test_locator_reproduces_a_linear_field_on_the_mesh():
    loc = TriangleLocator(NODE_X, NODE_Y, TRIANGLES)
    rng = np.random.default_rng(1)
    qx, qy = rng.random((2, 4, 25))          # any shape in, the same shape out
    tri, w = loc.locate(qx, qy)
    assert tri.shape == qx.shape and w.shape == qx.shape + (3,)
    assert (tri != -1).all()
    np.testing.assert_allclose(loc.interpolate(_linear(NODE_X, NODE_Y), tri, w),
                               _linear(qx, qy))


def test_locator_counts_nodes_and_edges_as_inside_and_the_rest_as_outside():
    loc = TriangleLocator(NODE_X, NODE_Y, TRIANGLES)
    qx = np.array([0.0, 1.0, 0.5, 0.5, 1.0, -0.1, 2.0, np.nan])
    qy = np.array([0.0, 1.0, 0.5, 0.0, 0.5, 0.5, 2.0, 0.5])
    tri, w = loc.locate(qx, qy)
    np.testing.assert_array_equal(tri[:5] != -1, True)
    np.testing.assert_array_equal(tri[5:], -1)
    np.testing.assert_array_equal(w[5:], 0.0)
    assert np.isnan(loc.interpolate(DEPTH, tri, w)[5:]).all()


def test_locator_takes_the_float_triangles_a_grid000_decodes_to():
    """xarray decodes grid000.nc's triangles as float64 (they carry a _FillValue)."""
    loc = TriangleLocator(NODE_X, NODE_Y, TRIANGLES.astype(float))
    tri, _ = loc.locate(np.array([0.7]), np.array([0.2]))
    assert tri[0] == 0


def test_locator_works_where_matplotlib_refuses_an_overlapping_mesh(folded_mesh):
    x, y, tris = folded_mesh
    from matplotlib.tri import Triangulation
    with pytest.raises(RuntimeError, match="invalid"):
        Triangulation(x, y, tris).get_trifinder()

    loc = TriangleLocator(x, y, tris)
    #               only tri 0   only tri 2  tris 0 and 1  outside
    qx = np.array([-4.9,       4.9,        9.75,         30.0])
    qy = np.array([-3.7,       3.7,        -2.36,        30.0])
    tri, w = loc.locate(qx, qy)
    assert tri[0] == 0 and tri[1] == 2 and tri[2] in (0, 1) and tri[3] == -1
    np.testing.assert_allclose(loc.interpolate(_linear(x, y), tri, w)[:3],
                               _linear(qx, qy)[:3])


def test_a_point_in_a_large_triangle_beside_small_ones_is_found():
    """Why bounding boxes and not nearest centroids: here the ten nearest
    centroids all belong to the strip of small triangles, not to the large
    triangle the point is actually in."""
    xs = [0.0, 0.0, -10.0]
    ys = [-5.0, 5.0, 0.0]
    tris = [[0, 1, 2]]
    for k in range(10):                      # a strip of slivers at x in [0.1, 0.2]
        y0 = -0.5 + 0.1 * k
        n = len(xs)
        xs += [0.1, 0.2, 0.1]
        ys += [y0, y0, y0 + 0.1]
        tris.append([n, n + 1, n + 2])
    loc = TriangleLocator(np.array(xs), np.array(ys), np.array(tris))
    tri, _ = loc.locate(np.array([-0.05]), np.array([0.0]))
    assert tri[0] == 0
