"""Invariants of the detection model and the station optimisers.

Tiny synthetic arrays - no hindcast, no model output, no HPC - which is the
point: the science should be checkable in milliseconds.
"""

import numpy as np
import pytest

from edna_sampling.detection import (greedy, greedy_fast, mask_below_seafloor,
                                     optimize_stations, valid_cell_mask)


# --- the paper's fast path must equal the reference -------------------------

@pytest.mark.parametrize("trial", range(20))
def test_greedy_fast_matches_the_reference_implementation(trial):
    """The paper figures call `greedy_fast`; `StatsData.greedy_algorithm` is the
    reference it claims to be a drop-in for. If they ever diverge, the published
    station choices depend on which one happened to run."""
    rng = np.random.default_rng(trial)
    n_rel = int(rng.integers(3, 12))
    n_y, n_x = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    detectable = rng.random((n_rel, n_y, n_x)) < rng.uniform(0.05, 0.6)

    for k in (1, 2, 3):
        ref_loc, ref_cov = greedy(detectable, k)
        fast_loc, fast_cov = greedy_fast(detectable, k)
        assert list(ref_loc) == list(fast_loc), f"k={k}"
        assert set(ref_cov) == set(fast_cov), f"k={k}"


def test_greedy_returns_exactly_k_stations_even_when_gain_is_exhausted():
    """Both implementations promise exactly k, which the coverage curves rely on
    when k exceeds the number of useful locations."""
    detectable = np.zeros((4, 3, 3), dtype=bool)
    detectable[:, 0, 0] = True          # one location covers everything
    for impl in (greedy, greedy_fast):
        locs, covered = impl(detectable, 3)
        assert len(locs) == 3
        assert covered == {0, 1, 2, 3}


def test_greedy_picks_the_location_covering_most_sources_first():
    detectable = np.zeros((5, 2, 2), dtype=bool)
    detectable[0:2, 0, 0] = True        # covers 2
    detectable[0:5, 1, 1] = True        # covers 5
    for impl in (greedy, greedy_fast):
        locs, _ = impl(detectable, 1)
        assert locs == [(1, 1)]


def test_greedy_second_pick_is_complementary_not_merely_popular():
    """The objective is *new* coverage: a location duplicating the first pick is
    worth nothing, however many sources it sees on its own."""
    detectable = np.zeros((6, 1, 3), dtype=bool)
    detectable[0:4, 0, 0] = True        # best single: 4
    detectable[0:4, 0, 1] = True        # a duplicate of it: adds 0
    detectable[4:6, 0, 2] = True        # adds the remaining 2
    for impl in (greedy, greedy_fast):
        locs, covered = impl(detectable, 2)
        assert locs[0] == (0, 0)
        assert locs[1] == (0, 2), "should take the complementary location"
        assert covered == {0, 1, 2, 3, 4, 5}


def test_optimize_stations_rejects_an_unknown_method():
    detectable = np.zeros((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="unknown optimizer"):
        optimize_stations(detectable, 1, method="annealing")


# --- the mesh-geometry helpers ---------------------------------------------------

# stats points: only in triangle 0, only in triangle 2, in the overlap, off the mesh
_X_STATS = np.array([-4.9, 4.9, 9.75, 30.0])
_Y_STATS = np.array([-3.7, 3.7, -2.36])


def test_valid_cell_mask_on_a_mesh_with_overlapping_triangles(folded_mesh):
    """Both helpers used matplotlib's trifinder, which refuses this mesh outright."""
    x, y, tris = folded_mesh
    mask = valid_cell_mask(x, y, tris, _X_STATS, _Y_STATS)
    assert mask.shape == (3, 4)
    # (x, y) pairs on the diagonal are the labelled points; x = 30 is off the mesh
    assert mask[0, 0] and mask[1, 1] and mask[2, 2]
    assert not mask[:, 3].any()


def test_mask_below_seafloor_masks_deep_layers_and_off_mesh_cells(folded_mesh):
    x, y, tris = folded_mesh
    depth = np.full(6, 5.0)
    z = np.array([1.0, 3.0, 5.0, 7.0])          # layer bottoms at 2, 4, 6, 8 m
    c = np.ones((1, 1, 3, 4, len(z)))
    masked, cell_depth = mask_below_seafloor(c, _X_STATS, _Y_STATS, z, x, y, tris, depth)
    inside = valid_cell_mask(x, y, tris, _X_STATS, _Y_STATS)

    np.testing.assert_allclose(cell_depth[inside], 5.0)
    assert np.isnan(cell_depth[~inside]).all()
    m = masked.mask[0, 0]
    # inside: the two layers whose bottom is above the 5 m bed stay
    np.testing.assert_array_equal(m[inside], [[False, False, True, True]] * inside.sum())
    assert m[~inside].all()
