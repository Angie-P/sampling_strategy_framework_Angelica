"""Shared fixtures. Deliberately free of hindcast data or model runs: everything
here must pass in seconds on a laptop with no access to an HPC filesystem."""

import numpy as np
import pytest


@pytest.fixture
def site_dict():
    """A minimal but complete site config, as parsed YAML."""
    return {
        "name": "testsite",
        "version": "v1",
        "domain": "testsite/poly.csv",
        "hindcast": "some_hindcast",
        "release_points": {"n": 8, "depth_range": [5, 30], "seed": 42},
        "source": {"shedding_rate_per_hour": 1.0e7, "copies_per_particle": 1000},
        "model": {
            "time_step": 120,
            "duration_hours": 47,
            "edna_half_life_hours": 6,
            "critical_friction_velocity": 0.009,
        },
        "stats": [{
            "name": "test_3d",
            "kind": "gridded_3d",
            "grid_center": [174.8, -36.3],
            "rows": 10, "cols": 10, "span": [0.05, 0.05],
            "layers": 20, "z_min": 0, "z_max": 20,
            "vertical_range_measured_relative_to": "surface",
        }],
        "chunking": {"release_groups_per_chunk": 2},
    }


@pytest.fixture
def profile_dict(tmp_path):
    return {
        "hindcasts": {"some_hindcast": {"input_dir": str(tmp_path / "hindcast"),
                                        "file_mask": "schout_*.nc"}},
        "output_root": str(tmp_path / "out"),
        "local": {"n_parallel_jobs": 2},
    }


@pytest.fixture
def folded_mesh():
    """The four folded elements of the SHYFEM Venice grid, recentred and scaled
    by 1e4, as (x, y, triangles). Node 3 sits just across edge 5-2 of the large
    triangle 0, so the three small triangles overlap it and that edge is a seam
    with nothing stitched to it. matplotlib's trifinder refuses this mesh.

    Points by where they fall: (-4.9, -3.7) only in triangle 0, (4.9, 3.7) only
    in triangle 2, (9.75, -2.36) in the overlap of triangles 0 and 1."""
    x = np.array([-6.6, 16.6, -6.7, 4.7, -22.1, 14.2])
    y = np.array([12.0, -1.2, 7.9, 0.4, -14.1, -5.0])
    tris = np.array([[5, 2, 4], [1, 3, 5], [1, 0, 3], [0, 2, 3]])
    return x, y, tris
