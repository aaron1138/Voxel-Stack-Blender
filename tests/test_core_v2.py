import sys
import os
import numpy as np
import pytest

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing_core import _lookup_distance_scale

@pytest.fixture
def sample_lut():
    """A sample distance LUT for testing, pre-sorted."""
    return np.array([
        [10.0, 0.8],  # dist < 10.0 -> scale 0.8
        [20.0, 1.0],  # 10.0 <= dist < 20.0 -> scale 0.8
        [50.0, 1.2],  # 20.0 <= dist < 50.0 -> scale 1.0
        [100.0, 1.5]  # 50.0 <= dist -> scale 1.2 (or 1.5 for last bucket)
    ], dtype=np.float32)


def test_lookup_distance_scale_empty_lut():
    """Tests that an empty LUT returns a default scale of 1.0."""
    empty_lut = np.empty((0, 2), dtype=np.float32)
    assert _lookup_distance_scale(30.0, empty_lut) == 1.0

def test_lookup_distance_scale_below_first_bucket(sample_lut):
    """Tests a distance smaller than the first entry in the LUT."""
    assert _lookup_distance_scale(5.0, sample_lut) == pytest.approx(0.8)

def test_lookup_distance_scale_within_buckets(sample_lut):
    """Tests distances that fall within various buckets."""
    # In the first bucket
    assert _lookup_distance_scale(10.0, sample_lut) == pytest.approx(0.8)
    assert _lookup_distance_scale(15.0, sample_lut) == pytest.approx(0.8)

    # In the second bucket
    assert _lookup_distance_scale(20.0, sample_lut) == pytest.approx(1.0)
    assert _lookup_distance_scale(49.9, sample_lut) == pytest.approx(1.0)

    # In the third bucket
    assert _lookup_distance_scale(50.0, sample_lut) == pytest.approx(1.2)
    assert _lookup_distance_scale(99.0, sample_lut) == pytest.approx(1.2)

def test_lookup_distance_scale_above_last_bucket(sample_lut):
    """Tests a distance larger than the last entry in the LUT."""
    # The logic should return the scale of the last bucket
    assert _lookup_distance_scale(100.0, sample_lut) == pytest.approx(1.5)
    assert _lookup_distance_scale(200.0, sample_lut) == pytest.approx(1.5)
