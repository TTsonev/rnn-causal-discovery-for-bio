import numpy as np
import pytest

from utils import pad_to_same_length


def test_pad_to_same_length_shapes_and_masking():
    sample_a = (
        [np.array([[1, 0], [2, 1], [3, 0]], dtype=float)],
        np.array([1.0], dtype=float),
        np.array([3]),
        np.array(["tx_a"]),
        np.array([2]),
        np.array([0.9], dtype=float),
        np.array([[0.1, 0.2]], dtype=float),
    )
    sample_b = (
        [np.array([[1, 1], [2, 0]], dtype=float)],
        np.array([0.0], dtype=float),
        np.array([2]),
        np.array(["tx_b"]),
        np.array([1]),
        np.array([0.1], dtype=float),
        np.array([[0.3, 0.4]], dtype=float),
    )

    x, y, lengths, mask, mask_targets, meta, y_cont, freq = pad_to_same_length([sample_a, sample_b])

    assert tuple(x.shape) == (2, 3, 2)
    assert tuple(y.shape) == (2, 1)
    assert tuple(lengths.shape) == (2,)
    assert tuple(mask.shape) == (2, 3, 2)
    assert mask[1, 2].sum().item() == 0.0
    assert tuple(mask_targets.shape) == (2, 1)
    assert tuple(meta.shape) == (2,)
    assert tuple(y_cont.shape) == (2, 1)
    assert freq.shape == (2, 1, 2)


def test_pad_to_same_length_handles_nan_targets():
    sample = (
        [np.array([[1, 0], [2, 1]], dtype=float)],
        np.array([np.nan], dtype=float),
        np.array([2]),
        np.array(["tx_nan"]),
        np.array([0]),
        np.array([0.3], dtype=float),
        np.array([[0.2, 0.8]], dtype=float),
    )

    _, y, _, _, mask_targets, _, _, _ = pad_to_same_length([sample])
    assert y[0, 0].item() == 0.0
    assert mask_targets[0, 0].item() == 0.0


def test_pad_to_same_length_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one"):
        pad_to_same_length([])
