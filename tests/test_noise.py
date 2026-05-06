import numpy as np

from nvsim.noise import generate_observed_counts


def test_poisson_false_returns_continuous_observed_values_without_overwriting_truth():
    true_u = np.array([[0.2, 1.7], [2.3, 0.4]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)
    true_u_before = true_u.copy()
    true_s_before = true_s.copy()

    observed = generate_observed_counts(true_u, true_s, capture_rate=0.5, poisson=False, dropout_rate=0.0)

    assert np.allclose(observed["unspliced"], true_u * 0.5)
    assert np.allclose(observed["spliced"], true_s * 0.5)
    assert not np.allclose(observed["unspliced"], np.round(observed["unspliced"]))
    assert np.allclose(true_u, true_u_before)
    assert np.allclose(true_s, true_s_before)


def test_poisson_true_returns_count_like_float_arrays():
    true_u = np.full((5, 3), 2.5)
    true_s = np.full((5, 3), 1.5)
    observed = generate_observed_counts(true_u, true_s, seed=4, poisson=True)

    assert observed["unspliced"].shape == true_u.shape
    assert observed["spliced"].shape == true_s.shape
    assert np.all(observed["unspliced"] >= 0.0)
    assert np.allclose(observed["unspliced"], np.round(observed["unspliced"]))
