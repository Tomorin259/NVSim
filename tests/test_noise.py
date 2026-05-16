import numpy as np
import pytest
from nvsim.noise import generate_observed_counts


def test_poisson_false_returns_continuous_observed_values_without_overwriting_truth():
    true_u = np.array([[0.2, 1.7], [2.3, 0.4]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)
    true_u_before = true_u.copy()
    true_s_before = true_s.copy()

    observed = generate_observed_counts(true_u, true_s, capture_rate=0.5, poisson=False, dropout_rate=0.0)

    assert np.allclose(observed["unspliced"], true_u * 0.5)
    assert np.allclose(observed["spliced"], true_s * 0.5)
    assert np.allclose(observed["capture_efficiency"], 0.5)
    assert not np.allclose(observed["unspliced"], np.round(observed["unspliced"]))
    assert np.allclose(true_u, true_u_before)
    assert np.allclose(true_s, true_s_before)


def test_poisson_true_returns_count_like_float_arrays():
    true_u = np.full((5, 3), 2.5)
    true_s = np.full((5, 3), 1.5)
    observed = generate_observed_counts(true_u, true_s, seed=4, poisson=True)

    assert observed["unspliced"].shape == true_u.shape
    assert observed["spliced"].shape == true_s.shape
    assert observed["capture_efficiency"].shape == (5,)
    assert np.all(observed["capture_efficiency"] == 1.0)
    assert np.all(observed["unspliced"] >= 0.0)
    assert np.allclose(observed["unspliced"], np.round(observed["unspliced"]))


def test_poisson_capture_can_use_cell_lognormal_efficiency():
    true_u = np.full((4, 2), 2.0)
    true_s = np.full((4, 2), 3.0)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=5,
        capture_rate=0.5,
        poisson=False,
        capture_efficiency_mode="cell_lognormal",
        capture_efficiency_cv=0.25,
    )

    eff = observed["capture_efficiency"]
    assert eff.shape == (4,)
    assert np.all((eff >= 0.0) & (eff <= 1.0))
    assert len(np.unique(np.round(eff, 6))) > 1
    assert np.allclose(observed["unspliced"], true_u * eff[:, None])
    assert np.allclose(observed["spliced"], true_s * eff[:, None])


def test_poisson_capture_cv_zero_matches_constant_mode():
    true_u = np.full((4, 2), 2.0)
    true_s = np.full((4, 2), 3.0)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=9,
        capture_rate=0.4,
        poisson=False,
        capture_efficiency_mode="cell_lognormal",
        capture_efficiency_cv=0.0,
    )

    assert np.allclose(observed["capture_efficiency"], 0.4)
    assert np.allclose(observed["unspliced"], true_u * 0.4)
    assert np.allclose(observed["spliced"], true_s * 0.4)


def test_cell_lognormal_capture_is_seed_reproducible():
    true_u = np.full((4, 2), 2.0)
    true_s = np.full((4, 2), 3.0)

    observed1 = generate_observed_counts(
        true_u,
        true_s,
        seed=29,
        capture_rate=0.5,
        capture_efficiency_mode="cell_lognormal",
        capture_efficiency_cv=0.2,
    )
    observed2 = generate_observed_counts(
        true_u,
        true_s,
        seed=29,
        capture_rate=0.5,
        capture_efficiency_mode="cell_lognormal",
        capture_efficiency_cv=0.2,
    )

    assert np.array_equal(observed1["capture_efficiency"], observed2["capture_efficiency"])
    assert np.array_equal(observed1["unspliced"], observed2["unspliced"])
    assert np.array_equal(observed1["spliced"], observed2["spliced"])


def test_binomial_capture_matches_velosim_style_round_then_capture():
    true_u = np.array([[0.2, 1.7], [2.3, 4.9]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=7,
        capture_rate=0.5,
        capture_model="binomial_capture",
        dropout_rate=0.0,
    )

    assert observed["unspliced"].shape == true_u.shape
    assert observed["spliced"].shape == true_s.shape
    assert np.allclose(observed["capture_efficiency"], 0.5)
    assert np.all(observed["unspliced"] >= 0.0)
    assert np.all(observed["spliced"] >= 0.0)
    assert np.all(observed["unspliced"] <= np.rint(true_u))
    assert np.all(observed["spliced"] <= np.rint(true_s))
    assert np.allclose(observed["unspliced"], np.round(observed["unspliced"]))


def test_binomial_capture_requires_capture_rate():
    true_u = np.array([[1.0]])
    true_s = np.array([[1.0]])

    try:
        generate_observed_counts(true_u, true_s, capture_model="binomial_capture")
    except ValueError as exc:
        assert "capture_rate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_binomial_capture_rejects_cell_lognormal_efficiency_mode():
    true_u = np.array([[1.0]])
    true_s = np.array([[1.0]])

    with pytest.raises(ValueError, match="capture_efficiency_mode"):
        generate_observed_counts(
            true_u,
            true_s,
            capture_model="binomial_capture",
            capture_rate=0.5,
            capture_efficiency_mode="cell_lognormal",
            capture_efficiency_cv=0.2,
        )


def test_binomial_capture_rate_one_returns_rounded_truth():
    true_u = np.array([[0.2, 1.7], [2.3, 4.9]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=11,
        capture_rate=1.0,
        capture_model="binomial_capture",
        dropout_rate=0.0,
    )

    assert np.array_equal(observed["unspliced"], np.rint(true_u))
    assert np.array_equal(observed["spliced"], np.rint(true_s))


def test_binomial_capture_rate_zero_returns_all_zeros():
    true_u = np.array([[0.2, 1.7], [2.3, 4.9]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=13,
        capture_rate=0.0,
        capture_model="binomial_capture",
        dropout_rate=0.0,
    )

    assert np.count_nonzero(observed["unspliced"]) == 0
    assert np.count_nonzero(observed["spliced"]) == 0


def test_dropout_rate_one_returns_all_zeros():
    true_u = np.array([[0.2, 1.7], [2.3, 4.9]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)

    observed = generate_observed_counts(
        true_u,
        true_s,
        seed=19,
        capture_rate=0.7,
        capture_model="poisson_capture",
        dropout_rate=1.0,
    )

    assert np.count_nonzero(observed["unspliced"]) == 0
    assert np.count_nonzero(observed["spliced"]) == 0


def test_same_seed_reproduces_noise_exactly():
    true_u = np.array([[0.2, 1.7], [2.3, 4.9]], dtype=float)
    true_s = np.array([[1.2, 0.6], [0.8, 2.1]], dtype=float)

    observed1 = generate_observed_counts(
        true_u,
        true_s,
        seed=23,
        capture_rate=0.5,
        capture_model="binomial_capture",
        dropout_rate=0.2,
    )
    observed2 = generate_observed_counts(
        true_u,
        true_s,
        seed=23,
        capture_rate=0.5,
        capture_model="binomial_capture",
        dropout_rate=0.2,
    )

    assert np.array_equal(observed1["unspliced"], observed2["unspliced"])
    assert np.array_equal(observed1["spliced"], observed2["spliced"])


def test_invalid_capture_model_raises_clear_error():
    true_u = np.array([[1.0]])
    true_s = np.array([[1.0]])

    with pytest.raises(ValueError, match="capture_model"):
        generate_observed_counts(true_u, true_s, capture_model="unknown")


def test_invalid_capture_efficiency_mode_raises_clear_error():
    true_u = np.array([[1.0]])
    true_s = np.array([[1.0]])

    with pytest.raises(ValueError, match="capture_efficiency_mode"):
        generate_observed_counts(true_u, true_s, capture_efficiency_mode="unknown")
