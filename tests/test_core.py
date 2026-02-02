# Copyright (c) 2025 Matthias C. Wurm / ForgottenForge.xyz
# Licensed under AGPL-3.0-or-later. See LICENSE for details.
# Commercial license available: nfo@forgottenforge.xyz
"""Tests for batch_susceptibility core module."""

import numpy as np
import pytest
from batch_susceptibility import BatchSusceptibility, SusceptibilityResult
from batch_susceptibility.core import susceptibility_from_losses, susceptibility_from_sweep


class TestSusceptibilityFromLosses:
    """Test the online (single-run re-batching) mode."""

    def test_iid_returns_alpha_near_minus_one(self):
        """i.i.d. data should give alpha ~ -1."""
        np.random.seed(42)
        data = np.random.randn(10000)
        result = susceptibility_from_losses(data)

        assert abs(result.alpha - (-1.0)) < 0.15

    def test_correlated_data_deviates(self):
        """Strongly correlated data should deviate from alpha = -1."""
        np.random.seed(42)
        N = 10000
        data = np.cumsum(np.random.randn(N))  # random walk
        result = susceptibility_from_losses(data)

        # Random walk: V(K) ~ K (alpha ~ +1), definitely not -1
        assert result.alpha > -0.5
        assert result.p_iid < 0.05

    def test_periodic_data_has_sigma_c(self):
        """Periodic signal should show K_c near the period."""
        np.random.seed(42)
        N = 10000
        period = 100
        t = np.arange(N)
        data = np.sin(2 * np.pi * t / period) + 0.3 * np.random.randn(N)

        result = susceptibility_from_losses(data)
        # K_c should be near the period (within factor 3)
        assert period / 3 < result.K_c < period * 3

    def test_result_has_all_fields(self):
        """Result should contain all expected fields."""
        data = np.random.randn(1000)
        result = susceptibility_from_losses(data)

        assert isinstance(result, SusceptibilityResult)
        assert isinstance(result.K_c, float)
        assert isinstance(result.kappa, float)
        assert isinstance(result.alpha, float)
        assert isinstance(result.alpha_se, float)
        assert isinstance(result.p_iid, float)
        assert len(result.K_values) > 0
        assert len(result.V_values) == len(result.K_values)
        assert len(result.chi_values) == len(result.K_values)

    def test_too_few_points_raises(self):
        """Should raise with fewer than 20 points."""
        bs = BatchSusceptibility()
        with pytest.raises(ValueError, match="at least 20"):
            bs.feed(np.random.randn(10))

    def test_summary_is_string(self):
        """Summary should return a readable string."""
        data = np.random.randn(1000)
        result = susceptibility_from_losses(data)
        s = result.summary()
        assert isinstance(s, str)
        assert "K_c" in s

    def test_regime_classification(self):
        """Test regime property."""
        np.random.seed(42)

        # i.i.d.
        result_iid = susceptibility_from_losses(np.random.randn(5000))
        assert result_iid.regime == "iid"

        # Correlated (random walk)
        result_rw = susceptibility_from_losses(np.cumsum(np.random.randn(5000)))
        assert result_rw.regime in ("correlated", "weakly-correlated")

    def test_custom_K_range(self):
        """Custom K_min and K_max should be respected."""
        data = np.random.randn(5000)
        result = susceptibility_from_losses(data, K_min=10, K_max=100)

        assert result.K_values[0] >= 10
        assert result.K_values[-1] <= 100


class TestSusceptibilityFromSweep:
    """Test the sweep (multiple training runs) mode."""

    def test_basic_sweep(self):
        """Sweep with decreasing loss should find a K_c."""
        batch_sizes = np.array([16, 32, 64, 128, 256, 512, 1024])
        # Loss decreases then plateaus
        losses = np.array([2.5, 2.0, 1.2, 0.8, 0.75, 0.74, 0.73])

        result = susceptibility_from_sweep(batch_sizes, losses)
        assert 32 <= result.K_c <= 256  # transition region

    def test_sweep_needs_4_points(self):
        """Should raise with fewer than 4 points."""
        with pytest.raises(ValueError, match="at least 4"):
            susceptibility_from_sweep(
                np.array([32, 64, 128]),
                np.array([1.0, 0.8, 0.7]),
            )

    def test_sweep_mismatched_lengths(self):
        """Should raise if lengths don't match."""
        with pytest.raises(ValueError, match="same length"):
            susceptibility_from_sweep(
                np.array([32, 64, 128, 256]),
                np.array([1.0, 0.8]),
            )


class TestBatchSusceptibilityClass:
    """Test the main BatchSusceptibility interface."""

    def test_feed_and_find(self):
        """Basic feed -> find_critical workflow."""
        bs = BatchSusceptibility()
        bs.feed(np.random.randn(5000))
        result = bs.find_critical()
        assert isinstance(result, SusceptibilityResult)

    def test_chaining(self):
        """feed() should return self for chaining."""
        bs = BatchSusceptibility()
        result = bs.feed(np.random.randn(5000)).find_critical()
        assert isinstance(result, SusceptibilityResult)

    def test_find_without_feed_raises(self):
        """Should raise if find_critical called without data."""
        bs = BatchSusceptibility()
        with pytest.raises(RuntimeError, match="No data"):
            bs.find_critical()

    def test_sweep_mode(self):
        """find_critical_from_sweep should work without feed."""
        bs = BatchSusceptibility()
        result = bs.find_critical_from_sweep(
            [32, 64, 128, 256, 512],
            [2.0, 1.5, 1.0, 0.9, 0.85],
        )
        assert isinstance(result, SusceptibilityResult)


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_constant_data(self):
        """Constant data should not crash."""
        data = np.ones(1000)
        bs = BatchSusceptibility()
        bs.feed(data)
        result = bs.find_critical()
        # Should return something, even if not meaningful
        assert isinstance(result, SusceptibilityResult)

    def test_very_long_sequence(self):
        """Should handle large datasets."""
        np.random.seed(42)
        data = np.random.randn(100000)
        result = susceptibility_from_losses(data)
        assert abs(result.alpha + 1) < 0.1  # should be ~i.i.d.

    def test_trending_data(self):
        """Data with a trend (like training loss)."""
        np.random.seed(42)
        N = 5000
        trend = 3.0 * np.exp(-np.arange(N) / 500)
        noise = 0.2 * np.random.randn(N)
        data = trend + noise

        result = susceptibility_from_losses(data)
        # Should detect the non-i.i.d. structure
        assert result.alpha != -1.0  # not exactly i.i.d.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
