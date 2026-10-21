# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for Demand technology model."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest
from pyochain import Some

from pstm.demand import SeriesDemand
from pstm.utils.dates import Dates

TZ = zi.ZoneInfo("Europe/Berlin")


class TestSeriesDemandCreation:
    """Test SeriesDemand instance creation."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def sample_series(self, dates):
        """Create sample power series."""
        return pd.Series(np.ones(len(dates)) * 1000, index=dates, name="power")

    def test_series_demand_creation(self, dates, sample_series) -> None:
        """Test SeriesDemand instance creation."""
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=1.0,
            series_thr=sample_series,
            series_thl=sample_series * 0.5,
            series_thh=sample_series * 0.0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        assert demand.factor_acp == 1.0

    def test_series_demand_with_numpy_array(self, dates) -> None:
        """Test SeriesDemand with numpy arrays."""
        acp = np.ones(len(dates)) * 2000
        thr = np.ones(len(dates)) * 1500
        thl = np.zeros(len(dates))
        thh = np.zeros(len(dates))

        demand = SeriesDemand(
            dates=dates,
            series_acp=acp,
            factor_acp=1.0,
            series_thr=thr,
            series_thl=thl,
            series_thh=thh,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        assert np.all(demand.acp["high"].to_numpy() == 2000)


class TestSeriesDemandFactors:
    """Test SeriesDemand scaling factors."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def sample_series(self, dates):
        """Create sample power series."""
        return pd.Series(np.ones(len(dates)) * 1000, index=dates, name="power")

    def test_acp_factor_scaling(self, dates, sample_series) -> None:
        """Test acp factor scales output correctly."""
        factor = 2.0
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=factor,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        # acp should be scaled by factor
        assert np.all(demand.acp["high"].to_numpy() == 1000 * factor)

    def test_thr_factor_scaling(self, dates, sample_series) -> None:
        """Test thr factor scales output correctly."""
        factor = 1.5
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series * 0,
            factor_acp=1.0,
            series_thr=sample_series,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=factor,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        # thr should be scaled by factor
        assert np.all(demand.thr["high"].to_numpy() == 1000 * factor)

    def test_thl_factor_scaling(self, dates, sample_series) -> None:
        """Test thl factor scales output correctly."""
        factor = 0.8
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series * 0,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=factor,
            factor_thh=1.0,
        )
        # thl should be scaled by factor
        assert np.all(demand.thl["high"].to_numpy() == 1000 * factor)

    def test_thh_factor_scaling(self, dates, sample_series) -> None:
        """Test thh factor scales output correctly."""
        factor = 0.5
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series * 0,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=factor,
        )
        # thh should be scaled by factor
        assert np.all(demand.thh["high"].to_numpy() == 1000 * factor)


class TestSeriesDemandReactivePower:
    """Test SeriesDemand reactive power calculation."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def sample_series(self, dates):
        """Create sample power series."""
        return pd.Series(np.ones(len(dates)) * 1000, index=dates, name="power")

    def test_default_cosphi(self, dates, sample_series) -> None:
        """Test default power factor of 0.9."""
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        assert demand.cosphi == 0.9

    def test_acq_calculation_from_cosphi(self, dates, sample_series) -> None:
        """Test reactive power calculation from power factor."""
        cosphi = 0.9
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
            cosphi=cosphi,
        )
        # Q follows from P as P tan(arccos(cosphi))
        expected_q = 1000 * np.tan(np.arccos(cosphi))
        assert np.allclose(demand.acq["high"].values, expected_q)

    def test_explicit_acq_series(self, dates, sample_series) -> None:
        """Test explicit reactive power series."""
        acq_series = sample_series * 0.5
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
            series_acq=Some(acq_series),
            factor_acq=1.0,
        )
        # When series_acq is provided, acq must come from series_acq * factor_acq
        # (= 1000 * 0.5 * 1.0 = 500), not from the active-power series.
        assert np.allclose(demand.acq["high"].values, 500.0)


class TestSeriesDemandAllOutputs:
    """Test that SeriesDemand sets all output columns."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def sample_series(self, dates):
        """Create sample power series."""
        return pd.Series(np.ones(len(dates)) * 1000, index=dates, name="power")

    def test_all_acp_columns_set(self, dates, sample_series) -> None:
        """Test all acp columns (high, base, low) are set."""
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series,
            factor_acp=1.0,
            series_thr=sample_series * 0,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        # All columns should be equal for demand
        assert np.allclose(demand.acp["high"].values, demand.acp["base"].values)
        assert np.allclose(demand.acp["high"].values, demand.acp["low"].values)

    def test_all_thr_columns_set(self, dates, sample_series) -> None:
        """Test all thr columns (high, base, low) are set."""
        demand = SeriesDemand(
            dates=dates,
            series_acp=sample_series * 0,
            factor_acp=1.0,
            series_thr=sample_series,
            series_thl=sample_series * 0,
            series_thh=sample_series * 0,
            factor_thr=1.0,
            factor_thl=1.0,
            factor_thh=1.0,
        )
        # All columns should be equal for demand
        assert np.allclose(demand.thr["high"].values, demand.thr["base"].values)
        assert np.allclose(demand.thr["high"].values, demand.thr["low"].values)
