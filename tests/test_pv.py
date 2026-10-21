# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for PV technology model."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest

from pstm.pv import PV
from pstm.utils.dates import Dates

TZ = zi.ZoneInfo("Europe/Berlin")


class TestPVCreation:
    """Test PV instance creation."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)

    @pytest.fixture
    def weather(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic weather data."""
        n = len(dates)
        return pd.DataFrame(
            {
                "ghi": np.maximum(0, 500 * np.sin(np.linspace(0, 4 * np.pi, n))),
                "dni": np.maximum(0, 800 * np.sin(np.linspace(0, 4 * np.pi, n))),
                "dhi": np.maximum(0, 100 * np.sin(np.linspace(0, 4 * np.pi, n))),
                "temp_air": 15 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)),
                "wind_speed": 3 + 2 * np.sin(np.linspace(0, 4 * np.pi, n)),
            },
            index=dates,
        )

    def test_pv_creation(self, dates: pd.DatetimeIndex) -> None:
        """Test basic PV creation."""
        pv = PV(
            dates=dates,
            power_inst=10000,  # 10 kWp
            efficiency_inv=0.95,
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        assert pv.power_inst == 10000
        assert pv.efficiency_inv == pytest.approx(0.95)
        assert pv.lat == pytest.approx(51.05)
        assert pv.lon == pytest.approx(13.73)

    def test_pv_default_parameters(self, dates: pd.DatetimeIndex) -> None:
        """Test PV default parameters."""
        pv = PV(
            dates=dates,
            power_inst=5000,
            efficiency_inv=0.97,
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        assert pv.surface_tilt == 30
        assert pv.surface_azimuth == 180
        assert pv.cosphi == pytest.approx(0.9)

    def test_pv_custom_tilt_azimuth(self, dates: pd.DatetimeIndex) -> None:
        """Test PV with custom tilt and azimuth."""
        pv = PV(
            dates=dates,
            power_inst=5000,
            efficiency_inv=0.97,
            lat=51.05,
            lon=13.73,
            alt=100,
            surface_tilt=45,
            surface_azimuth=90,
        )
        assert pv.surface_tilt == 45
        assert pv.surface_azimuth == 90

    def test_pv_output_dataframes(self, dates: pd.DatetimeIndex) -> None:
        """Test PV output DataFrames are initialized."""
        pv = PV(
            dates=dates,
            power_inst=10000,
            efficiency_inv=0.95,
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        assert hasattr(pv, "acp")
        assert hasattr(pv, "acq")
        assert len(pv.acp) == len(dates)
        assert len(pv.acq) == len(dates)


class TestPVRun:
    """Test PV run method."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing (shorter for faster tests)."""
        return Dates.date_range(tz=TZ, year=2024)[:96]  # 1 day

    @pytest.fixture
    def weather(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic weather data for one day."""
        n = len(dates)
        rng = np.random.default_rng(0)
        # Simulate daily solar pattern
        hours = np.array([d.hour + d.minute / 60 for d in dates])
        solar_factor = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        return pd.DataFrame(
            {
                "ghi": 800 * solar_factor,
                "dni": 600 * solar_factor,
                "dhi": 200 * solar_factor,
                "temp_air": 15 + 10 * np.sin((hours - 6) * np.pi / 12),
                "wind_speed": 3 + rng.random(n),
            },
            index=dates,
        )

    def test_pv_run_produces_output(self, dates: pd.DatetimeIndex, weather: pd.DataFrame) -> None:
        """Test that PV run produces non-zero output."""
        pv = PV(
            dates=dates,
            power_inst=10000,
            efficiency_inv=0.95,
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        pv.run(weather).unwrap()

        # PV output should be negative (generation convention)
        # At least some values should be non-zero during sunny hours
        assert np.any(pv.acp["low"]["L1"].to_numpy() != 0)

    def test_pv_output_within_limits(self, dates: pd.DatetimeIndex, weather: pd.DataFrame) -> None:
        """Test PV output is within rated power limits."""
        power_inst = 10000
        pv = PV(
            dates=dates,
            power_inst=power_inst,
            efficiency_inv=0.95,
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        pv.run(weather).unwrap()

        # Output should not exceed rated power (negative values for generation)
        assert np.all(pv.acp["low"]["L1"].to_numpy() >= -power_inst)


class TestPVFromEfficiency:
    """Test PV creation from efficiency and area."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    def test_from_efficiency_and_area(self, dates: pd.DatetimeIndex) -> None:
        """Test creating PV from efficiency and area."""
        pv = PV.from_efficiency_and_area(
            dates=dates,
            efficiency=0.20,
            efficiency_inv=0.95,
            area=50,  # 50 m²
            lat=51.05,
            lon=13.73,
            alt=100,
        )
        # 50 m² * 0.20 * 1000 W/m² = 10 kW
        assert pv.power_inst == pytest.approx(10000, rel=0.01)
