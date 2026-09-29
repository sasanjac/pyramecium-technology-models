# Copyright (c) 2018-2025 Sasan Jacob Rasti
"""Unit tests for Wind technology model."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest

from pstm.wind import MissingArgumentsError
from pstm.wind import Wind
from pstm.utils.dates import date_range

TZ = zi.ZoneInfo("Europe/Berlin")


class TestWindCreation:
    """Test Wind turbine instance creation."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    def test_wind_creation_with_turbine_type(self, dates: pd.DatetimeIndex) -> None:
        """Test Wind creation with turbine type."""
        wind = Wind(
            dates=dates,
            hub_height=135,
            turbine_type="E-126/4200",
            lat=53.5,
            lon=9.5,
        )
        assert wind.hub_height == 135
        assert wind.turbine_type == "E-126/4200"

    def test_wind_creation_with_power_inst(self, dates: pd.DatetimeIndex) -> None:
        """Test Wind creation with power_inst."""
        wind = Wind(
            dates=dates,
            hub_height=100,
            power_inst=3000000,  # 3 MW
            lat=53.5,
            lon=9.5,
        )
        assert wind.power_inst == 3000000

    def test_wind_creation_missing_arguments(self, dates: pd.DatetimeIndex) -> None:
        """Test Wind creation fails without turbine type or power_inst."""
        with pytest.raises(MissingArgumentsError):
            Wind(
                dates=dates,
                hub_height=100,
                lat=53.5,
                lon=9.5,
            )

    def test_wind_default_cosphi(self, dates: pd.DatetimeIndex) -> None:
        """Test default cosphi value."""
        wind = Wind(
            dates=dates,
            hub_height=100,
            power_inst=2000000,
            lat=53.5,
            lon=9.5,
        )
        assert wind.cosphi == 0.9


class TestWindRun:
    """Test Wind run method."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def weather(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic weather data.
        
        windpowerlib requires Multi-level column index with (variable, height).
        """
        n = len(dates)
        # Simulate varying wind speeds
        wind_speed = 8 + 4 * np.sin(np.linspace(0, 4 * np.pi, n))
        # Create multi-level columns as required by windpowerlib
        columns = pd.MultiIndex.from_tuples([
            ("wind_speed", 10),
            ("temperature", 2),
            ("pressure", 0),
            ("roughness_length", 0),
        ])
        data = np.column_stack([
            wind_speed,  # wind_speed m/s
            15 + 5 * np.random.rand(n) + 273.15,  # temperature K
            101300 + 1000 * np.random.rand(n),  # pressure Pa
            np.full(n, 0.1),  # roughness_length m
        ])
        return pd.DataFrame(data, index=dates, columns=columns)

    def test_wind_run_produces_output(
        self, dates: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> None:
        """Test that Wind run produces output."""
        wind = Wind(
            dates=dates,
            hub_height=100,
            power_inst=2000000,
            lat=53.5,
            lon=9.5,
        )
        wind.run(weather)
        
        # Wind output should be non-zero (negative for generation)
        assert np.any(wind.acp["low"]["L1"].values != 0)

    def test_wind_output_within_rated_power(
        self, dates: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> None:
        """Test Wind output is within rated power limits."""
        power_inst = 2000000
        wind = Wind(
            dates=dates,
            hub_height=100,
            power_inst=power_inst,
            lat=53.5,
            lon=9.5,
        )
        wind.run(weather)
        
        # Output (negative) should not exceed rated power
        assert np.all(wind.acp["low"]["L1"].values >= -power_inst * 1.1)  # 10% margin

    def test_wind_reactive_power(
        self, dates: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> None:
        """Test Wind reactive power is calculated."""
        wind = Wind(
            dates=dates,
            hub_height=100,
            power_inst=2000000,
            lat=53.5,
            lon=9.5,
            cosphi=0.9,
        )
        wind.run(weather)
        
        # Reactive power should be calculated
        assert np.any(wind.acq["low"]["L1"].values != 0)


class TestWindSiteIndependence:
    """The output depends on the weather at the site, not on its latitude.

    The former full-load-hour correction subtracted 1.692 h per km of latitude
    distance to the Elbe mouth. It took one coefficient of the multilinear
    regression of Pflugfelder, Kramer and Weber (Applied Energy 361, 2024)
    without its intercept and the other regressors, with the opposite sign of
    the distance effect, and applied an ERA5-calibrated bias to a TRY-based
    model chain. It is removed.
    """

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def weather(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate synthetic weather data.
        
        windpowerlib requires Multi-level column index with (variable, height).
        """
        n = len(dates)
        columns = pd.MultiIndex.from_tuples([
            ("wind_speed", 10),
            ("temperature", 2),
            ("pressure", 0),
            ("roughness_length", 0),
        ])
        data = np.column_stack([
            np.full(n, 10.0),  # Constant wind m/s
            np.full(n, 15.0 + 273.15),  # temperature K
            np.full(n, 101300.0),  # pressure Pa
            np.full(n, 0.1),  # roughness_length m
        ])
        return pd.DataFrame(data, index=dates, columns=columns)

    def test_identical_weather_gives_identical_output(
        self, dates: pd.DatetimeIndex, weather: pd.DataFrame
    ) -> None:
        """Two sites under the same weather generate the same energy."""
        # Coastal location (near Elbe mouth)
        wind_coastal = Wind(
            dates=dates,
            hub_height=100,
            power_inst=2000000,
            lat=53.86,
            lon=9.5,
        )
        
        # Inland location (southern Germany)
        wind_inland = Wind(
            dates=dates,
            hub_height=100,
            power_inst=2000000,
            lat=48.0,
            lon=11.5,
        )
        
        # Both should produce output without errors
        wind_coastal.run(weather)
        wind_inland.run(weather)

        coastal_generation = wind_coastal.acp["low"]["L1"].sum()
        inland_generation = wind_inland.acp["low"]["L1"].sum()
        assert coastal_generation < 0
        assert coastal_generation == pytest.approx(inland_generation, rel=1e-12)


class TestWindModelChainData:
    """The model chain of a turbine follows its ``modelchain_data``."""

    def test_default_is_the_hellman_chain(self) -> None:
        wind = Wind(dates=date_range(tz=TZ, year=2023)[:4], hub_height=100, power_inst=2e6, lat=54.8, lon=13.8)
        assert wind.mc.wind_speed_model == "hellman"

    def test_a_multi_height_chain_interpolates_between_levels(self) -> None:
        from pstm.wind import MODELCHAIN_DATA

        dates = date_range(tz=TZ, year=2023)[:4]
        data = {**MODELCHAIN_DATA, "wind_speed_model": "log_interpolation_extrapolation"}
        wind = Wind(dates=dates, hub_height=100, power_inst=2e6, lat=54.8, lon=13.8, modelchain_data=data)
        assert wind.mc.wind_speed_model == "log_interpolation_extrapolation"
        columns = pd.MultiIndex.from_tuples(
            [("wind_speed", 75), ("wind_speed", 150), ("temperature", 75), ("pressure", 75), ("roughness_length", 0)],
        )
        weather = pd.DataFrame([[6.0, 8.0, 283.0, 101_000.0, 0.1]] * 4, index=dates, columns=columns)
        v_hub = np.asarray(wind.mc.wind_speed_hub(weather), dtype=float)
        expected = 6.0 + (8.0 - 6.0) * np.log(100 / 75) / np.log(150 / 75)
        np.testing.assert_allclose(v_hub, expected)
