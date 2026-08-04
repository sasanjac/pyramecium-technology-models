# Copyright (c) 2018-2025 Sasan Jacob Rasti
"""Unit tests for Heat Pump technology models."""

from __future__ import annotations

import datetime as dt
import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest

from pstm.heat_pump import AirHeatPump
from pstm.heat_pump import BrineHeatPump
from pstm.utils.dates import date_range

TZ = zi.ZoneInfo("Europe/Berlin")


class TestAirHeatPump:
    """Test Air Heat Pump model."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]  # 1 day

    @pytest.fixture
    def temp_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate temperature series."""
        hours = np.array([d.hour + d.minute / 60 for d in dates])
        temp = 5 + 10 * np.sin((hours - 6) * np.pi / 12)  # Vary between -5 and 15°C
        return pd.Series(temp, index=dates)

    @pytest.fixture
    def thermal_demand(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate thermal demand series."""
        return pd.Series(np.full(len(dates), 3000.0), index=dates)  # 3 kW constant

    def test_air_hp_creation_low_efficiency(self, dates: pd.DatetimeIndex) -> None:
        """Test Air HP creation with low efficiency."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="low",
            tz=TZ,
        )
        assert hp.power_inst == 5000
        assert hp.target_temp == 35
        assert hp.efficiency == "low"

    def test_air_hp_creation_high_efficiency(self, dates: pd.DatetimeIndex) -> None:
        """Test Air HP creation with high efficiency."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=8000,
            target_temp=45,
            efficiency="high",
            tz=TZ,
        )
        assert hp.efficiency == "high"

    def test_air_hp_run_without_demand(
        self, dates: pd.DatetimeIndex, temp_series: pd.Series
    ) -> None:
        """Test Air HP run without demand profile."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        hp.run(temp_series)

        # Thermal output should be non-zero
        assert np.any(hp.thr["low"].values != 0)
        # Electrical consumption should be non-zero
        assert np.any(hp.acp["high"]["L1"].values != 0)

    def test_air_hp_run_with_demand(
        self,
        dates: pd.DatetimeIndex,
        temp_series: pd.Series,
        thermal_demand: pd.Series,
    ) -> None:
        """Test Air HP run with demand profile."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        hp.run(temp_series, thermal_demand)

        # Thermal output should match demand
        assert np.allclose(hp.thr["low"].values, -thermal_demand.values)

    def test_air_hp_cop_calculation(self, dates: pd.DatetimeIndex) -> None:
        """Test COP is calculated correctly."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        # COP should increase as temp difference decreases
        temp_diff_small = pd.Series([10.0], index=dates[:1])
        temp_diff_large = pd.Series([30.0], index=dates[:1])

        cop_small = hp.calc_cop(temp_diff_small).iloc[0]
        cop_large = hp.calc_cop(temp_diff_large).iloc[0]

        assert cop_small > cop_large  # Higher COP at smaller temp diff


class TestBrineHeatPump:
    """Test Brine (Ground Source) Heat Pump model."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def temp_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate temperature series."""
        hours = np.array([d.hour + d.minute / 60 for d in dates])
        temp = 10 + 5 * np.sin((hours - 6) * np.pi / 12)  # Ground source: more stable
        return pd.Series(temp, index=dates)

    def test_brine_hp_creation(self, dates: pd.DatetimeIndex) -> None:
        """Test Brine HP creation."""
        hp = BrineHeatPump(
            dates=dates,
            power_inst=6000,
            target_temp=35,
            efficiency="high",
            tz=TZ,
        )
        assert hp.power_inst == 6000
        assert hp.efficiency == "high"

    def test_brine_hp_run(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        """Test Brine HP run."""
        hp = BrineHeatPump(
            dates=dates,
            power_inst=6000,
            target_temp=35,
            efficiency="high",
            tz=TZ,
        )
        hp.run(temp_series)

        # Should produce thermal output
        assert np.any(hp.thr["low"].values != 0)

    def test_brine_hp_higher_cop_than_air(self, dates: pd.DatetimeIndex) -> None:
        """Test Brine HP has higher COP than Air HP at same temp diff."""
        air_hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        brine_hp = BrineHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )

        temp_diff = pd.Series([20.0], index=dates[:1])

        cop_air = air_hp.calc_cop(temp_diff).iloc[0]
        cop_brine = brine_hp.calc_cop(temp_diff).iloc[0]

        # Brine HP should have higher COP (more efficient ground source)
        assert cop_brine > cop_air


class TestHeatPumpEfficiencyLevels:
    """Test heat pump efficiency levels."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    def test_efficiency_levels_affect_cop(self, dates: pd.DatetimeIndex) -> None:
        """Test that efficiency levels affect COP."""
        hp_low = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="low",
            tz=TZ,
        )
        hp_normal = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        hp_high = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="high",
            tz=TZ,
        )

        temp_diff = pd.Series([25.0], index=dates[:1])

        cop_low = hp_low.calc_cop(temp_diff).iloc[0]
        cop_normal = hp_normal.calc_cop(temp_diff).iloc[0]
        cop_high = hp_high.calc_cop(temp_diff).iloc[0]

        assert cop_low < cop_normal < cop_high

    def test_cosphi_default(self, dates: pd.DatetimeIndex) -> None:
        """Test default cosphi value."""
        hp = AirHeatPump(
            dates=dates,
            power_inst=5000,
            target_temp=35,
            efficiency="normal",
            tz=TZ,
        )
        assert hp.cosphi == 0.7  # Default for heat pumps
