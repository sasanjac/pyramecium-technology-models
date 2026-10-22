# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for Power2Heat technology model."""

from __future__ import annotations

from typing import TYPE_CHECKING
import zoneinfo as zi

import numpy as np
import pytest

from pstm.p2h import Power2Heat
from pstm.utils.dates import Dates

if TYPE_CHECKING:
    import pandas as pd

TZ = zi.ZoneInfo("Europe/Berlin")


class TestPower2HeatCreation:
    """Test Power2Heat instance creation."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    def test_p2h_room_heating(self, dates: pd.DatetimeIndex) -> None:
        """Test Power2Heat creation for room heating."""
        p2h = Power2Heat(
            dates=dates,
            power_inst=10000,
            efficiency=0.95,
            heating_type="room",
        )
        assert p2h.power_inst == 10000
        assert p2h.efficiency == pytest.approx(0.95)
        assert p2h.heating_type == "room"

    def test_p2h_process_low(self, dates: pd.DatetimeIndex) -> None:
        """Test Power2Heat creation for low-temp process heating."""
        p2h = Power2Heat(
            dates=dates,
            power_inst=50000,
            efficiency=0.92,
            heating_type="process_low",
        )
        assert p2h.heating_type == "process_low"

    def test_p2h_process_high(self, dates: pd.DatetimeIndex) -> None:
        """Test Power2Heat creation for high-temp process heating."""
        p2h = Power2Heat(
            dates=dates,
            power_inst=100000,
            efficiency=0.88,
            heating_type="process_high",
        )
        assert p2h.heating_type == "process_high"


class TestPower2HeatOutputs:
    """Test Power2Heat thermal and electrical outputs."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    def test_room_heating_thr_output(self, dates: pd.DatetimeIndex) -> None:
        """Test room heating sets thr output correctly."""
        power_inst = 10000
        efficiency = 0.95
        p2h = Power2Heat(
            dates=dates,
            power_inst=power_inst,
            efficiency=efficiency,
            heating_type="room",
        )
        # thr.low should be negative (heat output)
        expected_thermal = -power_inst * efficiency
        assert np.all(p2h.thr["low"].to_numpy() == expected_thermal)

    def test_process_low_thl_output(self, dates: pd.DatetimeIndex) -> None:
        """Test process_low heating sets thl output correctly."""
        power_inst = 50000
        efficiency = 0.92
        p2h = Power2Heat(
            dates=dates,
            power_inst=power_inst,
            efficiency=efficiency,
            heating_type="process_low",
        )
        # thl.low should be negative (heat output)
        expected_thermal = -power_inst * efficiency
        assert np.all(p2h.thl["low"].to_numpy() == expected_thermal)

    def test_process_high_thh_output(self, dates: pd.DatetimeIndex) -> None:
        """Test process_high heating sets thh output correctly."""
        power_inst = 100000
        efficiency = 0.88
        p2h = Power2Heat(
            dates=dates,
            power_inst=power_inst,
            efficiency=efficiency,
            heating_type="process_high",
        )
        # thh.low should be negative (heat output)
        expected_thermal = -power_inst * efficiency
        assert np.all(p2h.thh["low"].to_numpy() == expected_thermal)


class TestPower2HeatElectricalConsumption:
    """Test Power2Heat electrical consumption."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    def test_acp_consumption(self, dates: pd.DatetimeIndex) -> None:
        """Test electrical power consumption (acp.high)."""
        power_inst = 10000
        p2h = Power2Heat(
            dates=dates,
            power_inst=power_inst,
            efficiency=0.95,
            heating_type="room",
        )
        # acp.high should be positive (consumption)
        assert np.all(p2h.acp["high"].to_numpy() == power_inst)

    def test_reactive_power_calculation(self, dates: pd.DatetimeIndex) -> None:
        """Test reactive power calculation from cosphi."""
        power_inst = 10000
        cosphi = 0.9
        p2h = Power2Heat(
            dates=dates,
            power_inst=power_inst,
            efficiency=0.95,
            heating_type="room",
            cosphi=cosphi,
        )
        # Q follows from P as P tan(arccos(cosphi))
        expected_q = power_inst * np.tan(np.arccos(cosphi))
        assert np.allclose(p2h.acq["high"].to_numpy(), expected_q)

    def test_default_cosphi(self, dates: pd.DatetimeIndex) -> None:
        """Test default power factor of 0.9."""
        p2h = Power2Heat(
            dates=dates,
            power_inst=10000,
            efficiency=0.95,
            heating_type="room",
        )
        assert p2h.cosphi == pytest.approx(0.9)


class TestPower2HeatInvalidInputs:
    """Test Power2Heat error handling."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    def test_invalid_heating_type(self, dates: pd.DatetimeIndex) -> None:
        """Test invalid heating type is an error."""
        p2h = Power2Heat.create(dates=dates, power_inst=10000, efficiency=0.95, heating_type="invalid")
        assert "is not a valid HeatingType" in str(p2h.unwrap_err())
