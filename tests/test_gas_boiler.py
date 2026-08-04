# Copyright (c) 2018-2025 Sasan Jacob Rasti
"""Unit tests for Gas Boiler technology model."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pytest

from pstm.gas_boiler import GasBoiler
from pstm.utils.dates import date_range

TZ = zi.ZoneInfo("Europe/Berlin")


class TestGasBoilerCreation:
    """Test Gas Boiler instance creation."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    def test_gas_boiler_room_heating_ch4(self, dates) -> None:
        """Test Gas Boiler creation for room heating with CH4."""
        boiler = GasBoiler(
            dates=dates,
            power_inst=20000,
            efficiency=0.92,
            heating_type="room",
            gas_type="CH4",
        )
        assert boiler.power_inst == 20000
        assert boiler.efficiency == 0.92
        assert boiler.heating_type == "room"
        assert boiler.gas_type == "CH4"

    def test_gas_boiler_room_heating_h2(self, dates) -> None:
        """Test Gas Boiler creation for room heating with H2."""
        boiler = GasBoiler(
            dates=dates,
            power_inst=15000,
            efficiency=0.90,
            heating_type="room",
            gas_type="H2",
        )
        assert boiler.gas_type == "H2"

    def test_gas_boiler_process_low(self, dates) -> None:
        """Test Gas Boiler creation for low-temp process heating."""
        boiler = GasBoiler(
            dates=dates,
            power_inst=50000,
            efficiency=0.85,
            heating_type="process_low",
            gas_type="CH4",
        )
        assert boiler.heating_type == "process_low"
        # Should set thl
        assert np.all(boiler.thl["low"].values == -50000)

    def test_gas_boiler_process_high(self, dates) -> None:
        """Test Gas Boiler creation for high-temp process heating."""
        boiler = GasBoiler(
            dates=dates,
            power_inst=100000,
            efficiency=0.80,
            heating_type="process_high",
            gas_type="CH4",
        )
        assert boiler.heating_type == "process_high"
        # Should set thh
        assert np.all(boiler.thh["low"].values == -100000)


class TestGasBoilerOutputs:
    """Test Gas Boiler thermal and gas outputs."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    def test_room_heating_thr_output(self, dates) -> None:
        """Test room heating sets thr output."""
        boiler = GasBoiler(
            dates=dates,
            power_inst=20000,
            efficiency=0.92,
            heating_type="room",
            gas_type="CH4",
        )
        # Room heating should set thr.low to negative (heat output)
        assert np.all(boiler.thr["low"].values == -20000)

    def test_gas_consumption_ch4(self, dates) -> None:
        """Test CH4 gas consumption calculation."""
        power_inst = 20000
        efficiency = 0.92
        boiler = GasBoiler(
            dates=dates,
            power_inst=power_inst,
            efficiency=efficiency,
            heating_type="room",
            gas_type="CH4",
        )
        # CH4 consumption should be power / efficiency
        expected_gas = power_inst / efficiency
        assert np.all(boiler.ch4["high"].values == pytest.approx(expected_gas, rel=0.01))

    def test_gas_consumption_h2(self, dates) -> None:
        """Test H2 gas consumption calculation."""
        power_inst = 15000
        efficiency = 0.90
        boiler = GasBoiler(
            dates=dates,
            power_inst=power_inst,
            efficiency=efficiency,
            heating_type="room",
            gas_type="H2",
        )
        # H2 consumption should be power / efficiency
        expected_gas = power_inst / efficiency
        assert np.all(boiler.h2["high"].values == pytest.approx(expected_gas, rel=0.01))


class TestGasBoilerInvalidInputs:
    """Test Gas Boiler error handling."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return date_range(tz=TZ, year=2024)[:96]

    def test_invalid_heating_type(self, dates) -> None:
        """Test invalid heating type raises error."""
        with pytest.raises(ValueError):
            # Need to bypass attrs validation by using object.__setattr__
            boiler = GasBoiler(
                dates=dates,
                power_inst=20000,
                efficiency=0.92,
                heating_type="invalid",  # type: ignore
                gas_type="CH4",
            )

    def test_invalid_gas_type(self, dates) -> None:
        """Test invalid gas type raises error."""
        with pytest.raises(ValueError):
            boiler = GasBoiler(
                dates=dates,
                power_inst=20000,
                efficiency=0.92,
                heating_type="room",
                gas_type="invalid",  # type: ignore
            )
