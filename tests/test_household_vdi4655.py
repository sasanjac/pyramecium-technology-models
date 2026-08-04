# Copyright (c) 2018-2025 Sasan Jacob Rasti
"""Unit tests for VDI4655 Household model."""

from __future__ import annotations

import pathlib

import pytest

from pstm.household_vdi4655 import (
    ELECTRICITY_DEMAND,
    ELECTRICITY_DEMAND_MFH,
    ELECTRICITY_DEMAND_OFH,
    INSULATION_STANDARDS,
    POWER_CONVERSION_FACTORS,
    POWER_FACTORS,
    WATER_DEMAND,
)


class TestElectricityDemandConstants:
    """Test electricity demand constants."""

    def test_ofh_demand_values(self) -> None:
        """Test OFH electricity demand per number of units."""
        assert ELECTRICITY_DEMAND_OFH[1] == 2350
        assert ELECTRICITY_DEMAND_OFH[2] == 2020
        assert ELECTRICITY_DEMAND_OFH[3] == 1650
        assert ELECTRICITY_DEMAND_OFH[4] == 1500
        assert ELECTRICITY_DEMAND_OFH[5] == 1400
        assert ELECTRICITY_DEMAND_OFH[6] == 1350

    def test_mfh_demand_constant(self) -> None:
        """Test MFH electricity demand is constant."""
        assert ELECTRICITY_DEMAND_MFH == 3000

    def test_electricity_demand_ofh_default(self) -> None:
        """Test OFH electricity demand default for >6 units."""
        # Should return the value for 6 units for any value > 6
        assert ELECTRICITY_DEMAND["OFH"][6] == 1350
        assert ELECTRICITY_DEMAND["OFH"][10] == 1350
        assert ELECTRICITY_DEMAND["OFH"][100] == 1350


class TestWaterDemand:
    """Test water demand constants."""

    def test_ofh_water_demand(self) -> None:
        """Test OFH water demand."""
        assert WATER_DEMAND["OFH"] == 500

    def test_mfh_water_demand(self) -> None:
        """Test MFH water demand."""
        assert WATER_DEMAND["MFH"] == 1000


class TestInsulationStandards:
    """Test insulation standard ranges."""

    def test_unrenovated_range(self) -> None:
        """Test unrenovated building range."""
        low, high = INSULATION_STANDARDS["unrenovated"]
        assert low == 250
        assert high == 300

    def test_leh_range(self) -> None:
        """Test LEH (low energy house) range."""
        low, high = INSULATION_STANDARDS["LEH"]
        assert low == 55
        assert high == 70

    def test_passive_house_range(self) -> None:
        """Test passive house range."""
        low, high = INSULATION_STANDARDS["PH"]
        assert low == 0
        assert high == 15

    def test_all_standards_present(self) -> None:
        """Test all insulation standards are defined."""
        expected = {
            "unrenovated",
            "WSVO77",
            "WSVO82",
            "WSVO95",
            "LEH",
            "KfW85",
            "KfW70",
            "KfW55",
            "KfW40",
            "PH",
        }
        assert set(INSULATION_STANDARDS.keys()) == expected

    def test_ranges_are_descending(self) -> None:
        """Test insulation standards have descending ranges (better insulation = lower demand)."""
        standards_ordered = [
            "unrenovated",
            "WSVO77",
            "WSVO82",
            "WSVO95",
            "LEH",
            "KfW85",
            "KfW70",
            "KfW55",
            "KfW40",
            "PH",
        ]
        prev_high = float("inf")
        for standard in standards_ordered:
            _, high = INSULATION_STANDARDS[standard]
            assert high < prev_high
            prev_high = high


class TestPowerConversionFactors:
    """Test power conversion factors."""

    def test_ofh_conversion_factor(self) -> None:
        """Test OFH power conversion factor."""
        # OFH: x W = y kWh * 60 min/h * 60 s/min / 2 s * 1000 W/kW
        assert POWER_CONVERSION_FACTORS["OFH"] == 1_800_000

    def test_mfh_conversion_factor(self) -> None:
        """Test MFH power conversion factor."""
        # MFH: x W = y kWh * 60 min/h / 15 min * 1000 W/kW
        assert POWER_CONVERSION_FACTORS["MFH"] == 4_000


class TestPowerFactors:
    """Test power factor parameters."""

    def test_capacitive_factors(self) -> None:
        """Test capacitive power factor parameters."""
        assert POWER_FACTORS[0]["a"] == 0.99
        assert POWER_FACTORS[0]["b"] == 1.605

    def test_inductive_factors(self) -> None:
        """Test inductive power factor parameters."""
        assert POWER_FACTORS[1]["a"] == 0.99
        assert POWER_FACTORS[1]["b"] == 4.904

    def test_two_factor_types(self) -> None:
        """Test exactly two factor types exist (capacitive and inductive)."""
        assert len(POWER_FACTORS) == 2
