# Copyright (c) 2018-2025 Sasan Jacob Rasti
"""Unit tests for generation module parameters and fixtures."""

from __future__ import annotations

import pytest

from pstm.generation import BUILD_TYPE, HP_PARAMS, HHParams
from pstm.heat_pump import AirHeatPump, BrineHeatPump


class TestHPParams:
    """Test heat pump parameter configurations."""

    def test_ofh_urban_2020(self) -> None:
        """Test OFH urban 2020 configuration."""
        params = HP_PARAMS["OFH"]["urban"][2020]
        assert params["efficiency"] == "low"
        assert params["hp_type"] == AirHeatPump

    def test_ofh_rural_2050(self) -> None:
        """Test OFH rural 2050 configuration."""
        params = HP_PARAMS["OFH"]["rural"][2050]
        assert params["efficiency"] == "high"
        assert params["hp_type"] == BrineHeatPump

    def test_mfh_urban_2040(self) -> None:
        """Test MFH urban 2040 configuration."""
        params = HP_PARAMS["MFH"]["urban"][2040]
        assert params["efficiency"] == "normal"
        assert params["hp_type"] == AirHeatPump

    def test_mfh_rural_2020(self) -> None:
        """Test MFH rural 2020 configuration."""
        params = HP_PARAMS["MFH"]["rural"][2020]
        assert params["efficiency"] == "normal"
        assert params["hp_type"] == BrineHeatPump

    def test_all_years_present_ofh_urban(self) -> None:
        """Test all target years are configured for OFH urban."""
        years = set(HP_PARAMS["OFH"]["urban"].keys())
        assert years == {2020, 2040, 2050}

    def test_all_years_present_mfh_rural(self) -> None:
        """Test all target years are configured for MFH rural."""
        years = set(HP_PARAMS["MFH"]["rural"].keys())
        assert years == {2020, 2040, 2050}


class TestBuildType:
    """Test building type configurations."""

    def test_ofh_2020(self) -> None:
        """Test OFH 2020 building type."""
        assert BUILD_TYPE["OFH"][2020] == "EH"

    def test_ofh_2040(self) -> None:
        """Test OFH 2040 building type."""
        assert BUILD_TYPE["OFH"][2040] == "LEH"

    def test_ofh_2050(self) -> None:
        """Test OFH 2050 building type."""
        assert BUILD_TYPE["OFH"][2050] == "LEH"

    def test_mfh_always_eh(self) -> None:
        """Test MFH always uses EH building type."""
        for year in [2020, 2040, 2050]:
            assert BUILD_TYPE["MFH"][year] == "EH"


class TestHHParams:
    """Test HHParams data class."""

    def test_hhparams_creation(self) -> None:
        """Test HHParams instance creation."""
        hp_params = {"efficiency": "high", "hp_type": BrineHeatPump}
        params = HHParams(
            house_type="OFH",
            hp_params=hp_params,
            build_type="LEH",
        )
        assert params.house_type == "OFH"
        assert params.hp_params == hp_params
        assert params.build_type == "LEH"

    def test_hhparams_mfh(self) -> None:
        """Test HHParams for MFH."""
        hp_params = {"efficiency": "normal", "hp_type": AirHeatPump}
        params = HHParams(
            house_type="MFH",
            hp_params=hp_params,
            build_type="EH",
        )
        assert params.house_type == "MFH"
        assert params.build_type == "EH"
