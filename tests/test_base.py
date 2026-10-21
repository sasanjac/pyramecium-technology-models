# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for the Tech base class."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest

from pstm.base import HeatingType
from pstm.base import Tech
from pstm.storage import Storage
from pstm.storage import StorageType
from pstm.utils.dates import Dates

TZ = zi.ZoneInfo("Europe/Berlin")


class TechCase:
    """Fixtures shared by the test classes of this module."""

    @pytest.fixture
    def dates(self) -> pd.DatetimeIndex:
        return Dates.date_range(tz=TZ, year=2024)[:96]


class TestHeat(TechCase):
    @pytest.mark.parametrize(
        ("heating_type", "frame"),
        [(HeatingType.ROOM, "thr"), (HeatingType.PROCESS_LOW, "thl"), (HeatingType.PROCESS_HIGH, "thh")],
    )
    def test_heat_selects_the_thermal_frame(
        self,
        dates: pd.DatetimeIndex,
        heating_type: HeatingType,
        frame: str,
    ) -> None:
        tech = Tech(dates=dates)
        assert tech.heat(heating_type) is getattr(tech, frame)


class TestResample(TechCase):
    def test_downsampling_onto_dates(self, dates: pd.DatetimeIndex) -> None:
        index = pd.date_range(dates[0], periods=4 * len(dates), freq="15min")
        target = pd.Series(np.repeat(np.arange(len(dates), dtype=np.float64), 4), index=index)
        resampled = Tech(dates=dates)._resample(target, index).unwrap()  # noqa: SLF001
        assert np.allclose(resampled.to_numpy(), np.arange(len(dates)))

    def test_downsampling_without_frequency_is_err(self, dates: pd.DatetimeIndex) -> None:
        irregular = pd.DatetimeIndex(list(dates))
        assert irregular.freq is None
        index = pd.date_range(dates[0], periods=4 * len(dates), freq="15min")
        target = pd.Series(np.zeros(len(index)), index=index)
        error = Tech(dates=irregular)._resample(target, index).unwrap_err()  # noqa: SLF001
        assert "frequency of the index is not defined" in str(error)


class TestCreate(TechCase):
    def test_create_converts_the_name_of_a_storage_type(self, dates: pd.DatetimeIndex) -> None:
        storage = Storage.create(
            dates=dates,
            power_inst_in=1,
            power_inst_out=1,
            capacity=1,
            efficiency_in=1,
            efficiency_out=1,
            self_discharge=0,
            storage_type="THR",
        ).unwrap()
        assert storage.storage_type is StorageType.THR

    def test_create_returns_an_unknown_storage_type_as_err(self, dates: pd.DatetimeIndex) -> None:
        storage = Storage.create(
            dates=dates,
            power_inst_in=1,
            power_inst_out=1,
            capacity=1,
            efficiency_in=1,
            efficiency_out=1,
            self_discharge=0,
            storage_type="XX",
        )
        assert "is not a valid StorageType" in str(storage.unwrap_err())
