# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for the heat pump technology models (Ruhnau et al. 2019)."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pandas as pd
import pytest
from pyochain import Some

from pstm.heat_pump import EFFICIENCY_OFFSETS
from pstm.heat_pump import FIELD_CORRECTION
from pstm.heat_pump import MIN_TEMP_DIFF
from pstm.heat_pump import AirHeatPump
from pstm.heat_pump import BrineHeatPump
from pstm.heat_pump import Efficiency
from pstm.heat_pump import HeatingSystem
from pstm.heat_pump import IndustrialHeatPump
from pstm.heat_pump import WaterHeatPump
from pstm.heat_pump import sink_temperature
from pstm.utils.dates import date_range
from pstm.utils.ground import fit_surface_harmonic
from pstm.utils.ground import kusuda_ground_temperature
from pstm.utils.ground import kusuda_temperature

TZ = zi.ZoneInfo("Europe/Berlin")


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return date_range(tz=TZ, year=2024)[:96]


@pytest.fixture
def temp_series(dates: pd.DatetimeIndex) -> pd.Series:
    hours = np.array([d.hour + d.minute / 60 for d in dates])
    return pd.Series(5 + 10 * np.sin((hours - 6) * np.pi / 12), index=dates)


def _air(dates: pd.DatetimeIndex, **kwargs: object) -> AirHeatPump:
    params: dict[str, object] = {"power_inst": 5000, "heating_system": "floor", "efficiency": "normal"}
    params.update(kwargs)
    return AirHeatPump(dates=dates, tz=TZ, **params)  # type: ignore[arg-type]


class TestSinkTemperature:
    """Eq. (6) of Ruhnau et al. (2019)."""

    def test_heating_curves(self) -> None:
        temp = pd.Series([-10.0, 0.0, 10.0])
        assert np.allclose(sink_temperature(HeatingSystem.RADIATOR, temp).unwrap(), [50.0, 40.0, 30.0])
        assert np.allclose(sink_temperature(HeatingSystem.FLOOR, temp).unwrap(), [35.0, 30.0, 25.0])
        assert np.allclose(sink_temperature(HeatingSystem.WATER, temp).unwrap(), [50.0, 50.0, 50.0])

    def test_unknown_system_is_err(self) -> None:
        error = sink_temperature("steam", pd.Series([0.0])).unwrap_err()  # type: ignore[arg-type]
        assert "Unknown heating system" in str(error)

    def test_class_converts_the_name_of_a_system(self, dates: pd.DatetimeIndex) -> None:
        assert _air(dates, heating_system="radiator").heating_system is HeatingSystem.RADIATOR
        assert _air(dates, efficiency="high").efficiency is Efficiency.HIGH


class TestCreate:
    """``create`` returns the error of a converter or validator as ``Err``."""

    def test_valid_arguments_are_ok(self, dates: pd.DatetimeIndex) -> None:
        hp = AirHeatPump.create(dates=dates, tz=TZ, power_inst=1, heating_system="floor", efficiency="normal")
        assert isinstance(hp.unwrap(), AirHeatPump)

    def test_unknown_system_is_err(self, dates: pd.DatetimeIndex) -> None:
        hp = AirHeatPump.create(dates=dates, tz=TZ, power_inst=1, heating_system="steam", efficiency="normal")
        assert "is not a valid HeatingSystem" in str(hp.unwrap_err())

    def test_constant_without_sink_temp_is_err(self, dates: pd.DatetimeIndex) -> None:
        hp = AirHeatPump.create(dates=dates, tz=TZ, power_inst=1, heating_system="constant", efficiency="normal")
        assert "sink_temp is required" in str(hp.unwrap_err())


class TestConstantSink:
    """Constant sink at ``sink_temp``, outside the residential scope of the source."""

    def test_sink_is_sink_temp_at_every_step(self) -> None:
        temp = pd.Series([-10.0, 0.0, 10.0])
        assert np.allclose(sink_temperature(HeatingSystem.CONSTANT, temp, Some(65.0)).unwrap(), [65.0, 65.0, 65.0])

    def test_temperature_difference_and_cop(self, dates: pd.DatetimeIndex) -> None:
        ground = pd.Series(8.0, index=dates)
        hp = BrineHeatPump(
            dates=dates,
            tz=TZ,
            power_inst=1,
            heating_system="constant",
            sink_temp=Some(65.0),
            efficiency="high",
            ground_temp=Some(ground),
        )
        temp = pd.Series(0.0, index=dates)
        temp_diff = hp.calc_temp_diff(temp).unwrap()
        assert np.allclose(temp_diff, 65.0 - (8.0 - 5.0))
        curve = 10.29 - 0.21 * 62.0 + 0.0012 * 62.0**2
        assert np.allclose(hp.calc_cop(temp_diff), FIELD_CORRECTION * curve + EFFICIENCY_OFFSETS["high"])

    def test_lower_bound_of_the_temperature_difference(self, dates: pd.DatetimeIndex) -> None:
        hp = _air(dates, heating_system="constant", sink_temp=Some(20.0))
        assert np.allclose(hp.calc_temp_diff(pd.Series([15.0, 30.0])).unwrap(), MIN_TEMP_DIFF)

    def test_constant_without_sink_temp_raises_in_init(self, dates: pd.DatetimeIndex) -> None:
        with pytest.raises(ValueError, match="sink_temp is required"):
            _air(dates, heating_system="constant")

    @pytest.mark.parametrize("heating_system", ["radiator", "floor", "water"])
    def test_sink_temp_on_another_sink_is_err(self, dates: pd.DatetimeIndex, heating_system: str) -> None:
        hp = AirHeatPump.create(
            dates=dates,
            tz=TZ,
            power_inst=1,
            heating_system=heating_system,
            efficiency="normal",
            sink_temp=Some(65.0),
        )
        assert "sink_temp is required" in str(hp.unwrap_err())

    def test_function_rejects_a_mismatched_sink_temp(self) -> None:
        assert "sink_temp is required" in str(sink_temperature(HeatingSystem.CONSTANT, pd.Series([0.0])).unwrap_err())
        mismatch = sink_temperature(HeatingSystem.FLOOR, pd.Series([0.0]), Some(65.0))
        assert "sink_temp is required" in str(mismatch.unwrap_err())

    def test_constant_sink_writes_thr(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        hp = _air(dates, heating_system="constant", sink_temp=Some(65.0))
        hp.run(temp_series).unwrap()
        assert np.all(hp.thr["low"].to_numpy() < 0)
        assert np.all(hp.thw["low"].to_numpy() == 0)
        assert hp.th is hp.thr


class TestResidentialCop:
    """COP = 0.85 * f(max(T_sink - T_source, 15 K)) + class offset."""

    def test_normal_class_is_the_source(self, dates: pd.DatetimeIndex) -> None:
        dt = pd.Series([30.0])
        assert _air(dates).calc_cop(dt).iloc[0] == pytest.approx(0.85 * (6.08 - 0.09 * 30 + 0.0005 * 900))
        brine = BrineHeatPump(dates=dates, tz=TZ, power_inst=1, heating_system="floor", efficiency="normal")
        assert brine.calc_cop(dt).iloc[0] == pytest.approx(0.85 * (10.29 - 0.21 * 30 + 0.0012 * 900))
        water = WaterHeatPump(dates=dates, tz=TZ, power_inst=1, heating_system="floor", efficiency="normal")
        assert water.calc_cop(dt).iloc[0] == pytest.approx(0.85 * (9.97 - 0.20 * 30 + 0.0012 * 900))

    def test_efficiency_class_adds_offset_after_correction(self, dates: pd.DatetimeIndex) -> None:
        dt = pd.Series([25.0])
        normal = _air(dates).calc_cop(dt).iloc[0]
        for cls, offset in EFFICIENCY_OFFSETS.items():
            assert _air(dates, efficiency=cls).calc_cop(dt).iloc[0] == pytest.approx(normal + offset)
        assert (
            _air(dates, efficiency="low").calc_cop(dt).iloc[0]
            < normal
            < _air(dates, efficiency="high").calc_cop(dt).iloc[0]
        )

    def test_temp_diff_bounded_below_by_15_k(self, dates: pd.DatetimeIndex) -> None:
        # Floor heating at 20 degC ambient: sink 20 degC, source 20 degC, lift 0 K -> 15 K.
        warm = pd.Series([20.0, -10.0])
        dt = _air(dates).calc_temp_diff(warm).unwrap()
        assert dt.iloc[0] == MIN_TEMP_DIFF
        assert dt.iloc[1] == pytest.approx(35.0 - (-10.0))

    def test_radiator_lift_above_floor(self, dates: pd.DatetimeIndex) -> None:
        temp = pd.Series([-5.0])
        radiator = _air(dates, heating_system="radiator").calc_temp_diff(temp).unwrap().iloc[0]
        floor = _air(dates, heating_system="floor").calc_temp_diff(temp).unwrap().iloc[0]
        assert radiator == pytest.approx(45.0 + 5.0)
        assert floor == pytest.approx(32.5 + 5.0)

    def test_water_source_is_constant_5_degc(self, dates: pd.DatetimeIndex) -> None:
        hp = WaterHeatPump(dates=dates, tz=TZ, power_inst=1, heating_system="water", efficiency="normal")
        dt = hp.calc_temp_diff(pd.Series([-10.0, 15.0])).unwrap()
        assert np.allclose(dt, [45.0, 45.0])

    def test_brine_uses_given_ground_temperature(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        ground = pd.Series(8.0, index=dates)
        hp = BrineHeatPump(
            dates=dates,
            tz=TZ,
            power_inst=1,
            heating_system="water",
            efficiency="normal",
            ground_temp=Some(ground),
        )
        assert np.allclose(hp.calc_temp_diff(temp_series).unwrap(), 50.0 - (8.0 - 5.0))

    def test_brine_without_soil_series_needs_a_year(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        hp = BrineHeatPump(dates=dates, tz=TZ, power_inst=1, heating_system="floor", efficiency="normal")
        assert "at least 365 days" in str(hp.run(temp_series).unwrap_err())

    def test_field_correction_constant(self) -> None:
        assert FIELD_CORRECTION == 0.85


class TestRun:
    def test_run_without_demand(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        hp = _air(dates)
        hp.run(temp_series).unwrap()
        cop = -hp.thr["low"].to_numpy() / hp.acp["high"]["L1"].to_numpy()
        expected = hp.calc_cop(hp.calc_temp_diff(temp_series).unwrap()).to_numpy()
        assert np.allclose(cop, expected)
        assert np.all(hp.thw["low"].to_numpy() == 0)

    def test_run_with_demand(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        demand = pd.Series(np.full(len(dates), 3000.0), index=dates)
        hp = _air(dates)
        hp.run(temp_series, Some(demand)).unwrap()
        assert np.allclose(hp.thr["low"].to_numpy(), -demand.to_numpy())

    def test_water_heating_writes_thw(self, dates: pd.DatetimeIndex, temp_series: pd.Series) -> None:
        hp = _air(dates, heating_system="water")
        hp.run(temp_series).unwrap()
        assert np.all(hp.thw["low"].to_numpy() < 0)
        assert np.all(hp.thr["low"].to_numpy() == 0)
        assert hp.th is hp.thw

    def test_brine_full_year_run(self) -> None:
        dates = date_range(tz=TZ, year=2023)
        day = np.arange(len(dates)) / 24
        temp = pd.Series(9 - 10 * np.cos(2 * np.pi * (day - 15) / 365), index=dates)
        hp = BrineHeatPump(dates=dates, tz=TZ, power_inst=1000, heating_system="floor", efficiency="normal")
        hp.run(temp).unwrap()
        cop = -hp.thr["low"].to_numpy()
        air = AirHeatPump(dates=dates, tz=TZ, power_inst=1000, heating_system="floor", efficiency="normal")
        air.run(temp).unwrap()
        # January: ground warmer than air, so the brine COP exceeds the air COP.
        assert cop[: 24 * 31].mean() > -air.thr["low"].to_numpy()[: 24 * 31].mean()

    def test_cosphi_default(self, dates: pd.DatetimeIndex) -> None:
        assert _air(dates).cosphi == 0.7


class TestIndustrialHeatPump:
    def test_constant_target_temperature_kept(self, dates: pd.DatetimeIndex) -> None:
        hp = IndustrialHeatPump(dates=dates, tz=TZ, power_inst=1, target_temp=90, efficiency="high")
        dt = hp.calc_temp_diff(pd.Series([10.0])).unwrap()
        assert dt.iloc[0] == 80.0
        assert hp.calc_cop(dt).iloc[0] == pytest.approx(80.84 / 80.0**0.76)


class TestKusuda:
    """Eq. (3) of Kusuda and Achenbach (1965)."""

    def test_hand_computed_value(self) -> None:
        # D = 0.025 ft2/h = 0.025 * 0.09290304 * 24 = 0.0557418 m2/d, tau = 365 d, z = 1.945 m.
        # damping = exp(-1.945 * sqrt(pi / (0.0557418 * 365))) = 0.465665
        # lag     = 1.945 / 2 * sqrt(365 / (pi * 0.0557418))  = 44.3987 d
        # T(100)  = 10 - 10 * 0.465665 * cos(2 pi / 365 * (100 - 30 - 44.3987)) = 5.7883 degC
        value = kusuda_temperature(100.0, mean=10.0, amplitude=10.0, phase=30.0)
        assert float(value) == pytest.approx(5.7883, abs=1e-4)

    def test_surface_is_undamped(self) -> None:
        value = kusuda_temperature(30.0, mean=10.0, amplitude=10.0, phase=30.0, depth=0.0)
        assert float(value) == pytest.approx(0.0)

    def test_fit_recovers_harmonic(self) -> None:
        dates = date_range(tz=zi.ZoneInfo("UTC"), year=2023)
        day = (dates.dayofyear.to_numpy() - 1) + dates.hour.to_numpy() / 24
        temp = pd.Series(9.5 - 9.0 * np.cos(2 * np.pi / 365 * (day - 20.0)), index=dates)
        mean, amplitude, phase = fit_surface_harmonic(temp).unwrap()
        assert mean == pytest.approx(9.5, abs=1e-2)
        assert amplitude == pytest.approx(9.0, abs=1e-2)
        assert phase == pytest.approx(20.0, abs=0.1)

    def test_fit_needs_a_year(self) -> None:
        temp = pd.Series(0.0, index=date_range(tz=zi.ZoneInfo("UTC"), year=2023)[: 24 * 364])
        assert "at least 365 days" in str(fit_surface_harmonic(temp).unwrap_err())
        assert kusuda_ground_temperature(temp).is_err()

    def test_ground_series_is_damped_and_delayed(self) -> None:
        dates = date_range(tz=zi.ZoneInfo("UTC"), year=2023)
        day = (dates.dayofyear.to_numpy() - 1) + dates.hour.to_numpy() / 24
        temp = pd.Series(9.5 - 9.0 * np.cos(2 * np.pi / 365 * (day - 20.0)), index=dates)
        ground = kusuda_ground_temperature(temp).unwrap()
        assert ground.max() - ground.min() == pytest.approx(2 * 9.0 * 0.465665, abs=0.05)
        assert ground.idxmin().dayofyear - 1 == pytest.approx(20 + 44.4, abs=1.5)
