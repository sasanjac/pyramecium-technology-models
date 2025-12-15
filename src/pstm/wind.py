# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import collections.abc as cabc
import pathlib
import typing as t

import attrs
import loguru
import numpy as np
import pandas as pd
import windpowerlib as wpl
import windpowerlib.modelchain as wmc

from pstm.base import Tech

loguru.logger.disable("windpowerlib")

SRC_PATH = pathlib.Path(__file__).parent.parent.parent
DEFAULT_TURBINE = wpl.WindTurbine(turbine_type="E-126/4200", hub_height=135)
DEFAULT_POWER_CURVE_WIND = DEFAULT_TURBINE.power_curve["wind_speed"]
DEFAULT_POWER_CURVE_POWER = DEFAULT_TURBINE.power_curve["value"] / DEFAULT_TURBINE.power_curve["value"].max()

FLH_REDUCTION_PER_KM = 1.692  # [1] Y. Pflugfelder, H. Kramer, und C. Weber, „A novel approach to generate bias-corrected regional wind infeed timeseries based on reanalysis data“, Applied Energy, Bd. 361, S. 122890, Mai 2024, doi: 10.1016/j.apenergy.2024.122890.
FULL_LOAD_HOURS_MAX = 4_000
FULL_LOAD_HOURS_MIN = 1_500
LAT_BASE = 53.86446  # Elbe mouth, near Hamburg
LAT_IN_KM = 111.32

MODELCHAIN_DATA = {
    "wind_speed_model": "hellman",
    "density_model": "ideal_gas",
    "temperature_model": "linear_gradient",
    "power_output_model": "power_curve",
    "density_correction": True,
    "obstacle_height": 0,
    "hellman_exp": 1 / 7,
}


class MissingArgumentsError(ValueError):
    def __init__(self) -> None:
        super().__init__("Specify either turbine type or nominal power.")


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Wind(Tech):
    hub_height: float
    turbine_type: str | None = None
    power_inst: float | None = None
    lat: float
    lon: float
    power_curve_wind: pd.Series | None = None
    power_curve_power: pd.Series | None = None
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        if self.turbine_type is not None:
            self.turbine = wpl.WindTurbine(turbine_type=self.turbine_type, hub_height=self.hub_height)
        elif self.power_inst is not None:
            power_curve = wpl.create_power_curve(
                wind_speed=DEFAULT_POWER_CURVE_WIND,
                power=DEFAULT_POWER_CURVE_POWER * self.power_inst,
            )
            self.turbine = wpl.WindTurbine(
                power_inst=self.power_inst,
                hub_height=self.hub_height,
                power_curve=power_curve,
            )
        else:
            raise MissingArgumentsError

        self.mc = wmc.ModelChain(self.turbine, **MODELCHAIN_DATA)

    def run(self, weather: pd.DataFrame) -> None:
        self.mc.run_model(weather)
        index = t.cast("pd.DatetimeIndex", weather.index)
        acp = self._resample_as_array(target=-self.mc.power_output, index=index)
        full_load_hours: float = acp.sum() / self.power_inst * 8760 / acp.size
        ocean_offset_km = abs(self.lat - LAT_BASE) * LAT_IN_KM
        flh_reduction = ocean_offset_km * FLH_REDUCTION_PER_KM
        full_load_hours_corr = full_load_hours - flh_reduction  # Correct full load hours based on distance to coast [1]
        if not (FULL_LOAD_HOURS_MIN < full_load_hours_corr < FULL_LOAD_HOURS_MAX):
            loguru.logger.warning(
                "Strange full load hours: {full_load_hours_corr}",
                full_load_hours_corr=full_load_hours_corr,
            )

        acp = acp / full_load_hours * full_load_hours_corr

        self.acp.loc[:, ("low", 1)] = acp
        self.acp.loc[:, ("base", 1)] = acp
        acq = acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", 1)] = acq
        self.acq.loc[:, ("high", 1)] = -acq

    @property
    def ac(self) -> pd.Series:
        return t.cast("pd.Series", self.mc.power_output)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class WindFarm(Tech):
    powers_inst: list[float]
    lat: float
    lon: float
    units: cabc.Sequence[wpl.WindTurbine] | None = None
    n_units: list[int] | list[None] | None = None
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        wind_turbine_fleet = pd.DataFrame(
            {
                "wind_turbine": self.units,
                "number_of_turbines": self.n_units,
                "total_capacity": self.powers_inst,
            },
        )
        self.wind_farm = wpl.WindFarm(wind_turbine_fleet=wind_turbine_fleet)
        self.mc = wpl.TurbineClusterModelChain(self.wind_farm, **MODELCHAIN_DATA)

    def run(self, weather: pd.DataFrame) -> None:
        self.mc.run_model(weather)
        index = t.cast("pd.DatetimeIndex", weather.index)
        acp = self._resample_as_array(target=-self.mc.power_output, index=index)
        full_load_hours: float = acp.sum() / sum(self.powers_inst) * 8760 / acp.size
        ocean_offset_km = abs(self.lat - LAT_BASE) * LAT_IN_KM
        flh_reduction = ocean_offset_km * FLH_REDUCTION_PER_KM
        full_load_hours_corr = full_load_hours - flh_reduction  # Correct full load hours based on distance to coast [1]
        if not (FULL_LOAD_HOURS_MIN < full_load_hours_corr < FULL_LOAD_HOURS_MAX):
            loguru.logger.warning(
                "Strange full load hours: {full_load_hours_corr}",
                full_load_hours_corr=full_load_hours_corr,
            )

        acp = acp / full_load_hours * full_load_hours_corr

        self.acp.loc[:, ("low", 1)] = acp
        self.acp.loc[:, ("base", 1)] = acp
        acq = acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", 1)] = acq
        self.acq.loc[:, ("high", 1)] = -acq

    @property
    def ac(self) -> pd.Series:
        return t.cast("pd.Series", self.mc.power_output)

    @classmethod
    def from_power_inst(
        cls,
        dates: pd.DatetimeIndex,
        power_inst: float,
        lat: float,
        lon: float,
    ) -> WindFarm:
        dataframe = pd.read_feather(SRC_PATH / "data/wind/turbines.feather")
        data = dataframe.sample()
        unit_data = {
            "turbine_type": data.turbine_type.to_numpy()[0],
            "hub_height": data.hub_height.to_numpy()[0],
        }
        units = [wpl.WindTurbine(**unit_data)]
        n_units = [None]
        powers_inst = [power_inst]
        return cls(dates=dates, units=units, n_units=n_units, powers_inst=powers_inst, lat=lat, lon=lon)

    @classmethod
    def from_powers_inst_and_hub_heights(
        cls,
        dates: pd.DatetimeIndex,
        powers_inst: list[float],
        hub_heights: list[float],
        lat: float,
        lon: float,
    ) -> WindFarm:
        dataframe = pd.read_feather(SRC_PATH / "data/wind/turbines.feather")
        units: cabc.MutableSequence[wpl.WindTurbine] = []
        for hub_height in hub_heights:
            condition = (dataframe.hub_height - hub_height).abs().argsort()
            data = dataframe.iloc[condition].iloc[0]
            unit_data = {
                "turbine_type": data.turbine_type,
                "hub_height": data.hub_height,
            }
            units.append(wpl.WindTurbine(**unit_data))

        n_units = [None for _ in units]
        return cls(dates=dates, units=units, n_units=n_units, powers_inst=powers_inst, lat=lat, lon=lon)
