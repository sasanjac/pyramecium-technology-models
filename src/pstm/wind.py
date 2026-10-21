# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import collections.abc as cabc
import pathlib
import typing as t

import attrs
import loguru
import numpy as np
import pandas as pd
from pyochain import NONE
from pyochain import Null
from pyochain import Option
from pyochain import Some
import windpowerlib as wpl
import windpowerlib.modelchain as wmc

from pstm.base import Tech

if t.TYPE_CHECKING:
    import numpy.typing as npt
    from pyochain import Result

loguru.logger.disable("windpowerlib")

SRC_PATH = pathlib.Path(__file__).parent.parent.parent
DEFAULT_TURBINE = wpl.WindTurbine(turbine_type="E-126/4200", hub_height=135)
DEFAULT_POWER_CURVE_WIND = DEFAULT_TURBINE.power_curve["wind_speed"]
DEFAULT_POWER_CURVE_POWER = DEFAULT_TURBINE.power_curve["value"] / DEFAULT_TURBINE.power_curve["value"].max()

FULL_LOAD_HOURS_MAX = 4_500
FULL_LOAD_HOURS_MIN = 1_500

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
    turbine_type: Option[str] = NONE
    power_inst: Option[float] = NONE
    lat: float
    lon: float
    power_curve_wind: Option[pd.Series] = NONE
    power_curve_power: Option[pd.Series] = NONE
    cosphi: float = attrs.field(default=0.9)
    modelchain_data: dict[str, t.Any] = attrs.field(factory=lambda: dict(MODELCHAIN_DATA))

    def __attrs_post_init__(self) -> None:
        match self.turbine_type, self.power_inst:
            case Some(turbine_type), _:
                self.turbine = wpl.WindTurbine(turbine_type=turbine_type, hub_height=self.hub_height)
            case Null(), Some(power_inst):
                power_curve = wpl.create_power_curve(
                    wind_speed=DEFAULT_POWER_CURVE_WIND,
                    power=DEFAULT_POWER_CURVE_POWER * power_inst,
                )
                self.turbine = wpl.WindTurbine(
                    power_inst=power_inst,
                    hub_height=self.hub_height,
                    power_curve=power_curve,
                )
            case _:
                # attrs reports invalid arguments only by raising; ``create`` returns it as ``Err``.
                raise MissingArgumentsError

        self.mc = wmc.ModelChain(self.turbine, **self.modelchain_data)

    def run(self, weather: pd.DataFrame) -> Result[None, ValueError]:
        """Generation profile for ``weather``.

        Returns:
            ``Err`` if the profile cannot be moved onto :attr:`dates`.
        """
        self.mc.run_model(weather)
        index = t.cast("pd.DatetimeIndex", weather.index)
        return self._resample_as_array(target=-self.mc.power_output, index=index).map(self._set_power)

    def _set_power(self, acp: npt.NDArray[np.float64]) -> None:
        power_inst = self.power_inst.unwrap_or(self.turbine.nominal_power)
        full_load_hours: float = -acp.sum() / power_inst * 8760 / acp.size  # acp is negative (generation)
        if not (FULL_LOAD_HOURS_MIN < full_load_hours < FULL_LOAD_HOURS_MAX):
            loguru.logger.warning(
                "Strange full load hours: {full_load_hours}",
                full_load_hours=full_load_hours,
            )

        self.acp.loc[:, ("low", "L1")] = acp
        self.acp.loc[:, ("base", "L1")] = acp
        acq = acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", "L1")] = acq
        self.acq.loc[:, ("high", "L1")] = -acq

    @property
    def ac(self) -> pd.Series:
        return t.cast("pd.Series", self.mc.power_output)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class WindFarm(Tech):
    powers_inst: list[float]
    lat: float
    lon: float
    units: Option[cabc.Sequence[wpl.WindTurbine]] = NONE
    n_units: Option[list[int] | list[None]] = NONE
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        wind_turbine_fleet = pd.DataFrame(
            {
                "wind_turbine": self.units.unwrap_or_none(),
                "number_of_turbines": self.n_units.unwrap_or_none(),
                "total_capacity": self.powers_inst,
            },
        )
        self.wind_farm = wpl.WindFarm(wind_turbine_fleet=wind_turbine_fleet)
        self.mc = wpl.TurbineClusterModelChain(self.wind_farm, **MODELCHAIN_DATA)

    def run(self, weather: pd.DataFrame) -> Result[None, ValueError]:
        """Generation profile for ``weather``.

        Returns:
            ``Err`` if the profile cannot be moved onto :attr:`dates`.
        """
        self.mc.run_model(weather)
        index = t.cast("pd.DatetimeIndex", weather.index)
        return self._resample_as_array(target=-self.mc.power_output, index=index).map(self._set_power)

    def _set_power(self, acp: npt.NDArray[np.float64]) -> None:
        full_load_hours: float = -acp.sum() / sum(self.powers_inst) * 8760 / acp.size  # acp is negative (generation)
        if not (FULL_LOAD_HOURS_MIN < full_load_hours < FULL_LOAD_HOURS_MAX):
            loguru.logger.warning(
                "Strange full load hours: {full_load_hours}",
                full_load_hours=full_load_hours,
            )

        self.acp.loc[:, ("low", "L1")] = acp
        self.acp.loc[:, ("base", "L1")] = acp
        acq = acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", "L1")] = acq
        self.acq.loc[:, ("high", "L1")] = -acq

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
        n_units: list[None] = [None]
        powers_inst = [power_inst]
        return cls(dates=dates, units=Some(units), n_units=Some(n_units), powers_inst=powers_inst, lat=lat, lon=lon)

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

        n_units: list[None] = [None for _ in units]
        return cls(dates=dates, units=Some(units), n_units=Some(n_units), powers_inst=powers_inst, lat=lat, lon=lon)
