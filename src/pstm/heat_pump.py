# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import abc
import datetime as dt
import enum
import typing as t

import attrs
import numpy as np
import pandas as pd
from pyochain import NONE
from pyochain import Err
from pyochain import Null
from pyochain import Ok
from pyochain import Option
from pyochain import Result
from pyochain import Some

from pstm.base import Tech
from pstm.utils.ground import Ground


class Efficiency(enum.StrEnum):
    """Efficiency class of a residential heat pump."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class IndustrialEfficiency(enum.StrEnum):
    """Efficiency class of an industrial heat pump."""

    NORMAL = "normal"
    HIGH = "high"


class HeatingSystem(enum.StrEnum):
    """Heat distribution of a residential heat pump; ``CONSTANT`` takes its temperature from ``sink_temp``."""

    RADIATOR = "radiator"
    FLOOR = "floor"
    WATER = "water"
    CONSTANT = "constant"

    def sink_temperature(self, temp_amb: pd.Series, sink_temp: Option[float] = NONE) -> Result[pd.Series, ValueError]:
        """Sink temperature of eq. (6) of Ruhnau et al. (2019), or a constant sink.

        Radiator heating supplies ``40 - 1.0 T_amb``, floor heating
        ``30 - 0.5 T_amb`` and hot-water heating a constant 50 degC.  The
        ``constant`` sink supplies ``sink_temp`` at every step, e.g. the supply
        temperature of a district-heating loop.  It lies outside the residential
        scope of the source.

        Args:
            temp_amb: Ambient air temperature in degC.
            sink_temp: Sink temperature in degC of the ``constant`` sink.

        Returns:
            ``Ok`` with the sink temperature in degC on the index of ``temp_amb``.
            ``Err`` if ``sink_temp`` is given for another sink than ``constant``
            or missing for it.
        """
        match self, sink_temp:
            case HeatingSystem.CONSTANT, Some(temp):
                return Ok(pd.Series(float(temp), index=temp_amb.index))
            case HeatingSystem.RADIATOR, Null():
                return Ok(40.0 - 1.0 * temp_amb)
            case HeatingSystem.FLOOR, Null():
                return Ok(30.0 - 0.5 * temp_amb)
            case HeatingSystem.WATER, Null():
                return Ok(pd.Series(WATER_HEATING_TEMP, index=temp_amb.index))
            case _:
                msg = f"sink_temp is required for the constant sink and only for it, got {self!r} with {sink_temp!r}."
                return Err(ValueError(msg))


EFFICIENCY_OFFSETS = {
    Efficiency.LOW: -0.5,
    Efficiency.NORMAL: 0,
    Efficiency.HIGH: 0.5,
}
"""Additive COP offset per efficiency class.

The classes are an assumption of this work for future devices; Ruhnau et al.
(2019) define none.  ``normal`` reproduces the source.
"""

MIN_TEMP_DIFF = 15.0
"""Lower bound in K of the source-to-sink temperature difference (Ruhnau et al. 2019)."""

FIELD_CORRECTION = 0.85
"""Factor on every COP that aligns the manufacturer curves with field measurements (Ruhnau et al. 2019)."""

BRINE_TEMP_DROP = 5.0
"""Temperature drop in K across the heat exchanger of ground- and groundwater-source units."""

GROUNDWATER_TEMP = 10.0
"""Groundwater temperature in degC."""

WATER_HEATING_TEMP = 50.0
"""Sink temperature in degC of domestic hot-water heating."""


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class HeatPump(Tech, abc.ABC):
    power_inst: float
    cosphi: float = attrs.field(default=0.7)
    tz: dt.tzinfo

    def run(self, temp: pd.Series, thermal_demand: Option[pd.Series] = NONE) -> Result[None, ValueError]:
        """Thermal and electrical profiles for the ambient temperature ``temp``.

        Without ``thermal_demand`` the pump runs at ``power_inst``; with it, the
        electrical input follows the demand.

        Returns:
            ``Err`` if the temperature difference cannot be evaluated or moved
            onto :attr:`dates`.
        """
        index = t.cast("pd.DatetimeIndex", temp.index)
        return (
            self.calc_temp_diff(temp)
            .and_then(lambda temp_diff: self._resample(target=temp_diff, index=index))
            .map(lambda temp_diff: self._set_power(self.calc_cop(temp_diff), thermal_demand))
        )

    def _set_power(self, cop: pd.Series, thermal_demand: Option[pd.Series]) -> None:
        match thermal_demand:
            case Some(demand):
                self._th.loc[:, "low"] = (-demand).to_numpy(dtype=np.float64)
                self.acp.loc[:, ("high", "L1")] = np.divide(demand, cop)
            case Null():
                self._th.loc[:, "low"] = (-cop * self.power_inst).to_numpy(dtype=np.float64)
                self.acp.loc[:, ("high", "L1")] = self.power_inst * np.ones(len(self.dates))

        self.acq.loc[:, ("high", "L1")] = (
            self.acp.loc[:, "high"].loc[:, "L1"] * np.tan(np.arccos(self.cosphi))
        ).to_numpy(
            dtype=np.float64,
        )

    @abc.abstractmethod
    def calc_temp_diff(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _th(self) -> pd.DataFrame:
        raise NotImplementedError


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class ResidentialHeatPump(HeatPump):
    """Residential heat pump after Ruhnau et al. (2019).

    [1] O. Ruhnau, L. Hirth, and A. Praktiknjo
    "Time series of heat demand and heat pump efficiency for energy system modeling,"
    Sci Data, vol. 6, no. 1, p. 189, Dec. 2019, doi: 10.1038/s41597-019-0199-y.

    The COP is ``0.85 * f(dT) + offset``.  ``f`` is the quadratic curve of
    eq. (4) of the source for the heat source of the subclass.  ``dT`` is
    the source-to-sink temperature difference of eq. (5), bounded below by
    15 K.  The sink follows the heating curve of eq. (6) for
    ``heating_system``.  The factor 0.85 aligns the manufacturer curves with
    field measurements, as in the source.

    ``offset`` comes from :data:`EFFICIENCY_OFFSETS`.  The efficiency
    classes are an assumption of this work for future devices; the source
    defines none, and ``normal`` reproduces it exactly.

    The ``constant`` sink holds ``sink_temp`` at every step.  It models a
    supply loop at a fixed temperature, e.g. a district-heating substation,
    and lies outside the residential scope of Ruhnau et al. (2019).  The
    curves of the source and the 15 K bound are applied to it unchanged.
    ``sink_temp`` is given exactly when ``heating_system`` is ``constant``.

    Room heating (radiator, floor) and the constant sink are written to
    ``thr``, hot-water heating to ``thw``.  The constant sink supplies a
    heating loop whose consumers draw space heat and hot water from one
    carrier.  In pyramecium that carrier is ``TH``, which the VDI 4655
    household fills from ``thr`` (``TW`` is the carrier of ``thw``).
    """

    heating_system: HeatingSystem = attrs.field(converter=HeatingSystem)
    efficiency: Efficiency = attrs.field(converter=Efficiency)
    sink_temp: Option[float] = attrs.field(default=NONE)

    @sink_temp.validator
    def _check_sink_temp(self, _attribute: attrs.Attribute, value: Option[float]) -> None:
        # attrs reports an invalid field only by raising; ``create`` returns it as ``Err``.
        match self.heating_system.sink_temperature(pd.Series(dtype=np.float64), value):
            case Err(error):
                raise error
            case Ok():
                pass

    @property
    def _th(self) -> pd.DataFrame:
        match self.heating_system:
            case HeatingSystem.WATER:
                return self.thw
            case HeatingSystem.RADIATOR | HeatingSystem.FLOOR | HeatingSystem.CONSTANT:
                return self.thr
            case _:
                t.assert_never(self.heating_system)

    @property
    def th(self) -> pd.DataFrame:
        """Thermal output frame of the configured heating system."""
        return self._th

    @abc.abstractmethod
    def source_temperature(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        """Heat source temperature in degC on the index of the ambient air temperature ``temp``."""
        raise NotImplementedError

    def calc_temp_diff(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        return (
            self.heating_system.sink_temperature(temp, self.sink_temp)
            .and_then(lambda sink: self.source_temperature(temp).map(lambda source: sink - source))
            .map(lambda temp_diff: temp_diff.clip(lower=MIN_TEMP_DIFF))
        )

    @abc.abstractmethod
    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        """Manufacturer COP curve of eq. (4) of the source."""
        raise NotImplementedError

    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        return FIELD_CORRECTION * self.cop_curve(temp_diff) + EFFICIENCY_OFFSETS[self.efficiency]


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class AirHeatPump(ResidentialHeatPump):
    """Air-source heat pump; the source is the ambient air."""

    def source_temperature(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        return Ok(temp)

    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        return 6.08 - 0.09 * temp_diff + 0.0005 * temp_diff**2


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class BrineHeatPump(ResidentialHeatPump):
    """Ground-source heat pump; the source is the ground temperature minus 5 K.

    Ruhnau et al. (2019) take the ground temperature from ERA-Interim soil
    level 4 (1.00 m to 2.89 m).  Without a soil series (``ground_temp`` is
    ``NONE``) the ground temperature at the midpoint of that layer follows from
    the air temperature through :meth:`pstm.utils.ground.Ground.kusuda_ground_temperature`,
    which needs at least one year of air temperature.
    """

    ground_temp: Option[pd.Series] = NONE

    def source_temperature(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        match self.ground_temp:
            case Some(ground_temp):
                ground: Result[pd.Series, ValueError] = Ok(ground_temp.reindex(temp.index))
            case Null():
                ground = Ground.kusuda_ground_temperature(temp)

        return ground.map(lambda ground_temp: ground_temp - BRINE_TEMP_DROP)

    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        return 10.29 - 0.21 * temp_diff + 0.0012 * temp_diff**2


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class WaterHeatPump(ResidentialHeatPump):
    """Groundwater-source heat pump; the source is a constant 10 degC minus 5 K."""

    def source_temperature(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        return Ok(pd.Series(GROUNDWATER_TEMP - BRINE_TEMP_DROP, index=temp.index))

    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        return 9.97 - 0.20 * temp_diff + 0.0012 * temp_diff**2


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class IndustrialHeatPump(HeatPump):
    """Based on the following publications.

    [1] C. Arpagaus, “Hochtemperatur-Wärmepumpen für industrielle Anwendungen," Zürich, May 08, 2019.
    [2] European Climate Foundation (ECF), “Roadmap 2050: A practical guide to a prosperous, low-carbon Europe,"
    Technical and Economic Analysis 1, Apr. 2010. Accessed: Jan. 10, 2023. [Online]. Available: https://www.roadmap2050.eu/attachments/files/Volume1_fullreport_PressPack.pdf
    """

    target_temp: float
    efficiency: IndustrialEfficiency = attrs.field(converter=IndustrialEfficiency)

    def calc_temp_diff(self, temp: pd.Series) -> Result[pd.Series, ValueError]:
        return Ok(self.target_temp - temp)

    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        match self.efficiency:
            case IndustrialEfficiency.HIGH:
                base_value = 80.84
            case IndustrialEfficiency.NORMAL:
                base_value = 68.455
            case _:
                t.assert_never(self.efficiency)

        return base_value / (temp_diff**0.76)

    @property
    def _th(self) -> pd.DataFrame:
        return self.thl
