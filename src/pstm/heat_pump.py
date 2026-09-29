# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import abc
import datetime as dt
import typing as t
from typing import Literal

import attrs
import numpy as np
import pandas as pd

from pstm.base import Tech
from pstm.utils.ground import kusuda_ground_temperature

EFFICIENCY_OFFSETS = {
    "low": -0.5,
    "normal": 0,
    "high": 0.5,
}
"""Additive COP offset per efficiency class.

The classes are an assumption of this work for future devices; Ruhnau et al.
(2019) define none.  ``normal`` reproduces the source.
"""

HeatingSystem = Literal["radiator", "floor", "water", "constant"]

HEATING_SYSTEMS: tuple[HeatingSystem, ...] = ("radiator", "floor", "water", "constant")
"""Sinks of :func:`sink_temperature`; ``constant`` takes its temperature from ``sink_temp``."""

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


def sink_temperature(
    heating_system: HeatingSystem,
    temp_amb: pd.Series,
    sink_temp: float | None = None,
) -> pd.Series:
    """Sink temperature of eq. (6) of Ruhnau et al. (2019), or a constant sink.

    Radiator heating supplies ``40 - 1.0 T_amb``, floor heating
    ``30 - 0.5 T_amb`` and hot-water heating a constant 50 degC.  The
    ``constant`` sink supplies ``sink_temp`` at every step, e.g. the supply
    temperature of a district-heating loop.  It lies outside the residential
    scope of the source.

    Args:
        heating_system: Heat distribution the pump supplies.
        temp_amb: Ambient air temperature in degC.
        sink_temp: Sink temperature in degC of the ``constant`` sink.

    Returns:
        The sink temperature in degC on the index of ``temp_amb``.

    Raises:
        ValueError: If ``heating_system`` is unknown, or if ``sink_temp`` is
            given for another sink than ``constant`` or missing for it.
    """
    if (heating_system == "constant") != (sink_temp is not None):
        msg = f"sink_temp is required for the constant sink and only for it, got {heating_system!r} with {sink_temp!r}."
        raise ValueError(msg)
    if heating_system == "constant":
        return pd.Series(float(t.cast("float", sink_temp)), index=temp_amb.index)
    if heating_system == "radiator":
        return 40.0 - 1.0 * temp_amb
    if heating_system == "floor":
        return 30.0 - 0.5 * temp_amb
    if heating_system == "water":
        return pd.Series(WATER_HEATING_TEMP, index=temp_amb.index)
    msg = f"Unknown heating system {heating_system!r}."
    raise ValueError(msg)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class HeatPump(Tech, abc.ABC):
    power_inst: float
    cosphi: float = attrs.field(default=0.7)
    tz: dt.tzinfo

    def run(self, temp: pd.Series, thermal_demand: pd.Series | None = None) -> None:
        temp_diff_raw = self.calc_temp_diff(temp)
        index = t.cast("pd.DatetimeIndex", temp.index)
        temp_diff = self._resample(target=temp_diff_raw, index=index)

        cop = self.calc_cop(temp_diff)
        if thermal_demand is None:
            self._th.loc[:, "low"] = (-cop * self.power_inst).to_numpy(dtype=np.float64)
            self.acp.loc[:, ("high", "L1")] = self.power_inst * np.ones(len(self.dates))
        else:
            self._th.loc[:, "low"] = (-thermal_demand).to_numpy(dtype=np.float64)
            self.acp.loc[:, ("high", "L1")] = np.divide(thermal_demand, cop)

        self.acq.loc[:, ("high", "L1")] = (
            self.acp.loc[:, "high"].loc[:, "L1"] * np.tan(np.arccos(self.cosphi))
        ).to_numpy(
            dtype=np.float64,
        )

    @abc.abstractmethod
    def calc_temp_diff(self, temp: pd.Series) -> pd.Series:
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

    heating_system: HeatingSystem = attrs.field(validator=attrs.validators.in_(HEATING_SYSTEMS))
    efficiency: Literal["high", "normal", "low"]
    sink_temp: float | None = attrs.field(default=None)

    @sink_temp.validator
    def _check_sink_temp(self, _attribute: attrs.Attribute, value: float | None) -> None:
        if (self.heating_system == "constant") != (value is not None):
            msg = (
                "sink_temp is required for heating_system 'constant' and only for it, "
                f"got {self.heating_system!r} with sink_temp={value!r}."
            )
            raise ValueError(msg)

    @property
    def _th(self) -> pd.DataFrame:
        return self.thw if self.heating_system == "water" else self.thr

    @property
    def th(self) -> pd.DataFrame:
        """Thermal output frame of the configured heating system."""
        return self._th

    @abc.abstractmethod
    def source_temperature(self, temp: pd.Series) -> pd.Series:
        """Heat source temperature in degC on the index of the ambient air temperature ``temp``."""
        raise NotImplementedError

    def calc_temp_diff(self, temp: pd.Series) -> pd.Series:
        temp_diff = sink_temperature(self.heating_system, temp, self.sink_temp) - self.source_temperature(temp)
        return temp_diff.clip(lower=MIN_TEMP_DIFF)

    @abc.abstractmethod
    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        """Manufacturer COP curve of eq. (4) of the source."""
        raise NotImplementedError

    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        return FIELD_CORRECTION * self.cop_curve(temp_diff) + EFFICIENCY_OFFSETS[self.efficiency]


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class AirHeatPump(ResidentialHeatPump):
    """Air-source heat pump; the source is the ambient air."""

    def source_temperature(self, temp: pd.Series) -> pd.Series:
        return temp

    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        return 6.08 - 0.09 * temp_diff + 0.0005 * temp_diff**2


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class BrineHeatPump(ResidentialHeatPump):
    """Ground-source heat pump; the source is the ground temperature minus 5 K.

    Ruhnau et al. (2019) take the ground temperature from ERA-Interim soil
    level 4 (1.00 m to 2.89 m).  Without a soil series (``ground_temp`` is
    None) the ground temperature at the midpoint of that layer follows from
    the air temperature through :func:`pstm.utils.ground.kusuda_ground_temperature`,
    which needs at least one year of air temperature.
    """

    ground_temp: pd.Series | None = None

    def source_temperature(self, temp: pd.Series) -> pd.Series:
        ground = kusuda_ground_temperature(temp) if self.ground_temp is None else self.ground_temp.reindex(temp.index)
        return ground - BRINE_TEMP_DROP

    def cop_curve(self, temp_diff: pd.Series) -> pd.Series:
        return 10.29 - 0.21 * temp_diff + 0.0012 * temp_diff**2


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class WaterHeatPump(ResidentialHeatPump):
    """Groundwater-source heat pump; the source is a constant 10 degC minus 5 K."""

    def source_temperature(self, temp: pd.Series) -> pd.Series:
        return pd.Series(GROUNDWATER_TEMP - BRINE_TEMP_DROP, index=temp.index)

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
    efficiency: Literal["high", "normal"]

    def calc_temp_diff(self, temp: pd.Series) -> pd.Series:
        return self.target_temp - temp

    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        if self.efficiency == "high":
            base_value = 80.84
        elif self.efficiency == "normal":
            base_value = 68.455
        else:
            raise ValueError

        return base_value / (temp_diff**0.76)

    @property
    def _th(self) -> pd.DataFrame:
        return self.thl
