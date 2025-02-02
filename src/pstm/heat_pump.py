# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import abc
import datetime as dt
import typing as t
from typing import Literal

import attrs
import numpy as np

from pstm.base import Tech

if t.TYPE_CHECKING:
    import pandas as pd

EFFICIENCY_OFFSETS = {
    "low": -0.5,
    "normal": 0,
    "high": 0.5,
}


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class HeatPump(Tech, abc.ABC):
    power_inst: float
    target_temp: float
    cosphi: float = attrs.field(default=0.7)
    tz: dt.tzinfo

    def run(self, temp: pd.Series, thermal_demand: pd.Series | None = None) -> None:
        temp_diff_raw = self.target_temp - temp
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
    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _th(self) -> pd.DataFrame:
        raise NotImplementedError


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class ResidentialHeatPump(HeatPump):
    """Based on the following publications.

    [1] O. Ruhnau, L. Hirth, and A. Praktiknjo
    "Time series of heat demand and heat pump efficiency for energy system modeling,"
    Sci Data, vol. 6, no. 1, p. 189, Dec. 2019, doi: 10.1038/s41597-019-0199-y.
    """

    efficiency: Literal["high", "normal", "low"]

    @property
    def _th(self) -> pd.DataFrame:
        return self.thr


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class AirHeatPump(ResidentialHeatPump):
    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        efficiency_offset = EFFICIENCY_OFFSETS.get(self.efficiency, 0)
        return 6.08 - 0.09 * temp_diff + 0.0005 * temp_diff**2 + efficiency_offset


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class BrineHeatPump(ResidentialHeatPump):
    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        efficiency_offset = EFFICIENCY_OFFSETS.get(self.efficiency, 0)
        return 10.29 - 0.21 * temp_diff + 0.0012 * temp_diff**2 + efficiency_offset


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class WaterHeatPump(ResidentialHeatPump):
    def calc_cop(self, temp_diff: pd.Series) -> pd.Series:
        efficiency_offset = EFFICIENCY_OFFSETS.get(self.efficiency, 0)
        return 9.97 - 0.20 * temp_diff + 0.0012 * temp_diff**2 + efficiency_offset


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class IndustrialHeatPump(HeatPump):
    """Based on the following publications.

    [1] C. Arpagaus, “Hochtemperatur-Wärmepumpen für industrielle Anwendungen," Zürich, May 08, 2019.
    [2] European Climate Foundation (ECF), “Roadmap 2050: A practical guide to a prosperous, low-carbon Europe,"
    Technical and Economic Analysis 1, Apr. 2010. Accessed: Jan. 10, 2023. [Online]. Available: https://www.roadmap2050.eu/attachments/files/Volume1_fullreport_PressPack.pdf
    """

    efficiency: Literal["high", "normal"]

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
