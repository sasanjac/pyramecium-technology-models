# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import enum
import typing as t

import attrs
import numpy as np
import pandas as pd
from pyochain import Ok
from pyochain import option

from pstm.utils.results import Constructible

if t.TYPE_CHECKING:
    import numpy.typing as npt
    from pyochain import Result


class HeatingType(enum.StrEnum):
    """Temperature level of the heat a device supplies."""

    ROOM = "room"
    PROCESS_LOW = "process_low"
    PROCESS_HIGH = "process_high"


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Tech(Constructible):
    dates: pd.DatetimeIndex
    acp: pd.DataFrame = attrs.field(init=False, repr=False)  # Active power

    @acp.default
    def _acp_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.zeros((len(self.dates), 3), dtype=np.float64),
            columns=pd.MultiIndex.from_arrays(
                [
                    ["high", "base", "low"],
                    ["L1", "L1", "L1"],
                ],
            ),
            index=self.dates,
        )

    acq: pd.DataFrame = attrs.field(init=False, repr=False)  # Reactive power

    @acq.default
    def _acq_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.zeros((len(self.dates), 3), dtype=np.float64),
            columns=pd.MultiIndex.from_arrays(
                [
                    ["high", "base", "low"],
                    ["L1", "L1", "L1"],
                ],
            ),
            index=self.dates,
        )

    thr: pd.DataFrame = attrs.field(init=False, repr=False)  # Room heating

    @thr.default
    def _thr_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    thw: pd.DataFrame = attrs.field(init=False, repr=False)  # Water heating

    @thw.default
    def _thw_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    thl: pd.DataFrame = attrs.field(init=False, repr=False)  # Low temp process heating

    @thl.default
    def _thl_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    thh: pd.DataFrame = attrs.field(init=False, repr=False)  # High temp process heating

    @thh.default
    def _thh_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    ch4: pd.DataFrame = attrs.field(init=False, repr=False)  # Methane

    @ch4.default
    def _ch4_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    h2: pd.DataFrame = attrs.field(init=False, repr=False)  # Hydrogen

    @h2.default
    def _h2_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    col: pd.DataFrame = attrs.field(init=False, repr=False)  # Cooling

    @col.default
    def _col_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": np.zeros(len(self.dates), dtype=np.float64),
                "base": np.zeros(len(self.dates), dtype=np.float64),
                "low": np.zeros(len(self.dates), dtype=np.float64),
            },
            index=self.dates,
        )

    def heat(self, heating_type: HeatingType) -> pd.DataFrame:
        """Thermal frame of ``heating_type``."""
        match heating_type:
            case HeatingType.ROOM:
                return self.thr
            case HeatingType.PROCESS_LOW:
                return self.thl
            case HeatingType.PROCESS_HIGH:
                return self.thh
            case _:
                t.assert_never(heating_type)

    def _resample(self, target: pd.Series, index: pd.DatetimeIndex) -> Result[pd.Series, ValueError]:
        """``target`` on ``index`` moved onto :attr:`dates`.

        Returns:
            ``Err`` if :attr:`dates` is not longer than ``index`` and has no frequency.
        """
        delta = self.dates[0] - index[0]
        target = target.set_axis(labels=index + delta)
        if len(self.dates) > len(index):
            return Ok(target.reindex(index=self.dates).interpolate(method="linear", limit_direction="both"))

        return (
            option(self.dates.freq)
            .ok_or(ValueError("The frequency of the index is not defined."))
            .map(lambda freq: target.resample(freq).mean().reindex(index=self.dates))
        )

    def _resample_as_array(
        self,
        target: pd.Series,
        index: pd.DatetimeIndex,
    ) -> Result[npt.NDArray[np.float64], ValueError]:
        return self._resample(target, index).map(lambda series: series.to_numpy(dtype=np.float64))
