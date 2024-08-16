# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import attrs
import numpy as np
import pandas as pd


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Tech:
    dates: pd.DatetimeIndex
    acp: pd.DataFrame = attrs.field(init=False, repr=False)  # Active power

    @acp.default
    def _acp_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.zeros((len(self.dates), 3)),
            columns=pd.MultiIndex.from_arrays(
                [
                    ["high", "base", "low"],
                    [1, 1, 1],
                ],
            ),
            index=self.dates,
        )

    acq: pd.DataFrame = attrs.field(init=False, repr=False)  # Reactive power

    @acq.default
    def _acq_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.zeros((len(self.dates), 3)),
            columns=pd.MultiIndex.from_arrays(
                [
                    ["high", "base", "low"],
                    [1, 1, 1],
                ],
            ),
            index=self.dates,
        )

    thr: pd.DataFrame = attrs.field(init=False, repr=False)  # Room heating

    @thr.default
    def _thr_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    thw: pd.DataFrame = attrs.field(init=False, repr=False)  # Water heating

    @thw.default
    def _thw_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    thl: pd.DataFrame = attrs.field(init=False, repr=False)  # Low temp process heating

    @thl.default
    def _thl_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    thh: pd.DataFrame = attrs.field(init=False, repr=False)  # High temp process heating

    @thh.default
    def _thh_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    ch4: pd.DataFrame = attrs.field(init=False, repr=False)  # Methane

    @ch4.default
    def _ch4_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    h2: pd.DataFrame = attrs.field(init=False, repr=False)  # Hydrogen

    @h2.default
    def _h2_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )

    col: pd.DataFrame = attrs.field(init=False, repr=False)  # Cooling

    @col.default
    def _col_default(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high": 0,
                "base": 0,
                "low": 0,
            },
            index=self.dates,
        )
