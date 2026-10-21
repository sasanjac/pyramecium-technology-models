# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import attrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from pyochain import NONE
from pyochain import Null
from pyochain import Option
from pyochain import Some

from pstm.base import Tech


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class SeriesDemand(Tech):
    series_acp: pd.Series | npt.NDArray[np.float64]
    factor_acp: float
    series_thr: pd.Series | npt.NDArray[np.float64]
    series_thl: pd.Series | npt.NDArray[np.float64]
    series_thh: pd.Series | npt.NDArray[np.float64]
    factor_thr: float
    factor_thl: float
    factor_thh: float
    series_acq: Option[pd.Series | npt.NDArray[np.float64]] = NONE
    factor_acq: float = attrs.field(default=1)
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        acp = self.series_acp * self.factor_acp
        self.acp.high = acp
        self.acp.base = acp
        self.acp.low = acp
        thr = self.series_thr * self.factor_thr
        self.thr.high = thr
        self.thr.base = thr
        self.thr.low = thr
        thl = self.series_thl * self.factor_thl
        self.thl.high = thl
        self.thl.base = thl
        self.thl.low = thl
        thh = self.series_thh * self.factor_thh
        self.thh.high = thh
        self.thh.base = thh
        self.thh.low = thh
        match self.series_acq:
            case Some(series_acq):
                acq = series_acq * self.factor_acq
            case Null():
                acq = acp * np.tan(np.arccos(self.cosphi))

        self.acq.high = acq
        self.acq.base = acq
        self.acq.low = acq
