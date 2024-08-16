# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import datetime as dt

import pandas as pd


def date_range(tz: dt.tzinfo, freq: dt.timedelta = dt.timedelta(hours=1), year: int = 2050) -> pd.DatetimeIndex:
    return pd.date_range(start=str(year), end=str(year + 1), freq=freq, tz=tz, inclusive="left")
