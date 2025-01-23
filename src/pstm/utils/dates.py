# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import datetime as dt

import pandas as pd


def date_range(tz: dt.tzinfo, freq: dt.timedelta = dt.timedelta(hours=1), year: int = 2050) -> pd.DatetimeIndex:
    return pd.date_range(start=str(year), end=str(year + 1), freq=freq, tz=tz, inclusive="left")
