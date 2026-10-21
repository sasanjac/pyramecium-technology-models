# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import datetime as dt

import pandas as pd


class Dates:
    @staticmethod
    def date_range(tz: dt.tzinfo, freq: dt.timedelta = dt.timedelta(hours=1), year: int = 2050) -> pd.DatetimeIndex:
        """Steps of ``year`` at ``freq`` in ``tz``.

        Returns:
            The steps from the start of ``year`` up to the first step of the following year, without it.
        """
        return pd.date_range(start=str(year), end=str(year + 1), freq=freq, tz=tz, inclusive="left")
