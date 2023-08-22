# :author: Jörg Dickert <joerg.dickert@tu-dresden.de>
# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2015-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import datetime as dt
import json
import typing as t
from collections.abc import Sequence

import cattrs
import numpy as np
import pandas as pd

from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import Constants
from pstm.dickert.baseline_profiles import BaselineProfiles
from pstm.dickert.cycle_profiles import CycleProfiles
from pstm.dickert.lighting_profiles import LightingProfiles
from pstm.dickert.on_off_profiles import OnOffProfiles
from pstm.dickert.process_profiles import ProcessProfiles

if t.TYPE_CHECKING:
    import pathlib

    import numpy.typing as npt


import attrs

from pstm.base import Tech
from pstm.utils import dates
from pstm.utils import geo


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Households:
    appliances: Sequence[Appliances]
    phase_distribution: tuple[float, float, float]

    def run(self, n_units: int, n_steps: int, lat: float, lon: float, altitude: float, year: int) -> None:
        with geo.GeoRef() as georef:
            self.tz = georef.get_time_zone(lat=lat, lon=lon)

        self.n_steps = n_steps
        self.year = year

        for app in self.appliances:
            app.run(
                n_units=n_units,
                n_steps=n_steps,
                phase_distribution=self.phase_distribution,
                lat=lat,
                lon=lon,
                altitude=altitude,
                year=year,
                tz=self.tz,
            )

        self.p = np.sum([app.p for app in self.appliances], axis=0)
        self.q = np.sum([app.q for app in self.appliances], axis=0)

    @classmethod
    def from_config(cls, config: dict[str, list[dict[str, t.Any]]]) -> Households:
        baselines_profiles = [BaselineProfiles(**e) for e in config["baseline_profiles"] if e["equipment_level"] > 0]
        cycle_profiles = [CycleProfiles(**e) for e in config["cycle_profiles"] if e["equipment_level"] > 0]
        on_off_profiles = [OnOffProfiles(**e) for e in config["on_off_profiles"] if e["equipment_level"] > 0]
        process_profiles = [ProcessProfiles(**e) for e in config["process_profiles"] if e["equipment_level"] > 0]
        lighting_profiles = [LightingProfiles(**e) for e in config["lighting_profiles"] if e["equipment_level"] > 0]
        return Households(
            appliances=baselines_profiles + cycle_profiles + on_off_profiles + process_profiles + lighting_profiles,  # type: ignore[operator]
            phase_distribution=config["phase_distribution"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, json_file_path: pathlib.Path) -> Households:
        with json_file_path.open(mode="r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        return cattrs.structure(data, Households)

    def get(self, index: pd.DatetimeIndex) -> Tech:
        p, self.p = (self.p[:, 0, :], self.p[:, 1:, :])
        q, self.q = (self.q[:, 0, :], self.q[:, 1:, :])
        step_length = Constants.MINUTES_PER_YEAR // self.n_steps
        _index = dates.date_range(tz=self.tz, year=self.year, freq=dt.timedelta(minutes=step_length))
        dfp = pd.DataFrame(p, index=_index)
        dfq = pd.DataFrame(q, index=_index)
        dfp = dfp.resample(rule=index.freq).mean()
        dfq = dfq.resample(rule=index.freq).mean()

        t = Tech(dates=index)
        t.acp = self._df_from_array(index=index, data=dfp.to_numpy())
        t.acq = self._df_from_array(index=index, data=dfq.to_numpy())
        return t

    def _df_from_array(self, *, index: pd.DatetimeIndex, data: npt.NDArray[np.float64]) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.stack(
                [
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                ],
                axis=1,
            ),
            columns=pd.MultiIndex.from_arrays(
                [
                    [
                        "high",
                        "high",
                        "high",
                        "base",
                        "base",
                        "base",
                        "low",
                        "low",
                        "low",
                    ],
                    [
                        1,
                        2,
                        3,
                        1,
                        2,
                        3,
                        1,
                        2,
                        3,
                    ],
                ],
            ),
            index=index,
        )
