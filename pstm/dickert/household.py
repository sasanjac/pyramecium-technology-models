# :author: Jörg Dickert <joerg.dickert@tu-dresden.de>
# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2015-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

if t.TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from pstm.dickert.appliance import Appliance

import attrs

from pstm.base import Tech
from pstm.dickert.baseline_profile import BaselineProfile
from pstm.dickert.cycle_profile import CycleProfile
from pstm.dickert.lighting_profile import LightingProfile
from pstm.dickert.on_off_profile import OnOffProfile
from pstm.dickert.process_profile import ProcessProfile

SEED = 999999999

GEN = np.random.default_rng(seed=SEED)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Households:
    appliances: Sequence[Appliance]
    phase_distribution: tuple[float, float, float]

    def run(self, n_units: int, n_steps: int) -> None:
        for app in self.appliances:
            app.run(n_units, n_steps, phase_distribution=self.phase_distribution, freq=self.freq)

        self.p = np.sum([app.p for app in self.appliances], axis=0)
        self.q = np.sum([app.q for app in self.appliances], axis=0)

    @classmethod
    def from_config(cls, config) -> Households:
        baseline_profile = BaselineProfile()
        cycle_profile = CycleProfile()
        on_off_profile = OnOffProfile()
        process_profile = ProcessProfile()
        lighting_profile = LightingProfile()
        return Households(
            dates=config.dates,
            appliances=[baseline_profile, cycle_profile, on_off_profile, process_profile, lighting_profile],
        )

    def get(self, dates: pd.DatetimeIndex) -> Tech:
        p, self.p = (self.p[:, 0, :], self.p[:, 1:, :])
        q, self.q = (self.q[:, 0, :], self.q[:, 1:, :])
        t = Tech(dates=dates)
        t.acp = self._df_from_array(dates, p)
        t.acq = self._df_from_array(dates, q)
        return t

    def _df_from_array(self, dates: pd.DatetimeIndex, data: npt.NDArray[np.float64]) -> pd.DataFrame:
        return pd.DataFrame(
            data=[
                np.ones(len(dates)),
                np.ones(len(dates)),
                np.ones(len(dates)),
                data[:, 0],
                data[:, 1],
                data[:, 2],
                np.ones(len(dates)),
                np.ones(len(dates)),
                np.ones(len(dates)),
            ],
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
            index=dates,
        )
