# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

import collections.abc as cabc
import datetime as dt
import json
import typing as t

import attrs
import cattrs
import numpy as np
import pandas as pd

from pstm.base import Tech
from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import Constants
from pstm.dickert.appliances import DistributionType
from pstm.dickert.appliances import Phase
from pstm.dickert.baseline_profiles import BaselineProfiles
from pstm.dickert.cycle_profiles import CycleProfiles
from pstm.dickert.lighting_profiles import LightingProfiles
from pstm.dickert.on_off_profiles import OnOffProfiles
from pstm.dickert.process_profiles import ProcessProfiles
from pstm.utils import dates
from pstm.utils import geo

if t.TYPE_CHECKING:
    import pathlib

    import numpy.typing as npt

    Array2DF = npt.NDArray[np.float64]

cattrs.register_structure_hook(dt.datetime, lambda v, _: dt.datetime.fromisoformat(v))
cattrs.register_structure_hook(dt.date, lambda v, _: dt.date.fromisoformat(v))
cattrs.register_structure_hook(dt.time, lambda v, _: dt.time.fromisoformat(v))

N_STEPS = 525_600


class AppliancesDict(t.TypedDict):
    description: str
    phase: Phase
    switch_on_current: float
    switch_on_time: int
    equipment_level: float
    active_power_distribution_type: DistributionType
    active_power_parameter_1: float
    active_power_parameter_2: float
    active_power_parameter_3: float
    reactive_power_share: float
    reactive_power_distribution_type: DistributionType
    reactive_power_parameter_1: float
    reactive_power_parameter_2: float
    reactive_power_parameter_3: float


class BaselineProfilesDict(AppliancesDict):
    power_variation: float
    power_variation_max: float


class OperationProfilesDict(AppliancesDict):
    active_power_parameter_4: float
    operation_distribution_type: DistributionType
    operation_parameter_1: float
    operation_parameter_2: float
    operation_parameter_3: float
    operation_variation: float


class CycleProfilesDict(OperationProfilesDict):
    period_distribution_type: DistributionType
    period_parameter_1: float
    period_parameter_2: float
    period_parameter_3: float
    period_variation: float


class OnOffProfilesDict(OperationProfilesDict):
    usage_distribution_type: DistributionType
    usage_parameter_1: float
    usage_parameter_2: float
    usage_parameter_3: float
    usage_variation: float
    time_on_distribution_types: tuple[
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
    ]
    time_on_parameters_1: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_parameters_3: tuple[float, float, float, float, float, float]
    probability_1: float
    probability_2: float
    probability_3: float
    probability_4: float


class ProcessProfilesDict(OnOffProfilesDict):
    active_power_2_distribution_type: DistributionType
    active_power_2_parameter_1: float
    active_power_2_parameter_2: float
    reactive_power_2_distribution_type: DistributionType
    reactive_power_2_parameter_1: float
    reactive_power_2_parameter_2: float
    operation_2_distribution_type: DistributionType
    operation_2_parameter_1: float
    operation_2_parameter_2: float


class LightingProfilesDict(AppliancesDict):
    lighting_distribution_types: tuple[
        DistributionType,
        DistributionType,
    ]
    lighting_parameters_1: tuple[dt.time, dt.time]
    lighting_parameters_2: tuple[dt.time, dt.time]
    time_on_distribution_types: tuple[
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
    ]
    time_on_parameters_1: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_off_distribution_types: tuple[
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
    ]
    time_off_parameters_1: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_off_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_variations: tuple[float, float, float, float]


class ConfigDict(t.TypedDict):
    baseline_profiles: cabc.Sequence[BaselineProfilesDict]
    cycle_profiles: cabc.Sequence[CycleProfilesDict]
    on_off_profiles: cabc.Sequence[OnOffProfilesDict]
    process_profiles: cabc.Sequence[ProcessProfilesDict]
    lighting_profiles: cabc.Sequence[LightingProfilesDict]
    phase_distribution: tuple[float, float, float]


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Households:
    baseline_profiles: cabc.Sequence[BaselineProfiles]
    cycle_profiles: cabc.Sequence[CycleProfiles]
    on_off_profiles: cabc.Sequence[OnOffProfiles]
    process_profiles: cabc.Sequence[ProcessProfiles]
    lighting_profiles: cabc.Sequence[LightingProfiles]
    phase_distribution: tuple[float, float, float]

    def __attrs_post_init__(self) -> None:
        self.p: Array2DF = np.empty(shape=(0, 0), dtype=np.float64)
        self.q: Array2DF = np.empty(shape=(0, 0), dtype=np.float64)

    def run(
        self,
        /,
        *,
        n_units: int,
        lat: float,
        lon: float,
        altitude: float,
        year: int,
        generator: np.random.Generator,
        baseline_only: bool = False,
    ) -> None:
        with geo.GeoRef(use_clc=False) as georef:
            self.tz = georef.get_time_zone(lat=lat, lon=lon)

        self.n_steps = Constants.MINUTES_PER_YEAR
        self.year = year

        appliances = self.appliances if not baseline_only else self.baseline_appliances
        for app in appliances:
            app.run(
                n_units=n_units,
                n_steps=self.n_steps,
                phase_distribution=self.phase_distribution,
                lat=lat,
                lon=lon,
                altitude=altitude,
                year=year,
                tz=self.tz,
                generator=generator,
            )

        self.p = np.sum([app.p for app in appliances], axis=0)
        self.q = np.sum([app.q for app in appliances], axis=0)

    @property
    def baseline_appliances(self) -> cabc.Sequence[Appliances]:
        return self.baseline_profiles + self.cycle_profiles  # type:ignore[operator, no-any-return]

    @property
    def appliances(self) -> cabc.Sequence[Appliances]:
        return (  # type:ignore[no-any-return]
            self.baseline_profiles  # type:ignore[operator]
            + self.cycle_profiles
            + self.on_off_profiles
            + self.process_profiles
            + self.lighting_profiles
        )

    @classmethod
    def from_config(cls, config: ConfigDict) -> Households:
        baseline_profiles = [BaselineProfiles(**e) for e in config["baseline_profiles"] if e["equipment_level"] > 0]
        cycle_profiles = [CycleProfiles(**e) for e in config["cycle_profiles"] if e["equipment_level"] > 0]
        on_off_profiles = [OnOffProfiles(**e) for e in config["on_off_profiles"] if e["equipment_level"] > 0]
        process_profiles = [ProcessProfiles(**e) for e in config["process_profiles"] if e["equipment_level"] > 0]
        lighting_profiles = [LightingProfiles(**e) for e in config["lighting_profiles"] if e["equipment_level"] > 0]
        return Households(
            baseline_profiles=baseline_profiles,
            cycle_profiles=cycle_profiles,
            on_off_profiles=on_off_profiles,
            process_profiles=process_profiles,
            lighting_profiles=lighting_profiles,
            phase_distribution=config["phase_distribution"],
        )

    @classmethod
    def from_json(cls, json_file_path: pathlib.Path) -> Households:
        with json_file_path.open(mode="r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        return cattrs.structure(data, Households)

    def get(self, index_target: pd.DatetimeIndex) -> Tech:
        p, self.p = (self.p[:, 0, :], self.p[:, 1:, :])
        q, self.q = (self.q[:, 0, :], self.q[:, 1:, :])
        step_length = Constants.MINUTES_PER_YEAR / self.n_steps
        index = dates.date_range(tz=self.tz, year=self.year, freq=dt.timedelta(minutes=step_length))
        dfp = pd.DataFrame(p, index=index)
        dfq = pd.DataFrame(q, index=index)
        freq = index_target.freq
        if freq is None:
            msg = "The frequency of the index is not set."
            raise ValueError(msg)

        dfp = dfp.resample(rule=freq).mean()
        dfq = dfq.resample(rule=freq).mean()

        t = Tech(dates=index_target)
        t.acp = self._df_from_array(index=index_target, data=dfp.to_numpy())
        t.acq = self._df_from_array(index=index_target, data=dfq.to_numpy())
        return t

    @staticmethod
    def _df_from_array(*, index: pd.DatetimeIndex, data: Array2DF) -> pd.DataFrame:
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
                        "L1",
                        "L2",
                        "L3",
                        "L1",
                        "L2",
                        "L3",
                        "L1",
                        "L2",
                        "L3",
                    ],
                ],
            ),
            index=index,
        )
