# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import typing as t
from collections import defaultdict

import attrs
import numpy as np
import pandas as pd

from pstm.base import Tech
from pstm.utils import dates
from pstm.utils import exceptions
from pstm.utils.geo import GeoRef

if t.TYPE_CHECKING:
    import numpy.typing as npt

    JSONTypes = bool | str | float | list["JSONTypes"] | dict[str, "JSONTypes"] | None

    Profiles = dict[str, list[float]]
    ProfilesMapping = dict[str, Profiles]
    Factors = dict[str, dict[str, float]]
    FactorsMapping = dict[str, Factors]
    TypeDays = list[str]
    TypeDaysMapping = dict[str, TypeDays]

SRC_PATH = pathlib.Path(__file__).parent.parent.parent

ELECTRICITY_DEMAND_OFH = {
    1: 2350,
    2: 2020,
    3: 1650,
    4: 1500,
    5: 1400,
    6: 1350,
}

ELECTRICITY_DEMAND_MFH = 3000

ELECTRICITY_DEMAND: dict[str, dict[int, float]] = {
    "OFH": defaultdict(lambda: ELECTRICITY_DEMAND_OFH[6], ELECTRICITY_DEMAND_OFH),
    "MFH": defaultdict(lambda: ELECTRICITY_DEMAND_MFH),
}

WATER_DEMAND = {
    "OFH": 500,
    "MFH": 1000,
}

INSULATION_STANDARDS = {
    "unrenovated": (250, 300),
    "WSVO77": (150, 250),
    "WSVO82": (100, 150),
    "WSVO95": (70, 100),
    "LEH": (55, 70),
    "KfW85": (45, 55),
    "KfW70": (35, 45),
    "KfW55": (25, 35),
    "KfW40": (15, 25),
    "PH": (0, 15),
}

POWER_FACTORS = [  # Moller et al - Probabilistic household load model for unbalance studies based on measurements - 2016
    {"a": 0.99, "b": 1.605},  # capacitive
    {"a": 0.99, "b": 4.904},  # inductive
]


def load_json_from_file(path: pathlib.Path) -> JSONTypes:
    with path.open(encoding="utf-8") as f:
        return t.cast("JSONTypes", json.load(f))


HOUSE_TYPES = t.Literal["MFH", "OFH"]
BUILDING_TYPES = t.Literal["EH", "LEH"]

DATA_PATH = SRC_PATH / "data/household/vdi4655"

POWER_CONVERSION_FACTORS = {
    "OFH": 1_800_000,
    "MFH": 4_000,
}  # OFH: x W = y kWh * 60 min/h * 60 s/min / 2 s * 1000 W/kW; MFH: x W = y kWh * 60 min/h / 15 min * 1000 W/kW
FREQS = {"OFH": 2, "MFH": 900}  # OFH: = 2 s; MFH: = 15 min = 900 s
PROFILE_SHIFT_LENGTH = 7200  # 8 * 15 min * 60 s


class MissingArgumentsError(ValueError):
    def __init__(self, zone: str) -> None:
        super().__init__(f"Either lat/lon or {zone} zone must be set.")


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Household(Tech):
    house_type: HOUSE_TYPES
    building_type: BUILDING_TYPES
    n_units: int
    heat_demand: float
    area: float
    tz: dt.tzinfo
    lat: float | None = None
    lon: float | None = None
    climate_zone: str | None = None
    data_path: pathlib.Path = DATA_PATH

    def __attrs_post_init__(self) -> None:
        self._init_data()
        self.factors = self.factors_mapping[self.house_type][self.building_type]
        self.profiles = self.profiles_mapping[self.house_type][self.building_type]
        self.power_conversion_factor = POWER_CONVERSION_FACTORS[self.house_type]
        self.freq = FREQS[self.house_type]
        if self.climate_zone is None:
            if self.lat is not None and self.lon is not None:
                with GeoRef(use_clc=False) as georef:
                    self.climate_zone = georef.get_dwd_try_zone(self.lat, self.lon)
            else:
                zone = "climate"
                raise MissingArgumentsError(zone)

        if self.tz is None:
            if self.lat is not None and self.lon is not None:
                with GeoRef(use_clc=False) as georef:
                    self.tz = georef.get_time_zone(lat=self.lat, lon=self.lon)
            else:
                zone = "time"
                raise MissingArgumentsError(zone)

        self.type_days = self.type_days_mapping[self.climate_zone]

    def _init_data(self) -> None:
        factors_mapping_paths = {
            house_type: {
                building_type: self.data_path / f"factors_{house_type}_{building_type}.json"
                for building_type in t.get_args(BUILDING_TYPES)
            }
            for house_type in t.get_args(HOUSE_TYPES)
        }
        self.factors_mapping = {
            house_type: {
                building_type: t.cast(
                    "FactorsMapping",
                    load_json_from_file(factors_mapping_paths[house_type][building_type]),
                )
                for building_type in t.get_args(BUILDING_TYPES)
            }
            for house_type in t.get_args(HOUSE_TYPES)
        }

        profiles_mapping_paths = {
            house_type: {
                building_type: self.data_path / f"e_profile_{house_type}_{building_type}.json"
                for building_type in t.get_args(BUILDING_TYPES)
            }
            for house_type in t.get_args(HOUSE_TYPES)
        }
        self.profiles_mapping = {
            house_type: {
                building_type: t.cast(
                    "ProfilesMapping",
                    load_json_from_file(profiles_mapping_paths[house_type][building_type]),
                )
                for building_type in t.get_args(BUILDING_TYPES)
            }
            for house_type in t.get_args(HOUSE_TYPES)
        }

        type_days_path = self.data_path / "type_days.json"
        self.type_days_mapping = t.cast("TypeDaysMapping", load_json_from_file(type_days_path))

    def run(self, *, thermal: bool = True, electrical: bool = True, random_shift: bool = False) -> None:
        index = dates.date_range(self.tz, freq=dt.timedelta(seconds=self.freq), year=self.dates.year[0])
        if thermal is True:
            water_thermal_demand = self._calculate_water_thermal_demand()
            heating_thermal_demand = self._calculate_heating_thermal_demand()
            if random_shift is True:
                water_thermal_demand = np.roll(
                    water_thermal_demand,
                    random.randint(-PROFILE_SHIFT_LENGTH // self.freq, PROFILE_SHIFT_LENGTH // self.freq),  # noqa: S311
                )
                heating_thermal_demand = np.roll(
                    heating_thermal_demand,
                    random.randint(-PROFILE_SHIFT_LENGTH // self.freq, PROFILE_SHIFT_LENGTH // self.freq),  # noqa: S311
                )

            self.thw.loc[:, "high"] = self._resample_as_array(
                target=pd.Series(data=water_thermal_demand),
                index=index,
            )
            self.thr.loc[:, "high"] = self._resample_as_array(
                target=pd.Series(data=heating_thermal_demand),
                index=index,
            )

        if electrical is True:
            active_electrical_demand = self._calculate_active_electrical_demand()
            if random_shift is True:
                active_electrical_demand = np.roll(active_electrical_demand, random.randint(-8, 8))  # noqa: S311

            self.acp.loc[:, ("high", "L1")] = self._resample_as_array(
                target=pd.Series(data=active_electrical_demand),
                index=index,
            )
            self.acq.loc[:, ("high", "L1")] = self._calculate_reactive_electrical_demand()

    def _calculate_water_thermal_demand(self) -> npt.NDArray[np.float64]:
        energy = WATER_DEMAND[self.house_type] * self.n_units
        return self._calculate_demand(energy=energy, profile_type="water", type_factor=self.n_units, offset=1)

    def _calculate_heating_thermal_demand(self) -> npt.NDArray[np.float64]:
        energy = self.heat_demand * self.area
        return self._calculate_demand(energy=energy, profile_type="heating", type_factor=1, offset=0)

    def _calculate_active_electrical_demand(self) -> npt.NDArray[np.float64]:
        energy = ELECTRICITY_DEMAND[self.house_type][self.n_units] * self.n_units
        return self._calculate_demand(energy=energy, profile_type="electricity", type_factor=self.n_units, offset=1)

    def _calculate_demand(
        self,
        *,
        energy: float,
        profile_type: str,
        type_factor: int,
        offset: int,
    ) -> npt.NDArray[np.float64]:
        if self.climate_zone is None:
            raise exceptions.ModelNotRunError

        factors = self.factors[f"f_{profile_type}"][self.climate_zone]
        profiles = self.profiles[f"e_{profile_type}"]

        energies_daily = {td: energy * (offset / 365 + type_factor * fac) for td, fac in factors.items()}
        energy_profiles_daily = {td: e * np.array(profiles[td]) for td, e in energies_daily.items()}
        energy_profiles = [energy_profiles_daily[td] for td in self.type_days]
        return np.concatenate(energy_profiles) * self.power_conversion_factor

    def _calculate_reactive_electrical_demand(self) -> npt.NDArray[np.float64]:
        acp = t.cast("npt.NDArray[np.float64]", self.acp.high.L1.to_numpy(dtype=np.float64))
        sign = random.randint(0, 1)  # capacitive or inductive  # noqa: S311
        cosphi = self._cosphi(sign, acp)
        x = np.tan(np.arccos(cosphi)).astype(np.float64)
        return (sign * 2 - 1) * acp * x  # type:ignore[no-any-return]

    @staticmethod
    def _cosphi(sign: int, acp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return POWER_FACTORS[sign]["a"] - np.exp(-POWER_FACTORS[sign]["b"] * acp)
