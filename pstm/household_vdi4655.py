# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

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

    JSONTypes = None | bool | str | float | list["JSONTypes"] | dict[str, "JSONTypes"]

    Profiles = dict[str, list[float]]
    ProfilesMapping = dict[str, Profiles]
    Factors = dict[str, dict[str, float]]
    FactorsMapping = dict[str, Factors]
    TypeDays = list[str]
    TypeDaysMapping = dict[str, TypeDays]

SRC_PATH = pathlib.Path(__file__).parent.parent

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

POWER_FACTORS = (
    [  # Moller et al - Probabilistic household load model for unbalance studies based on measurements - 2016
        {"a": 0.99, "b": 1.605},  # capacitive
        {"a": 0.99, "b": 4.904},  # inductive
    ]
)


def load_json_from_file(path: pathlib.Path) -> JSONTypes:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


HOUSE_TYPES = t.Literal["MFH", "OFH"]
BUILDING_TYPES = t.Literal["EH", "LEH"]

FACTORS_MAPPING_PATHS = {
    house_type: {
        building_type: SRC_PATH / f"data/household/vdi4655/factors_{house_type}_{building_type}.json"
        for building_type in t.get_args(BUILDING_TYPES)
    }
    for house_type in t.get_args(HOUSE_TYPES)
}
FACTORS_MAPPING = {
    house_type: {
        building_type: t.cast(
            "FactorsMapping",
            load_json_from_file(FACTORS_MAPPING_PATHS[house_type][building_type]),
        )
        for building_type in t.get_args(BUILDING_TYPES)
    }
    for house_type in t.get_args(HOUSE_TYPES)
}

PROFILES_MAPPING_PATHS = {
    house_type: {
        building_type: SRC_PATH / f"data/household/vdi4655/e_profile_{house_type}_{building_type}_15m.json"
        for building_type in t.get_args(BUILDING_TYPES)
    }
    for house_type in t.get_args(HOUSE_TYPES)
}
PROFILES_MAPPING = {
    house_type: {
        building_type: t.cast(
            "ProfilesMapping",
            load_json_from_file(PROFILES_MAPPING_PATHS[house_type][building_type]),
        )
        for building_type in t.get_args(BUILDING_TYPES)
    }
    for house_type in t.get_args(HOUSE_TYPES)
}

TYPE_DAYS_PATH = SRC_PATH / "data/household/vdi4655/type_days.json"
TYPE_DAYS_MAPPING = t.cast("TypeDaysMapping", load_json_from_file(TYPE_DAYS_PATH))


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

    def __attrs_post_init__(self) -> None:
        self.factors = FACTORS_MAPPING[self.house_type][self.building_type]
        self.profiles = PROFILES_MAPPING[self.house_type][self.building_type]
        if self.climate_zone is None:
            if self.lat is not None and self.lon is not None:
                with GeoRef() as georef:
                    self.climate_zone = georef.get_dwd_try_zone(self.lat, self.lon)
            else:
                zone = "climate"
                raise MissingArgumentsError(zone)

        if self.tz is None:
            if self.lat is not None and self.lon is not None:
                with GeoRef() as georef:
                    self.tz = georef.get_time_zone(lat=self.lat, lon=self.lon)
            else:
                zone = "time"
                raise MissingArgumentsError(zone)

        self.type_days = TYPE_DAYS_MAPPING[self.climate_zone]

    def run(self, *, thermal: bool = True, electrical: bool = True, random_shift: bool = False) -> None:
        index = dates.date_range(self.tz, freq=dt.timedelta(minutes=15), year=self.dates.year[0])
        if thermal is True:
            water_thermal_demand = self._calculate_water_thermal_demand()
            heating_thermal_demand = self._calculate_heating_thermal_demand()
            thermal_demand = water_thermal_demand + heating_thermal_demand
            if random_shift is True:
                thermal_demand = np.roll(thermal_demand, random.randint(-8, 8))

            th_raw = pd.Series(data=thermal_demand, index=index)
            th = pd.Series(data=th_raw, index=self.dates).interpolate(
                method="spline",
                order=1,
                limit_direction="both",
            )
            self.thr.loc[:, ("high", 1)] = th.to_numpy()

        if electrical is True:
            active_electrical_demand = self._calculate_active_electrical_demand()
            if random_shift is True:
                active_electrical_demand = np.roll(active_electrical_demand, random.randint(-8, 8))

            acp_raw = pd.Series(data=active_electrical_demand, index=index)
            acp = pd.Series(data=acp_raw, index=self.dates).interpolate(
                method="spline",
                order=1,
                limit_direction="both",
            )
            self.acp.loc[:, ("high", 1)] = acp.to_numpy()
            reactive_electrical_demand = self._calculate_reactive_electrical_demand()
            acq_raw = pd.Series(data=reactive_electrical_demand, index=index)
            acq = pd.Series(data=acq_raw, index=self.dates).interpolate(
                method="spline",
                order=1,
                limit_direction="both",
            )
            self.acq.loc[:, ("high", 1)] = acq.to_numpy()

    def _calculate_water_thermal_demand(self) -> npt.NDArray[np.float64]:
        energy = WATER_DEMAND[self.house_type] * self.n_units
        return self._calculate_demand(energy, "water")

    def _calculate_heating_thermal_demand(self) -> npt.NDArray[np.float64]:
        energy = self.heat_demand * self.area
        return self._calculate_demand(energy, "heating")

    def _calculate_active_electrical_demand(self) -> npt.NDArray[np.float64]:
        energy = ELECTRICITY_DEMAND[self.house_type][self.n_units] * self.n_units
        return self._calculate_demand(energy, "electricity")

    def _calculate_demand(self, energy: float, profile_type: str) -> npt.NDArray[np.float64]:
        if self.climate_zone is None:
            raise exceptions.ModelNotRunError

        factors = self.factors[f"f_{profile_type}"][self.climate_zone]
        profiles = self.profiles[f"e_{profile_type}"]

        energies_daily = {td: energy * (1 / 365 + self.n_units * fac) for td, fac in factors.items()}
        energy_profiles_daily = {td: e * np.array(profiles[td]) for td, e in energies_daily.items()}
        energy_profiles = [energy_profiles_daily[td] for td in self.type_days]
        return np.concatenate(energy_profiles) * 60 / 15  # kWh = kW * 15 min

    def _calculate_reactive_electrical_demand(self) -> npt.NDArray[np.float64]:
        acp = self.acp.high[1].to_numpy()
        sign = random.randint(0, 1)  # capacitive or inductive
        cosphi = self._cosphi(sign, acp)
        return (sign * 2 - 1) * acp * np.tan(np.arccos(cosphi))

    def _cosphi(self, sign: int, acp: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return POWER_FACTORS[sign]["a"] - np.exp(-POWER_FACTORS[sign]["b"] * acp)
