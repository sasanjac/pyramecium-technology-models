# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2018-2025 Guntram Preßmair

from __future__ import annotations

import datetime as dt
import typing as t

import attrs
import numpy as np
import pandas as pd

from pstm.base import Tech
from pstm.utils import dates

if t.TYPE_CHECKING:
    TypeDay = t.Literal["WD", "WE"]
    DrivingCategory = t.Literal["low", "medium", "high"]
    Profile = t.Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    NTrips = t.Literal[0, 1, 2, 3]

SEED = 999999999

GEN = np.random.default_rng(seed=SEED)


DAILY_MEAN_DISTANCES: dict[
    Profile,
    float,
] = {  # G. Preßmair, “Modellierung und Simulation von Lastprofilen batterieelektrischer Fahrzeuge zur Auslegung von Ladestationen in Wohnhausanlagen,” Mar. 2020.
    1: 15,
    2: 16,
    3: 21,
    4: 22,
    5: 29,
    6: 35,
    7: 39,
    8: 52,
    9: 55,
    10: 62,
    11: 63,
    12: 79,
}
N_TRIPS_PROBS: dict[
    TypeDay,
    dict[DrivingCategory, tuple[float, float, float, float]],
] = {  # G. Preßmair, “Modellierung und Simulation von Lastprofilen batterieelektrischer Fahrzeuge zur Auslegung von Ladestationen in Wohnhausanlagen,” Mar. 2020.
    "WD": {
        "low": (0.500, 0.395, 0.097, 0.008),
        "medium": (0.346, 0.574, 0.064, 0.016),
        "high": (0.355, 0.549, 0.088, 0.008),
    },
    "WE": {
        "low": (0.437, 0.501, 0.062, 0.000),
        "medium": (0.458, 0.480, 0.062, 0.000),
        "high": (0.334, 0.500, 0.166, 0.000),
    },
}
CHARGING_TIMES: dict[
    TypeDay,
    dict[NTrips, tuple[float, float]],
] = {  # G. Preßmair, “Modellierung und Simulation von Lastprofilen batterieelektrischer Fahrzeuge zur Auslegung von Ladestationen in Wohnhausanlagen,” Mar. 2020.
    "WD": {
        0: (0.0, 24.0),
        1: (9.5, 24.0),
        2: (11.0, 24.0),
        3: (13.0, 24.0),
    },
    "WE": {
        0: (0.0, 24.0),
        1: (9.0, 23.5),
        2: (12.0, 24.0),
        3: (13.0, 24.0),
    },
}
MED_DISTANCE_THRESH = 10_000
HIGH_DISTANCE_THRESH = 20_000
WD_THRESH = 5
DAYS_PER_WEEK = 7


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Driver:
    daily_mean_distance: float

    def distance_distribution(self, n_samples: int = 1) -> np.ndarray:
        sigma2 = np.log(self.daily_mean_distance / self.daily_mean_distance**2 + 1)
        mu = np.log(self.daily_mean_distance) - sigma2 / 2
        return GEN.lognormal(mu, np.sqrt(sigma2), n_samples)

    def number_of_trips(self, td: TypeDay) -> NTrips:
        p = N_TRIPS_PROBS[td][self.driving_category]
        return t.cast("NTrips", GEN.choice(a=(0, 1, 2, 3), p=p))

    @property
    def driving_category(self) -> DrivingCategory:
        if self.daily_mean_distance * 365 < MED_DISTANCE_THRESH:
            return "low"

        if self.daily_mean_distance * 365 < HIGH_DISTANCE_THRESH:
            return "medium"

        return "high"

    @property
    def daily_distance_wd(self) -> float:
        sigma2 = np.log(1 / self.daily_mean_distance + 1)
        mu = np.log(self.daily_mean_distance) - sigma2 / 2
        return float(GEN.lognormal(mu, np.sqrt(sigma2)))

    @property
    def daily_distance_we(self) -> float:
        sigma2 = np.log(2 / self.daily_mean_distance + 1)
        mu = np.log(self.daily_mean_distance) - sigma2 / 2
        return float(GEN.lognormal(mu, np.sqrt(sigma2)))

    @classmethod
    def from_random(cls) -> Driver:
        profile: Profile = GEN.integers(low=1, high=12)  # type: ignore[assignment]
        return cls.from_profile(profile=profile)

    @classmethod
    def from_profile(cls, profile: Profile) -> Driver:
        daily_mean_distance = DAILY_MEAN_DISTANCES[profile]
        return cls(daily_mean_distance=daily_mean_distance)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Car:
    mileage: float  # kWh/100 km
    losses: float  # p.u.
    capacity: float  # kWh
    charging_power: float  # kW


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class ChargePoint:
    charging_power_max: float  # kW


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class EVSystem(Tech):
    driver: Driver
    car: Car
    charge_point: ChargePoint
    tz: dt.tzinfo

    @staticmethod
    def charged_internally() -> bool:
        return t.cast("bool", GEN.choice(a=[False, True], p=[0.15, 0.85]))

    def charging_start_point(self, td: TypeDay) -> float:
        start, end = CHARGING_TIMES[td][self.driver.number_of_trips(td)]
        return self.charging_time(start=start, end=end)

    @staticmethod
    def charging_time(start: float, end: float) -> float:
        mean = (end - start) / 2
        sigma2 = np.log(1 / mean + 1)
        mu = np.log(mean) - sigma2 / 2
        return float(GEN.lognormal(mu, np.sqrt(sigma2)))

    def energy(self, td: TypeDay) -> float:
        if (self.driver.number_of_trips(td) > 0) and self.charged_internally():
            match td:
                case "WD":
                    distance = self.driver.daily_distance_wd
                case "WE":
                    distance = self.driver.daily_distance_we

            return distance * self.car.mileage / 100 * (1 + self.car.losses)

        return 0

    def power(self, td: TypeDay) -> np.ndarray:
        rv = np.zeros(24 * 4)
        energy = self.energy(td)
        csp = 24 * 4 + 1
        while csp > 24 * 4:
            csp = int(self.charging_start_point(td) * 4 // 1)

        n_full = int((energy * 4) // self.car.charging_power)
        last_power = ((energy * 4) / self.car.charging_power - n_full) * self.car.charging_power
        rv[csp : csp + n_full] = self.car.charging_power
        rv[csp + n_full] = last_power
        return rv

    def run(self) -> None:
        index = dates.date_range(self.tz, freq=dt.timedelta(minutes=15), year=self.dates.year[0])
        type_day_start = int(self.dates.weekday[0])
        tds: list[TypeDay] = ["WD" if (i + type_day_start) % DAYS_PER_WEEK < WD_THRESH else "WE" for i in range(365)]
        acp = pd.Series(np.concatenate([self.power(td) for td in tds]), index=index, name="p_el")
        self.acp.loc[:, "high"] = acp.resample(self.dates.freq).mean()


CARS = {  # G. Preßmair, “Modellierung und Simulation von Lastprofilen batterieelektrischer Fahrzeuge zur Auslegung von Ladestationen in Wohnhausanlagen,” Mar. 2020.
    "BMW i3": Car(mileage=13.52, losses=0.15, capacity=27.2, charging_power=10.5),
    "VW e-Golf": Car(mileage=15.49, losses=0.108, capacity=31.5, charging_power=7.1),
    "Hyundai Ioniq": Car(mileage=13.13, losses=0.104, capacity=28, charging_power=7.04),
    "Renault Zoe": Car(mileage=16.15, losses=0.207, capacity=41, charging_power=20.6),
}
