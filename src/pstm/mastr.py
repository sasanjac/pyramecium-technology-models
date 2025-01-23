# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from open_mastr import Mastr

if TYPE_CHECKING:
    from pstm.utils.geo import GeoRef

THRESHOLD = 0.01
LAT_MIN = 47.271679
LAT_MAX = 55.0846
LON_MIN = 5.866944
LON_MAX = 15.04193


@dataclass
class MaStR:
    georef: GeoRef

    def __post_init__(self) -> None:
        self.db = Mastr()
        # if (datetime.date.today - self.db.date) > datetime.timedelta(days=7):
        with self.db.engine.connect() as connection:
            table = "wind_extended"
            units = pd.read_sql(sql=table, con=connection)
            units = units[units["Lage"] == "Windkraft an Land"]
            units = units[["Laengengrad", "Breitengrad", "Nettonennleistung", "Nabenhoehe"]]
            units = units[units["Laengengrad"] > LON_MIN]
            units = units[units["Laengengrad"] < LON_MAX]
            units = units[units["Breitengrad"] > LAT_MIN]
            units = units[units["Breitengrad"] < LAT_MAX]
            units = units[units["Nettonennleistung"] > units["Nettonennleistung"].quantile(0.25)]
            units = units[units["Nabenhoehe"] > units["Nabenhoehe"].quantile(0.25)]
            self.units_wind = units.dropna()

        voronoi = [
            self.georef.get_voronoi(lat=unit.Breitengrad, lon=unit.Laengengrad)
            for _, unit in self.units_wind.iterrows()
        ]
        self.units_wind["voronoi"] = voronoi

    def installed_wind_farms(self, lat: float, lon: float, power_installed: float) -> list[WindFarm]:
        voronois = self.georef.get_voronois(lat=lat, lon=lon)
        for voronoi in voronois:
            installed_units = self.units_wind[self.units_wind["voronoi"] == voronoi]
            if not installed_units.empty:
                break

        power_installed_current = installed_units.Nettonennleistung.sum() * 1e3  # kW in W
        unit_factor = math.ceil(power_installed * 1e6 / power_installed_current)
        power_factor = power_installed * 1e6 / (power_installed_current * unit_factor)
        return [
            WindFarm(
                lat=unit.Breitengrad,
                lon=unit.Laengengrad,
                hub_heights=[unit.Nabenhoehe] * unit_factor,
                powers_installed=[unit.Nettonennleistung * 1e3 * power_factor] * unit_factor,
            )
            for _, unit in installed_units.iterrows()
        ]


# @dataclass
# class MaStR_:

#     def __post_init__(self) -> None:

#     def get_for_plz(self, plz: int, column: str | None = None) -> pd.Series:
#         if column is None:


@dataclass
class WindFarm:
    lat: float
    lon: float
    powers_installed: list[float]
    hub_heights: list[float]


# @dataclass
# class InstalledUnits:

#     def __post_init__(self) -> None:

#     def interpolate(self, power_installed_current: float) -> npt.NDArray[np.float64]:
#         return self._interpolate(

#     def _interpolate(
#         self,
#         power_installed_current: float,
#         sample: npt.NDArray[np.float64],
#     ) -> npt.NDArray[np.float64]:
#         if abs(sample.sum() - power_installed_current) / power_installed_current < THRESHOLD:
