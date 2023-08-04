# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import pandas as pd
import pyproj

from pstm.utils import dates

if TYPE_CHECKING:
    import datetime
    import pathlib

    from pstm.utils.geo import GeoRef


@dataclass
class WeatherGenerator:
    lat: float
    lon: float
    ghi: pd.Series  # in W/m^2
    dhi: pd.Series  # in W/m^2
    temp_air: pd.Series  # in K
    speed_wind: pd.Series  # in m/s
    pressure: pd.Series  # in Pa
    roughness_length: float
    tz: datetime.tzinfo
    freq: str = field(default="1h", repr=False)
    year: int = 2050

    def __post_init__(self) -> None:
        self.times = dates.date_range(self.tz, self.freq, year=self.year)

    @property
    def weather_at_pv_module(self) -> pd.DataFrame:
        dataframe = pd.concat([self.dhi, self.ghi, self.temp_air_celsius, self.speed_wind], axis=1)
        dataframe.index = self.times
        dataframe.columns = pd.Index(["dhi", "ghi", "temp_air", "wind_speed"])
        return dataframe

    @property
    def weather_at_wind_turbine(self) -> pd.DataFrame:
        rl = self.speed_wind.copy()
        rl[:] = self.roughness_length
        dataframe = pd.concat([self.speed_wind, self.temp_air, self.pressure, rl], axis=1)
        dataframe.columns = pd.MultiIndex.from_arrays(
            [
                ["wind_speed", "temperature", "pressure", "roughness_length"],
                [10, 2, 0, 0],
            ],
        )
        dataframe.index = self.times
        return dataframe

    @property
    def temp_air_celsius(self) -> pd.Series:
        return self.temp_air - 273.15

    def to_feather(self, file_path: pathlib.Path) -> None:
        dataframe = pd.concat([self.dhi, self.ghi, self.speed_wind, self.temp_air, self.pressure], axis=1)
        dataframe.columns = pd.Index(
            [
                "dhi",
                "ghi",
                "sw",
                "ta",
                "p",
            ],
        )
        dataframe = dataframe.reset_index()
        dataframe.to_feather(file_path)
        metadata = {
            "tz": self.tz,
            "lat": self.lat,
            "lon": self.lon,
            "rl": self.roughness_length,
            "freq": self.freq,
            "year": self.year,
        }
        with file_path.with_suffix(".meta").open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, fp=file_handle)

    @classmethod
    def from_dwd(  # noqa: PLR0913
        cls,
        dwd_file_path: pathlib.Path,
        georef: GeoRef,
        lat: float,
        lon: float,
        tz: datetime.tzinfo | None = None,
        index: pd.DatetimeIndex | None = None,
        freq: str = "1h",
        year: int = 2050,
    ) -> WeatherGenerator:
        weather = pd.read_csv(dwd_file_path, skiprows=34, sep=r"\s+")
        weather = weather.dropna()
        if tz is None:
            tz = georef.get_time_zone(lat=lat, lon=lon)

        if index is None:
            index = dates.date_range(tz, "1h", year=year)

        weather.index = index
        if freq != "1h":
            new_index = dates.date_range(tz, freq, year=year)
            weather = weather.resample(freq).mean()
            weather = weather.reindex(new_index, method="ffill")

        roughness_length = georef.get_roughness_length(lat, lon)
        return WeatherGenerator(
            lat=lat,
            lon=lon,
            tz=tz,
            ghi=weather["B"] + weather["D"],
            dhi=weather["D"],
            temp_air=weather["t"] + 273.15,
            speed_wind=weather["WG"],
            pressure=weather["p"] * 100,
            roughness_length=roughness_length,
            freq=freq,
            year=year,
        )

    @classmethod
    def from_feather(cls, file_path: pathlib.Path) -> WeatherGenerator:
        dataframe = pd.read_feather(file_path)
        dataframe.index = pd.Index(dataframe.pop("index"))
        with file_path.with_suffix(".meta").open(encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)

        transformer = pyproj.Transformer.from_crs("EPSG:3034", "EPSG:4326")
        lat, lon = transformer.transform(xx=metadata["lat"], yy=metadata["lon"])
        return WeatherGenerator(
            lat=lat,
            lon=lon,
            tz=metadata["tz"],
            roughness_length=metadata["rl"],
            freq=metadata["freq"],
            year=metadata["year"],
            ghi=dataframe.ghi,
            dhi=dataframe.dhi,
            temp_air=dataframe.ta,
            speed_wind=dataframe.sw,
            pressure=dataframe.p,
        )
