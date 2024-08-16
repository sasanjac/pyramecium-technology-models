# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import datetime as dt
import json
import typing as t
import zoneinfo
from dataclasses import dataclass
from dataclasses import field

import aiofiles
import aiohttp
import netCDF4
import numpy as np
import pandas as pd
import pyproj

from pstm.utils import dates

if t.TYPE_CHECKING:
    import pathlib

    import numpy.typing as npt

    from pstm.utils.geo import GeoRef

    Array: t.TypeAlias = npt.NDArray[np.float64]


NEWA_BASE_FILE_NAME = "NEWA_WEATHER_{lat}-{lon}_{year}.nc"
NEWA_BASE_URL = "https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/get-data-point?latitude={lat}&longitude={lon}&height=50&height=75&height=100&height=150&height=200&height=250&height=500&variable=HGT&variable=LU_INDEX&variable=LANDMASK&variable=ZNT&variable=T2&variable=WS&variable=T&variable=PD&dt_start={year}-01-01T00:00:00&dt_stop={year2}-01-01T00:00:00"
R_SPEC_AIR = 287.0500676
DOWNLOAD_OK = 200


@dataclass
class WeatherGenerator:
    lat: float
    lon: float
    alt: float
    ghi: pd.Series  # in W/m^2
    dhi: pd.Series  # in W/m^2
    temp_air: pd.Series  # in K
    speed_wind: pd.Series  # in m/s
    pressure: pd.Series  # in Pa
    roughness_length: float
    tz: dt.tzinfo
    freq: dt.timedelta = field(default=dt.timedelta(hours=1), repr=False)
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
        dataframe = dataframe.reset_index().drop(["index"], axis=1)
        dataframe.to_feather(file_path)
        metadata = {
            "tz": str(self.tz),
            "lat": self.lat,
            "lon": self.lon,
            "rl": self.roughness_length,
            "freq": str(self.freq.seconds),
            "year": self.year,
        }
        with file_path.with_suffix(".meta").open("w", encoding="utf-8") as file_handle:
            json.dump(metadata, fp=file_handle)

    @classmethod
    def from_dwd(
        cls,
        dwd_file_path: pathlib.Path,
        georef: GeoRef,
        lat: float,
        lon: float,
        alt: float | None = None,
        tz: dt.tzinfo | None = None,
        index: pd.DatetimeIndex | None = None,
        freq: dt.timedelta = dt.timedelta(hours=1),
        year: int = 2050,
        skiprows: int = 32,
    ) -> WeatherGenerator:
        weather = pd.read_csv(dwd_file_path, skiprows=skiprows, sep=r"\s+")
        weather = weather.drop(0, axis=0)
        weather = weather.dropna()
        if tz is None:
            tz = georef.get_time_zone(lat=lat, lon=lon)

        if alt is None:
            alt = georef.get_altitude(lat=lat, lon=lon)

        if index is None:
            index = dates.date_range(tz, freq=freq, year=year)

        weather.index = index
        if freq != dt.timedelta(hours=1):
            new_index = dates.date_range(tz, freq, year=year)
            weather = weather.resample(freq).mean()
            weather = weather.reindex(new_index, method="ffill")

        roughness_length = georef.get_roughness_length(lat, lon)
        return WeatherGenerator(
            lat=lat,
            lon=lon,
            alt=alt,
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
        with file_path.with_suffix(".meta").open(encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)

        freq = dt.timedelta(seconds=int(metadata["freq"]))
        alt = metadata["alt"]
        tz = zoneinfo.ZoneInfo(metadata["tz"])
        dataframe.index = dates.date_range(year=metadata["year"], freq=freq, tz=tz)
        transformer = pyproj.Transformer.from_crs("EPSG:3034", "EPSG:4326")
        lat, lon = transformer.transform(xx=metadata["lat"], yy=metadata["lon"])
        return WeatherGenerator(
            lat=lat,
            lon=lon,
            alt=alt,
            tz=tz,
            roughness_length=metadata["rl"],
            freq=freq,
            year=metadata["year"],
            ghi=dataframe.ghi,
            dhi=dataframe.dhi,
            temp_air=dataframe.ta,
            speed_wind=dataframe.sw,
            pressure=dataframe.p,
        )


@dataclass
class NEWA:
    index: pd.DatetimeIndex
    roughness_length: npt.NDArray[np.float64]
    wind_speed_50: npt.NDArray[np.float64]
    wind_speed_75: npt.NDArray[np.float64]
    wind_speed_100: npt.NDArray[np.float64]
    wind_speed_150: npt.NDArray[np.float64]
    wind_speed_200: npt.NDArray[np.float64]
    wind_speed_250: npt.NDArray[np.float64]
    wind_speed_500: npt.NDArray[np.float64]
    wind_power_density_50: npt.NDArray[np.float64]
    wind_power_density_75: npt.NDArray[np.float64]
    wind_power_density_100: npt.NDArray[np.float64]
    wind_power_density_150: npt.NDArray[np.float64]
    wind_power_density_200: npt.NDArray[np.float64]
    wind_power_density_250: npt.NDArray[np.float64]
    wind_power_density_500: npt.NDArray[np.float64]
    temperature_2: npt.NDArray[np.float64]
    temperature_50: npt.NDArray[np.float64]
    temperature_75: npt.NDArray[np.float64]
    temperature_100: npt.NDArray[np.float64]
    temperature_150: npt.NDArray[np.float64]
    temperature_200: npt.NDArray[np.float64]
    temperature_250: npt.NDArray[np.float64]
    temperature_500: npt.NDArray[np.float64]

    @property
    def weather(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.data, columns=self.columns, index=self.index)

    @property
    def density_50(self) -> npt.NDArray[np.float64]:
        return self.density(wind_power_density=self.wind_power_density_50, wind_speed=self.wind_speed_50)

    @property
    def pressure_50(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_50, density=self.density_50)

    @property
    def density_75(self) -> npt.NDArray[np.float64]:
        return self.density(wind_power_density=self.wind_power_density_75, wind_speed=self.wind_speed_75)

    @property
    def pressure_75(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_75, density=self.density_75)

    @property
    def density_100(self) -> npt.NDArray[np.float64]:
        return self.density(
            wind_power_density=self.wind_power_density_100,
            wind_speed=self.wind_speed_100,
        )

    @property
    def pressure_100(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_100, density=self.density_100)

    @property
    def density_150(self) -> npt.NDArray[np.float64]:
        return self.density(
            wind_power_density=self.wind_power_density_150,
            wind_speed=self.wind_speed_150,
        )

    @property
    def pressure_150(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_150, density=self.density_150)

    @property
    def density_200(self) -> npt.NDArray[np.float64]:
        return self.density(
            wind_power_density=self.wind_power_density_200,
            wind_speed=self.wind_speed_200,
        )

    @property
    def pressure_200(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_200, density=self.density_200)

    @property
    def density_250(self) -> npt.NDArray[np.float64]:
        return self.density(
            wind_power_density=self.wind_power_density_250,
            wind_speed=self.wind_speed_250,
        )

    @property
    def pressure_250(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_250, density=self.density_250)

    @property
    def density_500(self) -> npt.NDArray[np.float64]:
        return self.density(
            wind_power_density=self.wind_power_density_500,
            wind_speed=self.wind_speed_500,
        )

    @property
    def pressure_500(self) -> npt.NDArray[np.float64]:
        return self.pressure(temperature=self.temperature_500, density=self.density_500)

    @property
    def data(self) -> npt.NDArray[np.float64]:
        data = [
            self.roughness_length,
            self.wind_speed_50,
            self.wind_speed_75,
            self.wind_speed_100,
            self.wind_speed_150,
            self.wind_speed_200,
            self.wind_speed_250,
            self.wind_speed_500,
            self.pressure_50,
            self.pressure_75,
            self.pressure_100,
            self.pressure_150,
            self.pressure_200,
            self.pressure_250,
            self.pressure_500,
            self.temperature_2,
            self.temperature_50,
            self.temperature_75,
            self.temperature_100,
            self.temperature_150,
            self.temperature_200,
            self.temperature_250,
            self.temperature_500,
        ]
        return np.stack(data, axis=1)

    @property
    def data_feather(self) -> npt.NDArray[np.float64]:
        data = [
            self.roughness_length,
            self.wind_speed_50,
            self.wind_speed_75,
            self.wind_speed_100,
            self.wind_speed_150,
            self.wind_speed_200,
            self.wind_speed_250,
            self.wind_speed_500,
            self.wind_power_density_50,
            self.wind_power_density_75,
            self.wind_power_density_100,
            self.wind_power_density_150,
            self.wind_power_density_200,
            self.wind_power_density_250,
            self.wind_power_density_500,
            self.temperature_2,
            self.temperature_50,
            self.temperature_75,
            self.temperature_100,
            self.temperature_150,
            self.temperature_200,
            self.temperature_250,
            self.temperature_500,
        ]
        return np.stack(data, axis=1)

    @property
    def columns(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [
                [
                    "roughness_length",
                    "wind_speed",
                    "wind_speed",
                    "wind_speed",
                    "wind_speed",
                    "wind_speed",
                    "wind_speed",
                    "wind_speed",
                    "pressure",
                    "pressure",
                    "pressure",
                    "pressure",
                    "pressure",
                    "pressure",
                    "pressure",
                    "temperature",
                    "temperature",
                    "temperature",
                    "temperature",
                    "temperature",
                    "temperature",
                    "temperature",
                    "temperature",
                ],
                [
                    0,
                    50,
                    75,
                    100,
                    150,
                    200,
                    250,
                    500,
                    50,
                    75,
                    100,
                    150,
                    200,
                    250,
                    500,
                    2,
                    50,
                    75,
                    100,
                    150,
                    200,
                    250,
                    500,
                ],
            ],
        )

    @property
    def columns_feather(self) -> pd.Index:
        return pd.Index(
            [
                "roughness_length-0",
                "wind_speed-50",
                "wind_speed-75",
                "wind_speed-100",
                "wind_speed-150",
                "wind_speed-200",
                "wind_speed-250",
                "wind_speed-500",
                "wind_power_density-50",
                "wind_power_density-75",
                "wind_power_density-100",
                "wind_power_density-150",
                "wind_power_density-200",
                "wind_power_density-250",
                "wind_power_density-500",
                "temperature-2",
                "temperature-50",
                "temperature-75",
                "temperature-100",
                "temperature-150",
                "temperature-200",
                "temperature-250",
                "temperature-500",
            ],
        )

    def to_feather(self, file_path: pathlib.Path) -> None:
        dataframe = pd.DataFrame(data=self.data_feather, columns=self.columns_feather, index=self.index)
        dataframe.reset_index().to_feather(file_path)

    @classmethod
    def from_feather(cls, file_path: pathlib.Path) -> NEWA:
        dataframe = pd.read_feather(file_path)
        return cls(
            index=dataframe.index,
            roughness_length=dataframe["roughness_length"],
            wind_speed_50=dataframe["wind_speed_50"],
            wind_speed_75=dataframe["wind_speed_75"],
            wind_speed_100=dataframe["wind_speed_100"],
            wind_speed_150=dataframe["wind_speed_150"],
            wind_speed_200=dataframe["wind_speed_200"],
            wind_speed_250=dataframe["wind_speed_250"],
            wind_speed_500=dataframe["wind_speed_500"],
            wind_power_density_50=dataframe["wind_power_density_50"],
            wind_power_density_75=dataframe["wind_power_density_75"],
            wind_power_density_100=dataframe["wind_power_density_100"],
            wind_power_density_150=dataframe["wind_power_density_150"],
            wind_power_density_200=dataframe["wind_power_density_200"],
            wind_power_density_250=dataframe["wind_power_density_250"],
            wind_power_density_500=dataframe["wind_power_density_500"],
            temperature_2=dataframe["temperature_2"],
            temperature_50=dataframe["temperature_50"],
            temperature_75=dataframe["temperature_75"],
            temperature_100=dataframe["temperature_100"],
            temperature_150=dataframe["temperature_150"],
            temperature_200=dataframe["temperature_200"],
            temperature_250=dataframe["temperature_250"],
            temperature_500=dataframe["temperature_500"],
        )

    @classmethod
    async def download(
        cls,
        lat: float,
        lon: float,
        year: int,
        data_path: pathlib.Path,
    ) -> pathlib.Path:
        file_path = data_path / NEWA_BASE_FILE_NAME.format(lat=lat, lon=lon, year=year)
        if not file_path.exists():
            url = NEWA_BASE_URL.format(lat=lat, lon=lon, year=year, year2=int(year) + 1)
            chunk_size = 16384
            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as response,
            ):
                if response.status != DOWNLOAD_OK:
                    msg = f"Error: {response.status}"
                    raise ValueError(msg)

                async with aiofiles.open(file_path, mode="wb") as file_handle:
                    async for data in response.content.iter_chunked(chunk_size):
                        await file_handle.write(data)

        return file_path

    @classmethod
    def from_nc(cls, file_path: pathlib.Path, tz: dt.tzinfo) -> NEWA:
        data = netCDF4.Dataset(file_path)
        np_index = np.datetime64(
            dt.datetime(1989, 1, 1, 0, 0, 0, tzinfo=tz).astimezone(dt.UTC).replace(tzinfo=None),
        ) + data.variables["time"][:].astype(
            "timedelta64[m]",
        )
        index = pd.DatetimeIndex(np_index).tz_localize("utc").tz_convert(tz)
        return cls(
            index=index,
            roughness_length=data.variables["ZNT"][:].data,
            wind_speed_50=data.variables["WS"][:, 0].data,
            wind_speed_75=data.variables["WS"][:, 1].data,
            wind_speed_100=data.variables["WS"][:, 2].data,
            wind_speed_150=data.variables["WS"][:, 3].data,
            wind_speed_200=data.variables["WS"][:, 4].data,
            wind_speed_250=data.variables["WS"][:, 5].data,
            wind_speed_500=data.variables["WS"][:, 6].data,
            wind_power_density_50=data.variables["PD"][:, 0].data,
            wind_power_density_75=data.variables["PD"][:, 1].data,
            wind_power_density_100=data.variables["PD"][:, 2].data,
            wind_power_density_150=data.variables["PD"][:, 3].data,
            wind_power_density_200=data.variables["PD"][:, 4].data,
            wind_power_density_250=data.variables["PD"][:, 5].data,
            wind_power_density_500=data.variables["PD"][:, 6].data,
            temperature_2=data.variables["T2"][:].data,
            temperature_50=data.variables["T"][:, 0].data,
            temperature_75=data.variables["T"][:, 1].data,
            temperature_100=data.variables["T"][:, 2].data,
            temperature_150=data.variables["T"][:, 3].data,
            temperature_200=data.variables["T"][:, 4].data,
            temperature_250=data.variables["T"][:, 5].data,
            temperature_500=data.variables["T"][:, 6].data,
        )

    @staticmethod
    def pressure(
        temperature: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:  # Wikipedia
        return temperature * R_SPEC_AIR * density

    @staticmethod
    def density(
        wind_power_density: npt.NDArray[np.float64],
        wind_speed: npt.NDArray[np.float64],
    ) -> npt.NDArray[
        np.float64
    ]:  # [1] A. Kalmikov, “Wind Power Fundamentals,” in Wind Energy Engineering, Elsevier, 2017, pp. 17-24. doi: 10.1016/B978-0-12-809451-8.00002-3.
        return 2 * wind_power_density / wind_speed**3
