# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import datetime as dt
import json
import pathlib
import tempfile
import typing as t
import zoneinfo as zi

from loguru import logger
import numpy as np
import pandas as pd
from pyochain import NONE
from pyochain import Err
from pyochain import Null
from pyochain import Ok
from pyochain import Some

from pstm.utils.dates import Dates
from pstm.utils.exceptions import MissingExtraError

try:
    import aiohttp
    import netCDF4
    import pyproj
    import requests
except ModuleNotFoundError as e:
    extra = "weather"
    raise MissingExtraError(extra) from e

if t.TYPE_CHECKING:
    from pyochain import Option
    from pyochain import Result

    from pstm.utils.geo import GeoRef

    type Array1DF = np.ndarray[tuple[int], np.dtype[np.float64]]


NEWA_BASE_FILE_NAME = "NEWA_WEATHER_{lat}-{lon}_{year}.nc"
NEWA_HEIGHTS = (50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 500.0)  # m, the levels of NEWA_BASE_URL
NEWA_BASE_FEATHER_FILE_NAME = "NEWA_WEATHER_{lat:.3f}-{lon:.3f}_{year}.feather"
NEWA_BASE_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "weather" / "newa"
NEWA_BASE_URL = "https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/get-data-point?latitude={lat}&longitude={lon}&height=50&height=75&height=100&height=150&height=200&height=250&height=500&variable=HGT&variable=LU_INDEX&variable=LANDMASK&variable=ZNT&variable=T2&variable=WS&variable=T&variable=PD&dt_start={year}-01-01T00:00:00&dt_stop={year2}-01-01T00:00:00"
R_SPEC_AIR = 287.0500676
DOWNLOAD_OK = 200
CHUNK_SIZE = 16384


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
        self.times = Dates.date_range(self.tz, self.freq, year=self.year)

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
            "alt": self.alt,
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
        alt: Option[float] = NONE,
        tz: Option[dt.tzinfo] = NONE,
        index: Option[pd.DatetimeIndex] = NONE,
        freq: dt.timedelta = dt.timedelta(hours=1),
        year: int = 2050,
        skiprows: int = 32,
    ) -> WeatherGenerator:
        weather = pd.read_csv(dwd_file_path, skiprows=skiprows, sep=r"\s+")
        weather = weather.drop(0, axis=0)
        weather = weather.dropna()
        tz_ = tz.unwrap_or_else(lambda: georef.get_time_zone(lat=lat, lon=lon))
        alt_ = alt.unwrap_or_else(lambda: georef.get_altitude(lat=lat, lon=lon))
        weather.index = index.unwrap_or_else(lambda: Dates.date_range(tz_, freq=freq, year=year))
        if freq != dt.timedelta(hours=1):
            new_index = Dates.date_range(tz_, freq, year=year)
            # resample().mean() leaves the newly created sub-hourly bins as NaN; reindex(method="ffill")
            # cannot fill them because the labels already exist, so forward-fill the values explicitly.
            weather = weather.resample(freq).mean().reindex(new_index).ffill()

        roughness_length = georef.get_roughness_length(lat, lon)
        return WeatherGenerator(
            lat=lat,
            lon=lon,
            alt=alt_,
            tz=tz_,
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
        tz = zi.ZoneInfo(metadata["tz"])
        dataframe.index = Dates.date_range(year=metadata["year"], freq=freq, tz=tz)
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
    roughness_length: Array1DF
    wind_speed_50: Array1DF
    wind_speed_75: Array1DF
    wind_speed_100: Array1DF
    wind_speed_150: Array1DF
    wind_speed_200: Array1DF
    wind_speed_250: Array1DF
    wind_speed_500: Array1DF
    wind_power_density_50: Array1DF
    wind_power_density_75: Array1DF
    wind_power_density_100: Array1DF
    wind_power_density_150: Array1DF
    wind_power_density_200: Array1DF
    wind_power_density_250: Array1DF
    wind_power_density_500: Array1DF
    temperature_2: Array1DF
    temperature_50: Array1DF
    temperature_75: Array1DF
    temperature_100: Array1DF
    temperature_150: Array1DF
    temperature_200: Array1DF
    temperature_250: Array1DF
    temperature_500: Array1DF

    @property
    def weather(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.data, columns=self.columns, index=self.index)

    @property
    def density_50(self) -> Array1DF:
        return self.density(wind_power_density=self.wind_power_density_50, wind_speed=self.wind_speed_50)

    @property
    def pressure_50(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_50, density=self.density_50)

    @property
    def density_75(self) -> Array1DF:
        return self.density(wind_power_density=self.wind_power_density_75, wind_speed=self.wind_speed_75)

    @property
    def pressure_75(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_75, density=self.density_75)

    @property
    def density_100(self) -> Array1DF:
        return self.density(
            wind_power_density=self.wind_power_density_100,
            wind_speed=self.wind_speed_100,
        )

    @property
    def pressure_100(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_100, density=self.density_100)

    @property
    def density_150(self) -> Array1DF:
        return self.density(
            wind_power_density=self.wind_power_density_150,
            wind_speed=self.wind_speed_150,
        )

    @property
    def pressure_150(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_150, density=self.density_150)

    @property
    def density_200(self) -> Array1DF:
        return self.density(
            wind_power_density=self.wind_power_density_200,
            wind_speed=self.wind_speed_200,
        )

    @property
    def pressure_200(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_200, density=self.density_200)

    @property
    def density_250(self) -> Array1DF:
        return self.density(
            wind_power_density=self.wind_power_density_250,
            wind_speed=self.wind_speed_250,
        )

    @property
    def pressure_250(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_250, density=self.density_250)

    @property
    def density_500(self) -> Array1DF:
        return self.density(
            wind_power_density=self.wind_power_density_500,
            wind_speed=self.wind_speed_500,
        )

    @property
    def pressure_500(self) -> Array1DF:
        return self.pressure(temperature=self.temperature_500, density=self.density_500)

    @property
    def data(self) -> Array1DF:
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
        return t.cast("Array1DF", np.stack(data, axis=1))

    @property
    def data_feather(self) -> Array1DF:
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
        return t.cast("Array1DF", np.stack(data, axis=1))

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
                "roughness_length_0",
                "wind_speed_50",
                "wind_speed_75",
                "wind_speed_100",
                "wind_speed_150",
                "wind_speed_200",
                "wind_speed_250",
                "wind_speed_500",
                "wind_power_density_50",
                "wind_power_density_75",
                "wind_power_density_100",
                "wind_power_density_150",
                "wind_power_density_200",
                "wind_power_density_250",
                "wind_power_density_500",
                "temperature_2",
                "temperature_50",
                "temperature_75",
                "temperature_100",
                "temperature_150",
                "temperature_200",
                "temperature_250",
                "temperature_500",
            ],
        )

    def to_feather(self, file_path: pathlib.Path) -> None:
        """Write the series with the index as the column ``time``, which :meth:`from_feather` restores."""
        dataframe = pd.DataFrame(data=self.data_feather, columns=self.columns_feather)
        dataframe.insert(0, "time", self.index)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_feather(file_path)

    @classmethod
    def from_feather(cls, file_path: pathlib.Path) -> NEWA:
        dataframe = pd.read_feather(file_path)
        dataframe = dataframe.set_index("time").rename_axis(None)
        return cls(
            index=t.cast("pd.DatetimeIndex", dataframe.index),
            roughness_length=dataframe["roughness_length_0"].to_numpy(),
            wind_speed_50=t.cast("Array1DF", dataframe["wind_speed_50"].to_numpy()),
            wind_speed_75=t.cast("Array1DF", dataframe["wind_speed_75"].to_numpy()),
            wind_speed_100=t.cast("Array1DF", dataframe["wind_speed_100"].to_numpy()),
            wind_speed_150=t.cast("Array1DF", dataframe["wind_speed_150"].to_numpy()),
            wind_speed_200=t.cast("Array1DF", dataframe["wind_speed_200"].to_numpy()),
            wind_speed_250=t.cast("Array1DF", dataframe["wind_speed_250"].to_numpy()),
            wind_speed_500=t.cast("Array1DF", dataframe["wind_speed_500"].to_numpy()),
            wind_power_density_50=t.cast("Array1DF", dataframe["wind_power_density_50"].to_numpy()),
            wind_power_density_75=t.cast("Array1DF", dataframe["wind_power_density_75"].to_numpy()),
            wind_power_density_100=t.cast("Array1DF", dataframe["wind_power_density_100"].to_numpy()),
            wind_power_density_150=t.cast("Array1DF", dataframe["wind_power_density_150"].to_numpy()),
            wind_power_density_200=t.cast("Array1DF", dataframe["wind_power_density_200"].to_numpy()),
            wind_power_density_250=t.cast("Array1DF", dataframe["wind_power_density_250"].to_numpy()),
            wind_power_density_500=t.cast("Array1DF", dataframe["wind_power_density_500"].to_numpy()),
            temperature_2=t.cast("Array1DF", dataframe["temperature_2"].to_numpy()),
            temperature_50=t.cast("Array1DF", dataframe["temperature_50"].to_numpy()),
            temperature_75=t.cast("Array1DF", dataframe["temperature_75"].to_numpy()),
            temperature_100=t.cast("Array1DF", dataframe["temperature_100"].to_numpy()),
            temperature_150=t.cast("Array1DF", dataframe["temperature_150"].to_numpy()),
            temperature_200=t.cast("Array1DF", dataframe["temperature_200"].to_numpy()),
            temperature_250=t.cast("Array1DF", dataframe["temperature_250"].to_numpy()),
            temperature_500=t.cast("Array1DF", dataframe["temperature_500"].to_numpy()),
        )

    @classmethod
    def from_api(
        cls,
        lat: float,
        lon: float,
        year: int,
        tz: dt.tzinfo,
    ) -> Result[NEWA, requests.HTTPError | ValueError]:
        """NEWA series of a site, downloaded from the API.

        Returns:
            ``Err`` with the HTTP error of the request, or with the error of :meth:`from_nc`.
        """
        url = NEWA_BASE_URL.format(lat=lat, lon=lon, year=year, year2=int(year) + 1)
        response = requests.get(url, timeout=600, stream=True)
        if not response.ok:
            msg = f"{response.status_code} {response.reason} for url: {url}"
            return Err(requests.HTTPError(msg, response=response))

        with tempfile.NamedTemporaryFile(suffix=".nc", delete_on_close=False) as buf:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    buf.write(chunk)

            buf.flush()
            match cls.from_nc(file_path=buf.name, tz=tz):
                case Ok(newa):
                    return Ok(newa)
                case Err(error):
                    return Err(error)

    @classmethod
    async def from_api_async(
        cls,
        lat: float,
        lon: float,
        year: int,
        tz: dt.tzinfo,
    ) -> Result[NEWA, ValueError]:
        """NEWA series of a site from the feather cache, downloaded on a miss.

        A download that times out is repeated after one second.

        Returns:
            ``Err`` with the error of :meth:`download_and_convert`.
        """
        lat = round(lat, 3)
        lon = round(lon, 3)
        file_path = NEWA_BASE_PATH / NEWA_BASE_FEATHER_FILE_NAME.format(lat=lat, lon=lon, year=year)
        if file_path.exists():
            return Ok(cls.from_feather(file_path=file_path))

        logger.trace("Downloading NEWA data for lat={lat}, lon={lon}, year={year}", lat=lat, lon=lon, year=year)
        url = NEWA_BASE_URL.format(lat=lat, lon=lon, year=year, year2=int(year) + 1)
        while True:
            try:
                return await cls.download_and_convert(url=url, file_path=file_path, tz=tz)
            except TimeoutError:  # raised by aiohttp
                await asyncio.sleep(1)

    @classmethod
    async def download_and_convert(cls, url: str, file_path: pathlib.Path, tz: dt.tzinfo) -> Result[NEWA, ValueError]:
        """Download a NEWA point file, convert it and write it to ``file_path``.

        Returns:
            ``Err`` if the API does not answer with status 200, or with the error of :meth:`from_nc`.
        """
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, timeout=aiohttp.ClientTimeout(600)) as response,
        ):
            if response.status != DOWNLOAD_OK:
                return Err(ValueError(f"Error: {response.status}"))

            with tempfile.NamedTemporaryFile(delete_on_close=False) as buf:
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    if chunk:
                        buf.write(chunk)

                buf.flush()
                logger.trace("Converting NEWA data to feather format")
                return cls.from_nc(file_path=buf.name, tz=tz).inspect(lambda data: data.to_feather(file_path=file_path))

    @classmethod
    def from_nc(cls, file_path: pathlib.Path | str, tz: dt.tzinfo) -> Result[NEWA, ValueError]:
        """Read a NEWA mesoscale time-series point file.

        The ``time`` variable counts minutes in UTC, so the index is built from
        it in UTC and converted to ``tz``.  The API includes the instant of
        ``dt_stop``, which belongs to the next year, and the files written in
        2024 end one step before it.  Every step at or after
        ``temporalExtentMaximum`` is dropped, so both give the same index.

        Returns:
            ``Err`` if the file holds other heights than :data:`NEWA_HEIGHTS`.
        """
        data = netCDF4.Dataset(file_path, mode="r")
        heights = [float(h) for h in np.asarray(data.variables["height"][:])]
        if heights != list(NEWA_HEIGHTS):
            data.close()
            msg = f"NEWA file {file_path} holds the heights {heights}, expected {list(NEWA_HEIGHTS)}."
            return Err(ValueError(msg))

        time = data.variables["time"]
        stamps = np.asarray(netCDF4.num2date(time[:], units=time.units, only_use_cftime_datetimes=False), dtype=object)
        index_utc = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in stamps.ravel()]).tz_localize("UTC")
        stop = pd.Timestamp(data.temporalExtentMaximum).tz_localize("UTC")
        keep = np.asarray(index_utc < stop)
        index = index_utc[keep].tz_convert(tz)

        def var(name: str, level: Option[int] = NONE) -> Array1DF:
            match level:
                case Some(level_):
                    values = data.variables[name][:, level_]
                case Null():
                    values = data.variables[name][:]
            return t.cast("Array1DF", np.asarray(values, dtype=np.float64)[keep])

        newa = cls(
            index=index,
            roughness_length=var("ZNT"),
            wind_speed_50=var("WS", Some(0)),
            wind_speed_75=var("WS", Some(1)),
            wind_speed_100=var("WS", Some(2)),
            wind_speed_150=var("WS", Some(3)),
            wind_speed_200=var("WS", Some(4)),
            wind_speed_250=var("WS", Some(5)),
            wind_speed_500=var("WS", Some(6)),
            wind_power_density_50=var("PD", Some(0)),
            wind_power_density_75=var("PD", Some(1)),
            wind_power_density_100=var("PD", Some(2)),
            wind_power_density_150=var("PD", Some(3)),
            wind_power_density_200=var("PD", Some(4)),
            wind_power_density_250=var("PD", Some(5)),
            wind_power_density_500=var("PD", Some(6)),
            temperature_2=var("T2"),
            temperature_50=var("T", Some(0)),
            temperature_75=var("T", Some(1)),
            temperature_100=var("T", Some(2)),
            temperature_150=var("T", Some(3)),
            temperature_200=var("T", Some(4)),
            temperature_250=var("T", Some(5)),
            temperature_500=var("T", Some(6)),
        )
        data.close()
        return Ok(newa)

    @staticmethod
    def pressure(
        temperature: Array1DF,
        density: Array1DF,
    ) -> Array1DF:  # Wikipedia
        return t.cast("Array1DF", temperature * R_SPEC_AIR * density)

    @staticmethod
    def density(
        wind_power_density: Array1DF,
        wind_speed: Array1DF,
    ) -> Array1DF:  # [1] A. Kalmikov, “Wind Power Fundamentals,” in Wind Energy Engineering, Elsevier, 2017, pp. 17-24. doi: 10.1016/B978-0-12-809451-8.00002-3.
        return t.cast("Array1DF", 2 * wind_power_density / wind_speed**3)
