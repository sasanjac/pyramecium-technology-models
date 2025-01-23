# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import pathlib
import random
import typing as t
import zoneinfo as zi

import aiohttp
import attrs
import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rio
import shapely.geometry as shg
from loguru import logger
from scipy.spatial import distance

from pstm.utils.roughness_lengths import ROUGHNESS_LENGTHS
from pstm.utils.roughness_lengths import ValidCLCs

if t.TYPE_CHECKING:
    import collections.abc as cabc
    import datetime as dt
    from types import TracebackType

    import numpy.typing as npt

    T = t.TypeVar("T")

SRC_PATH = pathlib.Path(__file__).parent.parent.parent.parent
ALTITUDE_SERVICE_URL = "https://api.opentopodata.org/v1/eudem25m?locations={{}},{{}}"
DEFAULT_ALTITUDE_FILE_PATH = SRC_PATH / "data/geo/altitude_germany_20m_epsg25832.tif"
DEFAULT_CLC_FILE_PATH = SRC_PATH / "data/geo/clc_europe_epsg3035.feather"
DEFAULT_DWD_TRY_FILES_PATH = SRC_PATH / "data/weather/weather"
DEFAULT_DWD_TRY_SCENARIO: t.Literal["mittel", "sommerwarm", "winterkalt"] = "mittel"
DEFAULT_DWD_TRY_YEAR = 2045
DEFAULT_DWD_TRY_ZONES_FILE_PATH = SRC_PATH / "data/geo/dwd_try_zones_epsg4326.feather"
DEFAULT_NEWA_FILES_PATH = SRC_PATH / "data/weather/weather"
DEFAULT_NEWA_YEAR = 2018
DEFAULT_TIME_ZONES_FILE_PATH = SRC_PATH / "data/geo/time_zones_epsg4326.feather"
DEFAULT_WEATHER_GEN_FILES_PATH = SRC_PATH / "data/weather/weather"
DEFAULT_ZIP_CODES_FILE_PATH = SRC_PATH / "data/geo/zip_codes_germany_epsg4326.feather"


async def get_altitude_from_api(lat: float, lon: float) -> float:
    async with (
        aiohttp.ClientSession() as session,
        session.get(ALTITUDE_SERVICE_URL.format(lat, lon), timeout=aiohttp.ClientTimeout(total=5)) as response,
    ):
        json_body = await response.json()
        return json_body["results"][0]["altitude"]  # type:ignore[no-any-return]


@attrs.define(auto_attribs=True, kw_only=True, slots=False, unsafe_hash=True)
class GeoRef:
    zip_codes_file_path: pathlib.Path = attrs.field(default=DEFAULT_ZIP_CODES_FILE_PATH, hash=True)
    altitude_file_path: pathlib.Path = attrs.field(default=DEFAULT_ALTITUDE_FILE_PATH, hash=True)
    clc_file_path: pathlib.Path = attrs.field(default=DEFAULT_CLC_FILE_PATH, hash=True)
    dwd_try_zones_file_path: pathlib.Path = attrs.field(default=DEFAULT_DWD_TRY_ZONES_FILE_PATH, hash=True)
    dwd_try_year: int = attrs.field(default=DEFAULT_DWD_TRY_YEAR, hash=True)
    dwd_try_scenario: t.Literal["mittel", "sommerwarm", "winterkalt"] = attrs.field(
        default=DEFAULT_DWD_TRY_SCENARIO,
        hash=True,
    )
    dwd_try_files_path: pathlib.Path = attrs.field(default=DEFAULT_DWD_TRY_FILES_PATH, hash=True)
    weather_gen_files_path: pathlib.Path = attrs.field(default=DEFAULT_WEATHER_GEN_FILES_PATH, hash=True)
    newa_files_path: pathlib.Path = attrs.field(default=DEFAULT_NEWA_FILES_PATH, hash=True)
    newa_year: int = attrs.field(default=DEFAULT_NEWA_YEAR, hash=True)
    time_zones_file_path: pathlib.Path = attrs.field(default=DEFAULT_TIME_ZONES_FILE_PATH, hash=True)
    reference_epsg: int = attrs.field(default=4326, hash=True)
    use_raw_dwd_try_files: bool = attrs.field(default=False, hash=True)
    use_clc: bool = attrs.field(default=True, hash=True)
    voronoi_file_path: pathlib.Path | None = attrs.field(default=None, hash=True)

    def get_zip_code(self, lat: float, lon: float) -> int:
        return self.get_value_for_coord(self._zip_codes, lat=lat, lon=lon)

    def get_zip_code_center(self, zip_code: str) -> tuple[float, float]:
        matching_zip_codes = self._zip_codes.value == str(zip_code)
        polygon = self._zip_codes[matching_zip_codes].geometry.to_numpy()[0]
        point = polygon.centroid
        return self._transformer_3035_t.transform(xx=point.y, yy=point.x)

    def get_dwd_try_zone(self, lat: float, lon: float) -> str:
        zone: int = self.get_value_for_coord(self._dwd_try_zones, lat=lat, lon=lon)
        return f"TRY{zone:02d}"

    def get_weather_gen_index(self, lat: float, lon: float) -> int:
        return self.get_value_for_coord(self._weather_gen_index, lat=lat, lon=lon)

    def get_newa_index(self, lat: float, lon: float) -> int:
        return self.get_value_for_coord(self._newa_index, lat=lat, lon=lon)

    def get_voronoi(self, lat: float, lon: float) -> int:
        return self.get_value_for_coord(self._voronoi, lat=lat, lon=lon)

    def get_voronois(self, lat: float, lon: float) -> cabc.Sequence[int]:
        lat, lon = self._transformer_3035.transform(xx=lat, yy=lon)
        coord = shg.Point(lon, lat)
        return self._voronoi.distance(coord).sort_values().index.tolist()  # type:ignore[no-any-return]

    def get_value_for_coord(self, shape: gpd.GeoDataFrame[T], lat: float, lon: float) -> T:  # type: ignore[type-var, no-any-unimported]
        lat_, lon_ = self._transformer_3035.transform(xx=lat, yy=lon)
        coord = shg.Point(lon_, lat_)
        data = shape[shape.geometry.contains(coord)].value
        if len(data) == 0:
            logger.warning("Coordinate {lat}, {lon} not found in shape. Using closest distance.", lat=lat, lon=lon)
            return shape.iloc[shape.distance(coord).argmin()].value  # type:ignore[no-any-return]

        if len(data) != 1:
            logger.warning(
                "Coordinate {lat}, {lon} returned more than one element. Using first element.",
                lat=lat,
                lon=lon,
            )

        return data.to_numpy()[0]  # type:ignore[no-any-return]

    def get_weather_gen_file(self, lat: float, lon: float) -> pathlib.Path:
        if self.use_raw_dwd_try_files:
            msg = "`use_raw_dwd_try_files` is set to `true`. Either set to `false` or use `get_dwd_try_file`."
            raise RuntimeError(msg)

        index = self.get_weather_gen_index(lat=lat, lon=lon)
        return (
            self.weather_gen_files_path
            / str(self.dwd_try_year)
            / self.dwd_try_scenario
            / f"dwd_try_{self.dwd_try_year}_{self.dwd_try_scenario}_{index:06d}_epsg3034.feather"
        )

    def get_newa_file(self, lat: float, lon: float) -> pathlib.Path:
        index = self.get_newa_index(lat=lat, lon=lon)
        return self.newa_files_path / f"newa_{self.newa_year}_{index:06d}_epsg3034.feather"

    def get_dwd_try_file(self, lat: float, lon: float) -> pathlib.Path:
        if not self.use_raw_dwd_try_files:
            msg = "`use_raw_dwd_try_files` is set to `false`. Either set to `true` or use `get_weather_gen_file`."
            raise RuntimeError(msg)

        lon, lat = self._transformer_3034.transform(xx=lat, yy=lon)
        closest_index = distance.cdist([(lat, lon)], self._dwd_try_nodes).argmin()
        return self._dwd_try_files[closest_index]  # type:ignore[no-any-return]

    def get_altitude(self, lat: float, lon: float) -> float:
        lat, lon = self._transformer_25832.transform(xx=lat, yy=lon)
        return next(iter(self._altitude.sample(((lat, lon),))))[0]  # type:ignore[no-any-return]

    def get_roughness_length(self, lat: float, lon: float) -> float:
        clc = self.get_clc(lat=lat, lon=lon)
        min_length = ROUGHNESS_LENGTHS[clc]["rmin"]
        max_length = ROUGHNESS_LENGTHS[clc]["rmax"]
        return random.uniform(min_length, max_length)  # noqa: S311

    def get_clc(self, lat: float, lon: float) -> ValidCLCs:
        return self.get_value_for_coord(self._clc, lat=lat, lon=lon)

    def get_time_zone(self, lat: float, lon: float) -> dt.tzinfo:
        tz_str: str = self.get_value_for_coord(self._time_zones, lat=lat, lon=lon)
        return zi.ZoneInfo(tz_str)

    def __enter__(self) -> t.Self:
        self._init_zip_codes_file()
        self._init_altitude_file()
        self._init_dwd_try_zones_file()
        if self.use_raw_dwd_try_files:
            self._init_dwd_try_files()
        else:
            self._init_weather_gen_index_file()

        if self.use_clc:
            self._init_clc_file_path()

        self._init_time_zones_file()
        self._init_voronoi_file()
        self._init_transformers()

        return self

    def _init_zip_codes_file(self) -> None:
        logger.info("Loading zip codes file...")
        self._zip_codes: gpd.GeoDataFrame[int] = gpd.read_feather(self.zip_codes_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        logger.info("Loading zip codes file. Done.")

    def _init_altitude_file(self) -> None:
        logger.info("Loading altitude file...")
        crs = rio.CRS.from_epsg(25832)
        self._altitude: gpd.GeoDataFrame[float] = rio.open(self.altitude_file_path, crs=crs)  # type:ignore[no-any-unimported]
        logger.info("Loading altitude file. Done.")

    def _init_dwd_try_zones_file(self) -> None:
        logger.info("Loading DWD TRY zones file...")
        self._dwd_try_zones: gpd.GeoDataFrame[int] = gpd.read_feather(self.dwd_try_zones_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        logger.info("Loading DWD TRY zones file. Done.")

    def _init_dwd_try_files(self) -> None:
        logger.info("Loading DWD TRY files...")
        self._dwd_try_files = [
            f
            for f in (self.dwd_try_files_path / str(self.dwd_try_year) / self.dwd_try_scenario).iterdir()
            if f.is_file() and f.suffix == ".dat"
        ]
        dwd_try_coords = [f.name.split("_")[1] for f in self._dwd_try_files]
        dwd_try_right = np.array([coord[: len(coord) // 2] for coord in dwd_try_coords], dtype=np.float64)
        dwd_try_height = np.array([coord[len(coord) // 2 :] for coord in dwd_try_coords], dtype=np.float64)
        self._dwd_try_nodes: npt.NDArray[np.float64] = np.array(list(zip(dwd_try_height, dwd_try_right, strict=True)))
        logger.info("Loading DWD TRY files. Done.")

    def _init_weather_gen_index_file(self) -> None:
        logger.info("Loading DWD TRY index file...")
        index_file_path = (
            self.weather_gen_files_path
            / str(self.dwd_try_year)
            / self.dwd_try_scenario
            / f"dwd_try_{self.dwd_try_year}_{self.dwd_try_scenario}_index_epsg3034.feather"
        )
        try:
            self._weather_gen_index: gpd.GeoDataFrame[int] = gpd.read_feather(index_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        except FileNotFoundError:
            logger.warning("Could not find DWD TRY index file. Some methods may not work properly.")

        logger.info("Loading DWD TRY index file. Done.")

    def _init_weather_gen_files(self) -> None:
        logger.info("Loading weather generators files...")
        files = [
            f
            for f in (self.weather_gen_files_path / str(self.dwd_try_year) / self.dwd_try_scenario).iterdir()
            if f.is_file() and f.suffix == ".dat" and "index" not in f.name
        ]
        self._weather_gen_files: dict[int, pathlib.Path] = {int(f.stem.split("_")[-1]): f for f in files}
        logger.info("Loading weather generators files. Done.")

    def _init_newa_index_file(self) -> None:
        logger.info("Loading NEWA index file...")
        index_file_path = self.newa_files_path / f"newa_{self.newa_year}_index_epsg3034.feather"
        self._newa_index: gpd.GeoDataFrame[int] = gpd.read_feather(index_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        logger.info("Loading NEWA index file. Done.")

    def _init_newa_files(self) -> None:
        logger.info("Loading NEWA files...")
        files = [
            f
            for d in self.newa_files_path.iterdir()
            if d.is_dir()
            for f in d.iterdir()
            if (f.is_file() and f.suffix == ".feather" and str(self.newa_year) in f.name and "index" not in f.name)
        ]
        self._newa_files: dict[int, pathlib.Path] = {int(f.stem.split("_")[-1]): f for f in files}
        logger.info("Loading NEWA files. Done.")

    def _init_clc_file_path(self) -> None:
        logger.info("Loading CLC file...")
        self._clc: gpd.GeoDataFrame[ValidCLCs] = gpd.read_feather(self.clc_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        logger.info("Loading CLC file. Done.")

    def _init_time_zones_file(self) -> None:
        logger.info("Loading time zones file...")
        self._time_zones: gpd.GeoDataFrame[str] = gpd.read_feather(self.time_zones_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
        logger.info("Loading time zones file. Done.")

    def _init_voronoi_file(self) -> None:
        if self.voronoi_file_path is not None:
            logger.info("Loading voronoi file...")
            self._voronoi: gpd.GeoDataFrame[str] = gpd.read_feather(self.voronoi_file_path).to_crs(epsg=3035)  # type:ignore[no-any-unimported]
            logger.info("Loading voronoi file. Done.")
        else:
            logger.info("Skipping voronoi file...")

    def _init_transformers(self) -> None:
        logger.info("Creating transformers...")
        self._transformer_3034 = pyproj.Transformer.from_crs(f"EPSG:{self.reference_epsg}", "EPSG:3034")
        self._transformer_3035 = pyproj.Transformer.from_crs(f"EPSG:{self.reference_epsg}", "EPSG:3035")
        self._transformer_3035_t = pyproj.Transformer.from_crs("EPSG:3035", f"EPSG:{self.reference_epsg}")
        self._transformer_4326 = pyproj.Transformer.from_crs(f"EPSG:{self.reference_epsg}", "EPSG:4326")
        self._transformer_25832 = pyproj.Transformer.from_crs(f"EPSG:{self.reference_epsg}", "EPSG:25832")
        logger.info("Creating transformers. Done.")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._close_altitude_file()

    def _close_altitude_file(self) -> None:
        self._altitude.close()
