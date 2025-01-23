# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import datetime as dt
import pathlib
import zoneinfo as zi
from typing import Literal

import click
from loguru import logger

from pstm.utils.dates import date_range
from pstm.utils.geo import GeoRef
from pstm.utils.weather import WeatherGenerator

# Example input file name: TRY2045_37205002705500_Jahr.dat
# Example input file name: dwd_try_{year}_{scenario}_{index}_epsg3034.feather


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the input files directory.",
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, path_type=pathlib.Path),
    help="Path to the output files directory.",
)
@click.option(
    "--year",
    type=click.INT,
    help="DWD TRY year.",
)
@click.option(
    "--scenario",
    help="DWD TRY scenario.",
)
@click.option(
    "--skiprows",
    type=click.INT,
    help="Rows to skip (2015: 32, 2045: 34).",
)
@click.option("--delete", is_flag=True, default=False, help="Whether to delete the source file.")
def convert(
    *,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    year: int,
    scenario: Literal["mittel", "sommerwarm", "winterkalt"],
    skiprows: int,
    delete: bool = False,
) -> None:
    dwd_files = [file for file in input_path.iterdir() if file.suffix == ".dat"]
    n_files = len(dwd_files)
    output_path.mkdir(exist_ok=True, parents=True)
    tz = zi.ZoneInfo("Europe/Berlin")
    date_index = date_range(tz, freq=dt.timedelta(hours=1), year=year)
    file_name_template = f"dwd_try_{year}_{scenario}_{{:06d}}_epsg3034.feather"
    with GeoRef(
        weather_gen_files_path=output_path,
        dwd_try_year=year,
        dwd_try_scenario=scenario,
        reference_epsg=3034,
    ) as georef:
        for i, file_path in enumerate(dwd_files):
            if file_path.suffix == ".dat":
                logger.debug("Converting ({i}/{n_files}): {name}...", i=i, n_files=n_files, name=file_path.stem)
                coord = file_path.stem.split("_")[1]
                lat = int(coord[len(coord) // 2 :])
                lon = int(coord[: len(coord) // 2])
                index = georef.get_weather_gen_index(lat=lat, lon=lon)
                output_file = output_path / str(year) / scenario / file_name_template.format(index)
                if output_file.exists():
                    continue

                wg = WeatherGenerator.from_dwd(
                    file_path,
                    georef=georef,
                    lat=lat,
                    lon=lon,
                    year=year,
                    tz=tz,
                    index=date_index,
                    freq=dt.timedelta(hours=1),
                    skiprows=skiprows,
                )

                wg.to_feather(output_file)
                if delete:
                    logger.debug("Deleting {file_path}...", file_path=file_path)
                    file_path.unlink()
                    logger.debug("Deleting {file_path}. Done.", file_path=file_path)


if __name__ == "__main__":
    convert()
