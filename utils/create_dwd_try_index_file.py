# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import pathlib

import click
import geopandas as gpd
import numpy as np
import shapely as sh
import shapely.geometry as shg
import shapely.ops as sho
from loguru import logger

# Example input file name: TRY2045_37205002705500_Jahr.dat
# Example input file name: dwd_try_{year}_{scenario}_index_epsg3034.feather


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the input files directory.",
)
@click.option(
    "--year",
    help="DWD TRY year.",
)
@click.option(
    "--scenario",
    help="DWD TRY scenario.",
)
def create(*, input_path: pathlib.Path, year: int, scenario: str) -> None:
    logger.info("Generating coords...")
    weather_coords = [f.name.split("_")[1] for f in input_path.iterdir() if f.suffix == ".dat"]
    weather_right = np.array([coord[: len(coord) // 2] for coord in weather_coords], dtype=np.float64)
    weather_height = np.array([coord[len(coord) // 2 :] for coord in weather_coords], dtype=np.float64)
    weather_nodes = np.array(list(zip(weather_right, weather_height, strict=True)))
    logger.info("Generating coords. Done.")
    logger.info("Calculating bounds...")
    minx, miny = weather_nodes.min(axis=0)
    maxx, maxy = weather_nodes.max(axis=0)
    logger.info("Calculating bounds. Done.")
    logger.info("Calculating voronoi...")
    points = sh.MultiPoint(weather_nodes)
    envelope = shg.box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    geometry = sho.voronoi_diagram(points, envelope=envelope)
    logger.info("Calculating voronoi. Done.")
    logger.info("Generating GeoDataFrame...")
    index = np.arange(0, len(weather_nodes))
    gdf = gpd.GeoDataFrame(geometry=list(geometry.geoms), data={"value": index}, crs="EPSG:3034")
    logger.info("Generating GeoDataFrame. Done.")
    output_file_path = input_path / f"dwd_try_{year}_{scenario}_index_epsg3034.feather"
    logger.info("Writing data to {output_file_path}...", output_file_path=output_file_path)
    gdf.to_feather(output_file_path)
    logger.info("Writing data to {output_file_path}. Done.", output_file_path=output_file_path)


if __name__ == "__main__":
    create()
