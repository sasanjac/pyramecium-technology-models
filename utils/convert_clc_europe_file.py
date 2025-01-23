# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import pathlib

import click
import geopandas as gpd
from loguru import logger

# Example input file name: U2018_CLC2018_V2020_20u1.gpkg (https://land.copernicus.eu/pan-european/corine-land-cover/clc2018)
# Proper output file name: clc_europe_epsg3035.feather


@click.command()
@click.option(
    "--file-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the input file.",
)
@click.option("--delete", is_flag=True, default=False, help="Whether to delete the source file.")
def convert(*, file_path: pathlib.Path, delete: bool = False) -> None:
    logger.info("Reading {file_path}...", file_path=file_path)
    gdf = gpd.read_file(file_path)
    logger.info("Reading {file_path}. Done.", file_path=file_path)
    logger.info("Manipulation data...")
    gdf = gdf.drop(["Remark", "Area_Ha", "ID"], axis=1)
    gdf["value"] = gdf.pop("Code_18")
    logger.info("Manipulation data. Done.")
    output_file_path = file_path.parent / "clc_europe_epsg3035.feather"
    logger.info("Writing data to {output_file_path}...", output_file_path=output_file_path)
    gdf.to_feather(output_file_path)
    logger.info("Writing data to {output_file_path}. Done.", output_file_path=output_file_path)
    if delete:
        logger.info("Deleting {file_path}...", file_path=file_path)
        file_path.unlink()
        logger.info("Deleting {file_path}. Done.", file_path=file_path)


if __name__ == "__main__":
    convert()
