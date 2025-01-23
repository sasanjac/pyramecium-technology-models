# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import pathlib

import click
import geopandas as gpd
from loguru import logger

# Example input file name: Postleitzahlengebiete_-_OSM.json (https://opendata-esri-de.opendata.arcgis.com/datasets/5b203df4357844c8a6715d7d411a8341_0)
# Proper output file name: zip_codes_germany_epsg4326.feather


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
    gdf = gdf.drop(
        [
            "OBJECTID",
            "ags",
            "ort",
            "landkreis",
            "bundesland",
            "einwohner",
            "note",
            "SHAPE_Length",
            "SHAPE_Area",
        ],
        axis=1,
    )
    gdf["value"] = gdf.pop("plz")
    logger.info("Manipulation data. Done.")
    output_file_path = file_path.parent / "zip_codes_germany_epsg4326.feather"
    logger.info("Writing data to {output_file_path}...", output_file_path=output_file_path)
    gdf.to_feather(output_file_path)
    logger.info("Writing data to {output_file_path}. Done.", output_file_path=output_file_path)
    if delete:
        logger.info("Deleting {file_path}...", file_path=file_path)
        file_path.unlink()
        logger.info("Deleting {file_path}. Done.", file_path=file_path)


if __name__ == "__main__":
    convert()
