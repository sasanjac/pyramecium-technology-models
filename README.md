# Pyramecium Technology Models

A toolbox to create multi-energy power time series for different technologies in steady-state simulations of cellular energy systems

## Installation

Installation is fairly easy:
```sh
pip install pyramecium-technology-models
```

## CRS

By default, EPSG:4326 is used.

## Get Started

Before being able to create time series, there are a few geo files that have to be obtained first. The files can be converted using the provided conversion scripts:

- Digital Terrain Models:
  -  Germany (20m and 50m): https://sonny.4lima.de/
  -  Europe (1" and 2"): https://sonny.4lima.de/
- Zip codes
  - Germany: https://opendata-esri-de.opendata.arcgis.com/datasets/5b203df4357844c8a6715d7d411a8341_0
- Meteorological Data
  - Germany (DWD test reference year time series; 2016 and 2045): https://kunden.dwd.de/obt/
- VDI4655 reference profiles: https://www.vdi.de/richtlinien/details/.vdi-4655-referenzlastprofile-von-wohngebaeuden-fuer-strom-heizung-und-trinkwarmwasser-sowie-referenzerzeugungsprofile-fuer-fotovoltaikanlagen
- Clorine Land Coverage
  - Germany: https://gdz.bkg.bund.de/index.php/default/open-data/corine-land-cover-5-ha-stand-2018-clc5-2018.html
  - Europe: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018
- Time zones: https://github.com/evansiroky/timezone-boundary-builder/releases