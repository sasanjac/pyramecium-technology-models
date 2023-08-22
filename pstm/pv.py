# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import attrs
import numpy as np
import pvlib

from pstm.base import Tech

if TYPE_CHECKING:
    import pandas as pd

GAMMA_TEMP = -0.004
E0 = 1000
TEMPERATURE_MODEL_PARAMETERS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
MODELCHAIN_DATA = {
    "dc_model": "pvwatts",
    "ac_model": "pvwatts",
    "aoi_model": "physical",
    "spectral_model": "no_loss",
    "temperature_model": "sapm",
    "dc_ohmic_model": "no_loss",
    "losses_model": "pvwatts",
}


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class PV(Tech):
    power_inst: float  # in W
    efficiency_inv: float
    lat: float
    lon: float
    alt: float
    tz: dt.tzinfo
    surface_tilt: int = attrs.field(default=30)
    surface_azimuth: int = attrs.field(default=180)
    gamma_temp: float = attrs.field(default=GAMMA_TEMP)
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        self.system = pvlib.pvsystem.PVSystem(
            surface_azimuth=self.surface_azimuth,
            surface_tilt=self.surface_tilt,
            module_parameters={"pdc0": self.power_inst, "gamma_pdc": self.gamma_temp},
            inverter_parameters={
                "pdc0": self.power_inst,
                "eta_inv_nom": self.efficiency_inv,
            },
            temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS,
        )
        self.loc = pvlib.location.Location(
            latitude=self.lat,
            longitude=self.lon,
            tz=self.dates.tz,
            altitude=self.alt,
        )
        self.mc = pvlib.modelchain.ModelChain(
            self.system,
            self.loc,
            **MODELCHAIN_DATA,
        )

    def run(
        self,
        weather: pd.DataFrame,
    ) -> None:
        self.mc.complete_irradiance(weather)
        self.mc.run_model(self.mc.results.weather)
        self.acp.low = -self.mc.results.ac.to_numpy()
        self.acp.base = self.acp.low
        self.acq.low = self.acp.low * np.tan(np.arccos(self.cosphi))
        self.acq.high = -self.acq.low

    @classmethod
    def from_efficiency_and_area(
        cls,
        dates: pd.DatetimeIndex,
        efficiency: float,
        area: float,
        efficiency_inv: float,
        lat: float,
        lon: float,
        alt: float,
        tz: dt.tzinfo,
    ) -> PV:
        power_inst = E0 * efficiency * area
        return cls(
            dates=dates,
            power_inst=power_inst,
            efficiency_inv=efficiency_inv,
            lat=lat,
            lon=lon,
            alt=alt,
            tz=tz,
        )
