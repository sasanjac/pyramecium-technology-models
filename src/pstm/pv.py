# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

import typing as t

import attrs
import numpy as np
import pvlib

from pstm.base import Tech

if t.TYPE_CHECKING:
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
            tz=str(self.dates.tz),
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
        index = t.cast("pd.DatetimeIndex", weather.index)
        acp = self._resample_as_array(target=-self.mc.results.ac, index=index)
        self.acp.loc[:, ("low", "L1")] = acp
        self.acp.loc[:, ("base", "L1")] = acp
        acq = acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", "L1")] = acq
        self.acq.loc[:, ("high", "L1")] = -acq

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
    ) -> PV:
        power_inst = E0 * efficiency * area
        return cls(
            dates=dates,
            power_inst=power_inst,
            efficiency_inv=efficiency_inv,
            lat=lat,
            lon=lon,
            alt=alt,
        )
