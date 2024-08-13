# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

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
        delta = self.dates[0] - weather.index[0]
        acp_raw = -self.mc.results.ac.set_axis(labels=weather.index + delta)
        if len(self.dates) > len(weather.index):
            acp = acp_raw.reindex(index=self.dates).interpolate(
                method="linear",
                limit_direction="both",
            )
        else:
            acp = acp_raw.resample(self.dates.freq).mean().reindex(index=self.dates)

        _acp = acp.to_numpy(dtype=np.float64)
        self.acp.loc[:, ("low", 1)] = _acp
        self.acp.loc[:, ("base", 1)] = _acp
        _acq = _acp * np.tan(np.arccos(self.cosphi))
        self.acq.loc[:, ("low", 1)] = _acq
        self.acq.loc[:, ("high", 1)] = -_acq

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
