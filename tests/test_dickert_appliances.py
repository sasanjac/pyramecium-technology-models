# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for the Dickert appliance base class."""

from __future__ import annotations

import zoneinfo as zi

import attrs
import numpy as np

from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import DistributionType

PARAMS = {
    "description": "test",
    "phase": 1,
    "switch_on_current": 0.0,
    "switch_on_time": 0,
    "equipment_level": 1.0,
    "active_power_distribution_type": DistributionType.NORMAL,
    "active_power_parameter_1": 0.0,
    "active_power_parameter_2": 0.0,
    "active_power_parameter_3": 0.0,
    "reactive_power_share": 0.0,
    "reactive_power_distribution_type": DistributionType.NORMAL,
    "reactive_power_parameter_1": 0.0,
    "reactive_power_parameter_2": 0.0,
    "reactive_power_parameter_3": 0.0,
}
RUN = {
    "n_units": 2,
    "n_steps": 3,
    "generator": np.random.default_rng(0),
    "phase_distribution": (1.0, 0.0, 0.0),
    "lat": 51.0,
    "lon": 13.7,
    "altitude": 100.0,
    "year": 2023,
    "tz": zi.ZoneInfo("Europe/Berlin"),
}


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class ConstantAppliance(Appliances):
    power: float = 1.0

    def _run(self, *, n_units: int, n_steps: int, **_: object) -> tuple[np.ndarray, np.ndarray]:
        p = np.full((n_steps, n_units), self.power)
        return p, np.zeros_like(p)


class TestRun:
    def test_power_on_the_phase_of_the_appliance(self) -> None:
        appliance = ConstantAppliance(**PARAMS)
        appliance.run(**RUN).unwrap()
        assert np.all(appliance.p[:, :, 0] == 1.0)
        assert np.all(appliance.p[:, :, 1:] == 0.0)

    def test_negative_active_power_is_err(self) -> None:
        appliance = ConstantAppliance(**PARAMS, power=-1.0)
        assert "can not be negative" in str(appliance.run(**RUN).unwrap_err())


class TestCreate:
    def test_level_outside_the_unit_interval_is_err(self) -> None:
        appliance = ConstantAppliance.create(**{**PARAMS, "equipment_level": 1.5})
        assert "has to be between 0 and 1" in str(appliance.unwrap_err())
