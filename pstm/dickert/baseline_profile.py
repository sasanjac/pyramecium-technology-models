# :author: Jörg Dickert <joerg.dickert@tu-dresden.de>
# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2015-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import typing as t

import attrs
import numpy as np
import numpy.typing as npt

from pstm.dickert.appliance import Appliance
from pstm.dickert.appliance import validate_level
from pstm.dickert.household import GEN

if t.TYPE_CHECKING:
    from pstm.dickert.appliance import DistributionType


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class BaselineProfile(Appliance):s
    power_variation: attrs.field(float, validator=validate_level)
    power_variation_max: float

    def _run(self, n_units: int, n_steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        _p_const = self.sim_distribution(
            distribution_type=self.active_power_distribution_type,
            p_1=self.active_power_parameter_1,
            p_2=self.active_power_parameter_2,
            n_units=n_units,
        )
        p_const = np.ones((n_steps, n_units)) * np.tile(_p_const, (n_steps, 1))
        t = np.linspace(1, n_steps, n_steps)
        p_var = self.power_variation * np.cos(2 * np.pi * (t / n_steps - 28 / 365))
        p = p_const + p_var
        for i in range(n_units):
            p[p[:, i] > _p_const[i], i] = self.power_variation_max * _p_const[i]

        p[p < 0] = 0
        rnd_index = GEN.uniform(0, 1, (1, n_units))
        p[:, rnd_index > self.equipment_level] = 0

        fac = self.sim_distribution(
            distribution_type=self.reactive_power_distribution_type,
            p_1=self.reactive_power_parameter_1,
            p_2=self.reactive_power_parameter_2,
            n_units=n_units,
        )
        q = p * np.tile(fac, (n_steps, 1))
        rnd_index = GEN.uniform(0, 1, (1, n_units))
        q[:, rnd_index > self.reactive_power_share] = 0

        return (p, q)