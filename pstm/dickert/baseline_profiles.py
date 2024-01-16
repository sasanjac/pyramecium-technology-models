# :author: Jörg Dickert <joerg.dickert@tu-dresden.de>
# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2015-2023.
# :license: BSD 3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import numpy.typing as npt

from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import Constants
from pstm.dickert.appliances import validate_level

if TYPE_CHECKING:
    import datetime as dt


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class BaselineProfiles(Appliances):
    power_variation: float = attrs.field(validator=validate_level)
    power_variation_max: float

    def _run(
        self,
        *,
        n_units: int,
        n_steps: int,
        generator: np.random.Generator,
        lat: float,  # noqa: ARG002
        lon: float,  # noqa: ARG002
        altitude: float,  # noqa: ARG002
        year: int,  # noqa: ARG002
        tz: dt.tzinfo,  # noqa: ARG002
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        _p_const = self._sim_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        p_const = np.ones((n_steps, n_units)) * np.tile(_p_const, (n_steps, 1))
        steps = np.linspace(1, n_steps, n_steps, dtype=np.int64)
        p_var = self.power_variation * np.cos(2 * np.pi * (steps / n_steps - 28 / Constants.DAYS_PER_YEAR))
        p = p_const + p_var[:, np.newaxis]
        for i in range(n_units):
            p[p[:, i] > _p_const[:, i], i] = self.power_variation_max * _p_const[:, i]
        p[p < 0] = 0
        return self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p,
            generator=generator,
        )
