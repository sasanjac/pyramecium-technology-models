# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

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
        p_const_base = self._sim_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        p_const = np.ones((n_steps, n_units)) * np.tile(p_const_base, (n_steps, 1))
        steps = np.linspace(1, n_steps, n_steps, dtype=np.int64)
        p_var = self.power_variation * np.cos(2 * np.pi * (steps / n_steps - 28 / Constants.DAYS_PER_YEAR))
        p = p_const + p_var[:, np.newaxis]
        for i in range(n_units):
            p[p[:, i] > p_const_base[:, i], i] = self.power_variation_max * p_const_base[:, i]

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
