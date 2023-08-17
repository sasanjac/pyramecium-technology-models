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
from pstm.dickert.appliance import DistributionType
from pstm.dickert.appliance import validate_level
from pstm.dickert.household import GEN

YEAR_IN_MINUTES = 525_600


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class CycleProfile(Appliance):
    active_power_parameter_4: float
    period_distribution_type: DistributionType
    period_parameter_1: float
    period_parameter_2: float
    period_parameter_3: float
    operation_distribution_type: DistributionType
    operation_parameter_1: float
    operation_parameter_2: float
    operation_parameter_3: float
    period_variation: attrs.field(float, validator=validate_level)
    operation_variation: attrs.field(float, validator=validate_level)

    def _run(self, *, n_units: int, n_steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        step_length = YEAR_IN_MINUTES / n_steps
        period_length = self._calc_period_length(
            n_units=n_units,
            n_steps=n_steps,
            step_length=step_length,
        )
        operation_length = self._calc_operation_length(
            n_units=n_units,
            step_length=step_length,
            period_length=period_length,
        )
        t_on = np.cumsum(period_length)
        t_off = t_on + operation_length
        shift = self.sim_distribution(
            distribution_type="uniform",
            p_1=0,
            p_2=np.max(period_length[0, :])
            n_steps=period_length.shape[0],
            n_units=1,
        )

    def _calc_period_length(
        self,
        *,
        n_units: int,
        n_steps: int,
        step_length: int,
    ) -> npt.NDArray[np.float64]:
        scatter_init = self.sim_distribution(
            distribution_type=self.period_distribution_type,
            p_1=self.period_parameter_1 / step_length,
            p_2=self.period_parameter_2 / step_length,
            n_units=n_units,
        )
        period_length = np.tile(scatter_init, (np.ceil(n_steps * 1.01 / np.min(scatter_init)), 1))
        scatter_individual = self.sim_distribution(
            distribution_type=self.period_distribution_type,
            p_1=0,
            p_2=self.period_parameter_3 / step_length,
            n_units=n_units,
            n_steps=period_length.shape[0],
        )
        period_length = period_length + scatter_individual

        period_length[period_length <= 0] = 1
        for unit in range(n_units):
            step_max = self._find_step_max(n_steps=n_steps, period_length=period_length, unit=unit)
            steps = np.linspace(0, step_max, step_max)

            if self.period_variation != 0:
                shift = self.period_variation * scatter_init[0, unit]
                step = shift * np.sin(2 * np.pi * (steps[:, np.newaxis] / step_max + 3 / 4 - 28 / 365))
                period_length[steps, unit] = period_length[steps, unit] + step

        period_length[period_length <= 0] = 1
        return period_length

    def _calc_operation_length(
        self,
        *,
        n_units: int,
        n_steps: int,
        step_length: int,
        period_length: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        scatter_init = self.sim_distribution(
            distribution_type=self.period_distribution_type,
            p_1=self.operation_parameter_1 / step_length,
            p_2=self.operation_parameter_2 / step_length,
            n_units=n_units,
        )
        operation_length = np.tile(scatter_init, period_length.shape[0], 1)
        scatter_individual = self.sim_distribution(
            distribution_type=self.operation_distribution_type,
            p_1=0,
            p_2=self.operation_parameter_3 / step_length,
            n_units=n_units,
            n_steps=operation_length.shape[0],
        )
        operation_length = operation_length + scatter_individual

        operation_length[operation_length <= 0] = 1
        for unit in range(n_units):
            step_max = self._find_step_max(n_steps=n_steps, period_length=period_length, unit=unit)
            steps = np.linspace(0, step_max, step_max)

            if self.operation_variation != 0:
                shift = self.operation_variation * scatter_init[0, unit]
                step = shift * np.sin(2 * np.pi * (steps[:, np.newaxis] / step_max + 3 / 4 - 28 / 365))
                period_length[steps, unit] = period_length[steps, unit] + step

        operation_length[operation_length <= 0] = 1
        return operation_length

    def _find_step_max(self, *, n_steps: int, period_length: npt.NDArray[np.float64], unit: int) -> int:
        return np.argmax(np.cumsum(period_length[:unit]) > n_steps)
