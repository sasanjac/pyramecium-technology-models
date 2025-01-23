# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger
from numpy.core.numeric import ones

from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import Constants
from pstm.dickert.appliances import DistributionType
from pstm.dickert.appliances import validate_level
from pstm.dickert.appliances import validate_pm_level

if TYPE_CHECKING:
    import datetime as dt


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class OperationProfiles(Appliances):
    active_power_parameter_4: float
    operation_distribution_type: DistributionType
    operation_parameter_1: float
    operation_parameter_2: float
    operation_parameter_3: float
    operation_variation: float = attrs.field(validator=validate_level)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class CycleProfiles(OperationProfiles):
    period_distribution_type: DistributionType
    period_parameter_1: float
    period_parameter_2: float
    period_parameter_3: float
    period_variation: float = attrs.field(validator=validate_pm_level)

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
        p_base = self._sim_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        period_length = self._calc_period_length(
            n_units=n_units,
            n_steps=n_steps,
            generator=generator,
        )
        operation_length = self._calc_operation_length(
            n_units=n_units,
            n_steps=n_steps,
            period_length=period_length,
            generator=generator,
        )
        shift = self._sim_distribution_round(
            distribution_type="unif",
            parameter_1=0,
            parameter_2=float(np.max(period_length[0, :])),
            n_steps=1,
            n_units=n_units,
            generator=generator,
        )
        time_on = np.cumsum(period_length, axis=0) + shift
        time_off = time_on + operation_length
        x, y = np.where(time_on[1:] < time_off[:-1])
        time_off[x, y] = time_on[x + 1, y]
        if np.any((time_on[-1, :]) - n_steps <= 0):
            logger.warning(
                "{description}: not enough on cycles for cycle operation - adjust safety factor!",
                description=self.description,
            )

        p = np.zeros((np.max(time_off[-1, :]).astype(np.int64) + 1, n_units))

        for unit in range(n_units):
            p[time_on[:, unit], unit] = p_base[:, unit]
            p[time_off[:, unit], unit] -= ones(time_off.shape[0]) * p_base[:, unit]

        p = np.cumsum(p, axis=0)
        p = p[np.max(period_length[0, :]) :, :]
        p = p[:n_steps, :]
        return self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p,
            generator=generator,
        )

    def _calc_period_length(
        self,
        *,
        n_units: int,
        n_steps: int,
        generator: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        scatter_init = self._sim_distribution_round(
            distribution_type=self.period_distribution_type,
            parameter_1=self.period_parameter_1 / step_length,
            parameter_2=self.period_parameter_2 / step_length,
            n_units=n_units,
            generator=generator,
        )
        period_length = np.tile(scatter_init, (np.ceil(n_steps * 1.01 / np.min(scatter_init)).astype(np.int64), 1))
        scatter_individual = self._sim_distribution_round(
            distribution_type=self.period_distribution_type,
            parameter_1=0,
            parameter_2=self.period_parameter_3 / step_length,
            n_units=n_units,
            n_steps=period_length.shape[0],
            generator=generator,
        )
        period_length += scatter_individual

        period_length[period_length <= 0] = 1
        for unit in range(n_units):
            step_max = self._find_step_max(n_steps=n_steps, period_length=period_length, unit=unit)
            steps = np.linspace(1, step_max, step_max, dtype=np.int64)

            if self.period_variation != 0:
                shift = self.period_variation * scatter_init[0, unit]
                step = (
                    shift
                    * np.sin(
                        2 * np.pi * (steps / step_max + 3 / 4 - 28 / Constants.DAYS_PER_YEAR),
                    )
                ).astype(np.int64)
                period_length[steps, unit] += step

        period_length[period_length <= 0] = 1
        return period_length

    def _calc_operation_length(
        self,
        *,
        n_units: int,
        n_steps: int,
        period_length: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        scatter_init = self._sim_distribution_round(
            distribution_type=self.operation_distribution_type,
            parameter_1=self.operation_parameter_1 / step_length,
            parameter_2=self.operation_parameter_2 / step_length,
            n_units=n_units,
            generator=generator,
        )
        operation_length = np.tile(scatter_init, (period_length.shape[0], 1))
        scatter_individual = self._sim_distribution_round(
            distribution_type=self.operation_distribution_type,
            parameter_1=0,
            parameter_2=self.operation_parameter_3 / step_length,
            n_units=n_units,
            n_steps=operation_length.shape[0],
            generator=generator,
        )
        operation_length += scatter_individual

        operation_length[operation_length <= 0] = 1
        for unit in range(n_units):
            step_max = self._find_step_max(n_steps=n_steps, period_length=period_length, unit=unit)
            steps = np.linspace(1, step_max, step_max, dtype=np.int64)

            if self.operation_variation != 0:
                shift = self.operation_variation * scatter_init[0, unit]
                step = shift * np.sin(
                    2 * np.pi * (steps[:, np.newaxis] / step_max + 3 / 4 - 28 / Constants.DAYS_PER_YEAR),
                )
                period_length[steps, unit] += step

        operation_length[operation_length <= 0] = 1
        return operation_length

    @staticmethod
    def _find_step_max(*, n_steps: int, period_length: npt.NDArray[np.int64], unit: int) -> np.int64:
        return np.argmax(np.cumsum(period_length[:, unit], axis=0) > n_steps)
