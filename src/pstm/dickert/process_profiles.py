# Copyright (c) 2015-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import numpy.typing as npt

from pstm.dickert.appliances import DistributionType
from pstm.dickert.on_off_profiles import OnOffProfiles

if TYPE_CHECKING:
    import datetime as dt


MAX_ITERATIONS = 2**16


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class ProcessProfiles(OnOffProfiles):
    active_power_2_distribution_type: DistributionType
    active_power_2_parameter_1: float
    active_power_2_parameter_2: float
    reactive_power_2_distribution_type: DistributionType
    reactive_power_2_parameter_1: float
    reactive_power_2_parameter_2: float
    operation_2_distribution_type: DistributionType
    operation_2_parameter_1: float
    operation_2_parameter_2: float

    def _run(  # noqa: PLR0914
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
        p1_base = self._sim_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        p2_base = self._sim_distribution(
            distribution_type=self.active_power_2_distribution_type,
            parameter_1=self.active_power_2_parameter_1,
            parameter_2=self.active_power_2_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        usage_frequency = self._sim_distribution(
            distribution_type=self.usage_distribution_type,
            parameter_1=self.usage_parameter_1,
            parameter_2=self.usage_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        time_on = self._calc_time_on_total(
            n_steps=n_steps,
            n_units=n_units,
            usage_frequency=usage_frequency,
            generator=generator,
        )
        operation_length_1 = self._calc_operation_length(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.operation_distribution_type,
            parameter_1=self.operation_parameter_1,
            parameter_2=self.operation_parameter_2,
            variation=self.operation_variation,
            time_on=time_on,
            generator=generator,
        )
        operation_length_2 = self._calc_operation_length(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.operation_2_distribution_type,
            parameter_1=self.operation_2_parameter_1,
            parameter_2=self.operation_2_parameter_2,
            variation=0,
            time_on=time_on,
            generator=generator,
        )
        time_off_1 = (time_on + operation_length_1) * np.ceil(
            time_on * operation_length_1 / (n_steps * np.max(operation_length_1)),
        ).astype(np.int64)
        time_off_2 = (time_off_1 + operation_length_2) * np.ceil(
            time_off_1 * operation_length_2 / (n_steps * np.max(operation_length_2)),
        ).astype(np.int64)
        idx = (time_on[1:, :] <= time_off_2[:-1, :]) & (time_on[1:, :] > 0)
        while np.any(idx):
            idx = np.concatenate([np.full(shape=(1, n_units), fill_value=False), idx], axis=0)
            x, y = np.where(idx)
            diff = time_off_1[x, y] - time_on[x, y]
            diff2 = time_off_2[x, y] - time_on[x, y]
            time_on[x, y] = time_off_2[x - 1, y] + 1
            time_off_1[x, y] = time_on[x, y] + diff
            time_off_2[x, y] = time_on[x, y] + diff2
            idx = (time_on[1:, :] <= time_off_2[:-1, :]) & (time_on[1:, :] > 0)

        p_1 = self._finalize_active_power(
            n_steps=n_steps,
            n_units=n_units,
            active_power=p1_base,
            time_on=time_on,
            time_off=time_off_1,
        )

        p_2 = self._finalize_active_power(
            n_steps=n_steps,
            n_units=n_units,
            active_power=p2_base,
            time_on=time_off_1,
            time_off=time_off_2,
        )

        _, q_1 = self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p_1,
            generator=generator,
        )
        if p_2.size == 0:
            p = p_1
            q = q_1
        else:
            p = p_1 + p_2
            _, q_2 = self._finalize_power(
                n_units=n_units,
                n_steps=n_steps,
                distribution_type=self.reactive_power_2_distribution_type,
                parameter_1=self.reactive_power_2_parameter_1,
                parameter_2=self.reactive_power_2_parameter_2,
                active_power=p_2,
                generator=generator,
            )
            q = q_1 + q_2

        p, _ = self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p,
            generator=generator,
        )

        return (p, q)
