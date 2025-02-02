# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

import datetime as dt

import attrs
import numpy as np
import numpy.typing as npt
from loguru import logger

from pstm.dickert.appliances import Constants
from pstm.dickert.appliances import DistributionType
from pstm.dickert.appliances import validate_level
from pstm.dickert.appliances import validate_level_sequence
from pstm.dickert.cycle_profiles import OperationProfiles

MAX_PARAMETER_SUM = 3
MAX_ITERATIONS = 2**16

NaT = dt.time(0)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class OnOffProfiles(OperationProfiles):
    usage_distribution_type: DistributionType
    usage_parameter_1: float
    usage_parameter_2: float
    usage_parameter_3: float
    usage_variation: float = attrs.field(validator=validate_level)
    time_on_distribution_types: tuple[
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
    ]
    time_on_parameters_1: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_parameters_3: tuple[float, float, float, float, float, float] = attrs.field(
        validator=validate_level_sequence,
    )
    probability_1: float = attrs.field(validator=validate_level)
    probability_2: float = attrs.field(validator=validate_level)
    probability_3: float = attrs.field(validator=validate_level)
    probability_4: float = attrs.field(validator=validate_level)
    use_probability: bool = attrs.field(init=False, repr=False)

    @use_probability.default
    def _use_probability_default(self) -> bool:
        return bool(
            np.all(
                [
                    num == 0
                    for num in [
                        self.probability_1,
                        self.probability_2,
                        self.probability_3,
                        self.probability_4,
                    ]
                ],
                axis=0,
            ),
        )

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
        p_base = self._sim_p_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
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
        operation_length = self._calc_operation_length(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.operation_distribution_type,
            parameter_1=self.operation_parameter_1,
            parameter_2=self.operation_parameter_2,
            variation=self.operation_variation,
            time_on=time_on,
            generator=generator,
        )
        time_off = (time_on + operation_length) * np.ceil(
            time_on * operation_length / (n_steps * np.max(operation_length)),
        ).astype(np.int64)
        idx = (time_on[1:, :] <= time_off[:-1, :]) & (time_on[1:, :] > 0)
        while np.sum(idx) > 0:
            idx = np.concatenate([np.zeros((1, n_units), dtype=np.bool_), idx], axis=0)
            x, y = np.where(idx)
            diff = time_off[x, y] - time_on[x, y]
            time_on[x, y] = time_off[x - 1, y] + 1
            time_off[x, y] = time_on[x, y] + diff
            idx = (time_on[1:, :] <= time_off[:-1, :]) & (time_on[1:, :] > 0)

        p = self._finalize_active_power(
            n_steps=n_steps,
            n_units=n_units,
            active_power=p_base,
            time_on=time_on,
            time_off=time_off,
        )
        return self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p,
            generator=generator,
        )

    def _calc_operation_length(
        self,
        *,
        n_units: int,
        n_steps: int,
        distribution_type: DistributionType,
        parameter_1: float,
        parameter_2: float,
        variation: float,
        time_on: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        operation_length = self._sim_distribution_round(
            distribution_type=distribution_type,
            parameter_1=parameter_1 / step_length,
            parameter_2=parameter_2 / step_length,
            n_steps=time_on.shape[0],
            n_units=time_on.shape[1],
            factor=1 / step_length,
            clear=False,
            generator=generator,
        )

        steps = np.linspace(1, n_steps, n_steps, dtype=np.int64)
        shift = np.sin(2 * np.pi / n_steps * steps - 2 * np.pi * 3 / 4 - 28 / Constants.DAYS_PER_YEAR * 2 * np.pi)
        time_on[time_on > (n_steps - 1)] = n_steps - 1
        for i in range(n_units):
            operation_length[time_on[:, i] != 0, i] += (
                operation_length[time_on[:, i] != 0, i] * shift[time_on[time_on[:, i] != 0, i]] * variation
            ).astype(np.int64)

        operation_length[operation_length <= 0] = 1
        return operation_length

    def _calc_time_on_total(
        self,
        *,
        n_steps: int,
        n_units: int,
        usage_frequency: npt.NDArray[np.float64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        time_on_weekdays = self._calc_time_on(
            n_steps=n_steps,
            n_units=n_units,
            n_days=Constants.WEEKDAYS_PER_YEAR,
            usage_frequency=usage_frequency,
            parameter_index_start=0,
            parameter_index_end=3,
            day_index_start=0,
            day_index_end=5,
            add_tail=True,
            generator=generator,
        )
        time_on_weekenddays = self._calc_time_on(
            n_steps=n_steps,
            n_units=n_units,
            n_days=Constants.WEEKENDDAYS_PER_YEAR,
            usage_frequency=usage_frequency,
            parameter_index_start=3,
            parameter_index_end=6,
            day_index_start=5,
            day_index_end=7,
            generator=generator,
        )
        time_on = np.sort(np.concatenate([time_on_weekdays, time_on_weekenddays], axis=0), axis=0)
        idx = (time_on[:-1, :] == time_on[1:, :]) & (time_on[1:, :] > 0)
        while np.any(idx):
            idx = np.concatenate([idx, np.full(shape=(1, n_units), fill_value=False)], axis=0)
            x, y = np.where(idx)
            time_on[x + 1, y] = time_on[x, y] + 1
            idx = (time_on[:-1, :] == time_on[1:, :]) & (time_on[1:, :] > 0)

        return time_on

    def _calc_time_on(
        self,
        *,
        n_steps: int,
        n_units: int,
        n_days: int,
        usage_frequency: npt.NDArray[np.float64],
        parameter_index_start: int,
        parameter_index_end: int,
        day_index_start: int,
        day_index_end: int,
        add_tail: bool = False,
        generator: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR // n_steps
        if np.sum(self.time_on_parameters_1[:parameter_index_end] == 0) < MAX_PARAMETER_SUM:
            time_on = np.zeros((n_days, n_units * 3), dtype=np.int64)
            for i in range(parameter_index_start, parameter_index_end):
                top_1 = self.time_on_parameters_1[i]
                top_2 = self.time_on_parameters_2[i]
                if NaT not in {top_1, top_2}:
                    samples_per_day = Constants.MINUTES_PER_DAY // step_length
                    steps = self._calc_steps(
                        day_index_start=day_index_start,
                        day_index_end=day_index_end,
                        samples_per_day=samples_per_day,
                        add_tail=add_tail,
                    )
                    idx = np.round(
                        self._time_as_float(top_1) * Constants.MINUTES_PER_HOUR * Constants.HOURS_PER_DAY / step_length
                        + np.sort(steps),
                    ).astype(np.int64)
                    prob = self._sim_distribution(
                        distribution_type=self.time_on_distribution_types[i],
                        parameter_1=0,
                        parameter_2=self._time_as_float(top_2)
                        * Constants.MINUTES_PER_HOUR
                        * Constants.HOURS_PER_DAY
                        / step_length,
                        n_steps=idx.shape[0],
                        n_units=1,
                        clear=False,
                        generator=generator,
                    )
                    j = i - parameter_index_start
                    time_on[:, n_units * j : n_units * (j + 1)] = np.tile(
                        idx[:, np.newaxis] + prob,
                        (1, time_on[:, n_units * j : n_units * (j + 1)].shape[1]),
                    )

            steps = np.linspace(1, n_days, n_days, dtype=np.int64)
            year = self.usage_variation * np.sin(
                2 * np.pi / n_days * steps + 2 * np.pi * 3 / 4 - 28 / Constants.DAYS_PER_YEAR * 2 * np.pi,
            )

            prob1 = np.concatenate(
                [
                    np.tile(self.time_on_parameters_3[i], (1, n_units)) * usage_frequency / n_days
                    for i in range(parameter_index_start, parameter_index_end)
                ],
                axis=1,
            )

            prob1_y = prob1 - prob1 * year[:, np.newaxis]
            if np.max(prob1) > 1:
                logger.warning(
                    "{description}: too many on/off cycles - reduce usage frequency or time on parameter 3!",
                    description=self.description,
                )

            prob2 = self._sim_distribution(
                distribution_type="unif",
                parameter_1=0.5,
                parameter_2=0.5,
                n_steps=n_days,
                n_units=3 * n_units,
                generator=generator,
            )
            idx = prob1_y > prob2
            time_on = idx * time_on
            return np.concatenate(
                [
                    time_on[:, 0 * n_units : 1 * n_units],
                    time_on[:, 1 * n_units : 2 * n_units],
                    time_on[:, 2 * n_units : 3 * n_units],
                ],
            )

        return np.zeros((n_days, n_units), dtype=np.int64)

    def _sim_p_distribution(
        self,
        *,
        distribution_type: DistributionType,
        parameter_1: float,
        parameter_2: float,
        n_units: int,
        n_steps: int = 1,
        factor: float = 1,
        clear: bool = True,
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        if distribution_type == "unif" and self.use_probability:
            p = np.zeros((n_steps, n_units), dtype=np.float64)
            prob = self._sim_distribution(
                distribution_type=distribution_type,
                parameter_1=0.5,
                parameter_2=0.5,
                n_units=n_units,
                clear=False,
                generator=generator,
            )
            n = np.cumsum(
                [
                    self.probability_1,
                    self.probability_2,
                    self.probability_3,
                    self.probability_4,
                ],
            )
            p[:, prob < n[0]] = self.active_power_parameter_1
            p[:, (prob >= n[0]) & (prob < n[1])] = self.active_power_parameter_2
            p[:, (prob >= n[1]) & (prob < n[2])] = self.active_power_parameter_3
            p[:, (prob >= n[2]) & (prob < n[3])] = self.active_power_parameter_4

            return p

        return self._sim_distribution(
            distribution_type=distribution_type,
            parameter_1=parameter_1,
            parameter_2=parameter_2,
            n_units=n_units,
            n_steps=n_steps,
            factor=factor,
            clear=clear,
            generator=generator,
        )
