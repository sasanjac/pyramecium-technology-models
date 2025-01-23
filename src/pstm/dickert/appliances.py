# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

import abc
import typing as t

import attrs
import numpy as np

if t.TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    import numpy.typing as npt


DistributionType = t.Literal["normal", "unif", "lognormal"]
Phase = t.Literal[0, 1, 2, 3]


class Constants:
    WEEKS_PER_YEAR: int = 52
    MINUTES_PER_YEAR: int = 525_600
    MINUTES_PER_DAY: int = 1440
    DAYS_PER_WEEK: int = 7
    MINUTES_PER_HOUR: int = 60
    HOURS_PER_DAY: int = 24
    DAYS_PER_YEAR: int = 365
    WEEKDAYS_PER_YEAR: int = 261
    WEEKENDDAYS_PER_YEAR: int = 104


def validate_pm_level(instance: Appliances, attribute: attrs.Attribute, value: float) -> float:  # noqa: ARG001
    if not -1 <= value <= 1:
        msg = f"Attribute {attribute.name} has to be between -1 and 1, is {value}"
        raise ValueError(msg)

    return value


def validate_level(instance: Appliances, attribute: attrs.Attribute, value: float) -> float:  # noqa: ARG001
    if not 0 <= value <= 1:
        msg = f"Attribute {attribute.name} has to be between 0 and 1, is {value}"
        raise ValueError(msg)

    return value


def validate_level_sequence(
    instance: Appliances,
    attribute: attrs.Attribute,
    value: Sequence[float],
) -> Sequence[float]:
    return tuple(validate_level(instance, attribute, level) for level in value)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Appliances:
    description: str
    phase: Phase
    switch_on_current: float
    switch_on_time: int
    equipment_level: float = attrs.field(validator=validate_level)
    active_power_distribution_type: DistributionType
    active_power_parameter_1: float
    active_power_parameter_2: float
    active_power_parameter_3: float
    reactive_power_share: float = attrs.field(validator=validate_level)
    reactive_power_distribution_type: DistributionType
    reactive_power_parameter_1: float = attrs.field(validator=validate_level)
    reactive_power_parameter_2: float = attrs.field(validator=validate_level)
    reactive_power_parameter_3: float = attrs.field(validator=validate_level)

    @staticmethod
    def _sim_distribution(
        *,
        distribution_type: DistributionType,
        parameter_1: float,
        parameter_2: float,
        generator: np.random.Generator,
        n_units: int,
        n_steps: int = 1,
        factor: float = 1,
        clear: bool = True,
    ) -> npt.NDArray[np.float64]:
        match distribution_type:
            case "normal":
                p = generator.normal(
                    loc=parameter_1,
                    scale=parameter_2,
                    size=(n_steps, n_units),
                )
            case "unif":
                p = generator.uniform(
                    low=parameter_1 - parameter_2,
                    high=parameter_1 + parameter_2,
                    size=(n_steps, n_units),
                )
            case "lognormal":
                p = (
                    generator.lognormal(
                        mean=parameter_1,
                        sigma=parameter_2,
                        size=(n_steps, n_units),
                    )
                    * factor
                )
            case _:
                p = np.zeros((n_steps, n_units))

        if clear:
            p[p < 0] = 0

        return p

    def _sim_distribution_round(
        self,
        *,
        distribution_type: DistributionType,
        parameter_1: float,
        parameter_2: float,
        generator: np.random.Generator,
        n_units: int,
        n_steps: int = 1,
        factor: float = 1,
        clear: bool = True,
    ) -> npt.NDArray[np.int64]:
        x = self._sim_distribution(
            distribution_type=distribution_type,
            parameter_1=parameter_1,
            parameter_2=parameter_2,
            generator=generator,
            n_units=n_units,
            n_steps=n_steps,
            factor=factor,
            clear=clear,
        )
        return np.round(x).astype(np.int64)

    @staticmethod
    def _time_as_float(time: dt.time) -> float:
        return time.hour / 24 + time.minute / (24 * 60)

    @staticmethod
    def _finalize_active_power(
        *,
        n_steps: int,
        n_units: int,
        active_power: npt.NDArray[np.float64],
        time_on: npt.NDArray[np.int64],
        time_off: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        n_rows = np.max([np.max([time_on, time_off]) + 1, n_steps])
        p = np.zeros((n_rows, n_units))
        for i in range(n_units):
            p[time_off[time_off[:, i] > 0, i], i] = -1
            p[time_on[time_on[:, i] > 0, i], i] += 1

        p = np.cumsum(p, axis=0)
        p[p > 1] = 1
        p = p[:n_steps, :]
        return p * active_power

    def run(
        self,
        *,
        n_units: int,
        n_steps: int,
        generator: np.random.Generator,
        phase_distribution: tuple[float, float, float],
        lat: float,
        lon: float,
        altitude: float,
        year: int,
        tz: dt.tzinfo,
    ) -> None:
        p, q = self._run(
            n_units=n_units,
            n_steps=n_steps,
            lat=lat,
            lon=lon,
            altitude=altitude,
            year=year,
            tz=tz,
            generator=generator,
        )
        if np.any(p < 0):
            msg = "active power can not be negative."
            raise ValueError(msg)

        if self.phase == 0:
            self.phase = generator.choice((1, 2, 3), p=phase_distribution)

        self.p = np.zeros((n_steps, n_units, 3))
        self.p[:, :, self.phase - 1] = p
        self.q = np.zeros((n_steps, n_units, 3))
        self.q[:, :, self.phase - 1] = q

    @abc.abstractmethod
    def _run(
        self,
        *,
        n_units: int,
        n_steps: int,
        generator: np.random.Generator,
        lat: float,
        lon: float,
        altitude: float,
        year: int,
        tz: dt.tzinfo,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Internal power calculation method.

        Arguments:
            n_units {int} -- amount of units
            n_steps {int} -- amount of time steps
            lat {float} -- latitude of household location
            lon {float} -- longitude of household location
            altitude {float} -- altitude of household location
            year {int} -- year of power profile
            tz {dt.tzinfo} -- timezone of household location

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] -- active and reactive power
        """

    def _finalize_power(
        self,
        *,
        n_units: int,
        n_steps: int,
        distribution_type: DistributionType,
        parameter_1: float,
        parameter_2: float,
        active_power: npt.NDArray[np.float64],
        generator: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rnd_index = generator.uniform(0, 1, n_units)
        active_power[:, rnd_index > self.equipment_level] = 0

        fac = self._sim_distribution(
            distribution_type=distribution_type,
            parameter_1=parameter_1,
            parameter_2=parameter_2,
            n_units=n_units,
            generator=generator,
        )
        reactive_power = active_power * np.tile(fac, (n_steps, 1))
        rnd_index = generator.uniform(0, 1, n_units)
        reactive_power[:, rnd_index > self.reactive_power_share] = 0

        return (active_power, reactive_power)

    @staticmethod
    def _calc_steps(
        *,
        day_index_start: int,
        day_index_end: int,
        samples_per_day: int,
        add_tail: bool = False,
    ) -> npt.NDArray[np.int64]:
        if add_tail:
            tail = [
                np.ones(1, dtype=np.int64)
                * (Constants.WEEKS_PER_YEAR * Constants.DAYS_PER_WEEK + day_index_start)
                * samples_per_day,
            ]
        else:
            tail = []

        return np.concatenate(
            [
                (np.arange(0, Constants.WEEKS_PER_YEAR) * Constants.DAYS_PER_WEEK + i) * samples_per_day
                for i in range(day_index_start, day_index_end)
            ]
            + tail,
        )
