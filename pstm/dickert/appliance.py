# :author: Jörg Dickert <joerg.dickert@tu-dresden.de>
# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2015-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import abc
import typing as t

import attrs
import numpy as np

from pstm.dickert.household import GEN

if t.TYPE_CHECKING:
    import numpy.typing as npt

    DistributionType = t.Literal["normal", "unif", "lognormal"]
    Phase = t.Literal[1, 2, 3]


def validate_level(level: float) -> float:
    if not 0 < level < 1:
        msg = "Value has to be between 0 and 1, is {level}"
        raise ValueError(msg)

    return level


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Appliance:
    description: str
    phase: Phase
    switch_on_current: float
    switch_on_time: int
    equipment_level: attrs.field(float, validator=validate_level)
    active_power_distribution_type: DistributionType
    active_power_parameter_1: float
    active_power_parameter_2: float
    active_power_parameter_3: float
    reactive_power_share: attrs.field(float, validator=validate_level)
    reactive_power_distribution_type: DistributionType
    reactive_power_parameter_1: attrs.field(float, validator=validate_level)
    reactive_power_parameter_2: attrs.field(float, validator=validate_level)
    reactive_power_parameter_3: attrs.field(float, validator=validate_level)

    def sim_distribution(
        self,
        *,
        distribution_type: DistributionType,
        p_1: float,
        p_2: float,
        n_units: int,
        n_steps: int = 1,
    ) -> npt.NDArray[np.float64]:
        match distribution_type:
            case "normal":
                p = GEN.normal(
                    loc=p_1,
                    scale=p_2,
                    size=(n_steps, n_units),
                )
            case "unif":
                p = GEN.uniform(
                    low=p_1 - p_2,
                    high=p_1 + p_2,
                    size=(n_steps, n_units),
                )
            case "lognormal":
                p = GEN.lognormal(
                    mean=p_1,
                    sigma=p_2,
                    size=(n_steps, n_units),
                )

        p[p < 0] = 0
        return p

    def run(self, *, n_units: int, n_steps: int, phase_distribution: tuple[float, float, float]) -> None:
        p, q = self._run(n_units, n_steps)
        if self.phase == 0:
            self.phase = GEN.choice((1, 2, 3), p=phase_distribution)

        self.p = np.zeros((n_steps, n_units, 3))
        self.p[:, :, self.phase] = p
        self.q = np.zeros((n_steps, n_units, 3))
        self.q[:, :, self.phase] = q

    @abc.abstractmethod
    def _run(self, *, n_units: int, n_steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Internal power calculation method.

        Arguments:
            n_units {int} -- amount of units
            n_steps {int} -- amount of time steps

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] -- active and reactive power
        """
