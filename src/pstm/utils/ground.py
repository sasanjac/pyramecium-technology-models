# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Undisturbed ground temperature after Kusuda and Achenbach (1965).

[1] T. Kusuda and P. R. Achenbach, "Earth temperature and thermal diffusivity
at selected stations in the United States," National Bureau of Standards,
NBS Report 8972, Jun. 1965, eq. (3).

The report models the earth temperature at depth ``z`` and day ``t`` as a
damped and delayed first harmonic of the surface temperature:

    T(z, t) = A - B0 * exp(-z * sqrt(pi / (D * tau)))
                     * cos(2 * pi / tau * (t - P0 - z / 2 * sqrt(tau / (pi * D))))

with the annual mean ``A``, the surface amplitude ``B0``, the phase constant
``P0`` (day of the surface minimum), the thermal diffusivity ``D`` and the
period ``tau = 365 d``.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from pyochain import Err
from pyochain import Ok

if t.TYPE_CHECKING:
    import numpy.typing as npt
    from pyochain import Result

FT2_PER_H_TO_M2_PER_D = 0.09290304 * 24.0
"""Conversion of a diffusivity from ft^2/h to m^2/d."""

KUSUDA_DIFFUSIVITY = 0.025 * FT2_PER_H_TO_M2_PER_D
"""Soil thermal diffusivity in m^2/d.

The report uses 0.025 ft^2/h as "an approximate median of all of the thermal
diffusivities"; most stations lie between 0.015 and 0.035 ft^2/h.
"""

KUSUDA_PERIOD = 365.0
"""Period of the annual temperature wave in days."""

ERA_INTERIM_LEVEL4_DEPTH = (1.00 + 2.89) / 2
"""Midpoint in m of ERA-Interim soil level 4 (1.00 m to 2.89 m).

Ruhnau et al. (2019) take the ground-source temperature from this layer.
"""

_MIN_DAYS = 365

class Ground:
    @staticmethod
    def kusuda_temperature(
        day: npt.ArrayLike,
        *,
        mean: float,
        amplitude: float,
        phase: float,
        depth: float = ERA_INTERIM_LEVEL4_DEPTH,
        diffusivity: float = KUSUDA_DIFFUSIVITY,
        period: float = KUSUDA_PERIOD,
    ) -> npt.NDArray[np.float64]:
        """Earth temperature of eq. (3) of Kusuda and Achenbach (1965).

        Args:
            day: Day of the year (0-based, fractional) at which the temperature is evaluated.
            mean: Annual mean surface temperature ``A`` in degC.
            amplitude: Amplitude ``B0`` of the surface temperature in K.
            phase: Phase constant ``P0`` in days, the day of the surface minimum.
            depth: Depth ``z`` in m.
            diffusivity: Soil thermal diffusivity ``D`` in m^2/d.
            period: Period ``tau`` in days.

        Returns:
            The earth temperature in degC at every entry of ``day``.
        """
        day_arr = np.asarray(day, dtype=np.float64)
        damping = np.exp(-depth * np.sqrt(np.pi / (diffusivity * period)))
        lag = depth / 2 * np.sqrt(period / (np.pi * diffusivity))
        return t.cast(
            "npt.NDArray[np.float64]",
            mean - amplitude * damping * np.cos(2 * np.pi / period * (day_arr - phase - lag)),
        )


    @staticmethod
    def _day_of_year(index: pd.DatetimeIndex) -> npt.NDArray[np.float64]:
        hours = index.hour.to_numpy() + index.minute.to_numpy() / 60 + index.second.to_numpy() / 3600
        return t.cast("npt.NDArray[np.float64]", (index.dayofyear.to_numpy() - 1) + hours / 24)


    @staticmethod
    def fit_surface_harmonic(
        temp: pd.Series,
        period: float = KUSUDA_PERIOD,
    ) -> Result[tuple[float, float, float], ValueError]:
        """Annual mean, amplitude and phase constant of a temperature series.

        ``A`` is the mean of the series.  ``B0`` and ``P0`` come from a
        least-squares fit of ``A - B0 * cos(2 * pi / tau * (t - P0))`` to the
        daily means, each placed at noon of its day.

        Args:
            temp: Temperature series in degC on a DatetimeIndex covering at least one year.
            period: Period ``tau`` in days.

        Returns:
            ``Ok`` with ``(A, B0, P0)`` in degC, K and days, or ``Err`` if the
            series covers fewer than 365 days.
        """
        index = t.cast("pd.DatetimeIndex", temp.index)
        daily = temp.groupby(index.normalize()).mean()
        if len(daily) < _MIN_DAYS:
            msg = f"The harmonic fit needs at least {_MIN_DAYS} days of temperature data, got {len(daily)}."
            return Err(ValueError(msg))
        daily_index = t.cast("pd.DatetimeIndex", daily.index)
        omega = 2 * np.pi / period
        day = (daily_index.dayofyear.to_numpy() - 1) + 0.5
        mean = float(temp.mean())
        design = np.column_stack((np.cos(omega * day), np.sin(omega * day)))
        (coef_cos, coef_sin), *_ = np.linalg.lstsq(design, daily.to_numpy(dtype=np.float64) - mean, rcond=None)
        # A - B0 cos(w (t - P0)) = A - B0 cos(w P0) cos(w t) - B0 sin(w P0) sin(w t)
        amplitude = float(np.hypot(coef_cos, coef_sin))
        phase = float(np.mod(np.arctan2(-coef_sin, -coef_cos) / omega, period))
        return Ok((mean, amplitude, phase))


    @staticmethod
    def kusuda_ground_temperature(
        temp_air: pd.Series,
        *,
        depth: float = ERA_INTERIM_LEVEL4_DEPTH,
        diffusivity: float = KUSUDA_DIFFUSIVITY,
        period: float = KUSUDA_PERIOD,
    ) -> Result[pd.Series, ValueError]:
        """Ground temperature at ``depth`` approximated from the air temperature.

        The surface parameters of eq. (3) are taken from the air temperature:
        ``A`` is its annual mean, which the report finds "very close to the
        annual average outdoor air temperature" of the earth.  ``B0`` and ``P0``
        come from the first-harmonic fit of :func:`fit_surface_harmonic`.  The
        report finds the surface amplitude poorly correlated with the air
        amplitude, so ``B0`` taken from the air temperature is an approximation.

        Args:
            temp_air: Air temperature in degC on a DatetimeIndex covering at least one year.
            depth: Depth ``z`` in m.
            diffusivity: Soil thermal diffusivity ``D`` in m^2/d.
            period: Period ``tau`` in days.

        Returns:
            ``Ok`` with the ground temperature in degC on the index of
            ``temp_air``, or ``Err`` if ``temp_air`` covers fewer than 365 days.
        """
        index = t.cast("pd.DatetimeIndex", temp_air.index)

        def ground(mean: float, amplitude: float, phase: float) -> pd.Series:
            values = kusuda_temperature(
                _day_of_year(index),
                mean=mean,
                amplitude=amplitude,
                phase=phase,
                depth=depth,
                diffusivity=diffusivity,
                period=period,
            )
            return pd.Series(values, index=index, name="temp_ground_celsius")

        return fit_surface_harmonic(temp_air, period=period).map_star(ground)
