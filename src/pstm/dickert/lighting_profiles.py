# Copyright (c) 2018-2025 Sasan Jacob Rasti
# Copyright (c) 2015-2025 Jörg Dickert

from __future__ import annotations

import datetime as dt
import typing as t

import attrs
import numpy as np
import numpy.typing as npt
import pvlib

from pstm.dickert.appliances import Appliances
from pstm.dickert.appliances import Constants
from pstm.dickert.appliances import DistributionType
from pstm.dickert.appliances import validate_level_sequence
from pstm.utils import dates

NaT = dt.time(0)


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class LightingProfiles(Appliances):
    lighting_distribution_types: tuple[
        DistributionType,
        DistributionType,
    ]
    lighting_parameters_1: tuple[dt.time, dt.time]
    lighting_parameters_2: tuple[dt.time, dt.time]
    time_on_distribution_types: tuple[
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
    ]
    time_on_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_off_distribution_types: tuple[
        DistributionType,
        DistributionType,
        DistributionType,
        DistributionType,
    ]
    time_off_parameters_1: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_off_parameters_2: tuple[
        dt.time,
        dt.time,
        dt.time,
        dt.time,
    ]
    time_on_variations: tuple[float, float, float, float] = attrs.field(
        validator=validate_level_sequence,
    )

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
        location = pvlib.location.Location(
            latitude=lat,
            longitude=lon,
            tz=str(tz),
            altitude=altitude,
        )
        times = dates.date_range(tz=tz, freq=dt.timedelta(days=1), year=year)
        s = location.get_sun_rise_set_transit(times, method="spa")
        sunrise = s.sunrise.apply(lambda x: x.hour * 60 + x.minute).to_numpy()
        sunset = s.sunset.apply(lambda x: x.hour * 60 + x.minute).to_numpy()
        p_const = self._sim_distribution(
            distribution_type=self.active_power_distribution_type,
            parameter_1=self.active_power_parameter_1,
            parameter_2=self.active_power_parameter_2,
            n_units=n_units,
            generator=generator,
        )
        p_wd_morning = self._calc_wd_morning(
            n_units=n_units,
            n_steps=n_steps,
            n_days=Constants.WEEKDAYS_PER_YEAR,
            day_index_start=0,
            day_index_end=5,
            sunrise=sunrise,
            generator=generator,
        )
        p_wd_evening = self._calc_wd_evening(
            n_units=n_units,
            n_steps=n_steps,
            n_days=Constants.WEEKDAYS_PER_YEAR,
            day_index_start=0,
            day_index_end=5,
            sunset=sunset,
            generator=generator,
        )
        p_we_morning = self._calc_we_morning(
            n_units=n_units,
            n_steps=n_steps,
            n_days=Constants.WEEKENDDAYS_PER_YEAR,
            day_index_start=5,
            day_index_end=7,
            sunrise=sunrise,
            generator=generator,
        )
        p_we_evening = self._calc_we_evening(
            n_units=n_units,
            n_steps=n_steps,
            n_days=Constants.WEEKENDDAYS_PER_YEAR,
            day_index_start=5,
            day_index_end=7,
            sunset=sunset,
            generator=generator,
        )
        p_base = p_wd_morning + p_wd_evening + p_we_morning + p_we_evening
        p_base[p_base > 1] = 1
        p = p_base * p_const
        return self._finalize_power(
            n_units=n_units,
            n_steps=n_steps,
            distribution_type=self.reactive_power_distribution_type,
            parameter_1=self.reactive_power_parameter_1,
            parameter_2=self.reactive_power_parameter_2,
            active_power=p,
            generator=generator,
        )

    def _calc_wd_evening(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunset: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        return self._calc_evening(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            parameter_index=1,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunset=sunset,
            generator=generator,
            add_tail=True,
        )

    def _calc_we_evening(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunset: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        return self._calc_evening(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            parameter_index=3,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunset=sunset,
            generator=generator,
        )

    def _calc_evening(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        parameter_index: t.Literal[1, 3],
        day_index_start: int,
        day_index_end: int,
        sunset: npt.NDArray[np.int64],
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.float64]:
        top_1 = self.time_on_parameters_1[parameter_index]
        top_2 = self.time_on_parameters_2[parameter_index]
        if NaT not in {top_1, top_2}:
            time_on_user = self._calc_time_on_user(
                n_units=n_units,
                n_steps=n_steps,
                n_days=n_days,
                time_on_distribution_types=self.time_on_distribution_types[parameter_index],
                time_on_parameters_1=top_1,
                time_on_parameters_2=top_2,
                day_index_start=day_index_start,
                day_index_end=day_index_end,
                add_tail=add_tail,
                generator=generator,
            )
        else:
            time_on_user = None

        time_off = self._calc_time_off_user(
            n_units=n_units,
            n_steps=n_steps,
            time_off_distribution_types=self.time_off_distribution_types[parameter_index],
            time_off_parameters_1=self.time_off_parameters_1[parameter_index],
            time_off_parameters_2=self.time_off_parameters_2[parameter_index],
            n_days=n_days,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            add_tail=add_tail,
            generator=generator,
        )

        time_on = self._calc_time_on_sunset(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunset=sunset,
            add_tail=add_tail,
            generator=generator,
        )

        if time_on_user is not None:
            time_on = np.max([time_on_user, time_on], axis=0)

        return self._calc_power_from_times(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            time_on=time_on,
            time_off=time_off,
            generator=generator,
        )

    def _calc_wd_morning(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunrise: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        return self._calc_morning(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            parameter_index=0,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunrise=sunrise,
            generator=generator,
            add_tail=True,
        )

    def _calc_we_morning(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunrise: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        return self._calc_morning(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            parameter_index=2,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunrise=sunrise,
            generator=generator,
        )

    def _calc_morning(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        parameter_index: t.Literal[0, 2],
        day_index_start: int,
        day_index_end: int,
        sunrise: npt.NDArray[np.int64],
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.float64]:
        time_on = self._calc_time_on_user(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            time_on_distribution_types=self.time_on_distribution_types[parameter_index],
            time_on_parameters_1=self.time_on_parameters_1[parameter_index],
            time_on_parameters_2=self.time_on_parameters_2[parameter_index],
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            add_tail=add_tail,
            generator=generator,
        )
        top_1 = self.time_off_parameters_1[parameter_index]
        top_2 = self.time_off_parameters_2[parameter_index]
        if NaT not in {top_1, top_2}:
            time_off_user = self._calc_time_off_user(
                n_units=n_units,
                n_steps=n_steps,
                time_off_distribution_types=self.time_off_distribution_types[parameter_index],
                time_off_parameters_1=top_1,
                time_off_parameters_2=top_2,
                n_days=n_days,
                day_index_start=day_index_start,
                day_index_end=day_index_end,
                add_tail=add_tail,
                generator=generator,
            )
        else:
            time_off_user = None

        time_off = self._calc_time_off_sunrise(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            sunrise=sunrise,
            add_tail=add_tail,
            generator=generator,
        )

        if time_off_user is not None:
            time_off = np.min([time_off_user, time_off], axis=0)

        return self._calc_power_from_times(
            n_units=n_units,
            n_steps=n_steps,
            n_days=n_days,
            time_on=time_on,
            time_off=time_off,
            generator=generator,
        )

    def _calc_power_from_times(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        time_on: npt.NDArray[np.int64],
        time_off: npt.NDArray[np.int64],
        generator: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        for i in range(n_units):
            idx = time_on[:, i] >= time_off[:, i]
            time_on[idx, i] = 0
            time_off[idx, i] = 0

        prob_1 = self._sim_distribution(
            distribution_type="unif",
            parameter_1=0,
            parameter_2=1,
            n_steps=n_days,
            n_units=n_units,
            clear=False,
            generator=generator,
        )
        prob_2 = self.time_on_variations[0]
        idx = prob_2 > prob_1
        time_on = (time_on * idx).astype(np.int64)
        time_off = (time_off * idx).astype(np.int64)
        return self._finalize_active_power(
            n_steps=n_steps,
            n_units=n_units,
            active_power=np.array([1.0]),
            time_on=time_on,
            time_off=time_off,
        )

    def _calc_time_on_user(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        time_on_distribution_types: DistributionType,
        time_on_parameters_1: dt.time,
        time_on_parameters_2: dt.time,
        day_index_start: int,
        day_index_end: int,
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        samples_per_day = int(Constants.MINUTES_PER_DAY / step_length)
        steps = self._calc_steps(
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            samples_per_day=samples_per_day,
            add_tail=add_tail,
        )
        idx = (
            self._time_as_float(time_on_parameters_1)
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length
            + np.sort(steps)[:, np.newaxis]
        ).astype(np.int64)
        prob = self._sim_distribution_round(
            distribution_type=time_on_distribution_types,
            parameter_1=0,
            parameter_2=self._time_as_float(time_on_parameters_2)
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            n_steps=n_days,
            n_units=n_units,
            clear=False,
            generator=generator,
        )
        time_on_weekday_mid = idx + prob
        time_on_weekday_mid[time_on_weekday_mid < 1] = 1
        return time_on_weekday_mid

    def _calc_time_off_user(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        time_off_distribution_types: DistributionType,
        time_off_parameters_1: dt.time,
        time_off_parameters_2: dt.time,
        day_index_start: int,
        day_index_end: int,
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        samples_per_day = int(Constants.MINUTES_PER_DAY / step_length)
        steps = self._calc_steps(
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            samples_per_day=samples_per_day,
            add_tail=add_tail,
        )
        idx = (
            self._time_as_float(time_off_parameters_1)
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length
            + np.sort(steps)[:, np.newaxis]
        ).astype(np.int64)
        prob = self._sim_distribution_round(
            distribution_type=time_off_distribution_types,
            parameter_1=0,
            parameter_2=self._time_as_float(time_off_parameters_2)
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            n_steps=n_days,
            n_units=n_units,
            clear=False,
            generator=generator,
        )
        return idx + prob

    def _calc_time_off_sunrise(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunrise: npt.NDArray[np.int64],
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        samples_per_day = Constants.MINUTES_PER_DAY / step_length
        steps = self._calc_steps(
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            samples_per_day=1,
            add_tail=add_tail,
        )
        idx = np.sort(steps)[:, np.newaxis]
        prob = self._sim_distribution(
            distribution_type=self.lighting_distribution_types[0],
            parameter_1=self._time_as_float(self.lighting_parameters_1[0])
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            parameter_2=self._time_as_float(self.lighting_parameters_2[0])
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            n_steps=n_days,
            n_units=n_units,
            clear=False,
            generator=generator,
        )
        return (idx * samples_per_day + sunrise[idx] + prob).astype(np.int64)

    def _calc_time_on_sunset(
        self,
        *,
        n_units: int,
        n_steps: int,
        n_days: int,
        day_index_start: int,
        day_index_end: int,
        sunset: npt.NDArray[np.int64],
        generator: np.random.Generator,
        add_tail: bool = False,
    ) -> npt.NDArray[np.int64]:
        step_length = Constants.MINUTES_PER_YEAR / n_steps
        samples_per_day = Constants.MINUTES_PER_DAY / step_length
        steps = self._calc_steps(
            day_index_start=day_index_start,
            day_index_end=day_index_end,
            samples_per_day=1,
            add_tail=add_tail,
        )
        idx = np.sort(steps)[:, np.newaxis]
        prob = self._sim_distribution(
            distribution_type=self.lighting_distribution_types[1],
            parameter_1=self._time_as_float(self.lighting_parameters_1[1])
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            parameter_2=self._time_as_float(self.lighting_parameters_2[1])
            * Constants.MINUTES_PER_HOUR
            * Constants.HOURS_PER_DAY
            / step_length,
            n_steps=n_days,
            n_units=n_units,
            clear=False,
            generator=generator,
        )
        return (idx * samples_per_day + sunset[idx] + prob).astype(np.int64)
