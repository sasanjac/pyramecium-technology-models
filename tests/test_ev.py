# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for Electric Vehicle technology model."""

from __future__ import annotations

import zoneinfo as zi

import numpy as np
import pytest

from pstm.ev import CARS
from pstm.ev import Car
from pstm.ev import ChargePoint
from pstm.ev import Driver
from pstm.ev import DrivingCategory
from pstm.ev import EVSystem
from pstm.ev import TypeDay
from pstm.utils.dates import Dates

TZ = zi.ZoneInfo("Europe/Berlin")


class TestDriver:
    """Test Driver class functionality."""

    def test_driver_creation(self) -> None:
        """Test Driver instance creation."""
        driver = Driver(daily_mean_distance=30)
        assert driver.daily_mean_distance == 30

    def test_driver_from_profile(self) -> None:
        """Test Driver creation from profile."""
        driver = Driver.from_profile(profile=6)
        assert driver.daily_mean_distance == 35

    def test_driver_from_random(self) -> None:
        """Test Driver creation from random profile."""
        driver = Driver.from_random()
        assert driver.daily_mean_distance > 0

    def test_driving_category_low(self) -> None:
        """Test low driving category classification."""
        # < 10_000 km/year means < ~27.4 km/day
        driver = Driver(daily_mean_distance=10)
        assert driver.driving_category is DrivingCategory.LOW

    def test_driving_category_medium(self) -> None:
        """Test medium driving category classification."""
        # 10_000 - 20_000 km/year means ~27.4 - ~54.8 km/day
        driver = Driver(daily_mean_distance=40)
        assert driver.driving_category is DrivingCategory.MEDIUM

    def test_driving_category_high(self) -> None:
        """Test high driving category classification."""
        # >= 20_000 km/year means >= ~54.8 km/day
        driver = Driver(daily_mean_distance=60)
        assert driver.driving_category is DrivingCategory.HIGH

    def test_distance_distribution(self) -> None:
        """Test distance distribution sampling."""
        driver = Driver(daily_mean_distance=30)
        distances = driver.distance_distribution(n_samples=100)
        assert len(distances) == 100
        assert np.all(distances > 0)

    def test_number_of_trips(self) -> None:
        """Test number of trips sampling."""
        driver = Driver(daily_mean_distance=30)
        n_trips = driver.number_of_trips(TypeDay.WD)
        assert n_trips in [0, 1, 2, 3]


class TestCar:
    """Test Car class functionality."""

    def test_car_creation(self) -> None:
        """Test Car instance creation."""
        car = Car(
            mileage=15.0,
            losses=0.10,
            capacity=40,
            charging_power=11.0,
        )
        assert car.mileage == 15.0
        assert car.losses == 0.10
        assert car.capacity == 40
        assert car.charging_power == 11.0


class TestChargePoint:
    """Test ChargePoint class functionality."""

    def test_charge_point_creation(self) -> None:
        """Test ChargePoint instance creation."""
        cp = ChargePoint(charging_power_max=22.0)
        assert cp.charging_power_max == 22.0


class TestEVSystem:
    """Test EVSystem technology model."""

    @pytest.fixture
    def dates(self):
        """Generate dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)[:96]

    @pytest.fixture
    def ev_system(self, dates):
        """Create EVSystem for testing."""
        driver = Driver(daily_mean_distance=30)
        car = Car(
            mileage=15.0,
            losses=0.10,
            capacity=40,
            charging_power=11.0,
        )
        charge_point = ChargePoint(charging_power_max=22.0)
        return EVSystem(
            dates=dates,
            driver=driver,
            car=car,
            charge_point=charge_point,
            tz=TZ,
        )

    def test_ev_system_creation(self, ev_system) -> None:
        """Test EVSystem instance creation."""
        assert ev_system.driver.daily_mean_distance == 30
        assert ev_system.car.charging_power == 11.0
        assert ev_system.charge_point.charging_power_max == 22.0

    def test_ev_system_power_array(self, ev_system) -> None:
        """Test power array generation for a day."""
        power = ev_system.power(TypeDay.WD)
        assert len(power) == 96  # 24 hours * 4 quarters
        assert np.all(power >= 0)
        assert np.all(power <= ev_system.car.charging_power)

    def test_ev_system_energy_reasonable(self, ev_system) -> None:
        """Test energy consumption is reasonable."""
        energy_wd = ev_system.energy(TypeDay.WD)
        energy_we = ev_system.energy(TypeDay.WE)
        # Energy should be non-negative
        assert energy_wd >= 0
        assert energy_we >= 0
        # Energy for one day should be bounded by battery capacity
        # (could be higher with losses but let's check reasonable bounds)
        assert energy_wd < 200  # kWh - reasonable upper bound
        assert energy_we < 200


class TestEVSystemRun:
    """Test EVSystem run method."""

    @pytest.fixture
    def dates_full_year(self):
        """Generate full year dates for testing."""
        return Dates.date_range(tz=TZ, year=2024)

    @pytest.fixture
    def ev_system_full(self, dates_full_year):
        """Create EVSystem for full year testing."""
        driver = Driver(daily_mean_distance=30)
        car = Car(
            mileage=15.0,
            losses=0.10,
            capacity=40,
            charging_power=11.0,
        )
        charge_point = ChargePoint(charging_power_max=22.0)
        return EVSystem(
            dates=dates_full_year,
            driver=driver,
            car=car,
            charge_point=charge_point,
            tz=TZ,
        )

    def test_ev_system_run(self, ev_system_full) -> None:
        """Test EVSystem run method populates acp."""
        ev_system_full.run()
        acp_high = ev_system_full.acp["high"]["L1"]
        # acp should have values
        assert not acp_high.isna().all()
        # All values should be non-negative
        assert np.all(acp_high.values >= 0)
        # Check max power constraint
        assert np.all(acp_high.values <= ev_system_full.car.charging_power)


class TestCarPresets:
    """Test predefined car models."""

    def test_bmw_i3_exists(self) -> None:
        """Test BMW i3 preset exists."""
        assert "BMW i3" in CARS
        assert CARS["BMW i3"].capacity == 27.2

    def test_vw_egolf_exists(self) -> None:
        """Test VW e-Golf preset exists."""
        assert "VW e-Golf" in CARS

    def test_renault_zoe_exists(self) -> None:
        """Test Renault Zoe preset exists."""
        assert "Renault Zoe" in CARS
