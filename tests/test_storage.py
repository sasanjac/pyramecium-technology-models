# Copyright (c) 2018-2025 Sasan Jacob Rasti

import zoneinfo as zi

from pstm.storage import Storage
from pstm.storage import StorageType
from pstm.utils.dates import date_range

POWER_INST_IN = 100
POWER_INST_OUT = 80
CAPACITY = 1000
EFFICIENCY_IN = 0.9
EFFICIENCY_OUT = 0.8
SELF_DISCHARGE = 0.05
STORAGE_TYPE = StorageType.EL
COSPHI = 0.9


def test_storage_run() -> None:
    # Create a Storage instance with some initial values
    storage = Storage(
        dates=date_range(tz=zi.ZoneInfo("Europe/Berlin"), year=2050),
        power_inst_in=POWER_INST_IN,
        power_inst_out=POWER_INST_OUT,
        capacity=CAPACITY,
        efficiency_in=EFFICIENCY_IN,
        efficiency_out=EFFICIENCY_OUT,
        self_discharge=SELF_DISCHARGE,
        storage_type=STORAGE_TYPE,
        cosphi=COSPHI,
    )

    # Call the run method
    storage.run()

    # Add assertions to check the behavior of the run method
    assert storage.power_inst_in == POWER_INST_IN
    assert storage.power_inst_out == POWER_INST_OUT
    assert storage.capacity == CAPACITY
    assert storage.efficiency_in == EFFICIENCY_IN
    assert storage.efficiency_out == EFFICIENCY_OUT
    assert storage.self_discharge == SELF_DISCHARGE
    assert storage.storage_type == STORAGE_TYPE
    assert storage.cosphi == COSPHI
