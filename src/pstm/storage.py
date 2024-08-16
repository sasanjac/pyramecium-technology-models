# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause
from __future__ import annotations

from typing import Literal

import attrs
import numpy as np

from pstm.base import Tech

FREQS = {
    "H": 3600,
}


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Storage(Tech):
    power_inst_in: float
    power_inst_out: float
    capacity: float
    efficiency_in: float
    efficiency_out: float
    self_discharge: float
    storage_type: Literal["EL", "THR", "THL", "THH", "CH4", "H2"]
    cosphi: float = attrs.field(default=0.9)

    def run(self) -> None:
        if self.storage_type == "EL":
            self.acp.high = self.power_inst_in
            self.acp.low = -self.power_inst_out
            self.acq.high = self.power_inst_in * np.tan(np.arccos(self.cosphi))
            self.acq.low = -self.power_inst_out * np.tan(np.arccos(self.cosphi))
        elif self.storage_type == "THR":
            self.thr.high = self.power_inst_in
            self.thr.low = -self.power_inst_out
        elif self.storage_type == "THL":
            self.thl.high = self.power_inst_in
            self.thl.low = -self.power_inst_out
        elif self.storage_type == "THH":
            self.thr.high = self.power_inst_in
            self.thr.low = -self.power_inst_out
        elif self.storage_type == "CH4":
            self.ch4.high = self.power_inst_in
            self.ch4.low = -self.power_inst_out
        elif self.storage_type == "H2":
            self.h2.high = self.power_inst_in
            self.h2.low = -self.power_inst_out
