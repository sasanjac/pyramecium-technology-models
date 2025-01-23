# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import enum

import attrs
import numpy as np

from pstm.base import Tech

FREQS = {
    "H": 3600,
}


class StorageType(enum.Enum):
    EL = "EL"
    THR = "THR"
    THL = "THL"
    THH = "THH"
    CH4 = "CH4"
    H2 = "H2"


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Storage(Tech):
    power_inst_in: float
    power_inst_out: float
    capacity: float
    efficiency_in: float
    efficiency_out: float
    self_discharge: float
    storage_type: StorageType
    cosphi: float = attrs.field(default=0.9)

    def run(self) -> None:
        if self.storage_type == StorageType.EL:
            self.acp.high = self.power_inst_in
            self.acp.low = -self.power_inst_out
            self.acq.high = self.power_inst_in * np.tan(np.arccos(self.cosphi))
            self.acq.low = -self.power_inst_out * np.tan(np.arccos(self.cosphi))
        elif self.storage_type == StorageType.THR:
            self.thr.high = self.power_inst_in
            self.thr.low = -self.power_inst_out
        elif self.storage_type == StorageType.THL:
            self.thl.high = self.power_inst_in
            self.thl.low = -self.power_inst_out
        elif self.storage_type == StorageType.THH:
            self.thr.high = self.power_inst_in
            self.thr.low = -self.power_inst_out
        elif self.storage_type == StorageType.CH4:
            self.ch4.high = self.power_inst_in
            self.ch4.low = -self.power_inst_out
        elif self.storage_type == StorageType.H2:
            self.h2.high = self.power_inst_in
            self.h2.low = -self.power_inst_out
