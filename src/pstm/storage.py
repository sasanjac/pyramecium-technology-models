# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import enum
import typing as t

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
    storage_type: StorageType = attrs.field(converter=StorageType)
    cosphi: float = attrs.field(default=0.9)

    def run(self) -> None:
        match self.storage_type:
            case StorageType.EL:
                self.acp.high = self.power_inst_in
                self.acp.low = -self.power_inst_out
                self.acq.high = self.power_inst_in * np.tan(np.arccos(self.cosphi))
                self.acq.low = -self.power_inst_out * np.tan(np.arccos(self.cosphi))
            case StorageType.THR:
                self.thr.high = self.power_inst_in
                self.thr.low = -self.power_inst_out
            case StorageType.THL:
                self.thl.high = self.power_inst_in
                self.thl.low = -self.power_inst_out
            case StorageType.THH:
                self.thh.high = self.power_inst_in
                self.thh.low = -self.power_inst_out
            case StorageType.CH4:
                self.ch4.high = self.power_inst_in
                self.ch4.low = -self.power_inst_out
            case StorageType.H2:
                self.h2.high = self.power_inst_in
                self.h2.low = -self.power_inst_out
            case _:
                t.assert_never(self.storage_type)
