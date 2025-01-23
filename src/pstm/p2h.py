# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

from typing import Literal

import attrs
import numpy as np

from pstm.base import Tech


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Power2Heat(Tech):
    power_inst: float
    efficiency: float
    heating_type: Literal["room", "process_low", "process_high"]
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        if self.heating_type == "room":
            self.thr.low = -self.power_inst * self.efficiency
        elif self.heating_type == "process_low":
            self.thl.low = -self.power_inst * self.efficiency
        elif self.heating_type == "process_high":
            self.thh.low = -self.power_inst * self.efficiency
        else:
            raise ValueError

        self.acp.high = self.power_inst
        self.acq.high = self.power_inst * np.tan(np.arccos(self.cosphi))
