# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

from typing import Literal

import attrs

from pstm.base import Tech


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class GasBoiler(Tech):
    power_inst: float
    efficiency: float
    heating_type: Literal["room", "process_low", "process_high"]
    gas_type: Literal["CH4", "H2"]

    def __attrs_post_init__(self) -> None:
        if self.heating_type == "room":
            self.thr.low = -self.power_inst
        elif self.heating_type == "process_low":
            self.thl.low = -self.power_inst
        elif self.heating_type == "process_high":
            self.thh.low = -self.power_inst
        else:
            raise ValueError

        if self.gas_type == "CH4":
            self.ch4.high = self.power_inst / self.efficiency
        elif self.gas_type == "H2":
            self.h2.high = self.power_inst / self.efficiency
        else:
            raise ValueError
