# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import enum
import typing as t

import attrs

from pstm.base import HeatingType
from pstm.base import Tech


class GasType(enum.StrEnum):
    CH4 = "CH4"
    H2 = "H2"


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class GasBoiler(Tech):
    power_inst: float
    efficiency: float
    heating_type: HeatingType = attrs.field(converter=HeatingType)
    gas_type: GasType = attrs.field(converter=GasType)

    def __attrs_post_init__(self) -> None:
        self.heat(self.heating_type).low = -self.power_inst
        match self.gas_type:
            case GasType.CH4:
                self.ch4.high = self.power_inst / self.efficiency
            case GasType.H2:
                self.h2.high = self.power_inst / self.efficiency
            case _:
                t.assert_never(self.gas_type)
