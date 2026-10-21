# Copyright (c) 2018-2026 Sasan Jacob Rasti

from __future__ import annotations

import attrs
import numpy as np

from pstm.base import HeatingType
from pstm.base import Tech


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Power2Heat(Tech):
    power_inst: float
    efficiency: float
    heating_type: HeatingType = attrs.field(converter=HeatingType)
    cosphi: float = attrs.field(default=0.9)

    def __attrs_post_init__(self) -> None:
        self.heat(self.heating_type).low = -self.power_inst * self.efficiency
        self.acp.high = self.power_inst
        self.acq.high = self.power_inst * np.tan(np.arccos(self.cosphi))
