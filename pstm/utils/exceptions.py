# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause
from __future__ import annotations


class ModelNotRunError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Run model first.")
