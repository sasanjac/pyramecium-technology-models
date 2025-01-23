# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations


class ModelNotRunError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Run model first.")
