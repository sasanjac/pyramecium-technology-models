# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations


class ModelNotRunError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Run model first.")


class MissingExtraError(ModuleNotFoundError):
    def __init__(self, extra: str) -> None:
        super().__init__(f"Install the extra {extra}: pip install pyramecium-technology-models[{extra}]")
