# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Unit tests for the MaStR catalog lookup."""

from __future__ import annotations

import pandas as pd
from pyochain import NONE
from pyochain import Some

from pstm.mastr_converter import MaStRCatalog


class TestGetValue:
    @staticmethod
    def _catalog() -> MaStRCatalog:
        catalog = object.__new__(MaStRCatalog)
        catalog._catalog = pd.DataFrame({"Id": [35, 36], "Wert": ["In Betrieb", "In Planung"]})  # noqa: SLF001
        return catalog

    def test_known_code(self) -> None:
        assert self._catalog().get_value(35.0) == Some("In Betrieb")

    def test_unknown_code_is_none(self) -> None:
        assert self._catalog().get_value(99.0) == NONE

    def test_missing_code_is_none(self) -> None:
        assert self._catalog().get_value(float("nan")) == NONE
