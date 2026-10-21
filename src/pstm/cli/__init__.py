# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

from pstm.utils.exceptions import MissingExtraError

try:
    import click  # noqa: F401
    import openpyxl  # noqa: F401
except ModuleNotFoundError as e:
    extra = "cli"
    raise MissingExtraError(extra) from e
