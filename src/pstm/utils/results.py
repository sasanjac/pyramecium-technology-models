# Copyright (c) 2018-2026 Sasan Jacob Rasti
"""Constructor of the attrs classes of pstm that returns a :class:`pyochain.Result`."""

from __future__ import annotations

import typing as t

from pyochain import Err
from pyochain import Ok

if t.TYPE_CHECKING:
    from pyochain import Result


class Constructible:
    """Adds :meth:`create` to an attrs class.

    attrs runs converters and validators in ``__init__``, which raise.
    :meth:`create` returns their ``ValueError`` as ``Err``, so a caller can
    build an instance from unchecked input without ``try``.
    """

    @classmethod
    def create(cls, **kwargs: t.Any) -> Result[t.Self, ValueError]:
        """Instance built from ``kwargs``.

        Returns:
            ``Ok`` with the instance, or ``Err`` with the ``ValueError`` of a
            converter or validator.
        """
        try:
            return Ok(cls(**kwargs))
        except ValueError as error:
            return Err(error)
