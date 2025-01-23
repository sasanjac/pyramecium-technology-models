# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

from typing import Literal
from typing import TypedDict


class RoughnessLength(TypedDict):
    rmin: float
    rmax: float
    most: float


ValidCLCs = Literal[
    "111",
    "112",
    "121",
    "122",
    "123",
    "124",
    "131",
    "132",
    "133",
    "141",
    "142",
    "211",
    "212",
    "213",
    "221",
    "222",
    "223",
    "231",
    "241",
    "242",
    "243",
    "244",
    "311",
    "312",
    "313",
    "321",
    "322",
    "323",
    "324",
    "331",
    "332",
    "333",
    "334",
    "335",
    "411",
    "412",
    "421",
    "422",
    "423",
    "511",
    "512",
    "521",
    "522",
    "523",
]


ROUGHNESS_LENGTH_111: RoughnessLength = {
    "rmin": 1.1,
    "rmax": 1.3,
    "most": 1.2,
}

ROUGHNESS_LENGTH_311: RoughnessLength = {
    "rmin": 0.6,
    "rmax": 1.2,
    "most": 0.75,
}

ROUGHNESS_LENGTH_141: RoughnessLength = {
    "rmin": 0.5,
    "rmax": 0.6,
    "most": 0.6,
}

ROUGHNESS_LENGTH_112: RoughnessLength = {
    "rmin": 0.3,
    "rmax": 0.5,
    "most": 0.5,
}

ROUGHNESS_LENGTH_242: RoughnessLength = {
    "rmin": 0.1,
    "rmax": 0.5,
    "most": 0.3,
}

ROUGHNESS_LENGTH_241: RoughnessLength = {
    "rmin": 0.1,
    "rmax": 0.3,
    "most": 0.1,
}

ROUGHNESS_LENGTH_122: RoughnessLength = {
    "rmin": 0.05,
    "rmax": 0.1,
    "most": 0.075,
}

ROUGHNESS_LENGTH_211: RoughnessLength = {
    "rmin": 0.05,
    "rmax": 0.05,
    "most": 0.05,
}

ROUGHNESS_LENGTH_321: RoughnessLength = {
    "rmin": 0.03,
    "rmax": 0.1,
    "most": 0.005,
}

ROUGHNESS_LENGTH_131: RoughnessLength = {
    "rmin": 0.005,
    "rmax": 0.005,
    "most": 0.005,
}

ROUGHNESS_LENGTH_335: RoughnessLength = {
    "rmin": 0.001,
    "rmax": 0.001,
    "most": 0.001,
}

ROUGHNESS_LENGTH_422: RoughnessLength = {
    "rmin": 0.0005,
    "rmax": 0.0005,
    "most": 0.0005,
}

ROUGHNESS_LENGTH_331: RoughnessLength = {
    "rmin": 0.0003,
    "rmax": 0.0003,
    "most": 0.0003,
}

ROUGHNESS_LENGTH_511: RoughnessLength = {
    "rmin": 0,
    "rmax": 0,
    "most": 0,
}

ROUGHNESS_LENGTHS: dict[ValidCLCs, RoughnessLength] = {
    "111": ROUGHNESS_LENGTH_111,
    "311": ROUGHNESS_LENGTH_311,
    "312": ROUGHNESS_LENGTH_311,
    "313": ROUGHNESS_LENGTH_311,
    "141": ROUGHNESS_LENGTH_141,
    "324": ROUGHNESS_LENGTH_141,
    "334": ROUGHNESS_LENGTH_141,
    "112": ROUGHNESS_LENGTH_112,
    "133": ROUGHNESS_LENGTH_112,
    "121": ROUGHNESS_LENGTH_112,
    "142": ROUGHNESS_LENGTH_112,
    "123": ROUGHNESS_LENGTH_112,
    "242": ROUGHNESS_LENGTH_242,
    "243": ROUGHNESS_LENGTH_242,
    "244": ROUGHNESS_LENGTH_242,
    "241": ROUGHNESS_LENGTH_241,
    "221": ROUGHNESS_LENGTH_241,
    "222": ROUGHNESS_LENGTH_241,
    "223": ROUGHNESS_LENGTH_241,
    "122": ROUGHNESS_LENGTH_122,
    "211": ROUGHNESS_LENGTH_211,
    "212": ROUGHNESS_LENGTH_211,
    "213": ROUGHNESS_LENGTH_211,
    "411": ROUGHNESS_LENGTH_211,
    "421": ROUGHNESS_LENGTH_211,
    "321": ROUGHNESS_LENGTH_321,
    "322": ROUGHNESS_LENGTH_321,
    "323": ROUGHNESS_LENGTH_321,
    "231": ROUGHNESS_LENGTH_321,
    "131": ROUGHNESS_LENGTH_131,
    "132": ROUGHNESS_LENGTH_131,
    "124": ROUGHNESS_LENGTH_131,
    "332": ROUGHNESS_LENGTH_131,
    "333": ROUGHNESS_LENGTH_131,
    "335": ROUGHNESS_LENGTH_335,
    "422": ROUGHNESS_LENGTH_422,
    "412": ROUGHNESS_LENGTH_422,
    "423": ROUGHNESS_LENGTH_422,
    "331": ROUGHNESS_LENGTH_331,
    "511": ROUGHNESS_LENGTH_511,
    "512": ROUGHNESS_LENGTH_511,
    "521": ROUGHNESS_LENGTH_511,
    "522": ROUGHNESS_LENGTH_511,
    "523": ROUGHNESS_LENGTH_511,
}
"""
[1] J. Silva, C. Ribeiro, R. Guedes, M.-C. Rua, and F. Ulrich,
“Roughness length classification of Corine Land Cover classes," Proceedings of EWEC 2007, Jan. 2007.
"""
