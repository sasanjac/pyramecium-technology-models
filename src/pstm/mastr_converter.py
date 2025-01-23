# Copyright (c) 2018-2025 Sasan Jacob Rasti

from __future__ import annotations

import pathlib
import typing as t

import attrs
import numpy as np
import pandas as pd

if t.TYPE_CHECKING:

    class InputData(t.TypedDict):
        path: pathlib.Path
        export_path: pathlib.Path
        columns_extra: t.NotRequired[list[str]]
        conds_extra: t.NotRequired[list[tuple[str, ...]]]
        n_files: t.NotRequired[int]


SRC_PATH = pathlib.Path(__file__).parent.parent.parent
CATALOG_PATH = pathlib.Path(SRC_PATH / "data/mastr/Katalogwerte.xml")
DEFAULT_COLUMNS = [
    "Postleitzahl",
    "Nettonennleistung",
]


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class MaStRCatalog:
    path: pathlib.Path

    def __attrs_post_init__(self) -> None:
        self._catalog = import_xml(self.path)

    def get_code(self, code_str: str) -> int:
        condition = self._catalog["Wert"] == code_str
        values = self._catalog[condition]["Id"].to_numpy(dtype=np.int64).flatten()
        return int(values[0]) if values.size > 0 else -1

    def get_value(self, code: float) -> float | None:
        try:
            condition = self._catalog["Id"] == int(code)
        except ValueError:
            return None
        else:
            values = self._catalog[condition]["Wert"].to_numpy(dtype=np.int64).flatten()[0]
            return int(values[0]) if values.size > 0 else -1


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class MaStRConverter:
    path: pathlib.Path
    export_path: pathlib.Path
    columns_extra: list[str] = attrs.field(factory=list)
    conds_extra: list[tuple[str, ...]] = attrs.field(factory=list)
    n_files: int | None = None

    def __post_init__(self) -> None:
        if self.n_files is None:
            self.dataframe = import_xml(self.path)
        else:
            self.dataframe = import_xmls(self.path, self.n_files)

        self.catalog = MaStRCatalog(path=CATALOG_PATH)
        columns = DEFAULT_COLUMNS + self.columns_extra
        conds = self._create_conds()
        self.dataframe = self._finalize(self.dataframe, columns, conds)

    def export(self) -> None:
        export_dataframe(self.dataframe, self.export_path)

    def _create_conds(self) -> pd.Series:
        code_in_operation = self.catalog.get_code("In Betrieb")
        code_in_planning = self.catalog.get_code("In Planung")
        conds = (self.dataframe["EinheitBetriebsstatus"] == code_in_operation) | (
            self.dataframe["EinheitBetriebsstatus"] == code_in_planning
        )
        for cond in self.conds_extra:
            k, v = cond
            conds &= self.dataframe[k] == self.catalog.get_code(v)

        return conds

    def _finalize(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
        conds: pd.Series,
    ) -> pd.DataFrame:
        dataframe = dataframe[conds]
        dataframe = dataframe[columns]
        dataframe["Postleitzahl"] = dataframe["Postleitzahl"].fillna(0)
        dataframe["Postleitzahl"] = dataframe["Postleitzahl"].astype(int)
        if "Hersteller" in columns:
            dataframe["Hersteller"] = dataframe["Hersteller"].apply(self.catalog.get_value)

        return dataframe


def import_xmls(path: pathlib.Path, n_files: int) -> pd.DataFrame:
    dfs = [import_xml(path.with_name(f"{path.name}_{i + 1}.xml")) for i in range(n_files)]
    return pd.concat(dfs)


def import_xml(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_xml(path, encoding="UTF-16")


def export_dataframe(dataframe: pd.DataFrame, export_path: pathlib.Path) -> None:
    dataframe.to_parquet(
        export_path,
        compression="gzip",
    )


def main() -> None:
    data: list[InputData] = [
        {
            "path": SRC_PATH / "data/mastr/EinheitenSolar",
            "export_path": SRC_PATH / "data/mastr/20210727-EinheitenSolar_stripped.parquet.gzip",
            "n_files": 21,
        },
        {
            "path": SRC_PATH / "data/mastr/EinheitenWind.xml",
            "export_path": SRC_PATH / "data/mastr/20210727-EinheitenWindOnshore_stripped.parquet.gzip",
            "columns_extra": ["Nabenhoehe", "Hersteller", "Typenbezeichnung"],
            "conds_extra": [("Lage", "Windkraft an Land")],
        },
        {
            "path": SRC_PATH / "data/mastr/EinheitenWind.xml",
            "export_path": SRC_PATH / "data/mastr/20210727-EinheitenWindOffshore_stripped.parquet.gzip",
            "columns_extra": ["Nabenhoehe", "Hersteller", "Typenbezeichnung"],
            "conds_extra": [("Lage", "Windkraft auf See")],
        },
    ]
    for datum in data:
        mastr = MaStRConverter(**datum)
        mastr.export()


if __name__ == "__main__":
    main()
