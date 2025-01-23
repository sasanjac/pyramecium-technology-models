# Copyright (c) 2018-2025 Sasan Jacob Rasti

import datetime as dt
import json
import pathlib
import typing as t

import attrs
import cattrs
import click
import numpy as np
import pandas as pd
from cattrs.preconf.json import make_converter

from pstm.dickert.households import ConfigDict
from pstm.dickert.households import Households

CONVERTER = make_converter()
CONVERTER.register_unstructure_hook(dt.time, lambda v: v.isoformat())
CONVERTER.register_structure_hook(dt.time, lambda v, _: dt.time.fromisoformat(v))
CONVERTER.register_unstructure_hook_factory(
    attrs.has,
    lambda cl: cattrs.gen.make_dict_unstructure_fn(
        cl=cl,
        converter=CONVERTER,
        _cattrs_omit_if_default=False,
        _cattrs_use_linecache=True,
        _cattrs_use_alias=False,
        _cattrs_include_init_false=False,
        **{a.name: cattrs.override(omit=True) for a in attrs.fields(cl) if not a.init},
    ),
)

JSONPrimitive = bool | float | int | str | None
JSONType = JSONPrimitive | list[JSONPrimitive] | dict[str, JSONPrimitive]

SRC_PATH = pathlib.Path(__file__).parent.parent.parent


class NpEncoder(json.JSONEncoder):
    def default(self, obj: JSONType | np.integer | np.floating | np.ndarray) -> JSONType:
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)  # type:ignore[no-any-return]


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the input file.",
)
def create(*, input_file: pathlib.Path) -> None:  # noqa: PLR0914
    for j in range(6):
        dataframe = pd.read_excel(input_file, sheet_name=f"HH-Typ{j + 1}")
        starts = np.argwhere(dataframe.iloc[:, [0]] == 1)[:, 0]
        ends = np.argwhere(dataframe.iloc[:, [0]] == -1)[:, 0]
        baseline_profiles = dataframe.iloc[starts[0] : ends[0] + 1].fillna(0)
        baseline_profiles_dc = [
            {
                "description": row.iloc[1].to_numpy()[0],
                "phase": int(row.iloc[2].to_numpy()[0]),
                "switch_on_current": row.iloc[3].to_numpy()[0],
                "switch_on_time": row.iloc[4].to_numpy()[0],
                "equipment_level": row.iloc[5].to_numpy()[0],
                "active_power_distribution_type": row.iloc[6].to_numpy()[0],
                "active_power_parameter_1": row.iloc[7].to_numpy()[0],
                "active_power_parameter_2": row.iloc[8].to_numpy()[0],
                "active_power_parameter_3": row.iloc[9].to_numpy()[0],
                "reactive_power_share": row.iloc[10].to_numpy()[0],
                "reactive_power_distribution_type": row.iloc[11].to_numpy()[0],
                "reactive_power_parameter_1": row.iloc[12].to_numpy()[0],
                "reactive_power_parameter_2": row.iloc[13].to_numpy()[0],
                "reactive_power_parameter_3": row.iloc[14].to_numpy()[0],
                "power_variation": row.iloc[15].to_numpy()[0],
                "power_variation_max": row.iloc[16].to_numpy()[0],
            }
            for _, row in baseline_profiles.iterrows()
        ]
        cycle_profiles = dataframe.iloc[starts[1] : ends[1] + 1].fillna(0)
        cycle_profiles_dc = [
            {
                "description": row.iloc[1].to_numpy()[0],
                "phase": int(row.iloc[2].to_numpy()[0]),
                "switch_on_current": row.iloc[3].to_numpy()[0],
                "switch_on_time": row.iloc[4].to_numpy()[0],
                "equipment_level": row.iloc[5].to_numpy()[0],
                "active_power_distribution_type": row.iloc[6].to_numpy()[0],
                "active_power_parameter_1": row.iloc[7].to_numpy()[0],
                "active_power_parameter_2": row.iloc[8].to_numpy()[0],
                "active_power_parameter_3": row.iloc[9].to_numpy()[0],
                "active_power_parameter_4": row.iloc[10].to_numpy()[0],
                "reactive_power_share": row.iloc[11].to_numpy()[0],
                "reactive_power_distribution_type": row.iloc[12].to_numpy()[0],
                "reactive_power_parameter_1": row.iloc[13].to_numpy()[0],
                "reactive_power_parameter_2": row.iloc[14].to_numpy()[0],
                "reactive_power_parameter_3": row.iloc[15].to_numpy()[0],
                "period_distribution_type": row.iloc[16].to_numpy()[0],
                "period_parameter_1": row.iloc[17].to_numpy()[0],
                "period_parameter_2": row.iloc[18].to_numpy()[0],
                "period_parameter_3": row.iloc[19].to_numpy()[0],
                "operation_distribution_type": row.iloc[20].to_numpy()[0],
                "operation_parameter_1": row.iloc[21].to_numpy()[0],
                "operation_parameter_2": row.iloc[22].to_numpy()[0],
                "operation_parameter_3": row.iloc[23].to_numpy()[0],
                "period_variation": row.iloc[24].to_numpy()[0],
                "operation_variation": row.iloc[25].to_numpy()[0],
            }
            for _, row in cycle_profiles.iterrows()
        ]
        on_off_profiles = dataframe.iloc[starts[2] : ends[2] + 1]
        n = len(on_off_profiles) // 4
        on_off_profiles_dc = [
            {
                "description": on_off_profiles.iloc[i * 4, [1]].fillna(0).to_numpy()[0],
                "phase": int(on_off_profiles.iloc[i * 4, [2]].fillna(0).to_numpy()[0]),
                "switch_on_current": on_off_profiles.iloc[i * 4, [3]].fillna(0).to_numpy()[0],
                "switch_on_time": on_off_profiles.iloc[i * 4, [4]].fillna(0).to_numpy()[0],
                "equipment_level": on_off_profiles.iloc[i * 4, [5]].fillna(0).to_numpy()[0],
                "active_power_distribution_type": on_off_profiles.iloc[i * 4, [6]].fillna("normal").to_numpy()[0],
                "active_power_parameter_1": on_off_profiles.iloc[i * 4, [7]].fillna(0).to_numpy()[0],
                "active_power_parameter_2": on_off_profiles.iloc[i * 4, [8]].fillna(0).to_numpy()[0],
                "active_power_parameter_3": on_off_profiles.iloc[i * 4, [9]].fillna(0).to_numpy()[0],
                "active_power_parameter_4": on_off_profiles.iloc[i * 4, [10]].fillna(0).to_numpy()[0],
                "reactive_power_share": on_off_profiles.iloc[i * 4, [11]].fillna(0).to_numpy()[0],
                "reactive_power_distribution_type": on_off_profiles.iloc[i * 4, [12]].fillna("normal").to_numpy()[0],
                "reactive_power_parameter_1": on_off_profiles.iloc[i * 4, [13]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_2": on_off_profiles.iloc[i * 4, [14]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_3": on_off_profiles.iloc[i * 4, [15]].fillna(0).to_numpy()[0],
                "usage_distribution_type": on_off_profiles.iloc[i * 4, [16]].fillna("normal").to_numpy()[0],
                "usage_parameter_1": on_off_profiles.iloc[i * 4, [17]].fillna(0).to_numpy()[0],
                "usage_parameter_2": on_off_profiles.iloc[i * 4, [18]].fillna(0).to_numpy()[0],
                "usage_parameter_3": on_off_profiles.iloc[i * 4, [19]].fillna(0).to_numpy()[0],
                "operation_distribution_type": on_off_profiles.iloc[i * 4, [20]].fillna("normal").to_numpy()[0],
                "operation_parameter_1": on_off_profiles.iloc[i * 4, [21]].fillna(0).to_numpy()[0],
                "operation_parameter_2": on_off_profiles.iloc[i * 4, [22]].fillna(0).to_numpy()[0],
                "operation_parameter_3": on_off_profiles.iloc[i * 4, [23]].fillna(0).to_numpy()[0],
                "time_on_distribution_types": (
                    on_off_profiles.iloc[i * 4, [25]].fillna("normal").to_numpy()[0],
                    on_off_profiles.iloc[i * 4, [26]].fillna("normal").to_numpy()[0],
                    on_off_profiles.iloc[i * 4, [27]].fillna("normal").to_numpy()[0],
                    on_off_profiles.iloc[i * 4, [28]].fillna("normal").to_numpy()[0],
                    on_off_profiles.iloc[i * 4, [29]].fillna("normal").to_numpy()[0],
                    on_off_profiles.iloc[i * 4, [30]].fillna("normal").to_numpy()[0],
                ),
                "usage_variation": on_off_profiles.iloc[i * 4, [31]].fillna(0).to_numpy()[0],
                "operation_variation": on_off_profiles.iloc[i * 4, [32]].fillna(0).to_numpy()[0],
                "probability_1": on_off_profiles.iloc[i * 4 + 1, [7]].fillna(0).to_numpy()[0],
                "probability_2": on_off_profiles.iloc[i * 4 + 1, [8]].fillna(0).to_numpy()[0],
                "probability_3": on_off_profiles.iloc[i * 4 + 1, [9]].fillna(0).to_numpy()[0],
                "probability_4": on_off_profiles.iloc[i * 4 + 1, [10]].fillna(0).to_numpy()[0],
                "time_on_parameters_1": (
                    on_off_profiles.iloc[i * 4 + 1, [25]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 1, [26]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 1, [27]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 1, [28]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 1, [29]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 1, [30]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_parameters_2": (
                    on_off_profiles.iloc[i * 4 + 2, [25]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 2, [26]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 2, [27]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 2, [28]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 2, [29]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    on_off_profiles.iloc[i * 4 + 2, [30]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_parameters_3": (
                    on_off_profiles.iloc[i * 4 + 3, [25]].fillna(0).to_numpy()[0],
                    on_off_profiles.iloc[i * 4 + 3, [26]].fillna(0).to_numpy()[0],
                    on_off_profiles.iloc[i * 4 + 3, [27]].fillna(0).to_numpy()[0],
                    on_off_profiles.iloc[i * 4 + 3, [28]].fillna(0).to_numpy()[0],
                    on_off_profiles.iloc[i * 4 + 3, [29]].fillna(0).to_numpy()[0],
                    on_off_profiles.iloc[i * 4 + 3, [30]].fillna(0).to_numpy()[0],
                ),
            }
            for i in range(n)
        ]
        process_profiles = dataframe.iloc[starts[3] : ends[3] + 1]
        n = len(process_profiles) // 4
        process_profiles_dc = [
            {
                "description": process_profiles.iloc[i * 4, [1]].fillna(0).to_numpy()[0],
                "phase": int(process_profiles.iloc[i * 4, [2]].fillna(0).to_numpy()[0]),
                "switch_on_current": process_profiles.iloc[i * 4, [3]].fillna(0).to_numpy()[0],
                "switch_on_time": process_profiles.iloc[i * 4, [4]].fillna(0).to_numpy()[0],
                "equipment_level": process_profiles.iloc[i * 4, [5]].fillna(0).to_numpy()[0],
                "active_power_distribution_type": process_profiles.iloc[i * 4, [6]].fillna("normal").to_numpy()[0],
                "active_power_parameter_1": process_profiles.iloc[i * 4, [7]].fillna(0).to_numpy()[0],
                "active_power_parameter_2": process_profiles.iloc[i * 4, [8]].fillna(0).to_numpy()[0],
                "active_power_parameter_3": process_profiles.iloc[i * 4, [9]].fillna(0).to_numpy()[0],
                "active_power_parameter_4": process_profiles.iloc[i * 4, [10]].fillna(0).to_numpy()[0],
                "reactive_power_share": process_profiles.iloc[i * 4, [11]].fillna(0).to_numpy()[0],
                "reactive_power_distribution_type": process_profiles.iloc[i * 4, [12]].fillna("normal").to_numpy()[0],
                "reactive_power_parameter_1": process_profiles.iloc[i * 4, [13]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_2": process_profiles.iloc[i * 4, [14]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_3": process_profiles.iloc[i * 4, [15]].fillna(0).to_numpy()[0],
                "usage_distribution_type": process_profiles.iloc[i * 4, [16]].fillna("normal").to_numpy()[0],
                "usage_parameter_1": process_profiles.iloc[i * 4, [17]].fillna(0).to_numpy()[0],
                "usage_parameter_2": process_profiles.iloc[i * 4, [18]].fillna(0).to_numpy()[0],
                "usage_parameter_3": process_profiles.iloc[i * 4, [19]].fillna(0).to_numpy()[0],
                "operation_distribution_type": process_profiles.iloc[i * 4, [20]].fillna("normal").to_numpy()[0],
                "operation_parameter_1": process_profiles.iloc[i * 4, [21]].fillna(0).to_numpy()[0],
                "operation_parameter_2": process_profiles.iloc[i * 4, [22]].fillna(0).to_numpy()[0],
                "operation_parameter_3": process_profiles.iloc[i * 4, [23]].fillna(0).to_numpy()[0],
                "time_on_distribution_types": (
                    process_profiles.iloc[i * 4, [25]].fillna("normal").to_numpy()[0],
                    process_profiles.iloc[i * 4, [26]].fillna("normal").to_numpy()[0],
                    process_profiles.iloc[i * 4, [27]].fillna("normal").to_numpy()[0],
                    process_profiles.iloc[i * 4, [28]].fillna("normal").to_numpy()[0],
                    process_profiles.iloc[i * 4, [29]].fillna("normal").to_numpy()[0],
                    process_profiles.iloc[i * 4, [30]].fillna("normal").to_numpy()[0],
                ),
                "usage_variation": process_profiles.iloc[i * 4, [31]].fillna(0).to_numpy()[0],
                "operation_variation": process_profiles.iloc[i * 4, [32]].fillna(0).to_numpy()[0],
                "probability_1": process_profiles.iloc[i * 4 + 1, [7]].fillna(0).to_numpy()[0],
                "probability_2": process_profiles.iloc[i * 4 + 1, [8]].fillna(0).to_numpy()[0],
                "probability_3": process_profiles.iloc[i * 4 + 1, [9]].fillna(0).to_numpy()[0],
                "probability_4": process_profiles.iloc[i * 4 + 1, [10]].fillna(0).to_numpy()[0],
                "time_on_parameters_1": (
                    process_profiles.iloc[i * 4 + 1, [25]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 1, [26]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 1, [27]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 1, [28]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 1, [29]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 1, [30]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_parameters_2": (
                    process_profiles.iloc[i * 4 + 2, [25]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 2, [26]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 2, [27]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 2, [28]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 2, [29]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    process_profiles.iloc[i * 4 + 2, [30]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_parameters_3": (
                    process_profiles.iloc[i * 4 + 3, [25]].fillna(0).to_numpy()[0],
                    process_profiles.iloc[i * 4 + 3, [26]].fillna(0).to_numpy()[0],
                    process_profiles.iloc[i * 4 + 3, [27]].fillna(0).to_numpy()[0],
                    process_profiles.iloc[i * 4 + 3, [28]].fillna(0).to_numpy()[0],
                    process_profiles.iloc[i * 4 + 3, [29]].fillna(0).to_numpy()[0],
                    process_profiles.iloc[i * 4 + 3, [30]].fillna(0).to_numpy()[0],
                ),
                "active_power_2_distribution_type": process_profiles.iloc[i * 4 + 2, [6]]
                .fillna("normal")
                .to_numpy()[0],
                "active_power_2_parameter_1": process_profiles.iloc[i * 4 + 2, [7]].fillna(0).to_numpy()[0],
                "active_power_2_parameter_2": process_profiles.iloc[i * 4 + 2, [8]].fillna(0).to_numpy()[0],
                "reactive_power_2_distribution_type": process_profiles.iloc[i * 4 + 2, [12]]
                .fillna("normal")
                .to_numpy()[0],
                "reactive_power_2_parameter_1": process_profiles.iloc[i * 4 + 2, [13]].fillna(0).to_numpy()[0],
                "reactive_power_2_parameter_2": process_profiles.iloc[i * 4 + 2, [14]].fillna(0).to_numpy()[0],
                "operation_2_distribution_type": process_profiles.iloc[i * 4 + 2, [20]].fillna("normal").to_numpy()[0],
                "operation_2_parameter_1": process_profiles.iloc[i * 4 + 2, [21]].fillna(0).to_numpy()[0],
                "operation_2_parameter_2": process_profiles.iloc[i * 4 + 2, [22]].fillna(0).to_numpy()[0],
            }
            for i in range(n)
        ]
        lighting_profiles = dataframe.iloc[starts[4] : ends[4] + 1]
        n = len(process_profiles) // 9
        lighting_profiles_dc = [
            {
                "description": lighting_profiles.iloc[i * 9, [1]].fillna(0).to_numpy()[0],
                "phase": lighting_profiles.iloc[i * 9, [2]].fillna(0).to_numpy()[0],
                "switch_on_current": lighting_profiles.iloc[i * 9, [3]].fillna(0).to_numpy()[0],
                "switch_on_time": lighting_profiles.iloc[i * 9, [4]].fillna(0).to_numpy()[0],
                "equipment_level": lighting_profiles.iloc[i * 9, [5]].fillna(0).to_numpy()[0],
                "active_power_distribution_type": lighting_profiles.iloc[i * 9, [6]].fillna("normal").to_numpy()[0],
                "active_power_parameter_1": lighting_profiles.iloc[i * 9, [7]].fillna(0).to_numpy()[0],
                "active_power_parameter_2": lighting_profiles.iloc[i * 9, [8]].fillna(0).to_numpy()[0],
                "active_power_parameter_3": lighting_profiles.iloc[i * 9, [9]].fillna(0).to_numpy()[0],
                "reactive_power_share": lighting_profiles.iloc[i * 9 + 1, [5]].fillna(0).to_numpy()[0],
                "reactive_power_distribution_type": lighting_profiles.iloc[i * 9 + 1, [6]]
                .fillna("normal")
                .to_numpy()[0],
                "reactive_power_parameter_1": lighting_profiles.iloc[i * 9 + 1, [7]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_2": lighting_profiles.iloc[i * 9 + 1, [8]].fillna(0).to_numpy()[0],
                "reactive_power_parameter_3": lighting_profiles.iloc[i * 9 + 1, [9]].fillna(0).to_numpy()[0],
                "lighting_distribution_types": (
                    lighting_profiles.iloc[i * 9, [12]].fillna(0).to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 1, [12]].fillna(0).to_numpy()[0],
                ),
                "lighting_parameters_1": (
                    lighting_profiles.iloc[i * 9, [13]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 1, [13]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "lighting_parameters_2": (
                    lighting_profiles.iloc[i * 9, [14]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 1, [14]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_distribution_types": (
                    lighting_profiles.iloc[i * 9, [17]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9, [18]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9, [19]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9, [20]].fillna("normal").to_numpy()[0],
                ),
                "time_on_parameters_1": (
                    lighting_profiles.iloc[i * 9 + 1, [17]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 1, [18]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 1, [19]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 1, [20]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_parameters_2": (
                    lighting_profiles.iloc[i * 9 + 2, [17]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 2, [18]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 2, [19]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 2, [20]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_off_distribution_types": (
                    lighting_profiles.iloc[i * 9 + 4, [17]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 4, [18]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 4, [19]].fillna("normal").to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 4, [20]].fillna("normal").to_numpy()[0],
                ),
                "time_off_parameters_1": (
                    lighting_profiles.iloc[i * 9 + 5, [17]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 5, [18]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 5, [19]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 5, [20]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_off_parameters_2": (
                    lighting_profiles.iloc[i * 9 + 6, [17]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 6, [18]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 6, [19]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                    lighting_profiles.iloc[i * 9 + 6, [20]].fillna(dt.time(0)).to_numpy()[0],  # type:ignore[call-overload]
                ),
                "time_on_variations": (
                    lighting_profiles.iloc[i * 9 + 7, [17]].fillna(0).to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 7, [18]].fillna(0).to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 7, [19]].fillna(0).to_numpy()[0],
                    lighting_profiles.iloc[i * 9 + 7, [20]].fillna(0).to_numpy()[0],
                ),
            }
            for i in range(n)
        ]
        hh = {
            "baseline_profiles": baseline_profiles_dc,
            "cycle_profiles": cycle_profiles_dc,
            "on_off_profiles": on_off_profiles_dc,
            "process_profiles": process_profiles_dc,
            "lighting_profiles": lighting_profiles_dc,
            "phase_distribution": (0.4, 0.3, 0.3),
        }
        hh_c = t.cast("ConfigDict", hh)
        hhs = Households.from_config(hh_c)
        hh_c = CONVERTER.unstructure(hhs)
        path = SRC_PATH / f"data/household/dickert/hh{j + 1}.json"
        with path.open(mode="w+", encoding="utf-8") as file_handle:
            json.dump(hh_c, file_handle, cls=NpEncoder)


if __name__ == "__main__":
    create()
