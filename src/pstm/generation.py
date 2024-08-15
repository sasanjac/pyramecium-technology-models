# :author: Sasan Jacob Rasti <sasan_jacob.rasti@tu-dresden.de>
# :copyright: Copyright (c) Institute of Electrical Power Systems and High Voltage Engineering - TU Dresden, 2022-2023.
# :license: BSD 3-Clause

from __future__ import annotations

import json
import pathlib
import random
import typing as t

import attrs
import numpy as np
import pandas as pd
import pytz
import scipy.linalg as sl
import scipy.optimize as so
import yaml
from loguru import logger
from tqdm.auto import tqdm

from pstm.heat_pump import AirHeatPump
from pstm.heat_pump import BrineHeatPump
from pstm.heat_pump import ResidentialHeatPump
from pstm.household_vdi4655 import Household
from pstm.mastr import MaStR
from pstm.pv import PV
from pstm.utils.dates import date_range
from pstm.utils.geo import GeoRef
from pstm.utils.weather import WeatherGenerator
from pstm.wind import Wind
from pstm.wind import WindFarm

if t.TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    PROFILES_TYPE = dict[str, dict[str, dict[str, dict[str, pd.Series | None]]]]

    class HPParams(t.TypedDict):
        efficiency: t.Literal["high", "normal", "low"]
        hp_type: type[ResidentialHeatPump]


DF_INDEX = date_range(tz=pytz.timezone("Europe/Berlin"))
N_STEPS = 24
E_START = 0
P_HH = 9999
C_HH = 35
C_PV = -7
ZERO = 0.0

aeq = [np.array([[1, -1, -1, 1]]) for _ in range(2 * N_STEPS)]
Aeq = sl.block_diag(*aeq)
hub = np.tril(np.ones(2 * N_STEPS))
Aub1 = np.zeros((2 * N_STEPS, 8 * N_STEPS))
Aub1[:, 2::4] = hub
Aub1[:, 3::4] = -hub
Aub2 = -Aub1
Aub = np.vstack([Aub1, Aub2])
bub3 = np.ones(2 * N_STEPS) * E_START
cost_function = np.tile(np.array([C_HH, C_PV, 1, 1]), 2 * N_STEPS)
lb = np.tile(np.array([0, 0, 0, 0]), 2 * N_STEPS)

FULL_LOAD_HOURS_MIN = 1_500
FULL_LOAD_HOURS_MAX = 4_000


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class HHParams:
    house_type: t.Literal["OFH", "MFH"]
    hp_params: HPParams
    build_type: t.Literal["EH", "LEH"]


HP_PARAMS: dict[str, dict[str, dict[t.Literal[2020, 2030, 2050], HPParams]]] = {
    "OFH": {
        "urban": {
            2020: {"efficiency": "low", "hp_type": AirHeatPump},
            2030: {"efficiency": "low", "hp_type": BrineHeatPump},
            2050: {"efficiency": "high", "hp_type": BrineHeatPump},
        },
        "rural": {
            2020: {"efficiency": "normal", "hp_type": BrineHeatPump},
            2030: {"efficiency": "high", "hp_type": BrineHeatPump},
            2050: {"efficiency": "high", "hp_type": BrineHeatPump},
        },
    },
    "MFH": {
        "urban": {
            2020: {"efficiency": "low", "hp_type": AirHeatPump},
            2030: {"efficiency": "normal", "hp_type": AirHeatPump},
            2050: {"efficiency": "high", "hp_type": BrineHeatPump},
        },
        "rural": {
            2020: {"efficiency": "normal", "hp_type": BrineHeatPump},
            2030: {"efficiency": "high", "hp_type": BrineHeatPump},
            2050: {"efficiency": "high", "hp_type": BrineHeatPump},
        },
    },
}

BUILD_TYPE: dict[t.Literal["OFH", "MFH"], dict[t.Literal[2020, 2030, 2050], t.Literal["EH", "LEH"]]] = {
    "OFH": {
        2020: "EH",
        2030: "LEH",
        2050: "LEH",
    },
    "MFH": {
        2020: "EH",
        2030: "EH",
        2050: "EH",
    },
}


class MissingArgumentsError(ValueError):
    def __init__(self) -> None:
        super().__init__("Either cop and dem or hh params must be set.")


@attrs.define(auto_attribs=True, kw_only=True, slots=False)
class Generator:
    grid_file_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    weather_gen_files_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    output_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    dem_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    profiles: dict[str, pd.DataFrame | pd.Series] = attrs.field(init=False, factory=dict)
    preoptimize_pv_bat: bool = attrs.field(default=False)
    params: dict[str, dict] = attrs.field(init=False, factory=dict)
    voronoi_file_path: pathlib.Path = attrs.field(converter=pathlib.Path)
    weather_path_template: str
    cop_fac: float = 1.0
    th_dem_fac: float = 1.0

    def __attrs_post_init__(self) -> None:
        self.output_path.mkdir(exist_ok=True, parents=True)

    def read_devices(self, grid_file_path: pathlib.Path) -> pd.DataFrame:
        return pd.read_excel(io=grid_file_path, sheet_name="Netzeinspeisung")

    def read_nodes(self, grid_file_path: pathlib.Path) -> pd.DataFrame:
        return pd.read_excel(io=grid_file_path, sheet_name="Knoten")

    def generate(self) -> None:
        logger.info("Loading data")
        dataframe = self.read_data(self.grid_file_path)
        with GeoRef(
            weather_gen_files_path=self.weather_gen_files_path,
            voronoi_file_path=self.voronoi_file_path,
        ) as georef:
            mastr = MaStR(georef=georef)
            self.generate_lv(dataframe, georef)
            profiles = self.aggregate_profiles(dataframe)
            self.generate_custom_vl(dataframe, georef, mastr, profiles, "20 kV")
            self.generate_custom_vl(dataframe, georef, mastr, profiles, "110 kV")
            self.generate_custom_vl(dataframe, georef, mastr, profiles, "220 kV")
            self.generate_custom_vl(dataframe, georef, mastr, profiles, "380 kV")
            self.dump()

    def read_data(self, grid_file_path: pathlib.Path) -> pd.DataFrame:
        logger.info("Loading grid data...")
        dataframe = pd.read_excel(io=grid_file_path, sheet_name="Sheet1")
        dataframe.REGION = dataframe.REGION.fillna("gen")
        dataframe.SUBREGION = dataframe.SUBREGION.fillna("gen")
        dataframe.NETZTYP = dataframe.NETZTYP.fillna("gen")
        dataframe.PVA = dataframe.PVA.fillna(0)
        dataframe.WON = dataframe.WON.fillna(0)
        dataframe.WOF = dataframe.WOF.fillna(0)
        dataframe.BAT = dataframe.BAT.fillna(0)
        dataframe.EV = dataframe.EV.fillna(0)
        dataframe.WP = dataframe.WP.fillna(0)
        logger.info("Loading grid data. Done.")
        return dataframe

    def generate_lv(self, dataframe: pd.DataFrame, georef: GeoRef) -> None:
        logger.info("Generating 0.4 kV profiles...")
        df_lv = dataframe[dataframe.Netzebene == "0,4 kV"]
        for _, row in tqdm(df_lv.iterrows(), total=df_lv.shape[0]):
            node = row.Knoten
            hh_params = self.determine_hh_params(row.NETZTYP)
            self.generate_for_row(row=row, node=node, georef=georef, hh_params=hh_params)

        logger.info("Generating 0.4 kV profiles. Done.")

    def dump(self) -> None:
        logger.info("Writing profiles to file...")
        for name, profile in tqdm(self.profiles.items()):
            profile.to_csv(self.output_path / f"{name}.csv", index_label="timestep")

        logger.info("Writing profiles to file. Done.")
        logger.info("Writing parameters to file...")
        for name, params in tqdm(self.params.items()):
            output_file_path = self.output_path / f"{name}.json"
            with output_file_path.open("w", encoding="utf-8") as f:
                json.dump(params, f)

        logger.info("Writing parameters to file. Done.")

    def generate_custom_vl(
        self,
        dataframe: pd.DataFrame,
        georef: GeoRef,
        mastr: MaStR,
        profiles: PROFILES_TYPE,
        voltage_level: str,
    ) -> None:
        logger.info("Generating {voltage_level} profiles...", voltage_level=voltage_level)
        df_vl = dataframe[dataframe.Netzebene == voltage_level]
        for _, row in tqdm(df_vl.iterrows(), total=df_vl.shape[0]):
            node = row.Knoten
            if node.endswith("_2_1") and dataframe.Knoten.str.contains(node.replace("_2_1", "_0_1")).any():
                continue

            dem = profiles[row.REGION][row.SUBREGION][row.NETZTYP]["dem"]
            cop = profiles[row.REGION][row.SUBREGION][row.NETZTYP]["cop"]
            self.generate_for_row(row=row, node=node, georef=georef, mastr=mastr, thermal_demand=dem, cop=cop)

        logger.info("Generating {voltage_level} profiles. Done.", voltage_level=voltage_level)

    def generate_for_row(
        self,
        row: pd.Series,
        node: str,
        georef: GeoRef,
        hh_params: HHParams | None = None,
        thermal_demand: pd.Series | None = None,
        cop: pd.Series | None = None,
        mastr: MaStR | None = None,
    ) -> None:
        weather = self.get_weather(georef, row)
        pv = self.generate_pv(row.PVA, georef, weather)
        cop, dem_h, params_h = self.generate_thermal_devs(
            -row.WP,
            georef,
            weather,
            hh_params=hh_params,
            thermal_demand=thermal_demand,
            cop=cop,
        )
        dem_e, pv = self.optimize_pv_bat(node, pv, row.BAT_S)
        params_bat = self.generate_battery(row.BAT_L)
        params = params_bat if params_h is None else params_h | params_bat

        if params:
            self.params[f"{node}_DEVICES"] = params

        if dem_e is not None:
            self.profiles[f"{node}_ELECTRICAL_DEMAND"] = dem_e

        if pv is not None:
            self.profiles[f"{node}_PV"] = pv

        if dem_h is not None:
            self.profiles[f"{node}_THERMAL_DEMAND"] = dem_h

        if cop is not None:
            self.profiles[f"{node}_COP"] = cop

        if mastr is not None:
            wind_on = self.generate_wind(lat=row.LAT, lon=row.LON, power_installed=row.WON, georef=georef, mastr=mastr)
            if wind_on is not None:
                self.profiles[f"{node}_WIND"] = wind_on

            wind_off = self.generate_wind_off(power_inst=row.WOF, weather=weather)
            if wind_off is not None:
                self.profiles[f"{node}_WIND_OFF"] = wind_off

    def generate_battery(self, q_bat: float) -> dict[str, dict[str, float]]:
        if q_bat == 0:
            return {}

        eta_bat_in = random.uniform(0.85, 0.95)  # noqa: S311
        eta_bat_out = eta_bat_in
        nu_bat = 0.04 / (30 * 24)  # 4 % / month
        return {"bat": {"e_el": q_bat, "eta_in": eta_bat_in, "eta_out": eta_bat_out, "nu": nu_bat}}

    def generate_pv(
        self,
        power_inst: float,
        georef: GeoRef,
        weather: WeatherGenerator,
    ) -> pd.DataFrame | None:
        if power_inst == 0:
            return None

        lat = weather.lat
        lon = weather.lon
        alt = georef.get_altitude(lat=lat, lon=lon)
        pv = PV(
            power_inst=power_inst * 1e6,
            efficiency_inv=1,
            lat=lat,
            lon=lon,
            alt=alt,
            dates=DF_INDEX,
        )
        pv_weather = weather.weather_at_pv_module
        pv.run(pv_weather)
        return self.create_df(power=-pv.acp.base * 1e-6)

    def generate_thermal_devs(
        self,
        p_wp: float,
        georef: GeoRef,
        weather: WeatherGenerator,
        hh_params: HHParams | None = None,
        thermal_demand: pd.Series | None = None,
        cop: pd.Series | None = None,
    ) -> tuple[pd.Series | None, pd.Series | None, dict[str, dict[str, float]] | None]:
        if p_wp == 0:
            return (None, None, None)

        p_hs = 3 * p_wp
        if thermal_demand is not None and cop is not None:
            dem = thermal_demand * p_wp
            dem.name = "p_th"
        elif hh_params is not None:
            hp_class = hh_params.hp_params["hp_type"]
            hp = hp_class(
                dates=DF_INDEX,
                power_inst=0,
                efficiency=hh_params.hp_params["efficiency"],
                target_temp=35,
                tz=weather.tz,
            )
            cop = hp.calc_cop(weather.temp_air_celsius)
            dem = self.create_thermal_demand(hh_params, georef, weather, cop, p_wp)
        else:
            raise MissingArgumentsError

        p_wp_th = cop * p_wp
        q_sp = float(20e-3 * p_wp_th.mean() * 4.2 * 20 * 998 / 3600)
        nu_q_sp = 1 / (20 * 24)  # 1 K / day
        params = {
            "wp": {"p_el": p_wp},
            "hs": {"p_el": p_hs},
            "wws": {"q_th": q_sp, "nu": nu_q_sp},
        }
        return (cop, dem, params)

    def create_thermal_demand(
        self,
        hh_params: HHParams,
        georef: GeoRef,
        weather: WeatherGenerator,
        cop: pd.Series,
        p_el: float,
    ) -> pd.Series:
        climate_zone = georef.get_dwd_try_zone(weather.lat, weather.lon)
        hh = Household(
            house_type=hh_params.house_type,
            building_type=hh_params.build_type,
            n_units=4,
            area=120,
            heat_demand=70,
            lat=weather.lat,
            lon=weather.lon,
            tz=weather.tz,
            climate_zone=climate_zone,
            dates=DF_INDEX,
        )
        hh.run(electrical=False, random_shift=True)
        hh_th = hh.thr.high.copy()
        q95 = hh_th.quantile(0.95)
        hh_th[hh_th > q95] = q95
        fac = p_el / np.divide(hh_th, cop).max()
        return hh.thr.high * fac * self.th_dem_fac

    def optimize_pv_bat(
        self,
        node: str,
        pv: pd.DataFrame | None,
        q_bat: float,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        dem_path = self.dem_path / (node + "_konventionell_kombiniert.csv")
        if not dem_path.is_file():
            dem_path = pathlib.Path(str(dem_path).replace("_0_1_", "_2_1_"))
            if not dem_path.is_file():
                return (None, pv)

        demc = pd.read_csv(dem_path)
        dem = demc.P_el.to_numpy(dtype=np.float64).flatten()
        if pv is None:
            return (self.create_df(power=dem), None)

        if q_bat == 0 or not self.preoptimize_pv_bat:
            return (self.create_df(power=dem), pv)

        gen = pv.p_el.to_numpy(dtype=np.float64).flatten()
        dem = np.hstack([dem, dem[:N_STEPS]])
        gen = np.hstack([gen, gen])
        beq = dem - gen
        e_start = 0.3 * q_bat
        p_buy: list[npt.NDArray[np.float64]] = []
        p_sell: list[npt.NDArray[np.float64]] = []
        for i in range(365):
            beq_i = beq[i * N_STEPS : (i + 2) * N_STEPS]
            bub1 = np.ones(2 * N_STEPS) * q_bat - e_start
            bub2 = np.zeros(2 * N_STEPS) + e_start
            bub = np.hstack([bub1, bub2])
            ub = np.tile(np.array([P_HH, 0, q_bat, q_bat]), 2 * N_STEPS)
            ub[1::4] = gen[i * N_STEPS : (i + 2) * N_STEPS]
            bounds = np.vstack([lb, ub]).T
            res = so.linprog(
                c=cost_function,
                A_eq=Aeq,
                b_eq=beq_i,
                A_ub=Aub,
                b_ub=bub,
                bounds=bounds,
                method="highs-ds",
            )
            if res.success:
                p_buy.append(res.x[0 : 4 * N_STEPS : 4])
                p_sell.append(res.x[1 : 4 * N_STEPS : 4])
                p_stor = res.x[2 : 4 * N_STEPS : 4]
                p_load = res.x[3 : 4 * N_STEPS : 4]
                e_start = e_start + sum(p_stor) - sum(p_load)
            else:
                raise RuntimeError

        p_buy_agg = np.hstack(p_buy)
        p_sell_agg = np.hstack(p_sell)
        return (self.create_df(power=p_buy_agg), self.create_df(power=p_sell_agg))

    def generate_wind_off(
        self,
        power_inst: float,
        weather: WeatherGenerator,
    ) -> pd.DataFrame | None:
        if power_inst == 0:
            return None

        wind_weather = weather.weather_at_wind_turbine
        wind = Wind(
            hub_height=140,
            turbine_type="V164/9500",
            dates=DF_INDEX,
        )
        unit = WindFarm(
            units=[wind.turbine],
            n_units=[None],
            powers_inst=[power_inst * 1e6],
            dates=DF_INDEX,
        )
        unit.run(wind_weather)
        return self.create_df(power=-unit.acp.low * 1e-6)

    def generate_wind(
        self,
        lat: float,
        lon: float,
        power_installed: float,
        georef: GeoRef,
        mastr: MaStR,
    ) -> pd.DataFrame | None:
        if power_installed == 0:
            return None

        wind_farms = mastr.installed_wind_farms(lat=lat, lon=lon, power_installed=power_installed)
        wind_farms_ = [
            WindFarm.from_powers_inst_and_hub_heights(
                dates=DF_INDEX,
                powers_inst=wind_farm.powers_installed,
                hub_heights=wind_farm.hub_heights,
            )
            for wind_farm in wind_farms
        ]
        weather_gen_ids = [
            georef.get_weather_gen_index(lat=wind_farm.lat, lon=wind_farm.lon) for wind_farm in wind_farms
        ]
        weather_gen_paths = [pathlib.Path(self.weather_path_template.format(idx)) for idx in weather_gen_ids]
        weather_gens = [
            WeatherGenerator.from_feather(weather_gen_files_path) for weather_gen_files_path in weather_gen_paths
        ]
        for wind_farm, weather_gen in zip(wind_farms_, weather_gens, strict=True):
            wind_farm.run(weather_gen.weather_at_wind_turbine)

        acp_low = np.sum([wind_farm.acp.low for wind_farm in wind_farms_])
        fac = power_installed / acp_low.min()
        power = acp_low * fac
        full_load_hours = power.sum() / power_installed
        if not (FULL_LOAD_HOURS_MIN < full_load_hours < FULL_LOAD_HOURS_MAX):
            logger.warning(
                "Strange full load hours: {full_load_hours}",
                full_load_hours=full_load_hours,
            )
        return self.create_df(power=power)  # W in MW

    def create_df(self, *, power: npt.NDArray[np.float64] | pd.Series, add_q: bool = True) -> pd.DataFrame:
        dataframe = pd.DataFrame()
        dataframe.index = DF_INDEX
        dataframe["p_el"] = power
        dataframe.fillna(0)
        if add_q:
            dataframe["q_min"] = dataframe["p_el"] * np.cos(0.9)
            dataframe["q_max"] = -dataframe["q_min"]

        return dataframe

    def aggregate_profiles(self, dataframe: pd.DataFrame) -> PROFILES_TYPE:
        logger.info("Aggregating profiles...")
        regions, subregions, grid_types = self.classify_dataframe(dataframe)
        profiles: PROFILES_TYPE = {
            reg: {subreg: {gt: {"dem": None, "cop": None} for gt in grid_types} for subreg in subregions}
            for reg in regions
        }
        profiles = self.aggregate_profiles_by_gt_reg_subreg_prio(dataframe, profiles)
        profiles = self.aggregate_profiles_by_gt_reg_prio(profiles)
        profiles = self.aggregate_profiles_by_gt_no_prio(profiles)
        profiles = self.aggregate_profiles_no_prio(profiles)
        logger.info("Aggregating profiles. Done.")
        return profiles

    @staticmethod
    def classify_dataframe(dataframe: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
        regions = set(dataframe.REGION)
        subregions = set(dataframe.SUBREGION)
        grid_types = set(dataframe.NETZTYP)
        return (regions, subregions, grid_types)

    def aggregate_profiles_by_gt_reg_subreg_prio(
        self,
        dataframe: pd.DataFrame,
        profiles: PROFILES_TYPE,
    ) -> PROFILES_TYPE:
        df_lv = dataframe[dataframe.Netzebene == "0,4 kV"]
        for reg, subregs in profiles.items():
            for subreg, gts in subregs.items():
                for gt in gts:
                    sub_df = df_lv[(reg == df_lv.REGION) & (subreg == df_lv.SUBREGION) & (gt == df_lv.NETZTYP)]
                    dems = []
                    cops = []
                    for _, row in sub_df.iterrows():
                        try:
                            dem = self.profiles[f"{row.Knoten}_THERMAL_DEMAND"]
                            dem = t.cast("pd.Series", dem)
                            cop = self.profiles[f"{row.Knoten}_COP"]
                            cop = t.cast("pd.Series", cop)
                            devices = self.params[f"{row.Knoten}_DEVICES"]
                        except KeyError:
                            continue

                        dems.append(dem / devices["wp"]["p_el"])
                        cops.append(cop)
                        dem_agg, cop_agg = self.aggregate(dems, cops)
                        profiles[reg][subreg][gt] = self.assign(dem_agg, cop_agg)

        return profiles

    def aggregate_profiles_by_gt_reg_prio(self, profiles: PROFILES_TYPE) -> PROFILES_TYPE:
        for reg, subregs in profiles.items():
            for subreg, gts in subregs.items():
                for gt in gts:
                    if profiles[reg][subreg][gt]["dem"] is None:
                        dems = [v[gt]["dem"] for v in profiles[reg].values()]
                        cops = [v[gt]["cop"] for v in profiles[reg].values()]
                        dem, cop = self.aggregate(dems, cops)
                        profiles[reg][subreg][gt] = self.assign(dem, cop)

        return profiles

    def aggregate_profiles_by_gt_no_prio(self, profiles: PROFILES_TYPE) -> PROFILES_TYPE:
        for reg, subregs in profiles.items():
            for subreg, gts in subregs.items():
                for gt in gts:
                    if profiles[reg][subreg][gt]["dem"] is None:
                        dems = [v2[gt]["dem"] for v1 in profiles.values() for v2 in v1.values()]
                        cops = [v2[gt]["cop"] for v1 in profiles.values() for v2 in v1.values()]
                        dem, cop = self.aggregate(dems, cops)
                        profiles[reg][subreg][gt] = self.assign(dem, cop)

        return profiles

    def aggregate_profiles_no_prio(self, profiles: PROFILES_TYPE) -> PROFILES_TYPE:
        for reg, subregs in profiles.items():
            for subreg, gts in subregs.items():
                for gt in gts:
                    if profiles[reg][subreg][gt]["dem"] is None:
                        dems = [v["dem"] for v in profiles[reg][subreg].values()]
                        cops = [v["cop"] for v in profiles[reg][subreg].values()]
                        dem, cop = self.aggregate(dems, cops)
                        profiles[reg][subreg][gt] = self.assign(dem, cop)

        return profiles

    def aggregate(
        self,
        dems: Sequence[pd.Series | None],
        cops: Sequence[pd.Series | None],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        dem = np.array([e.tolist() for e in dems if e is not None])
        cop = np.array([e.tolist() for e in cops if e is not None])
        return dem, cop

    def assign(self, dem: npt.NDArray[np.float64], cop: npt.NDArray[np.float64]) -> dict[str, pd.Series | None]:
        return {"dem": self.to_mean_list(dem, name="p_th"), "cop": self.to_mean_list(cop, name="eta")}

    @staticmethod
    def to_mean_list(arr: np.ndarray, name: str) -> pd.Series | None:
        if len(arr) > 0:
            return pd.Series(arr.mean(axis=0), name=name, index=DF_INDEX)

        return None

    def get_weather(self, georef: GeoRef, row: pd.Series) -> WeatherGenerator:
        lat = float(row.LAT)
        lon = float(row.LON)
        wgid = georef.get_weather_gen_index(lat=lat, lon=lon)
        wgpath = pathlib.Path(self.weather_path_template.format(wgid))
        return WeatherGenerator.from_feather(wgpath)

    @staticmethod
    def determine_hh_params(grid_type: str) -> HHParams:
        if grid_type[:2] in ("S1", "S2"):
            loc = "rural"
            house_type: t.Literal["OFH", "MFH"] = "OFH"
        elif grid_type[:2] in ("S5", "S6"):
            loc = "urban"
            house_type = "MFH"
        else:
            loc = random.choice(["rural", "urban"])  # noqa: S311
            house_type = random.choice(["OFH", "MFH"])  # noqa: S311

        year: t.Literal[2020, 2030, 2050] = random.choice([2020, 2030, 2050])  # noqa: S311
        return HHParams(
            house_type=house_type,
            hp_params=HP_PARAMS[house_type][loc][year],
            build_type=BUILD_TYPE[house_type][year],
        )


def main() -> None:
    config_file = pathlib.Path("config_generation.yml")
    with config_file.open(encoding="utf8") as f:
        config = yaml.safe_load(f)
        random.seed(config.get("seed"))
        gen = Generator(**config.get("generator", {}))
        gen.generate()


if __name__ == "__main__":
    main()
