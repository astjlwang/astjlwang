#!/usr/bin/env python3
"""
从 HyperLEDA + GLADE+ 自动筛选近邻盘星系样本。

筛选条件（按提问者定义）：
1) 盘星系，且主轴角直径 D25 > 2 arcmin
2) 倾角 < 20 deg（这里定义为：edge-on=0, face-on=90）
3) 远离银盘：|b| >= 20 deg
4) 有距离和质量测量：dL < 50 Mpc，M* > 1e10 Msun

数据来源：
- HyperLEDA (VizieR: VII/237/pgc)：形态、logD25、logR25、PA
- GLADE+   (VizieR: VII/291/gladep)：dL、M*、天球坐标

说明：
- GLADE+ 中 M* 的单位是 1e10 Msun，因此阈值 1e10 Msun 对应 M* > 1。
- 优先用 PGC 直接匹配；对于 GLADE+ 中缺失 PGC 的对象，用坐标最近邻兜底匹配。
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u


def query_hyperleda(min_d25_arcmin: float) -> pd.DataFrame:
    """查询 HyperLEDA 的盘星系候选并返回 DataFrame。"""
    # HyperLEDA 的 logD25 定义为 log10(0.1 arcmin)。
    # D25(arcmin) > min_d25_arcmin => logD25 > log10(min_d25_arcmin * 10)
    logd25_cut = math.log10(min_d25_arcmin * 10.0)

    viz = Vizier(
        columns=[
            "PGC",
            "RAJ2000",
            "DEJ2000",
            "MType",
            "logD25",
            "logR25",
            "PA",
        ],
        column_filters={
            "OType": "=G",
            "MType": "*S*",   # 以 S 系列形态为盘星系近似（S0/SA/SB/SAB/Sc/...）
            "logD25": f">{logd25_cut:.4f}",
        },
    )
    viz.ROW_LIMIT = -1
    tables = viz.query_constraints(catalog="VII/237/pgc")
    if len(tables) == 0:
        raise RuntimeError("未能从 HyperLEDA 拉取到数据。")

    df = tables[0].to_pandas()
    df["PGC"] = pd.to_numeric(df["PGC"], errors="coerce").astype("Int64")
    df["logD25"] = pd.to_numeric(df["logD25"], errors="coerce")
    df["logR25"] = pd.to_numeric(df["logR25"], errors="coerce")
    return df


def query_glade(max_distance_mpc: float, min_mstar_1e10: float) -> pd.DataFrame:
    """查询 GLADE+ 的距离/质量候选并返回 DataFrame。"""
    viz = Vizier(
        columns=[
            "GLADE+",
            "PGC",
            "HyperLEDA",
            "RAJ2000",
            "DEJ2000",
            "dL",
            "M*",
            "Type",
        ],
        column_filters={
            "Type": "=G",
            "dL": f"<{max_distance_mpc}",
            # GLADE+ 的 M* 单位是 1e10 Msun
            "M*": f">{min_mstar_1e10}",
        },
    )
    viz.ROW_LIMIT = -1
    tables = viz.query_constraints(catalog="VII/291/gladep")
    if len(tables) == 0:
        raise RuntimeError("未能从 GLADE+ 拉取到数据。")

    df = tables[0].to_pandas()
    df["PGC"] = pd.to_numeric(df["PGC"], errors="coerce").astype("Int64")
    df["dL"] = pd.to_numeric(df["dL"], errors="coerce")
    df["M*"] = pd.to_numeric(df["M*"], errors="coerce")
    return df


def compute_user_inclination_and_galactic_b(
    hleda_df: pd.DataFrame,
    q0: float,
) -> pd.DataFrame:
    """
    计算用户定义倾角与银纬。

    - 标准天文倾角 i_std：face-on=0, edge-on=90
    - 用户倾角 i_user：edge-on=0, face-on=90
      即 i_user = 90 - i_std
    """
    out = hleda_df.copy()
    out = out[out["logR25"].notna()].copy()

    # logR25 = log10(a/b) => q=b/a=1/(10^logR25)
    q_obs = 1.0 / np.power(10.0, out["logR25"].to_numpy(dtype=float))
    cos2_i_std = (q_obs**2 - q0**2) / (1.0 - q0**2)
    cos2_i_std = np.clip(cos2_i_std, 0.0, 1.0)
    i_std_deg = np.degrees(np.arccos(np.sqrt(cos2_i_std)))
    out["incl_user_deg"] = 90.0 - i_std_deg
    out["axis_ratio_q_b_over_a"] = q_obs
    out["axis_ratio_a_over_b"] = 1.0 / q_obs

    c_icrs = SkyCoord(
        out["RAJ2000"].astype(str).to_numpy(),
        out["DEJ2000"].astype(str).to_numpy(),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )
    out["b_deg"] = c_icrs.galactic.b.deg

    # 便于阅读：把 logD25 转成 arcmin
    out["D25_arcmin"] = np.power(10.0, out["logD25"].to_numpy(dtype=float) - 1.0)
    return out


def match_by_pgc(glade_df: pd.DataFrame, hleda_df: pd.DataFrame) -> pd.DataFrame:
    """按 PGC 直接匹配。"""
    merged = glade_df.merge(hleda_df, on="PGC", how="inner", suffixes=("_glade", "_hleda"))
    merged["match_method"] = "PGC"
    merged["match_sep_arcsec"] = np.nan
    return merged


def match_by_coordinates(
    glade_df_unmatched: pd.DataFrame,
    hleda_df: pd.DataFrame,
    max_sep_arcsec: float,
) -> pd.DataFrame:
    """
    用天球坐标最近邻兜底匹配（不依赖 scipy）。
    仅用于 GLADE+ 中 PGC 缺失或未匹配的对象。
    """
    if len(glade_df_unmatched) == 0 or len(hleda_df) == 0:
        return pd.DataFrame()

    g = glade_df_unmatched.reset_index(drop=True).copy()
    h = hleda_df.reset_index(drop=True).copy()

    cg = SkyCoord(g["RAJ2000"].to_numpy(dtype=float) * u.deg, g["DEJ2000"].to_numpy(dtype=float) * u.deg)
    ch = SkyCoord(
        h["RAJ2000"].astype(str).to_numpy(),
        h["DEJ2000"].astype(str).to_numpy(),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )

    # 当前样本规模较小，直接广播计算角距矩阵即可。
    sep = cg[:, None].separation(ch[None, :])  # (Ng, Nh)
    sep_arcsec = sep.to_value(u.arcsec)
    idx_min = np.argmin(sep_arcsec, axis=1)
    sep_min = sep_arcsec[np.arange(len(g)), idx_min]

    ok = sep_min < max_sep_arcsec
    if not np.any(ok):
        return pd.DataFrame()

    g_ok = g.loc[ok].reset_index(drop=True).rename(
        columns={"RAJ2000": "RAJ2000_glade", "DEJ2000": "DEJ2000_glade"}
    )
    h_ok = h.iloc[idx_min[ok]].reset_index(drop=True)
    merged = pd.concat([g_ok, h_ok.add_prefix("hleda_")], axis=1)

    # 统一字段命名，便于与 PGC 匹配结果拼接
    rename_map = {
        "hleda_PGC": "PGC_hleda",
        "hleda_RAJ2000": "RAJ2000_hleda",
        "hleda_DEJ2000": "DEJ2000_hleda",
        "hleda_MType": "MType",
        "hleda_logD25": "logD25",
        "hleda_logR25": "logR25",
        "hleda_PA": "PA",
        "hleda_incl_user_deg": "incl_user_deg",
        "hleda_axis_ratio_q_b_over_a": "axis_ratio_q_b_over_a",
        "hleda_axis_ratio_a_over_b": "axis_ratio_a_over_b",
        "hleda_b_deg": "b_deg",
        "hleda_D25_arcmin": "D25_arcmin",
    }
    merged = merged.rename(columns=rename_map)
    merged["PGC"] = merged["PGC"].fillna(merged["PGC_hleda"])
    merged["match_method"] = "COORD"
    merged["match_sep_arcsec"] = sep_min[ok]
    return merged


def apply_final_filters(
    merged_df: pd.DataFrame,
    max_user_incl_deg: float,
    min_abs_b_deg: float,
) -> pd.DataFrame:
    """在合并后执行最终倾角与银纬筛选。"""
    out = merged_df.copy()
    out = out[out["incl_user_deg"] < max_user_incl_deg]
    out = out[out["b_deg"].abs() >= min_abs_b_deg]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选满足条件的近邻盘星系样本")
    parser.add_argument("--min-d25-arcmin", type=float, default=2.0, help="主轴角直径下限 (arcmin)")
    parser.add_argument("--max-distance-mpc", type=float, default=50.0, help="距离上限 (Mpc)")
    parser.add_argument("--min-mstar-msun", type=float, default=1e10, help="恒星质量下限 (Msun)")
    parser.add_argument("--max-user-incl-deg", type=float, default=20.0, help="用户定义倾角上限 (deg)")
    parser.add_argument("--min-abs-b-deg", type=float, default=20.0, help="|银纬|下限 (deg)")
    parser.add_argument("--q0", type=float, default=0.2, help="盘星系本征厚度 q0 (默认 0.2)")
    parser.add_argument("--max-match-sep-arcsec", type=float, default=10.0, help="坐标兜底匹配最大角距 (arcsec)")
    parser.add_argument("--output", type=str, default="selected_nearby_disk_galaxies.csv", help="输出 CSV 文件名")
    args = parser.parse_args()

    min_mstar_1e10 = args.min_mstar_msun / 1e10

    print("1) 查询 HyperLEDA ...")
    hleda = query_hyperleda(min_d25_arcmin=args.min_d25_arcmin)
    hleda = compute_user_inclination_and_galactic_b(hleda_df=hleda, q0=args.q0)
    print(f"   HyperLEDA 候选数: {len(hleda)}")

    print("2) 查询 GLADE+ ...")
    glade = query_glade(max_distance_mpc=args.max_distance_mpc, min_mstar_1e10=min_mstar_1e10)
    print(f"   GLADE+ 候选数: {len(glade)}")

    print("3) 先按 PGC 匹配 ...")
    merged_pgc = match_by_pgc(glade, hleda)
    print(f"   PGC 匹配数: {len(merged_pgc)}")

    matched_glade_ids = set(merged_pgc["GLADE+"].tolist())
    glade_unmatched = glade[~glade["GLADE+"].isin(matched_glade_ids)].copy()

    print("4) 对未匹配对象做坐标兜底匹配 ...")
    merged_coord = match_by_coordinates(
        glade_df_unmatched=glade_unmatched,
        hleda_df=hleda,
        max_sep_arcsec=args.max_match_sep_arcsec,
    )
    print(f"   坐标兜底匹配数: {len(merged_coord)}")

    merged_all = pd.concat([merged_pgc, merged_coord], ignore_index=True, sort=False)
    merged_all = merged_all.drop_duplicates(subset=["GLADE+"], keep="first")

    print("5) 最终按倾角与银纬筛选 ...")
    final_df = apply_final_filters(
        merged_df=merged_all,
        max_user_incl_deg=args.max_user_incl_deg,
        min_abs_b_deg=args.min_abs_b_deg,
    ).copy()

    # 统一输出质量单位到 Msun
    final_df["Mstar_Msun"] = final_df["M*"] * 1e10

    keep_cols = [
        "GLADE+",
        "PGC",
        "HyperLEDA",
        "MType",
        "D25_arcmin",
        "axis_ratio_a_over_b",
        "incl_user_deg",
        "b_deg",
        "dL",
        "M*",
        "Mstar_Msun",
        "PA",
        "RAJ2000_glade",
        "DEJ2000_glade",
        "match_method",
        "match_sep_arcsec",
    ]
    for c in keep_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan

    final_df = final_df[keep_cols].rename(
        columns={
            "dL": "distance_Mpc",
            "M*": "Mstar_1e10Msun",
            "RAJ2000_glade": "RA_deg",
            "DEJ2000_glade": "Dec_deg",
        }
    )
    final_df = final_df.sort_values(["distance_Mpc", "Mstar_Msun"], ascending=[True, False]).reset_index(drop=True)
    final_df.to_csv(args.output, index=False)

    print(f"完成：筛选后样本数 = {len(final_df)}")
    print(f"已输出到: {args.output}")


if __name__ == "__main__":
    main()
