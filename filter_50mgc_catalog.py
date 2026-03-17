from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table


# =========================
# User-configurable inputs
# =========================
CATALOG_PATH = Path("catalog.fits")
OUTPUT_TXT = Path("50mgc_selected_targets.txt")

MIN_ABS_GAL_B_DEG = 20.0
MIN_HALO_RADIUS_ARCMIN = 10.0
MIN_LOG_MHALO = 10.0
MAX_EDGEON_ANGLE_DEG = 20.0

# Disk inclination convention used below:
# - standard inclination: 0 deg = face-on, 90 deg = edge-on
# The requested "side-on" angle is then edge_on_angle = 90 - inclination.
INCLINATION_KIND = "standard"

# Intrinsic disk thickness for axis-ratio -> inclination conversion.
Q0 = 0.20

# Treat T-type >= -3 as disk galaxies (S0 and later).
DISK_TTYPE_MIN = -3.0

# Cosmology consistent with the earlier halo relations you showed.
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.0493)
C_KMS = 299792.458


def hm2sm_K18(log_mhalo):
    """K18-style SHMR: log Mh -> log M*."""
    logM1 = 11.39
    logepsilon = -1.685
    alpha = -1.740
    delta = 4.335
    gamma = 0.531

    x = np.asarray(log_mhalo, dtype=float) - logM1
    expx = np.exp(np.clip(x, -700, 700))
    termA = -np.log10(10 ** (alpha * x) + 1.0)

    y = 10 ** (-x)
    denom = np.where(y > 50.0, np.inf, 1.0 + np.exp(y))
    termB = delta * np.power(np.log10(1.0 + expx), gamma) / denom
    return logepsilon + logM1 + termA + termB - (-np.log10(2.0))


_HM_GRID = np.linspace(7.0, 16.0, 5000)
_SM_GRID = hm2sm_K18(_HM_GRID)


def sm2hm_K18(log_mstar):
    """Inverse of hm2sm_K18 using monotonic interpolation."""
    log_mstar = np.asarray(log_mstar, dtype=float)
    return np.interp(log_mstar, _SM_GRID, _HM_GRID, left=np.nan, right=np.nan)


def z2Delta(z):
    """Bryan & Norman (1998) virial overdensity."""
    Omz = COSMO.Om(z)
    x = Omz - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2


def hm2rvir_kpc(log_mhalo, z):
    """log Mh -> virial radius in kpc."""
    mass_msun = 10 ** np.asarray(log_mhalo, dtype=float)
    rho_c = COSMO.critical_density(z).to_value(u.Msun / u.kpc ** 3)
    Delta = z2Delta(z)
    return np.power(3.0 * mass_msun / (4.0 * np.pi * Delta * rho_c), 1.0 / 3.0)


def decode_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def normalize_object_columns(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(decode_bytes)
    return df


def load_catalog(path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}. 请把 catalog.fits 放在当前工作目录，"
            "或者修改 CATALOG_PATH。"
        )

    tab = Table.read(path)
    df = tab.to_pandas()
    return normalize_object_columns(df)


def require_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"catalog 缺少必要列: {missing}")


def infer_disk_mask(df):
    mask = pd.Series(False, index=df.index)

    if "t_type" in df.columns:
        t_type = pd.to_numeric(df["t_type"], errors="coerce")
        mask |= t_type >= DISK_TTYPE_MIN

    if "best_type" in df.columns:
        best_type = (
            df["best_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        disk_keywords = [
            "s0",
            "sa",
            "sb",
            "sc",
            "sd",
            "sm",
            "spiral",
            "disk",
            "late",
            "lenticular",
        ]
        keyword_mask = pd.Series(False, index=df.index)
        for key in disk_keywords:
            keyword_mask |= best_type.str.contains(key, regex=False)
        mask |= keyword_mask

    return mask


def edgeon_from_inclination(inclination_deg, kind="standard"):
    inclination_deg = pd.to_numeric(inclination_deg, errors="coerce")
    if kind == "standard":
        return 90.0 - inclination_deg
    if kind == "edgeon_angle":
        return inclination_deg
    raise ValueError("INCLINATION_KIND 只能是 'standard' 或 'edgeon_angle'")


def inclination_from_axis_ratio(axis_ratio, q0=Q0):
    q = pd.to_numeric(axis_ratio, errors="coerce")
    q = q.clip(lower=q0, upper=1.0)
    cos2i = (q ** 2 - q0 ** 2) / (1.0 - q0 ** 2)
    cos2i = cos2i.clip(lower=0.0, upper=1.0)
    return np.degrees(np.arccos(np.sqrt(cos2i)))


def infer_edgeon_angle(df):
    explicit_inclination_cols = [
        "inclination",
        "inclination_deg",
        "incl_deg",
        "incl_leda",
        "leda_inclination",
    ]
    for col in explicit_inclination_cols:
        if col in df.columns:
            angle = edgeon_from_inclination(df[col], kind=INCLINATION_KIND)
            return angle, col

    if "logr25" in df.columns:
        axis_ratio = 10.0 ** (-pd.to_numeric(df["logr25"], errors="coerce"))
        inclination = inclination_from_axis_ratio(axis_ratio, q0=Q0)
        return 90.0 - inclination, "logr25"

    axis_ratio_cols = [
        "ba",
        "b_a",
        "axis_ratio",
        "b_over_a",
        "minor_major_ratio",
        "minor_to_major",
    ]
    for col in axis_ratio_cols:
        if col in df.columns:
            inclination = inclination_from_axis_ratio(df[col], q0=Q0)
            return 90.0 - inclination, col

    raise KeyError(
        "没有找到可用于侧向筛选的列。请确认 catalog 中至少有以下之一："
        " inclination / inclination_deg / incl_leda / logr25 / ba / axis_ratio"
    )


def main():
    df = load_catalog(CATALOG_PATH)
    require_columns(df, ["ra", "dec", "bestdist", "logmass"])

    # Sky position -> Galactic latitude
    coords = SkyCoord(
        ra=pd.to_numeric(df["ra"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(df["dec"], errors="coerce").to_numpy() * u.deg,
        frame="icrs",
    )
    df["gal_b_deg"] = coords.galactic.b.deg

    # Stellar mass -> halo mass (empirical SHMR)
    df["logmass"] = pd.to_numeric(df["logmass"], errors="coerce")
    df["log_mhalo"] = sm2hm_K18(df["logmass"].to_numpy())

    # Use heliocentric velocity only to estimate the small redshift dependence of rho_c(z).
    if "v_h" in df.columns:
        z = np.clip(pd.to_numeric(df["v_h"], errors="coerce").fillna(0.0).to_numpy() / C_KMS, 0.0, None)
    else:
        z = np.zeros(len(df), dtype=float)
    df["z_for_halo"] = z

    # Halo radius in physical kpc and angular arcmin.
    df["rvir_kpc"] = hm2rvir_kpc(df["log_mhalo"].to_numpy(), z)
    bestdist_mpc = pd.to_numeric(df["bestdist"], errors="coerce")
    df["rvir_arcmin"] = np.degrees(
        np.arctan2(df["rvir_kpc"].to_numpy() / 1000.0, bestdist_mpc.to_numpy())
    ) * 60.0

    # Disk selection
    df["is_disk"] = infer_disk_mask(df)

    # Side-on selection
    edge_on_angle_deg, inclination_source = infer_edgeon_angle(df)
    df["edge_on_angle_deg"] = edge_on_angle_deg
    df["inclination_source"] = inclination_source

    # Quality and science cuts
    finite_mask = (
        df["gal_b_deg"].notna()
        & df["bestdist"].notna()
        & df["logmass"].notna()
        & df["log_mhalo"].notna()
        & df["rvir_arcmin"].notna()
        & df["edge_on_angle_deg"].notna()
    )

    science_mask = (
        (np.abs(df["gal_b_deg"]) > MIN_ABS_GAL_B_DEG)
        & (df["rvir_arcmin"] >= MIN_HALO_RADIUS_ARCMIN)
        & (df["log_mhalo"] >= MIN_LOG_MHALO)
        & (df["edge_on_angle_deg"] <= MAX_EDGEON_ANGLE_DEG)
        & df["is_disk"]
    )

    selected = df.loc[finite_mask & science_mask].copy()

    sort_cols = ["rvir_arcmin", "log_mhalo"]
    selected = selected.sort_values(sort_cols, ascending=[False, False])
    selected.to_csv(OUTPUT_TXT, sep="\t", index=False)

    print(f"catalog rows: {len(df)}")
    print(f"inclination source used: {inclination_source}")
    print(f"selected rows: {len(selected)}")
    print(f"output saved to: {OUTPUT_TXT.resolve()}")

    preview_cols = [
        col
        for col in [
            "objname",
            "ra",
            "dec",
            "bestdist",
            "logmass",
            "log_mhalo",
            "rvir_kpc",
            "rvir_arcmin",
            "gal_b_deg",
            "edge_on_angle_deg",
            "t_type",
            "best_type",
        ]
        if col in selected.columns
    ]
    if preview_cols:
        print("\nPreview:")
        print(selected[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
