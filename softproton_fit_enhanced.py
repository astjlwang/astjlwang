#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版 residual soft proton 拟合脚本（MOS1 / MOS2 / PN 分开）

相对原始版本的改进：
1. 支持 wstat/cstat 统计量（解决高能低计数问题）
2. 可选自适应 binning
3. 自动拟合质量诊断（reduced chi2 / goodness-of-fit）
4. 参数物理合理性检查
5. 成分贡献比例图

完整模型：
    total_model =
        rsp_photon[
            LHB
            + tbabs * ( powerlaw + MWhalo )
            + gaussian_1
            + gaussian_2
        ]
        + rsp_softproton[ bknpower ]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sherpa.astro.ui as ui


# =========================
# 用户配置区
# =========================
DATASETS = {
    "MOS1": {
        "pi": "/home/wjl/stacking/softproton/NGC5908/mos1S001-grp.pi",
        "emin": 0.5,
        "emax": 6.0,
        "diag_rsp": "mos1-diag.rsp",
    },
    "MOS2": {
        "pi": "/home/wjl/stacking/softproton/NGC5908/mos2S002-grp.pi",
        "emin": 0.5,
        "emax": 6.0,
        "diag_rsp": "mos2-diag.rsp",
    },
    "PN": {
        "pi": "/home/wjl/stacking/softproton/NGC5908/pnS003-grp.pi",
        "emin": 0.5,
        "emax": 5.0,
        "diag_rsp": "pn-diag.rsp",
    },
}

RSP_DIAG_DIR = "/home/wjl/jupyterlab/rsp_diag"

# ---- 统计/拟合策略 ----
# 推荐：使用 "wstat"（有背景谱时）或 "cstat"（无背景/手动建模背景时）
# 使用 wstat/cstat 时应设置 USE_SUBTRACT = False
# 如果用 chi2gehrels/chi2datavar，可设 USE_SUBTRACT = True
STAT = "wstat"
METHOD = "levmar"
USE_SUBTRACT = False

# ---- binning 选项 ----
# "none"   : 不额外分组（推荐 wstat/cstat 使用）
# "counts" : group_counts，需设 GROUP_MIN_COUNTS
# "snr"    : group_snr，需设 GROUP_MIN_SNR
# "adapt"  : group_adapt，需设 GROUP_ADAPT_MIN
GROUPING_METHOD = "none"
GROUP_MIN_COUNTS = 25
GROUP_MIN_SNR = 3
GROUP_ADAPT_MIN = 20

MAKE_PLOTS = True

# ---- tbabs ----
NH_INIT = 0.0121
NH_MIN, NH_MAX = 0.005, 0.2
THAW_NH = False

# ---- CXB: tbabs * powerlaw ----
CXB_INDEX = 1.46
CXB_NORM_INIT, CXB_NORM_MIN, CXB_NORM_MAX = 4e-7, 1e-8, 1e-3

# ---- LHB ----
LHB_KT = 0.1
LHB_ABUND = 1.0
LHB_NORM_INIT = 3e-6
LHB_NORM_MIN = 1e-8
LHB_NORM_MAX = 1e-2

# ---- MW halo ----
MWHALO_KT_INIT = 0.3
MWHALO_KT_MIN = 0.1
MWHALO_KT_MAX = 0.6
MWHALO_ABUND = 1.0
MWHALO_NORM_INIT = 3e-6
MWHALO_NORM_MIN = 1e-7
MWHALO_NORM_MAX = 1e-2

# ---- Soft proton broken powerlaw (MOS/PN 分开) ----
SP_CONFIG = {
    "MOS": {
        "bindl_init": 0.4, "bindl_min": 0.1, "bindl_max": 1.4,
        "break_init": 3.0, "break_min": 2.5, "break_max": 4.0, "thaw_break": False,
        "bindh_init": 1.0, "bindh_min": 0.5, "bindh_max": 2.5,
        "bnorm_init": 5e-3, "bnorm_min": 1e-7, "bnorm_max": 1e-1,
    },
    "PN": {
        "bindl_init": 0.6, "bindl_min": 0.1, "bindl_max": 1.8,
        "break_init": 3.0, "break_min": 2.5, "break_max": 4.0, "thaw_break": False,
        "bindh_init": 1.2, "bindh_min": 0.6, "bindh_max": 2.5,
        "bnorm_init": 5e-4, "bnorm_min": 1e-6, "bnorm_max": 1e-1,
    },
}

# ---- Gaussian lines ----
GAUSS1_E_INIT, GAUSS1_E_MIN, GAUSS1_E_MAX = 1.49, 1.47, 1.51
GAUSS1_SIGMA = 0.0
GAUSS1_NORM_INIT, GAUSS1_NORM_MIN, GAUSS1_NORM_MAX = 1e-6, 0.0, 1e-2

GAUSS2_E_INIT, GAUSS2_E_MIN, GAUSS2_E_MAX = 1.75, 1.72, 1.78
GAUSS2_SIGMA = 0.0
GAUSS2_NORM_INIT, GAUSS2_NORM_MIN, GAUSS2_NORM_MAX = 1e-6, 0.0, 1e-2

OUTDIR = "./softproton_fit_plots"


# =========================
# 辅助函数
# =========================

def safe_delete_data(dsid):
    try:
        ui.delete_data(dsid)
    except Exception:
        pass


def safe_show_model(dsid):
    try:
        ui.show_model(id=dsid)
    except TypeError:
        ui.show_model(dsid)


def safe_get_fit_plot(dsid):
    try:
        return ui.get_fit_plot(id=dsid)
    except TypeError:
        return ui.get_fit_plot(dsid)


def safe_get_delchi_plot(dsid):
    try:
        return ui.get_delchi_plot(id=dsid)
    except TypeError:
        return ui.get_delchi_plot(dsid)


def safe_get_model_plot(dsid):
    try:
        return ui.get_model_plot(id=dsid)
    except TypeError:
        return ui.get_model_plot(dsid)


def get_family(cam):
    return "PN" if cam.upper() == "PN" else "MOS"


def get_sp_cfg(cam):
    return SP_CONFIG[get_family(cam)]


def warn_if_low_energy_components_are_outside_band(cam, cfg):
    if cfg["emin"] > 1.78:
        print(
            f"[WARN] {cam}: emin={cfg['emin']:.3f} keV > 1.78 keV; "
            f"1.49/1.75 keV Gaussians 和 LHB/MWhalo 在拟合能段外，"
            f"低能成分基本无法被约束。"
        )


def get_diag_rsp_path(cfg):
    path = os.path.join(RSP_DIAG_DIR, cfg["diag_rsp"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 diag rsp 文件: {path}")
    return path


def build_unit_arf_from_current_rmf(dsid, name):
    rmf = ui.get_rmf(dsid)
    if rmf is None:
        raise RuntimeError(f"{dsid}: 没有 RMF，无法构造 unit ARF")
    if not hasattr(rmf, "energ_lo") or not hasattr(rmf, "energ_hi"):
        raise RuntimeError(f"{dsid}: RMF 缺少 energ_lo/energ_hi")
    elo = np.asarray(rmf.energ_lo, dtype=float)
    ehi = np.asarray(rmf.energ_hi, dtype=float)
    if elo.size == 0 or ehi.size == 0 or elo.size != ehi.size:
        raise RuntimeError(f"{dsid}: RMF 能量网格无效")

    exposure = None
    try:
        exposure = ui.get_data(dsid).exposure
    except Exception:
        pass

    return ui.create_arf(
        elo=elo, ehi=ehi,
        specresp=np.ones_like(elo, dtype=float),
        exposure=exposure, name=name,
    )


def apply_energy_filter(dsid, emin, emax):
    ui.set_analysis(dsid, "energy")
    ui.ignore_bad(dsid)
    ui.ignore_id(dsid)
    ui.notice_id(dsid, emin, emax)


def apply_grouping(dsid):
    if GROUPING_METHOD == "counts":
        ui.group_counts(dsid, GROUP_MIN_COUNTS)
        print(f"  [GROUPING] group_counts({GROUP_MIN_COUNTS})")
    elif GROUPING_METHOD == "snr":
        ui.group_snr(dsid, GROUP_MIN_SNR)
        print(f"  [GROUPING] group_snr({GROUP_MIN_SNR})")
    elif GROUPING_METHOD == "adapt":
        ui.group_adapt(dsid, GROUP_ADAPT_MIN)
        print(f"  [GROUPING] group_adapt({GROUP_ADAPT_MIN})")
    else:
        print(f"  [GROUPING] none (raw channels)")


def prepare_soft_proton_helper_dataset(src_id, helper_id, diag_rsp_path, emin, emax):
    safe_delete_data(helper_id)
    ui.copy_data(src_id, helper_id)
    ui.load_rmf(helper_id, diag_rsp_path)
    unit_arf = build_unit_arf_from_current_rmf(helper_id, f"{helper_id}_unitarf")
    ui.set_arf(helper_id, unit_arf)
    apply_energy_filter(helper_id, emin, emax)

    rsp_sp = ui.get_response(helper_id)
    if rsp_sp is None:
        raise RuntimeError(f"{helper_id}: 无法获取 soft proton 响应")
    return rsp_sp


def extract_component_model_plot(src_id, plot_id, core_model, emin, emax):
    safe_delete_data(plot_id)
    ui.copy_data(src_id, plot_id)
    apply_energy_filter(plot_id, emin, emax)

    rsp = ui.get_response(plot_id)
    if rsp is None:
        safe_delete_data(plot_id)
        raise RuntimeError(f"{plot_id}: 无法获取响应")

    ui.set_full_model(plot_id, rsp(core_model))
    mp = safe_get_model_plot(plot_id)

    xlo = np.asarray(mp.xlo, dtype=float)
    xhi = np.asarray(mp.xhi, dtype=float)
    y = np.asarray(mp.y, dtype=float)

    safe_delete_data(plot_id)
    return xlo, xhi, y


# =========================
# 拟合诊断
# =========================

# 期望的物理参数范围 (Kuntz & Snowden 2008; Snowden et al. 2008; ESAS cookbook)
PARAM_EXPECTED_RANGES = {
    "LHB_kT":       (0.08, 0.12, "keV"),
    "LHB_norm":     (1e-8, 1e-4, ""),
    "MWhalo_kT":    (0.10, 0.60, "keV"),
    "MWhalo_norm":  (1e-7, 1e-3, ""),
    "CXB_norm":     (1e-8, 1e-5, ""),
    "SP_PhoIndx1_MOS":  (0.1, 1.4, ""),
    "SP_PhoIndx1_PN":   (0.1, 1.8, ""),
    "SP_PhoIndx2_MOS":  (0.5, 2.5, ""),
    "SP_PhoIndx2_PN":   (0.6, 2.5, ""),
    "SP_norm":      (1e-7, 0.1, ""),
}


def check_parameter_reasonableness(cam, result):
    """检查拟合参数是否在文献推荐的物理范围内"""
    if "error" in result:
        return []

    family = get_family(cam)
    warnings = []

    checks = [
        ("MWhalo_kT",    result.get("mwhalo_kT"),   "MWhalo_kT"),
        ("MWhalo_norm",  result.get("mwhalo_norm"),  "MWhalo_norm"),
        ("LHB_norm",     result.get("lhb_norm"),     "LHB_norm"),
        ("CXB_norm",     result.get("cxb_norm"),     "CXB_norm"),
        ("SP_PhoIndx1",  result.get("sp_bindl"),     f"SP_PhoIndx1_{family}"),
        ("SP_PhoIndx2",  result.get("sp_bindh"),     f"SP_PhoIndx2_{family}"),
        ("SP_norm",      result.get("sp_bnorm"),     "SP_norm"),
    ]

    for label, val, range_key in checks:
        if val is None or range_key not in PARAM_EXPECTED_RANGES:
            continue
        lo, hi, unit = PARAM_EXPECTED_RANGES[range_key]
        if val < lo:
            warnings.append(
                f"  {label} = {val:.4g} 低于预期范围 [{lo:.4g}, {hi:.4g}] {unit}"
            )
        elif val > hi:
            warnings.append(
                f"  {label} = {val:.4g} 高于预期范围 [{lo:.4g}, {hi:.4g}] {unit}"
            )

    return warnings


def compute_fit_quality(cam):
    """计算拟合质量指标"""
    quality = {}

    try:
        stats = ui.get_stat_info()
        for s in stats:
            if hasattr(s, 'ids') and cam in s.ids:
                quality["statval"] = s.statval
                quality["dof"] = s.dof
                quality["stat_name"] = s.name if hasattr(s, 'name') else STAT
                if s.dof > 0:
                    quality["reduced"] = s.statval / s.dof
                break
        else:
            si = ui.get_stat_info(cam)
            if isinstance(si, list):
                si = si[0]
            quality["statval"] = si.statval
            quality["dof"] = si.dof
            quality["stat_name"] = STAT
            if si.dof > 0:
                quality["reduced"] = si.statval / si.dof
    except Exception as e:
        quality["error"] = str(e)

    return quality


def print_fit_quality(cam, quality):
    """打印拟合质量诊断"""
    print(f"\n--- {cam} 拟合质量诊断 ---")

    if "error" in quality:
        print(f"  [ERROR] 无法获取统计信息: {quality['error']}")
        return

    stat_name = quality.get("stat_name", STAT)
    statval = quality.get("statval", np.nan)
    dof = quality.get("dof", np.nan)
    reduced = quality.get("reduced", np.nan)

    print(f"  统计量: {stat_name}")
    print(f"  stat = {statval:.2f}, dof = {dof}")
    print(f"  reduced stat = {reduced:.3f}")

    if "chi2" in stat_name.lower():
        if reduced > 1.5:
            print(f"  [WARN] reduced chi2 = {reduced:.3f} > 1.5 → 拟合不佳，"
                  f"可能缺少模型成分或参数约束不当")
        elif reduced < 0.7:
            print(f"  [WARN] reduced chi2 = {reduced:.3f} < 0.7 → 可能过拟合"
                  f"或误差估计过大")
        else:
            print(f"  [OK] reduced chi2 在合理范围内")
    elif "cstat" in stat_name.lower() or "wstat" in stat_name.lower():
        if reduced > 1.5:
            print(f"  [WARN] reduced stat = {reduced:.3f} > 1.5 → 可能拟合不佳")
        elif reduced < 0.5:
            print(f"  [INFO] reduced stat = {reduced:.3f} 较低，可能 bins 太多或过拟合")
        else:
            print(f"  [OK] 统计量在合理范围内")
        print(f"  [TIP] 对于 cstat/wstat，建议用 Sherpa 的 goodness() 或 XSPEC 的"
              f" 'goodness 1000' 做 MC 模拟检验")


# =========================
# 绘图
# =========================

def make_plot(
    cam, helper_id, cfg, outpng,
    continuum_core_model, sky_core_model,
    gauss_core_model, sp_core_model,
):
    fp = safe_get_fit_plot(cam)
    dp = fp.dataplot
    mp = fp.modelplot

    try:
        delchi = safe_get_delchi_plot(cam)
        has_delchi = True
    except Exception:
        has_delchi = False

    cont_xlo, cont_xhi, cont_y = extract_component_model_plot(
        cam, f"{cam}_PLOTCONT", continuum_core_model, cfg["emin"], cfg["emax"])
    sky_xlo, sky_xhi, sky_y = extract_component_model_plot(
        cam, f"{cam}_PLOTSKY", sky_core_model, cfg["emin"], cfg["emax"])
    ga_xlo, ga_xhi, ga_y = extract_component_model_plot(
        cam, f"{cam}_PLOTGA", gauss_core_model, cfg["emin"], cfg["emax"])
    sp_xlo, sp_xhi, sp_y = extract_component_model_plot(
        helper_id, f"{cam}_PLOTSP", sp_core_model, cfg["emin"], cfg["emax"])

    n_panels = 3 if has_delchi else 2
    height_ratios = [3, 1, 1] if has_delchi else [3, 1]

    fig = plt.figure(figsize=(9, 7.5 if has_delchi else 6))
    gs = plt.GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.05)
    ax_main = fig.add_subplot(gs[0])

    # 数据点
    x_err = [dp.x - dp.xlo, dp.xhi - dp.x]
    ax_main.errorbar(
        dp.x, dp.y, xerr=x_err, yerr=dp.yerr,
        fmt="o", ms=2.3, capsize=0,
        color="#1f77b4", alpha=0.90, label=f"{cam} Data",
    )

    # 总模型
    x_edge = np.hstack([mp.xlo[0], mp.xhi])
    y_model = np.hstack([mp.y[0], mp.y])
    ax_main.step(x_edge, y_model, where="pre",
                 color="black", linewidth=2.3, label="Total Model")

    # 吸收后的 powerlaw continuum
    cont_edge = np.hstack([cont_xlo[0], cont_xhi])
    cont_step = np.hstack([cont_y[0], cont_y])
    ax_main.step(cont_edge, cont_step, where="pre",
                 color="#00BFFF", linewidth=2.0, linestyle="--",
                 label="Absorbed PL continuum")

    # sky background
    sky_edge = np.hstack([sky_xlo[0], sky_xhi])
    sky_step = np.hstack([sky_y[0], sky_y])
    ax_main.step(sky_edge, sky_step, where="pre",
                 color="#32CD32", linewidth=2.0, linestyle="-.",
                 label="LHB + MW hot gas")

    # Gaussian lines
    ga_edge = np.hstack([ga_xlo[0], ga_xhi])
    ga_step = np.hstack([ga_y[0], ga_y])
    ax_main.step(ga_edge, ga_step, where="pre",
                 color="#FF8C00", linewidth=2.0, linestyle=":",
                 label="Gaussian lines")

    # soft proton
    sp_edge = np.hstack([sp_xlo[0], sp_xhi])
    sp_step = np.hstack([sp_y[0], sp_y])
    ax_main.step(sp_edge, sp_step, where="pre",
                 color="#FF1493", linewidth=2.0, linestyle=(0, (6, 2, 1, 2)),
                 label="Soft proton")

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlim(cfg["emin"], cfg["emax"])
    ax_main.set_ylabel(r"$\mathrm{Counts}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$")
    ax_main.legend(loc="best", fontsize=8.5, frameon=False)
    ax_main.grid(alpha=0.25)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # 残差
    if has_delchi:
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)
        x_err_delchi = [delchi.x - delchi.xlo, delchi.xhi - delchi.x]
        ax_res.errorbar(
            delchi.x, delchi.y, xerr=x_err_delchi, yerr=delchi.yerr,
            fmt="o", ms=2.0, capsize=0, color="#2ca02c", alpha=0.85,
        )
        ax_res.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax_res.axhline(2, color="r", linestyle=":", linewidth=0.8)
        ax_res.axhline(-2, color="r", linestyle=":", linewidth=0.8)
        ax_res.set_xscale("log")
        ax_res.set_ylabel(r"$\Delta\chi$")
        ax_res.set_ylim(-4, 4)
        ax_res.grid(alpha=0.25)
        plt.setp(ax_res.get_xticklabels(), visible=False)

    # 成分贡献比例图
    ratio_ax_idx = 2 if has_delchi else 1
    ax_ratio = fig.add_subplot(gs[ratio_ax_idx], sharex=ax_main)

    total_y = mp.y.copy()
    total_y[total_y <= 0] = 1e-30

    def _safe_interp(comp_xlo, comp_y, ref_x):
        comp_x_mid = 0.5 * (comp_xlo[:-1] + comp_xlo[1:]) if len(comp_xlo) > len(comp_y) else comp_xlo
        if len(comp_x_mid) == len(comp_y):
            return np.interp(ref_x, comp_x_mid, comp_y, left=0, right=0)
        return np.interp(ref_x, comp_xlo[:len(comp_y)], comp_y, left=0, right=0)

    ref_x = 0.5 * (np.asarray(mp.xlo) + np.asarray(mp.xhi))

    for comp_name, comp_xlo, comp_y, comp_color in [
        ("Absorbed PL", cont_xlo, cont_y, "#00BFFF"),
        ("Sky BG", sky_xlo, sky_y, "#32CD32"),
        ("Gauss", ga_xlo, ga_y, "#FF8C00"),
        ("Soft proton", sp_xlo, sp_y, "#FF1493"),
    ]:
        comp_mid = 0.5 * (comp_xlo + np.roll(comp_xlo, -1))
        interp_y = np.interp(ref_x, comp_mid[:len(comp_y)], comp_y, left=0, right=0)
        ratio = interp_y / total_y
        ax_ratio.step(
            np.hstack([mp.xlo[0], np.asarray(mp.xhi)]),
            np.hstack([ratio[0], ratio]),
            where="pre", linewidth=1.5, label=comp_name, color=comp_color,
        )

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel(r"$\mathrm{Energy\ (keV)}$")
    ax_ratio.set_ylabel("Fraction")
    ax_ratio.set_ylim(0, 1.15)
    ax_ratio.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax_ratio.grid(alpha=0.25)

    fig.suptitle(f"{cam}  Soft Proton Fit  [{cfg['emin']}-{cfg['emax']} keV]  stat={STAT}",
                 fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpng, dpi=150)
    plt.close(fig)


# =========================
# 核心拟合
# =========================

def fit_one_camera(cam, cfg):
    print(f"\n{'='*60}")
    print(f"  Fitting {cam} ({cfg['emin']}-{cfg['emax']} keV)")
    print(f"  stat={STAT}, method={METHOD}, subtract={USE_SUBTRACT}")
    print(f"{'='*60}")

    warn_if_low_energy_components_are_outside_band(cam, cfg)
    diag_rsp_path = get_diag_rsp_path(cfg)
    helper_id = f"{cam}_SPHELP"
    spcfg = get_sp_cfg(cam)

    safe_delete_data(cam)
    safe_delete_data(helper_id)

    # 1) 读谱
    ui.load_pha(cam, cfg["pi"], use_errors=True)

    # 2) 背景处理
    if USE_SUBTRACT:
        ui.subtract(cam)
        print(f"  [INFO] 已做背景相减 (subtract)")
    else:
        print(f"  [INFO] 不做背景相减 (适用于 wstat/cstat)")

    # 3) 能段过滤
    apply_energy_filter(cam, cfg["emin"], cfg["emax"])

    # 4) 分组
    apply_grouping(cam)

    # 5) 正常 photon 响应
    rsp_ph = ui.get_response(cam)
    if rsp_ph is None:
        raise RuntimeError(f"{cam}: 无法获取 photon 响应")

    # 6) SP helper response
    rsp_sp = prepare_soft_proton_helper_dataset(
        cam, helper_id, diag_rsp_path, cfg["emin"], cfg["emax"])

    print(f"  photon response: OK")
    print(f"  SP helper dataset: {helper_id}")
    print(f"  SP family: {get_family(cam)}")

    # 7) 创建模型组件
    abs_name = f"Abs_{cam}"
    cxb_name = f"CXB_{cam}"
    lhb_name = f"LHB_{cam}"
    mwh_name = f"MWhalo_{cam}"
    g1_name = f"G1_{cam}"
    g2_name = f"G2_{cam}"
    sp_name = f"SP_{cam}"

    ui.create_model_component("xstbabs", abs_name)
    ui.create_model_component("xspowerlaw", cxb_name)
    ui.create_model_component("xsapec", lhb_name)
    ui.create_model_component("xsapec", mwh_name)
    ui.create_model_component("xsgaussian", g1_name)
    ui.create_model_component("xsgaussian", g2_name)
    ui.create_model_component("xsbknpower", sp_name)

    Abs    = ui.get_model_component(abs_name)
    CXB    = ui.get_model_component(cxb_name)
    LHB    = ui.get_model_component(lhb_name)
    MWhalo = ui.get_model_component(mwh_name)
    G1     = ui.get_model_component(g1_name)
    G2     = ui.get_model_component(g2_name)
    SP     = ui.get_model_component(sp_name)

    # 8) 设置参数
    Abs.nH.set(val=NH_INIT, min=NH_MIN, max=NH_MAX)
    if THAW_NH:
        Abs.nH.thaw()
    else:
        Abs.nH.freeze()

    CXB.PhoIndex.set(val=CXB_INDEX)
    CXB.PhoIndex.freeze()
    CXB.norm.set(val=CXB_NORM_INIT, min=CXB_NORM_MIN, max=CXB_NORM_MAX)
    CXB.norm.thaw()

    LHB.kT.set(val=LHB_KT); LHB.kT.freeze()
    LHB.Abundanc.set(val=LHB_ABUND); LHB.Abundanc.freeze()
    LHB.Redshift.set(val=0.0); LHB.Redshift.freeze()
    LHB.norm.set(val=LHB_NORM_INIT, min=LHB_NORM_MIN, max=LHB_NORM_MAX)
    LHB.norm.thaw()

    MWhalo.kT.set(val=MWHALO_KT_INIT, min=MWHALO_KT_MIN, max=MWHALO_KT_MAX)
    MWhalo.kT.thaw()
    MWhalo.Abundanc.set(val=MWHALO_ABUND); MWhalo.Abundanc.freeze()
    MWhalo.Redshift.set(val=0.0); MWhalo.Redshift.freeze()
    MWhalo.norm.set(val=MWHALO_NORM_INIT, min=MWHALO_NORM_MIN, max=MWHALO_NORM_MAX)
    MWhalo.norm.thaw()

    G1.LineE.set(val=GAUSS1_E_INIT, min=GAUSS1_E_MIN, max=GAUSS1_E_MAX)
    G1.LineE.thaw()
    G1.Sigma.set(val=GAUSS1_SIGMA); G1.Sigma.freeze()
    G1.norm.set(val=GAUSS1_NORM_INIT, min=GAUSS1_NORM_MIN, max=GAUSS1_NORM_MAX)
    G1.norm.thaw()

    G2.LineE.set(val=GAUSS2_E_INIT, min=GAUSS2_E_MIN, max=GAUSS2_E_MAX)
    G2.LineE.thaw()
    G2.Sigma.set(val=GAUSS2_SIGMA); G2.Sigma.freeze()
    G2.norm.set(val=GAUSS2_NORM_INIT, min=GAUSS2_NORM_MIN, max=GAUSS2_NORM_MAX)
    G2.norm.thaw()

    SP.PhoIndx1.set(val=spcfg["bindl_init"], min=spcfg["bindl_min"], max=spcfg["bindl_max"])
    SP.PhoIndx1.thaw()
    SP.BreakE.set(val=spcfg["break_init"], min=spcfg["break_min"], max=spcfg["break_max"])
    if spcfg["thaw_break"]:
        SP.BreakE.thaw()
    else:
        SP.BreakE.freeze()
    SP.PhoIndx2.set(val=spcfg["bindh_init"], min=spcfg["bindh_min"], max=spcfg["bindh_max"])
    SP.PhoIndx2.thaw()
    SP.norm.set(val=spcfg["bnorm_init"], min=spcfg["bnorm_min"], max=spcfg["bnorm_max"])
    SP.norm.thaw()

    # 9) 完整模型
    continuum_core_model = Abs * CXB
    sky_core_model = LHB + Abs * MWhalo
    gauss_core_model = G1 + G2
    sp_core_model = SP

    full_model = (rsp_ph(continuum_core_model + sky_core_model + gauss_core_model)
                  + rsp_sp(sp_core_model))
    ui.set_full_model(cam, full_model)

    safe_show_model(cam)

    # 10) 拟合
    ui.fit(cam)
    safe_show_model(cam)

    # 11) 拟合质量
    quality = compute_fit_quality(cam)
    print_fit_quality(cam, quality)

    statval = quality.get("statval", np.nan)
    dof = quality.get("dof", np.nan)

    # 12) 画图
    outpng = os.path.join(OUTDIR, f"softproton_fit_{cam}.png")
    plot_error = None

    if MAKE_PLOTS:
        try:
            make_plot(
                cam=cam, helper_id=helper_id, cfg=cfg, outpng=outpng,
                continuum_core_model=continuum_core_model,
                sky_core_model=sky_core_model,
                gauss_core_model=gauss_core_model,
                sp_core_model=sp_core_model,
            )
            print(f"  [INFO] plot saved: {outpng}")
        except Exception as exc:
            plot_error = repr(exc)
            outpng = None
            print(f"  [WARN] 绘图失败: {plot_error}")
    else:
        outpng = None

    result = {
        "camera": cam,
        "emin": cfg["emin"],
        "emax": cfg["emax"],
        "nh": Abs.nH.val,
        "cxb_index_fixed": CXB.PhoIndex.val,
        "cxb_norm": CXB.norm.val,
        "lhb_norm": LHB.norm.val,
        "mwhalo_kT": MWhalo.kT.val,
        "mwhalo_norm": MWhalo.norm.val,
        "g1_e": G1.LineE.val,
        "g1_norm": G1.norm.val,
        "g2_e": G2.LineE.val,
        "g2_norm": G2.norm.val,
        "sp_bindl": SP.PhoIndx1.val,
        "sp_break": SP.BreakE.val,
        "sp_bindh": SP.PhoIndx2.val,
        "sp_bnorm": SP.norm.val,
        "statval": statval,
        "dof": dof,
        "reduced_stat": quality.get("reduced", np.nan),
        "plot": outpng,
        "plot_error": plot_error,
    }

    # 13) 参数合理性检查
    warnings = check_parameter_reasonableness(cam, result)
    if warnings:
        print(f"\n  [PARAM CHECK] {cam} 参数合理性警告:")
        for w in warnings:
            print(f"    {w}")
    else:
        print(f"\n  [PARAM CHECK] {cam} 所有参数在预期范围内")

    return result


def fit_soft_proton_per_camera():
    os.makedirs(OUTDIR, exist_ok=True)

    ui.clean()
    ui.set_stat(STAT)
    ui.set_method(METHOD)

    print(f"\n统计量: {STAT}")
    print(f"优化方法: {METHOD}")
    print(f"背景相减: {USE_SUBTRACT}")
    print(f"分组方式: {GROUPING_METHOD}")

    results = []
    for cam, cfg in DATASETS.items():
        try:
            res = fit_one_camera(cam, cfg)
            results.append(res)
        except Exception as exc:
            print(f"[ERROR] {cam} 拟合失败: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "camera": cam,
                "emin": cfg["emin"],
                "emax": cfg["emax"],
                "error": str(exc),
            })

    # ===== 总结 =====
    print(f"\n{'='*70}")
    print(f"  Soft Proton Fit Summary  (stat={STAT})")
    print(f"{'='*70}")

    for r in results:
        if "error" in r:
            print(f"\n{r['camera']}: FAILED -> {r['error']}")
            continue

        reduced = r.get("reduced_stat", np.nan)
        extra = ""
        if r.get("plot_error"):
            extra = f" | plot_warn={r['plot_error']}"

        print(
            f"\n{r['camera']} ({r['emin']}-{r['emax']} keV):"
            f"\n  LHB_norm={r['lhb_norm']:.5g}"
            f"  MWhalo_kT={r['mwhalo_kT']:.5g}"
            f"  MWhalo_norm={r['mwhalo_norm']:.5g}"
            f"\n  G1E={r['g1_e']:.5g}  G1norm={r['g1_norm']:.5g}"
            f"  G2E={r['g2_e']:.5g}  G2norm={r['g2_norm']:.5g}"
            f"\n  SP: bindl={r['sp_bindl']:.5g}"
            f"  bbreak={r['sp_break']:.5g}"
            f"  bindh={r['sp_bindh']:.5g}"
            f"  bnorm={r['sp_bnorm']:.5g}"
            f"\n  CXB_norm={r['cxb_norm']:.5g}"
            f"  stat={r['statval']:.3f}  dof={r['dof']}"
            f"  reduced={reduced:.3f}"
            f"{extra}"
        )

        warnings = check_parameter_reasonableness(r['camera'], r)
        if warnings:
            for w in warnings:
                print(f"    [!] {w}")

    # ESAS proton 命令参考
    print(f"\n{'='*70}")
    print(f"  ESAS proton 命令初值 (speccontrol=2)")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            continue
        print(
            f"{r['camera']}: "
            f"bindl={r['sp_bindl']:.5g}  "
            f"bbreak={r['sp_break']:.5g}  "
            f"bindh={r['sp_bindh']:.5g}  "
            f"bnorm={r['sp_bnorm']:.5g}"
        )

    print(f"\n图像输出目录: {OUTDIR}")

    # bnorm 解读
    print(f"\n{'='*70}")
    print(f"  bnorm 解读")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            continue
        bn = r["sp_bnorm"]
        if bn < 1e-5:
            level = "极微弱（可忽略）"
        elif bn < 1e-3:
            level = "轻微"
        elif bn < 1e-2:
            level = "中等"
        else:
            level = "较严重"
        print(f"  {r['camera']}: bnorm={bn:.5g} → SP 污染程度: {level}")

    return results


if __name__ == "__main__":
    fit_soft_proton_per_camera()
