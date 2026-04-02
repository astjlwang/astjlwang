#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门拟合 residual soft proton（MOS1 / MOS2 / PN 分开）

完整模型：
    total_model =
        rsp_photon[
            LHB
            + tbabs * ( powerlaw + MWhalo )
            + gaussian_1
            + gaussian_2
        ]
        + rsp_softproton[ bknpower ]

其中：
    rsp_photon     = 原始谱文件自带的正常 ARF + RMF
    rsp_softproton = unit ARF + diag.rsp

说明：
1. soft proton 使用 broken power law（xsbknpower）
2. soft proton 不走普通 photon effective area，而走 unit ARF + diag.rsp
3. 使用 helper dataset，绕开你本地 Sherpa 对多 response-id 的兼容问题
4. 最终图中额外画出：
   - 吸收后的 powerlaw continuum
   - LHB + MWhalo sky background
   - Gaussian lines
   - soft proton
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sherpa.astro.ui as ui


# =========================
# 你只需要改这里
# =========================
DATASETS = {
    "MOS1": {
        "pi": "/home/wjl/stacking/softproton/UGCA145/mos1S001-grp.pi",
        "emin": 0.5,
        "emax": 6.0,
        "diag_rsp": "mos1-diag.rsp",
    },
    "MOS2": {
        "pi": "/home/wjl/stacking/softproton/UGCA145/mos2S002-grp.pi",
        "emin": 0.5,
        "emax": 6.0,
        "diag_rsp": "mos2-diag.rsp",
    },
    "PN": {
        "pi": "/home/wjl/stacking/softproton/UGCA145/pnS003-grp.pi",
        "emin": 0.5,
        "emax": 5.0,
        "diag_rsp": "pn-diag.rsp",
    },
}

RSP_DIAG_DIR = "/home/wjl/jupyterlab/rsp_diag"

USE_SUBTRACT = True

STAT = "chi2gehrels"
METHOD = "levmar"

MAKE_PLOTS = True

NH_INIT = 0.0918
NH_MIN, NH_MAX = 0.005, 0.2
THAW_NH = False

CXB_INDEX = 1.46
CXB_NORM_INIT, CXB_NORM_MIN, CXB_NORM_MAX = 4e-7, 1e-8, 1e-3

LHB_KT = 0.1
LHB_ABUND = 1.0
LHB_NORM_INIT = 3e-6
LHB_NORM_MIN = 1e-8
LHB_NORM_MAX = 1e-2

MWHALO_KT_INIT = 0.3
MWHALO_KT_MIN = 0.1
MWHALO_KT_MAX = 0.6
MWHALO_ABUND = 1.0
MWHALO_NORM_INIT = 3e-6
MWHALO_NORM_MIN = 1e-7
MWHALO_NORM_MAX = 1e-2

SP_CONFIG = {
    "MOS": {
        "bindl_init": 0.4,
        "bindl_min": 0.1,
        "bindl_max": 1.0,

        "break_init": 3.0,
        "break_min": 2.5,
        "break_max": 4.0,
        "thaw_break": False,

        "bindh_init": 1.0,
        "bindh_min": 0.5,
        "bindh_max": 1.8,

        "bnorm_init": 5e-3,
        "bnorm_min": 1e-7,
        "bnorm_max": 1e-1,
    },
    "PN": {
        "bindl_init": 0.6,
        "bindl_min": 0.1,
        "bindl_max": 1.5,

        "break_init": 3.0,
        "break_min": 2.5,
        "break_max": 4.0,
        "thaw_break": False,

        "bindh_init": 1.2,
        "bindh_min": 0.6,
        "bindh_max": 2.5,

        "bnorm_init": 5e-4,
        "bnorm_min": 1e-6,
        "bnorm_max": 1e-1,
    },
}

GAUSS1_E_INIT = 1.49
GAUSS1_E_MIN = 1.47
GAUSS1_E_MAX = 1.51
GAUSS1_SIGMA = 0.0
GAUSS1_NORM_INIT = 1e-6
GAUSS1_NORM_MIN = 0.0
GAUSS1_NORM_MAX = 1e-2

GAUSS2_E_INIT = 1.75
GAUSS2_E_MIN = 1.72
GAUSS2_E_MAX = 1.78
GAUSS2_SIGMA = 0.0
GAUSS2_NORM_INIT = 1e-6
GAUSS2_NORM_MIN = 0.0
GAUSS2_NORM_MAX = 1e-2

OUTDIR = "./softproton_fit_plots"
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
            f"[WARN] {cam}: 当前 emin={cfg['emin']:.3f} keV，高于 1.78 keV；"
            f"1.49/1.75 keV 两条 Gaussian 不在拟合能段内，LHB/MWhalo 也几乎不会被当前能段约束。"
            f"若要真正拟合这些低能成分，请把 emin 调到 <= 1.45 keV。"
        )


def get_diag_rsp_path(cfg):
    path = os.path.join(RSP_DIAG_DIR, cfg["diag_rsp"])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 diag rsp 文件: {path}\n"
            f"请确认 RSP_DIAG_DIR 和文件名设置正确。"
        )
    return path


def build_unit_arf_from_current_rmf(dsid, name):
    rmf = ui.get_rmf(dsid)
    if rmf is None:
        raise RuntimeError(f"{dsid}: 当前没有 RMF，无法构造 unit ARF")

    if not hasattr(rmf, "energ_lo") or not hasattr(rmf, "energ_hi"):
        raise RuntimeError(f"{dsid}: RMF 缺少 energ_lo/energ_hi")

    elo = np.asarray(rmf.energ_lo, dtype=float)
    ehi = np.asarray(rmf.energ_hi, dtype=float)

    if elo.size == 0 or ehi.size == 0 or elo.size != ehi.size:
        raise RuntimeError(
            f"{dsid}: RMF energ_lo/energ_hi 无效: "
            f"len(elo)={elo.size}, len(ehi)={ehi.size}"
        )

    exposure = None
    try:
        exposure = ui.get_data(dsid).exposure
    except Exception:
        exposure = None

    unit_arf = ui.create_arf(
        elo=elo,
        ehi=ehi,
        specresp=np.ones_like(elo, dtype=float),
        exposure=exposure,
        name=name,
    )
    return unit_arf


def apply_energy_filter(dsid, emin, emax):
    ui.set_analysis(dsid, "energy")
    ui.ignore_bad(dsid)
    ui.ignore_id(dsid)
    ui.notice_id(dsid, emin, emax)


def prepare_soft_proton_helper_dataset(src_id, helper_id, diag_rsp_path, emin, emax):
    safe_delete_data(helper_id)

    ui.copy_data(src_id, helper_id)

    ui.load_rmf(helper_id, diag_rsp_path)

    unit_arf = build_unit_arf_from_current_rmf(helper_id, f"{helper_id}_unitarf")
    ui.set_arf(helper_id, unit_arf)

    apply_energy_filter(helper_id, emin, emax)

    rsp_sp = ui.get_response(helper_id)
    if rsp_sp is None:
        raise RuntimeError(f"{helper_id}: 无法获取 soft proton 响应模型")

    return rsp_sp


def get_component_model_values(dsid, rsp_func, core_model):
    """
    在已有数据集上，临时替换 full_model 来获取某个成分的模型曲线，
    然后恢复原模型。这样避免了 copy_data 导致的 RMF 验证失败。
    """
    saved_model = ui.get_model(dsid)

    ui.set_full_model(dsid, rsp_func(core_model))
    mp = safe_get_model_plot(dsid)

    xlo = np.asarray(mp.xlo, dtype=float)
    xhi = np.asarray(mp.xhi, dtype=float)
    y = np.asarray(mp.y, dtype=float)

    ui.set_full_model(dsid, saved_model)
    return xlo, xhi, y


def make_plot(
    cam,
    helper_id,
    cfg,
    outpng,
    rsp_ph,
    rsp_sp,
    continuum_core_model,
    sky_core_model,
    gauss_core_model,
    sp_core_model,
    full_model,
):
    fp = safe_get_fit_plot(cam)
    dp = fp.dataplot
    mp = fp.modelplot
    delchi = safe_get_delchi_plot(cam)

    cont_xlo, cont_xhi, cont_y = get_component_model_values(
        cam, rsp_ph, continuum_core_model
    )

    sky_xlo, sky_xhi, sky_y = get_component_model_values(
        cam, rsp_ph, sky_core_model
    )

    ga_xlo, ga_xhi, ga_y = get_component_model_values(
        cam, rsp_ph, gauss_core_model
    )

    sp_xlo, sp_xhi, sp_y = get_component_model_values(
        helper_id, rsp_sp, sp_core_model
    )

    ui.set_full_model(cam, full_model)

    fig = plt.figure(figsize=(8.2, 6.4))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    x_err = [dp.x - dp.xlo, dp.xhi - dp.x]
    ax_main.errorbar(
        dp.x, dp.y,
        xerr=x_err, yerr=dp.yerr,
        fmt="o", ms=2.3, capsize=0,
        color="#1f77b4", alpha=0.90, label=f"{cam} Data"
    )

    x_edge = np.hstack([mp.xlo[0], mp.xhi])
    y_model = np.hstack([mp.y[0], mp.y])
    ax_main.step(
        x_edge, y_model, where="pre",
        color="black", linewidth=2.3, label="Total Model"
    )

    cont_edge = np.hstack([cont_xlo[0], cont_xhi])
    cont_step = np.hstack([cont_y[0], cont_y])
    ax_main.step(
        cont_edge, cont_step, where="pre",
        color="#00BFFF", linewidth=2.0, linestyle="--",
        label="Absorbed PL continuum"
    )

    sky_edge = np.hstack([sky_xlo[0], sky_xhi])
    sky_step = np.hstack([sky_y[0], sky_y])
    ax_main.step(
        sky_edge, sky_step, where="pre",
        color="#32CD32", linewidth=2.0, linestyle="-.",
        label="LHB + MW hot gas"
    )

    ga_edge = np.hstack([ga_xlo[0], ga_xhi])
    ga_step = np.hstack([ga_y[0], ga_y])
    ax_main.step(
        ga_edge, ga_step, where="pre",
        color="#FF8C00", linewidth=2.0, linestyle=":",
        label="Gaussian lines"
    )

    sp_edge = np.hstack([sp_xlo[0], sp_xhi])
    sp_step = np.hstack([sp_y[0], sp_y])
    ax_main.step(
        sp_edge, sp_step, where="pre",
        color="#FF1493", linewidth=2.0, linestyle=(0, (6, 2, 1, 2)),
        label="Soft proton component"
    )

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlim(cfg["emin"], cfg["emax"])
    ax_main.set_ylabel(r"$\mathrm{Counts}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$")
    ax_main.legend(loc="best", fontsize=8.8, frameon=False)
    ax_main.grid(alpha=0.25)

    x_err_delchi = [delchi.x - delchi.xlo, delchi.xhi - delchi.x]
    ax_res.errorbar(
        delchi.x, delchi.y,
        xerr=x_err_delchi, yerr=delchi.yerr,
        fmt="o", ms=2.0, capsize=0,
        color="#2ca02c", alpha=0.85
    )
    ax_res.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax_res.axhline(2, color="r", linestyle=":", linewidth=0.8)
    ax_res.axhline(-2, color="r", linestyle=":", linewidth=0.8)
    ax_res.set_xscale("log")
    ax_res.set_xlabel(r"$\mathrm{Energy\ (keV)}$")
    ax_res.set_ylabel(r"$\Delta\chi$")
    ax_res.set_ylim(-2.5, 2.5)
    ax_res.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.show()
    plt.close(fig)


def fit_one_camera(cam, cfg):
    print(f"\n===== Fitting {cam} ({cfg['emin']}-{cfg['emax']} keV) =====")
    warn_if_low_energy_components_are_outside_band(cam, cfg)

    diag_rsp_path = get_diag_rsp_path(cfg)
    helper_id = f"{cam}_SPHELP"
    spcfg = get_sp_cfg(cam)

    safe_delete_data(cam)
    safe_delete_data(helper_id)

    ui.load_pha(cam, cfg["pi"], use_errors=True)

    if USE_SUBTRACT:
        ui.subtract(cam)

    apply_energy_filter(cam, cfg["emin"], cfg["emax"])

    rsp_ph = ui.get_response(cam)
    if rsp_ph is None:
        raise RuntimeError(f"{cam}: 无法获取原始 photon 响应")

    rsp_sp = prepare_soft_proton_helper_dataset(
        src_id=cam,
        helper_id=helper_id,
        diag_rsp_path=diag_rsp_path,
        emin=cfg["emin"],
        emax=cfg["emax"],
    )

    print(f"{cam}: photon response OK")
    print(f"{cam}: soft-proton helper dataset = {helper_id}")
    print(f"{cam}: type(rsp_ph) = {type(rsp_ph)}")
    print(f"{cam}: type(rsp_sp) = {type(rsp_sp)}")
    print(f"{cam}: using {get_family(cam)} soft-proton priors")

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

    Abs = ui.get_model_component(abs_name)
    CXB = ui.get_model_component(cxb_name)
    LHB = ui.get_model_component(lhb_name)
    MWhalo = ui.get_model_component(mwh_name)
    G1 = ui.get_model_component(g1_name)
    G2 = ui.get_model_component(g2_name)
    SP = ui.get_model_component(sp_name)

    Abs.nH.set(val=NH_INIT, min=NH_MIN, max=NH_MAX)
    if THAW_NH:
        Abs.nH.thaw()
    else:
        Abs.nH.freeze()

    CXB.PhoIndex.set(val=CXB_INDEX)
    CXB.PhoIndex.freeze()
    CXB.norm.set(val=CXB_NORM_INIT, min=CXB_NORM_MIN, max=CXB_NORM_MAX)
    CXB.norm.thaw()

    LHB.kT.set(val=LHB_KT)
    LHB.kT.freeze()
    LHB.Abundanc.set(val=LHB_ABUND)
    LHB.Abundanc.freeze()
    LHB.Redshift.set(val=0.0)
    LHB.Redshift.freeze()
    LHB.norm.set(val=LHB_NORM_INIT, min=LHB_NORM_MIN, max=LHB_NORM_MAX)
    LHB.norm.thaw()

    MWhalo.kT.set(val=MWHALO_KT_INIT, min=MWHALO_KT_MIN, max=MWHALO_KT_MAX)
    MWhalo.kT.thaw()
    MWhalo.Abundanc.set(val=MWHALO_ABUND)
    MWhalo.Abundanc.freeze()
    MWhalo.Redshift.set(val=0.0)
    MWhalo.Redshift.freeze()
    MWhalo.norm.set(val=MWHALO_NORM_INIT, min=MWHALO_NORM_MIN, max=MWHALO_NORM_MAX)
    MWhalo.norm.thaw()

    G1.LineE.set(val=GAUSS1_E_INIT, min=GAUSS1_E_MIN, max=GAUSS1_E_MAX)
    G1.LineE.thaw()
    G1.Sigma.set(val=GAUSS1_SIGMA)
    G1.Sigma.freeze()
    G1.norm.set(val=GAUSS1_NORM_INIT, min=GAUSS1_NORM_MIN, max=GAUSS1_NORM_MAX)
    G1.norm.thaw()

    G2.LineE.set(val=GAUSS2_E_INIT, min=GAUSS2_E_MIN, max=GAUSS2_E_MAX)
    G2.LineE.thaw()
    G2.Sigma.set(val=GAUSS2_SIGMA)
    G2.Sigma.freeze()
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

    continuum_core_model = Abs * CXB
    sky_core_model = LHB + Abs * MWhalo
    gauss_core_model = G1 + G2
    sp_core_model = SP

    full_model = rsp_ph(continuum_core_model + sky_core_model + gauss_core_model) + rsp_sp(sp_core_model)
    ui.set_full_model(cam, full_model)

    safe_show_model(cam)

    ui.fit(cam)
    safe_show_model(cam)

    try:
        s = ui.get_stat_info(cam)[0]
        statval = s.statval
        dof = s.dof
    except Exception:
        statval = np.nan
        dof = np.nan

    outpng = os.path.join(OUTDIR, f"softproton_fit_{cam}.png")
    plot_error = None

    if MAKE_PLOTS:
        try:
            make_plot(
                cam=cam,
                helper_id=helper_id,
                cfg=cfg,
                outpng=outpng,
                rsp_ph=rsp_ph,
                rsp_sp=rsp_sp,
                continuum_core_model=continuum_core_model,
                sky_core_model=sky_core_model,
                gauss_core_model=gauss_core_model,
                sp_core_model=sp_core_model,
                full_model=full_model,
            )
            print(f"[INFO] {cam}: plot saved to {outpng}")
        except Exception as exc:
            plot_error = repr(exc)
            outpng = None
            print(f"[WARN] {cam}: 拟合完成，但绘图失败: {plot_error}")
    else:
        outpng = None

    return {
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
        "plot": outpng,
        "plot_error": plot_error,
    }


def fit_soft_proton_per_camera():
    os.makedirs(OUTDIR, exist_ok=True)

    ui.clean()
    ui.set_stat(STAT)
    ui.set_method(METHOD)

    results = []
    for cam, cfg in DATASETS.items():
        try:
            res = fit_one_camera(cam, cfg)
            results.append(res)
        except Exception as exc:
            print(f"[ERROR] {cam} 拟合失败: {exc}")
            results.append({
                "camera": cam,
                "emin": cfg["emin"],
                "emax": cfg["emax"],
                "error": str(exc),
            })

    print("\n================= Soft Proton Fit Summary =================")
    for r in results:
        if "error" in r:
            print(f"{r['camera']}: FAILED -> {r['error']}")
            continue

        extra = ""
        if r.get("plot_error"):
            extra = f" | plot_warn={r['plot_error']}"

        print(
            f"{r['camera']} ({r['emin']}-{r['emax']} keV): "
            f"LHB_norm={r['lhb_norm']:.5g}, "
            f"MWhalo_kT={r['mwhalo_kT']:.5g}, "
            f"MWhalo_norm={r['mwhalo_norm']:.5g}, "
            f"G1E={r['g1_e']:.5g}, G1norm={r['g1_norm']:.5g}, "
            f"G2E={r['g2_e']:.5g}, G2norm={r['g2_norm']:.5g}, "
            f"bindl={r['sp_bindl']:.5g}, "
            f"bbreak={r['sp_break']:.5g}, "
            f"bindh={r['sp_bindh']:.5g}, "
            f"bnorm={r['sp_bnorm']:.5g}, "
            f"CXB_norm={r['cxb_norm']:.5g}, "
            f"stat={r['statval']:.3f}, dof={r['dof']}"
            f"{extra}"
        )

    print("\n----- 可作为 ESAS proton 命令初值（speccontrol=2） -----")
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
    return results


if __name__ == "__main__":
    fit_soft_proton_per_camera()
