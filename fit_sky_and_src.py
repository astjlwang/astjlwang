#!/usr/bin/env python3
"""
Joint Spectrum Fitting (v15 adaptation)

This script performs a two-step Sherpa fitting workflow:
1. Jointly fit MOS sky background spectra with an extended background model.
2. Fit the source spectrum reusing the fitted sky background components.

The plotting section for the source fit has been enhanced to visualise:
    - the observed data
    - the total best-fit model
    - key physical/model components (including tbabs*vrnei)
    - residuals with ±2σ guide lines in the delchi panel
"""

import os
from re import A  # retained from reference snippet

import glob
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

try:
    import scienceplots

    plt.style.use(["science"])
except Exception:  # pragma: no cover - cosmetic
    print("scienceplots not installed")

import sherpa.astro.ui as ui


def safe_mkdir(dirname: str) -> None:
    try:
        os.makedirs(dirname, exist_ok=True)
        print(f"Directory '{dirname}' created or already exists")
    except Exception as exc:  # pragma: no cover - filesystem safety
        print(f"Error creating directory: {exc}")


safe_mkdir("joint_spectrum_fitting_2T_basedon_region_v15")


####################################################################


# ---------- 1. 文件路径 ----------
bkg1_pi = "/home/wjl/odf/skyback/mos1S001-grp.pi"
bkg2_pi = "/home/wjl/odf/skyback/mos2S002-grp.pi"
src_pi = "/home/wjl/odf/sp_resolved/sus/mos2S002-grp.pi"

# ---------- 2. 加载背景谱 ----------
ui.load_pha("BKG_1", bkg1_pi, use_errors=True)
ui.load_pha("BKG_2", bkg2_pi, use_errors=True)
ui.subtract("BKG_1")
ui.subtract("BKG_2")

bkg1_backscal = ui.get_backscal("BKG_1")
bkg2_backscal = ui.get_backscal("BKG_2")
print(f"BKG_1 BACKSCAL = {bkg1_backscal}")
print(f"BKG_2 BACKSCAL = {bkg2_backscal}")

# ---------- 3. sky background 模型 ----------
# Local Hot Bubble
ui.create_model_component("xsapec", "LHB")
LHB.kT = 0.1
LHB.kT.freeze()
LHB.Abundanc = 1.0
LHB.Abundanc.freeze()
LHB.norm = 3e-6  # norm 可拟合
LHB.norm.thaw()

# MW halo / hot bubble: apec with kT in [0.1, 0.6], abundance fixed to 1
ui.create_model_component("xsapec", "MWhalo")
MWhalo.kT.set(val=0.3, min=0.1, max=0.6)  # 初始0.3 keV，自由范围0.1–0.6 keV
MWhalo.kT.thaw()  # 允许拟合温度
MWhalo.Abundanc = 1.0
MWhalo.Abundanc.freeze()  # 丰度固定为1
MWhalo.norm.set(val=3e-6, min=1e-7, max=1e-3)
MWhalo.norm.thaw()

# CXB: 冻结 index=1.46
ui.create_model_component("xspowerlaw", "CXB")
CXB.PhoIndex = 1.46
CXB.PhoIndex.freeze()
CXB.norm = 4e-7
CXB.norm.thaw()

# 吸收层（Galactic absorption, tbabs），skyback拟合时允许变化
ui.create_model_component("xstbabs", "SkyAbs")
SkyAbs.nH.set(val=0.5, min=0.05, max=2.0)  # 初始0.5，自由范围0.05–2.0
SkyAbs.nH.thaw()

# Instrumental lines
ui.create_model_component("xsgaussian", "InstLine_1")
InstLine_1.LineE.set(val=1.49, min=1.47, max=1.51)
InstLine_1.Sigma.set(val=0.0)
InstLine_1.Sigma.freeze()
InstLine_1.norm = 1e-5
InstLine_1.norm.thaw()

ui.create_model_component("xsgaussian", "InstLine_2")
InstLine_2.LineE.set(val=1.755, min=1.73, max=1.78)
InstLine_2.Sigma.set(val=0.0)
InstLine_2.Sigma.freeze()
InstLine_2.norm = 1e-5
InstLine_2.norm.thaw()

# ✅ 新增 Soft Proton 分量
ui.create_model_component("xspowerlaw", "SoftProton")
SoftProton.PhoIndex.set(val=0.4, min=0.1, max=1.0)  # index 可拟合
SoftProton.norm.set(val=1e-5, min=0.0, max=1e-2)  # norm 可拟合
# 两个参数在 skyback 拟合时都是 thaw 状态

# ---------- 4. sky 模型定义 ----------
ui.create_model_component("scale1d", "sky_scale")
sky_scale.c0 = 1.0
sky_scale.c0.freeze()

sky_model = sky_scale * (LHB + SkyAbs * (MWhalo + CXB)) + InstLine_1 + InstLine_2 + SoftProton

ui.set_source("BKG_1", sky_model)
ui.set_source("BKG_2", sky_model)

# ---------- 5. 拟合设置 ----------
ui.set_stat("chi2gehrels")
ui.set_method("levmar")

# 拟合能段: 0.5-6.0 keV
for bid in ["BKG_1", "BKG_2"]:
    ui.ignore_bad(bid)
    ui.notice_id(bid, 0.5, 6.0)

# ---------- 6. joint fit ----------
print("\nFitting sky backgrounds (joint, 0.5–6.0 keV, with SoftProton)...")
ui.fit("BKG_1", "BKG_2")
ui.show_model("BKG_1")

# ---------- 7. 绘制 skyback 拟合结果（改进版） ----------
fig = plt.figure(figsize=(7, 7))
gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 1], hspace=0)
ax_main = fig.add_subplot(gs[0:3])
ax_res = fig.add_subplot(gs[3], sharex=ax_main)

colors = ["tab:blue", "tab:orange"]
labels = ["MOS1", "MOS2"]
for i, bid in enumerate(["BKG_1", "BKG_2"]):
    fp = ui.get_fit_plot(bid)
    dp = fp.dataplot
    mp = fp.modelplot

    # ---- 绘制观测数据 ----
    ax_main.errorbar(
        dp.x, dp.y, yerr=dp.yerr, fmt=".", color="tab:blue", label="Data" if i == 0 else None
    )

    # ---- 绘制总模型 ----
    xedge = np.hstack([mp.xlo[0], mp.xhi])
    ymod = np.hstack([mp.y[0], mp.y])
    ax_main.step(
        xedge, ymod, where="pre", color="orange", lw=1.8, label="Total Model" if i == 0 else None
    )

    # ---- 各个成分绘制 ----
    E = mp.x  # 模型能量 bin 中心

    def eval_comp(comp_name: str) -> np.ndarray:
        return ui.eval_model_component(bid, comp_name)

    # 各模型分量及颜色
    comps = {
        "LHB": ("purple", "LHB"),
        "SkyAbs*MWhalo": ("green", "MW Halo"),
        "SkyAbs*CXB": ("red", "CXB (abs)"),
        "SoftProton": ("black", "Soft Proton"),
        "InstLine_1": ("gray", "Instrumental Lines"),
        "InstLine_2": ("gray", None),  # 第二条灰色不再重复 legend
    }

    for cname, (color, label) in comps.items():
        try:
            y_vals = eval_comp(cname)
            ax_main.plot(E, y_vals, ls="--", lw=1.2, color=color, label=label)
        except Exception as exc:  # pragma: no cover - plotting support
            print(f"Component {cname} evaluation failed: {exc}")

    # ---- 残差图 ----
    delp = ui.get_delchi_plot(bid)
    ax_res.errorbar(delp.x, delp.y, yerr=delp.yerr, fmt=".", color=colors[i])

ax_main.set_yscale("log")
ax_main.set_ylabel("Counts / keV")
ax_main.legend(loc="upper right", fontsize=9, frameon=False)
ax_main.set_xlim(0.5, 6.0)

ax_res.axhline(0, color="k", ls="--")
ax_res.set_xlabel("Energy (keV)")
ax_res.set_ylabel("delchi")
ax_res.set_ylim(-2, 2)  # ±2σ 范围
plt.tight_layout()
plt.show()


def fit_and_plot_source() -> None:
    # ---------- 8. 加载源谱 ----------
    ui.load_pha("SRC_1", src_pi, use_errors=True)
    ui.subtract("SRC_1")
    ui.ignore_bad("SRC_1")
    ui.notice_id("SRC_1", 0.5, 6.0)

    src_backscal = ui.get_backscal("SRC_1")
    print(f"SRC_1 BACKSCAL = {src_backscal}")
    sky_backscal = bkg2_backscal
    scale_factor = src_backscal / sky_backscal
    print(f"scale factor (src/sky) = {scale_factor}")

    # ---------- 9. 构造源模型 ----------
    ui.create_model_component("scale1d", "sky_scale_src")
    sky_scale_src.c0 = scale_factor
    sky_scale_src.c0.freeze()

    # sky 模型参数冻结情况：
    #   LHB, MWhalo, CXB, SkyAbs, sky_scale: freeze
    #   InstLine_1/2: norm thaw
    #   SoftProton: PhoIndex & norm thaw  (在源拟合中允许继续变化)
    LHB.norm.freeze()
    MWhalo.kT.freeze()
    MWhalo.Abundanc.freeze()
    MWhalo.norm.freeze()
    CXB.norm.freeze()
    SkyAbs.nH.freeze()
    sky_scale.c0.freeze()
    InstLine_1.norm.thaw()
    InstLine_2.norm.thaw()
    # SoftProton index/norm 保持 thaw（不冻结）

    # 源的天体物理部分
    # 源的吸收模型（tbabs），允许在 1.0–3.0 ×10^22 cm^-2 范围内自由变化
    ui.create_model_component("xstbabs", "SrcAbs")
    SrcAbs.nH.set(val=2.0, min=1.0, max=3.0)  # 初始值2.0，自由范围1–3
    SrcAbs.nH.thaw()  # 允许自由拟合

    ui.create_model_component("xsvrnei", "SrcNEI")
    SrcNEI.norm = 1e-4

    SrcNEI.kT.set(val=0.8, min=0.3, max=3.0)
    SrcNEI.kT_init.set(val=5.0, min=0.3, max=10.0)
    SrcNEI.kT_init.freeze()

    SrcNEI.Mg.thaw()
    SrcNEI.Si.thaw()
    SrcNEI.S.thaw()
    SrcNEI.Ar.thaw()
    SrcNEI.Ca.thaw()

    ui.create_model_component("xsgaussian", "Src_InstLine_3")
    Src_InstLine_3.LineE.set(val=1.245, min=1.22, max=1.27)
    Src_InstLine_3.Sigma.set(val=0.0)
    Src_InstLine_3.Sigma.freeze()
    Src_InstLine_3.norm = 1e-5
    Src_InstLine_3.norm.thaw()

    # 源模型 = sky_scale_src*(sky_model) + SrcAbs*SrcNEI + Src_InstLine_3
    full_src_model = sky_scale_src * (
        LHB + SkyAbs * (MWhalo + CXB) + InstLine_1 + InstLine_2 + SoftProton
    ) + SrcAbs * SrcNEI + Src_InstLine_3
    ui.set_source("SRC_1", full_src_model)

    # ---------- 10. 源拟合 ----------
    print("\nFitting source (with scaled sky, SoftProton free, 0.5–6.0 keV)...")
    ui.set_stat("chi2gehrels")
    ui.set_method("levmar")
    ui.fit("SRC_1")
    ui.show_model("SRC_1")

    # ---------- 11. 绘图 ----------
    fig = plt.figure(figsize=(6.2, 6.5))
    gs = plt.GridSpec(4, 1, height_ratios=[3.2, 3.2, 3.2, 1.1], hspace=0)
    ax_main = fig.add_subplot(gs[0:3])
    ax_del = fig.add_subplot(gs[3], sharex=ax_main)

    fit_plot = ui.get_fit_plot("SRC_1")
    dataplot = fit_plot.dataplot
    modelplot = fit_plot.modelplot

    ax_main.errorbar(
        dataplot.x,
        dataplot.y,
        yerr=dataplot.yerr,
        fmt="o",
        ms=3.5,
        capsize=2,
        color="#1f77b4",
        label="Data",
    )

    model_edge = np.hstack([modelplot.xlo[0], modelplot.xhi])
    model_y_extended = np.hstack([modelplot.y[0], modelplot.y])
    ax_main.step(
        model_edge,
        model_y_extended,
        where="pre",
        color="#ff7f0e",
        linewidth=1.4,
        label="Total model",
    )

    pha_data = ui.get_data("SRC_1")
    mask = getattr(pha_data, "mask", None)

    component_specs = [
        ("SrcAbs*SrcNEI", "#d62728", "tbabs*vrnei"),
        ("Src_InstLine_3", "#2ca02c", "Src_InstLine_3"),
        ("sky_scale_src*LHB", "#9467bd", "Scaled LHB"),
        ("sky_scale_src*SkyAbs*MWhalo", "#17becf", "Scaled MW halo"),
        ("sky_scale_src*SkyAbs*CXB", "#bcbd22", "Scaled CXB (abs)"),
        ("sky_scale_src*SoftProton", "#7f7f7f", "Scaled Soft Proton"),
        ("sky_scale_src*InstLine_1", "#8c564b", "Scaled InstLine_1"),
        ("sky_scale_src*InstLine_2", "#8c564b", "Scaled InstLine_2"),
    ]

    def _plot_component(component_expr: str, color: str, label: str) -> None:
        try:
            comp_data = ui.calc_model_component("SRC_1", component_expr)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"Component {component_expr} evaluation failed: {exc}")
            return

        comp_y = getattr(comp_data, "y", None)
        if comp_y is None:
            return

        comp_y = np.asarray(comp_y, dtype=float)
        if mask is not None and comp_y.shape == mask.shape:
            comp_y = comp_y[mask]

        if comp_y.size == 0 or not np.any(np.isfinite(comp_y)):
            return

        comp_y = np.nan_to_num(comp_y, nan=0.0)
        if np.allclose(comp_y, 0.0):
            return

        comp_edge = np.hstack([modelplot.xlo[0], modelplot.xhi])
        comp_y_extended = np.hstack([comp_y[0], comp_y])
        ax_main.step(
            comp_edge,
            comp_y_extended,
            where="pre",
            color=color,
            linewidth=1.2,
            alpha=0.95,
            label=label,
        )

    for expr, color, label in component_specs:
        _plot_component(expr, color, label)

    ax_main.set_xscale("linear")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Counts / keV")
    ax_main.set_xlim(0.5, 6.0)

    handles, labels = ax_main.get_legend_handles_labels()
    # De-duplicate legend entries while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax_main.legend(unique_handles, unique_labels, loc="best", fontsize=8)

    delchi_plot = ui.get_delchi_plot("SRC_1")
    ax_del.errorbar(
        delchi_plot.x,
        delchi_plot.y,
        yerr=delchi_plot.yerr,
        fmt="o",
        ms=3,
        color="#9467bd",
        ecolor="#9467bd",
    )
    ax_del.axhline(0, color="k", ls="--", linewidth=0.8)
    ax_del.axhline(2, color="#d62728", ls=":", linewidth=0.8)
    ax_del.axhline(-2, color="#d62728", ls=":", linewidth=0.8)
    ax_del.set_xlabel("Energy (keV)")
    ax_del.set_ylabel("delchi")
    ax_del.set_ylim(-2.2, 2.2)

    plt.tight_layout()
    plt.show()

    ui.save(filename="SRC_1_fitting_results_softp.sherpa", clobber=True)
    print("\nDone. Results saved to SRC_1_fitting_results_softp.sherpa")


if __name__ == "__main__":
    fit_and_plot_source()
