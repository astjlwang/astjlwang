#!/usr/bin/env python3
"""
Sherpa-based source spectrum fitting script.

This script performs the following steps:
1. Load and prepare the source spectrum.
2. Configure the full source model (scaled sky background + astrophysical source + instrumental line).
3. Fit the model to the source spectrum.
4. Plot the data, best-fit model, and selected model components (SrcAbs*SrcNEI and Src_InstLine_3).
5. Plot the delchi residuals with ±2σ reference lines.

The code assumes the required Sherpa data products (PHA files, background scaling,
and pre-defined sky model components) have been prepared earlier in the workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from sherpa.astro import ui


def fit_and_plot_source():
    # ---------- 8. 加载源谱 ----------
    ui.load_pha('SRC_1', src_pi, use_errors=True)
    ui.subtract('SRC_1')
    ui.ignore_bad('SRC_1')
    ui.notice_id('SRC_1', 0.5, 6.0)

    src_backscal = ui.get_backscal('SRC_1')
    print(f"SRC_1 BACKSCAL = {src_backscal}")
    sky_backscal = bkg2_backscal
    scale_factor = src_backscal / sky_backscal
    print(f"scale factor (src/sky) = {scale_factor}")

    # ---------- 9. 构造源模型 ----------
    ui.create_model_component('scale1d', 'sky_scale_src')
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
    ui.create_model_component('xstbabs', 'SrcAbs')
    SrcAbs.nH.set(val=2.0, min=1.0, max=3.0)  # 初始值2.0，自由范围1–3
    SrcAbs.nH.thaw()  # 允许自由拟合

    ui.create_model_component('xsvrnei', 'SrcNEI')
    SrcNEI.norm = 1e-4

    SrcNEI.kT.set(val=0.8, min=0.3, max=3.0)
    SrcNEI.kT_init.set(val=5.0, min=0.3, max=10.0)
    SrcNEI.KT_init.freeze()

    SrcNEI.Mg.thaw()
    SrcNEI.Si.thaw()
    SrcNEI.S.thaw()
    SrcNEI.Ar.thaw()
    SrcNEI.Ca.thaw()

    ui.create_model_component('xsgaussian', 'Src_InstLine_3')
    Src_InstLine_3.LineE.set(val=1.245, min=1.22, max=1.27)
    Src_InstLine_3.Sigma.set(val=0.0)
    Src_InstLine_3.Sigma.freeze()
    Src_InstLine_3.norm = 1e-5
    Src_InstLine_3.norm.thaw()

    # 源模型 = sky_scale_src*(sky_model) + SrcAbs*SrcNEI + Src_InstLine_3
    full_src_model = sky_scale_src * (
        LHB + SkyAbs * (MWhalo + CXB) + InstLine_1 + InstLine_2 + SoftProton
    ) + SrcAbs * SrcNEI + Src_InstLine_3
    ui.set_source('SRC_1', full_src_model)

    # ---------- 10. 源拟合 ----------
    print("\nFitting source (with scaled sky, SoftProton free, 0.5–6.0 keV)...")
    ui.set_stat('chi2gehrels')
    ui.set_method('levmar')
    ui.fit('SRC_1')
    ui.show_model('SRC_1')

    # ---------- 11. 绘图 ----------
    fig = plt.figure(figsize=(6, 6))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 1], hspace=0)
    ax_main = fig.add_subplot(gs[0:3])
    ax_del = fig.add_subplot(gs[3], sharex=ax_main)

    plot = ui.get_fit_plot('SRC_1')
    x = plot.dataplot.x
    y = plot.dataplot.y
    yerr = plot.dataplot.yerr
    ax_main.errorbar(
        x, y, yerr=yerr, fmt='o', ms=3.5, capsize=2, color='#1f77b4', label='SRC data'
    )

    m = plot.modelplot
    model_edge = np.hstack([m.xlo[0], m.xhi])
    model_y_extended = np.hstack([m.y[0], m.y])
    ax_main.step(
        model_edge,
        model_y_extended,
        where='pre',
        color='#ff7f0e',
        linewidth=1.2,
        label='Full model',
    )

    pha_data = ui.get_data('SRC_1')
    mask = getattr(pha_data, 'mask', None)

    def _component_curve(component, color, label):
        comp_obj = ui.calc_model_component('SRC_1', component)
        comp_vals = np.asarray(getattr(comp_obj, 'y', comp_obj))
        if mask is not None:
            comp_vals = comp_vals[mask]
        if comp_vals.size == 0:
            return
        comp_edge = np.hstack([m.xlo[0], m.xhi])
        comp_y_extended = np.hstack([comp_vals[0], comp_vals])
        ax_main.step(
            comp_edge,
            comp_y_extended,
            where='pre',
            color=color,
            linewidth=1.4,
            label=label,
        )

    _component_curve(SrcAbs * SrcNEI, '#d62728', 'SrcAbs * SrcNEI')
    _component_curve(Src_InstLine_3, '#2ca02c', 'Src_InstLine_3')

    ax_main.set_xscale('linear')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Counts / keV')
    ax_main.legend(loc='best', fontsize=9)
    ax_main.set_xlim(0.5, 6.0)

    delchi = ui.get_delchi_plot('SRC_1')
    ax_del.errorbar(
        delchi.x,
        delchi.y,
        yerr=delchi.yerr,
        fmt='o',
        ms=3,
        color='#9467bd',
        ecolor='#9467bd',
    )
    ax_del.axhline(0, color='k', ls='--', linewidth=0.8)
    ax_del.axhline(2, color='r', ls=':', linewidth=0.8)
    ax_del.axhline(-2, color='r', ls=':', linewidth=0.8)
    ax_del.set_xlabel('Energy (keV)')
    ax_del.set_ylabel('delchi')
    ax_del.set_ylim(-2.2, 2.2)

    plt.tight_layout()
    plt.show()

    ui.save(filename='SRC_1_fitting_results_softp.sherpa', clobber=True)
    print("\nDone. Results saved to SRC_1_fitting_results_softp.sherpa")


if __name__ == "__main__":
    fit_and_plot_source()
