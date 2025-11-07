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


def _resolve_param(model, candidate_names):
    for name in candidate_names:
        for par in getattr(model, 'pars', []):
            if par.name.lower() == name.lower():
                return par
        if hasattr(model, name):
            attr = getattr(model, name)
            if hasattr(attr, 'thaw'):
                return attr
    raise AttributeError(
        f"Parameter {candidate_names} not found on model {getattr(model, 'name', model)}"
    )


def _set_par_value(par_expr, value, freeze=False):
    try:
        par = ui.get_par(par_expr)
    except Exception:
        return False

    val = value
    if par.min is not None:
        val = max(par.min, val)
    if par.max is not None:
        val = min(par.max, val)

    ui.set_par(par_expr, val)

    if freeze:
        ui.freeze(par_expr)
    else:
        ui.thaw(par_expr)

    return True


def _thaw_params(param_specs):
    thawed = []
    for model, names in param_specs:
        for par in names:
            resolved = _resolve_param(model, par if isinstance(par, (list, tuple)) else [par])
            resolved.thaw()
            thawed.append(resolved.name)
    return thawed


def _fit_stage(stage_label, param_specs):
    thawed_names = _thaw_params(param_specs)
    print(f"\n--- {stage_label} ---")
    print("Thawed parameters: " + ", ".join(thawed_names))
    ui.fit('SRC_1')
    ui.show_model('SRC_1')


def _reset_source_initials():
    _set_par_value('SrcAbs.nH', 2.0, freeze=True)

    _set_par_value('SrcNEI.norm', 1e-4, freeze=False)
    _set_par_value('SrcNEI.kT', 0.8, freeze=True)

    for tau_name in ['SrcNEI.Tau', 'SrcNEI.tau', 'SrcNEI.Tau_u', 'SrcNEI.Tau_l']:
        if _set_par_value(tau_name, 3e13, freeze=True):
            break

    for par_name in ['Mg', 'Si', 'S', 'Ar', 'Ca']:
        _set_par_value(f'SrcNEI.{par_name}', 1.0, freeze=True)

    _set_par_value('Src_InstLine_3.norm', 1e-5, freeze=False)
    _set_par_value('Src_InstLine_3.LineE', 1.245, freeze=False)
    _set_par_value('InstLine_1.norm', 1e-5, freeze=False)
    _set_par_value('InstLine_2.norm', 1e-5, freeze=False)


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
    SrcAbs.nH.freeze()

    ui.create_model_component('xsvrnei', 'SrcNEI')
    SrcNEI.norm = 1e-4

    SrcNEI.kT.set(val=0.8, min=0.3, max=3.0)
    SrcNEI.kT_init.set(val=5.0, min=0.3, max=10.0)
    SrcNEI.KT_init.freeze()

    SrcNEI.kT.freeze()
    try:
        SrcNEI.Tau.freeze()
    except AttributeError:
        _resolve_param(SrcNEI, ['Tau', 'tau', 'Tau_u', 'Tau_l']).freeze()

    for par_name in ['Mg', 'Si', 'S', 'Ar', 'Ca']:
        if hasattr(SrcNEI, par_name):
            getattr(SrcNEI, par_name).freeze()

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

    _reset_source_initials()

    # ---------- 10. 源拟合 ----------
    ui.set_stat('chi2gehrels')
    ui.set_method('levmar')
    print("\nInitial fit with all relevant SrcAbs/SrcNEI parameters frozen...")
    ui.fit('SRC_1')
    ui.show_model('SRC_1')

    stage1_params = [
        (SrcAbs, ['nH']),
        (SrcNEI, ['kT']),
        (SrcNEI, [['Tau', 'tau', 'Tau_u', 'Tau_l']]),
        (SrcNEI, ['S']),
        (SrcNEI, ['Si']),
    ]

    stage2_params = [
        (SrcNEI, ['Ar']),
        (SrcNEI, ['Ca']),
    ]

    _fit_stage("Stage 1: thaw nH, kT, Tau, S, Si", stage1_params)
    _fit_stage("Stage 2: thaw Ar, Ca", stage2_params)

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

    def _component_curve(component_expr, color, label=None, linestyle='-'):
        restored = False
        try:
            comp_plot = ui.get_model_component_plot('SRC_1', component_expr)
        except Exception:
            ui.set_source('SRC_1', component_expr)
            comp_plot = ui.get_model_plot('SRC_1')
            restored = True
        comp_edge = np.hstack([comp_plot.xlo[0], comp_plot.xhi])
        comp_y_extended = np.hstack([comp_plot.y[0], comp_plot.y])
        ax_main.step(
            comp_edge,
            comp_y_extended,
            where='pre',
            color=color,
            linewidth=1.4,
            linestyle=linestyle,
            label=label,
        )
        if restored:
            ui.set_source('SRC_1', full_src_model)
            ui.get_fit_plot('SRC_1')

    _component_curve(SrcAbs * SrcNEI, 'purple', 'SrcAbs * SrcNEI')

    gaussian_components = [
        (sky_scale_src * InstLine_1, 'Instrumental lines'),
        (sky_scale_src * InstLine_2, None),
        (Src_InstLine_3, None),
    ]

    for comp_expr, lbl in gaussian_components:
        _component_curve(comp_expr, '#7f7f7f', lbl, linestyle='--')

    ax_main.set_xscale('linear')
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Counts / keV')
    ax_main.set_xlim(0.5, 6.0)
    ax_main.legend(loc='best', fontsize=9)

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
