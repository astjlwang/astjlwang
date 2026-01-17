#!/usr/bin/env python3

# fit_sky_and_src.py (修正版)
# ===============================================================
# 在原基础上:
#   1. 拟合能段固定 0.5–6.0 keV
#   2. skyback 模型新增 SoftProton: powerlaw(index∈[0.1,1.0])
#   3. sky 拟合完绘制两条 MOS 背景数据+模型+残差
#   4. 源拟合时沿用同一个 SoftProton, 不冻结（index+norm自由）
#   5. 绘图统一为 LaTeX 样式，主图/残差布局一致，横坐标使用 log 刻度
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import sherpa.astro.ui as ui

# ---------- 1. 文件路径 ----------
bkg1_pi = '/home/wjl/odf/skyback/mos1S001-grp.pi'
bkg2_pi = '/home/wjl/odf/skyback/mos2S002-grp.pi'
src_pi = '/home/wjl/odf/sp_resolved/sus/mos2S002-grp.pi'

# ---------- 2. 加载背景谱 ----------
ui.load_pha('BKG_1', bkg1_pi, use_errors=True)
ui.load_pha('BKG_2', bkg2_pi, use_errors=True)
ui.subtract('BKG_1')
ui.subtract('BKG_2')

bkg1_backscal = ui.get_backscal('BKG_1')
bkg2_backscal = ui.get_backscal('BKG_2')
print(f"BKG_1 BACKSCAL = {bkg1_backscal}")
print(f"BKG_2 BACKSCAL = {bkg2_backscal}")

# ---------- 3. sky background 模型 ----------
# Local Hot Bubble
ui.create_model_component('xsapec', 'LHB')
LHB.kT = 0.1
LHB.kT.freeze()
LHB.Abundanc = 1.0
LHB.Abundanc.freeze()
LHB.norm = 3e-6
LHB.norm.thaw()

# MW halo / hot bubble
ui.create_model_component('xsapec', 'MWhalo')
MWhalo.kT.set(val=0.3, min=0.1, max=0.6)
MWhalo.kT.thaw()
MWhalo.Abundanc = 1.0
MWhalo.Abundanc.freeze()
MWhalo.norm.set(val=3e-6, min=1e-7, max=1e-3)
MWhalo.norm.thaw()

# CXB: 冻结 index
ui.create_model_component('xspowerlaw', 'CXB')
CXB.PhoIndex = 1.46
CXB.PhoIndex.freeze()
CXB.norm = 4e-7
CXB.norm.thaw()

# Galactic absorption
ui.create_model_component('xstbabs', 'SkyAbs')
SkyAbs.nH.set(val=0.5, min=0.05, max=2.0)
SkyAbs.nH.thaw()

# Instrumental lines
ui.create_model_component('xsgaussian', 'InstLine_1')
InstLine_1.LineE.set(val=1.49, min=1.47, max=1.51)
InstLine_1.Sigma.set(val=0.0)
InstLine_1.Sigma.freeze()
InstLine_1.norm = 1e-5
InstLine_1.norm.thaw()

ui.create_model_component('xsgaussian', 'InstLine_2')
InstLine_2.LineE.set(val=1.755, min=1.73, max=1.78)
InstLine_2.Sigma.set(val=0.0)
InstLine_2.Sigma.freeze()
InstLine_2.norm = 1e-5
InstLine_2.norm.thaw()

# Soft proton component
ui.create_model_component('xspowerlaw', 'SoftProton')
SoftProton.PhoIndex.set(val=0.4, min=0.1, max=1.0)
SoftProton.norm.set(val=1e-5, min=0.0, max=1e-2)

# ---------- 4. sky 模型定义 ----------
ui.create_model_component('scale1d', 'sky_scale')
sky_scale.c0 = 1.0
sky_scale.c0.freeze()

sky_model = (
    sky_scale * (LHB + SkyAbs * (MWhalo + CXB))
    + InstLine_1
    + InstLine_2
    + SoftProton
)

ui.set_source('BKG_1', sky_model)
ui.set_source('BKG_2', sky_model)

# ---------- 5. 拟合设置 ----------
ui.set_stat('chi2gehrels')
ui.set_method('levmar')

for bid in ['BKG_1', 'BKG_2']:
    ui.ignore_bad(bid)
    ui.notice_id(bid, 0.5, 6.0)

# ---------- 6. joint fit ----------
print("\nFitting sky backgrounds (joint, 0.5–6.0 keV, with SoftProton)...")
ui.fit('BKG_1', 'BKG_2')
ui.show_model('BKG_1')


def _plot_sky_background_results():
    fig = plt.figure(figsize=(6, 8.0))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.05)
    ax_main = fig.add_subplot(gs[0:3])
    ax_res = fig.add_subplot(gs[3], sharex=ax_main)

    data_colors = ['#1f77b4', '#2ca02c']
    labels = ['MOS1', 'MOS2']

    for idx, bid in enumerate(['BKG_1', 'BKG_2']):
        fp = ui.get_fit_plot(bid)
        dp = fp.dataplot
        mp = fp.modelplot

        ax_main.errorbar(
            dp.x,
            dp.y,
            yerr=dp.yerr,
            fmt='o',
            ms=1.2,
            capsize=1.2,
            color=data_colors[idx],
            label=rf'$\mathrm{{{labels[idx]}\ Data}}$',
        )

        x_edge = np.hstack([mp.xlo[0], mp.xhi])
        y_model = np.hstack([mp.y[0], mp.y])
        ax_main.step(
            x_edge,
            y_model,
            where='pre',
            color='#ff7f0e',
            linewidth=2.0,
            label=r'$\mathrm{Total\ Model}$' if idx == 0 else None,
        )

        model_energy = mp.x

        component_specs = {
            'LHB': ('purple', r'$\mathrm{LHB}$'),
            'SkyAbs*MWhalo': ('green', r'$\mathrm{MW\ Halo}$'),
            'SkyAbs*CXB': ('red', r'$\mathrm{CXB}$'),
            'SoftProton': ('black', r'$\mathrm{Soft\ Proton}$'),
            'InstLine_1': ('#7f7f7f', r'$\mathrm{Inst.\ Lines}$'),
            'InstLine_2': ('#7f7f7f', None),
        }

        for comp_name, (color, label) in component_specs.items():
            try:
                comp_y = ui.eval_model_component(bid, comp_name)
                ax_main.plot(
                    model_energy,
                    comp_y,
                    linestyle='--',
                    linewidth=1.0,
                    color=color,
                    alpha=0.7,
                    label=label if idx == 0 else None,
                )
            except Exception as exc:
                print(f"Component {comp_name} evaluation failed for {bid}: {exc}")

        delchi = ui.get_delchi_plot(bid)
        ax_res.errorbar(
            delchi.x,
            delchi.y,
            yerr=delchi.yerr,
            fmt='o',
            ms=1.8,
            color=data_colors[idx],
            ecolor=data_colors[idx],
            alpha=0.9,
        )

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlim(0.5, 6.0)
    ax_main.set_ylabel(r'$\mathrm{Counts}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$')
    ax_main.legend(loc='best', fontsize=9, frameon=False)

    ax_res.set_xscale('log')
    ax_res.set_xlabel(r'$\mathrm{Energy\ (keV)}$')
    ax_res.set_ylabel(r'$\Delta\chi$')
    ax_res.set_ylim(-2.2, 2.2)
    ax_res.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax_res.axhline(2, color='r', linestyle=':', linewidth=0.8)
    ax_res.axhline(-2, color='r', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.show()


_plot_sky_background_results()


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
            resolved = _resolve_param(
                model, par if isinstance(par, (list, tuple)) else [par]
            )
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

    _set_par_value('SrcNEI1.norm', 1e-4, freeze=False)
    _set_par_value('SrcNEI1.kT', 0.8, freeze=True)

    for tau_name in ['SrcNEI1.Tau', 'SrcNEI1.tau', 'SrcNEI1.Tau_u', 'SrcNEI1.Tau_l']:
        if _set_par_value(tau_name, 3e11, freeze=True):
            break

    for par_name in ['Mg', 'Si', 'S', 'Ar', 'Ca']:
        _set_par_value(f'SrcNEI1.{par_name}', 1.0, freeze=True)

    _set_par_value('SrcNEI2.norm', 1e-5, freeze=False)
    _set_par_value('SrcNEI2.kT', 0.3, freeze=True)
    for tau_name in ['SrcNEI2.Tau', 'SrcNEI2.tau', 'SrcNEI2.Tau_u', 'SrcNEI2.Tau_l']:
        if _set_par_value(tau_name, 4e12, freeze=True):
            break

    _set_par_value('Src_InstLine_3.norm', 1e-5, freeze=False)
    _set_par_value('Src_InstLine_3.LineE', 1.245, freeze=False)
    _set_par_value('InstLine_1.norm', 1e-5, freeze=False)
    _set_par_value('InstLine_2.norm', 1e-5, freeze=False)


def fit_and_plot_source():
    ui.load_pha('SRC_1', src_pi, use_errors=True)
    ui.subtract('SRC_1')
    ui.ignore_bad('SRC_1')
    ui.notice_id('SRC_1', 0.5, 6.0)

    src_backscal = ui.get_backscal('SRC_1')
    print(f"SRC_1 BACKSCAL = {src_backscal}")
    sky_backscal = bkg2_backscal
    scale_factor = src_backscal / sky_backscal
    print(f"scale factor (src/sky) = {scale_factor}")

    ui.create_model_component('scale1d', 'sky_scale_src')
    sky_scale_src.c0 = scale_factor
    sky_scale_src.c0.freeze()

    LHB.norm.freeze()
    MWhalo.kT.freeze()
    MWhalo.Abundanc.freeze()
    MWhalo.norm.freeze()
    CXB.norm.freeze()
    SkyAbs.nH.freeze()
    sky_scale.c0.freeze()
    InstLine_1.norm.thaw()
    InstLine_2.norm.thaw()

    ui.create_model_component('xstbabs', 'SrcAbs')
    SrcAbs.nH.set(val=2.0, min=1.0, max=3.0)
    SrcAbs.nH.freeze()

    ui.create_model_component('xsvrnei', 'SrcNEI1')
    SrcNEI1.norm = 1e-4
    SrcNEI1.kT.set(val=0.8, min=0.3, max=3.0)
    if hasattr(SrcNEI1, 'kT_init'):
        SrcNEI1.kT_init.set(val=5.0, min=0.3, max=10.0)
        SrcNEI1.kT_init.freeze()
    SrcNEI1.kT.freeze()
    SrcNEI1.Tau.set(val=4e11, min=1e11, max=4e13)
    for par_name in ['Mg', 'Si', 'S', 'Ar', 'Ca']:
        if hasattr(SrcNEI1, par_name):
            getattr(SrcNEI1, par_name).freeze()

    ui.create_model_component('xsnei', 'SrcNEI2')
    SrcNEI2.norm = 1e-5
    SrcNEI2.kT.set(val=0.3, min=0.1, max=0.7)
    SrcNEI2.kT.freeze()
    SrcNEI2.Tau.set(val=4e12, min=1e11, max=4e13)
    SrcNEI2.Tau.freeze()
    for attr_name in ['Abundanc', 'Redshift', 'Switch']:
        if hasattr(SrcNEI2, attr_name):
            getattr(SrcNEI2, attr_name).freeze()

    ui.create_model_component('xsgaussian', 'Src_InstLine_3')
    Src_InstLine_3.LineE.set(val=1.245, min=1.22, max=1.27)
    Src_InstLine_3.Sigma.set(val=0.0)
    Src_InstLine_3.Sigma.freeze()
    Src_InstLine_3.norm = 1e-5
    Src_InstLine_3.norm.thaw()

    full_src_model = sky_scale_src * (
        LHB + SkyAbs * (MWhalo + CXB) + InstLine_1 + InstLine_2 + SoftProton
    ) + SrcAbs * (SrcNEI1 + SrcNEI2) + Src_InstLine_3

    ui.set_source('SRC_1', full_src_model)

    _reset_source_initials()

    ui.set_stat('chi2gehrels')
    ui.set_method('levmar')
    print("\nInitial fit with all relevant SrcAbs/SrcNEI parameters frozen...")
    ui.fit('SRC_1')
    ui.show_model('SRC_1')

    stage1_params = [
        (SrcAbs, ['nH']),
        (SrcNEI1, ['kT']),
        (SrcNEI1, ['Tau']),
        (SrcNEI1, ['Si']),
        (SrcNEI1, ['S']),
        (SrcNEI2, ['kT']),
        (SrcNEI2, ['Tau']),
    ]

    stage2_params = [
        (SrcNEI1, ['Ar', 'Ca']),
    ]

    _fit_stage(
        "Stage 1: thaw nH, SrcNEI1 (kT, Tau, Si, S), SrcNEI2 (kT, Tau)",
        stage1_params,
    )
    _fit_stage("Stage 2: thaw SrcNEI1 Ar, Ca", stage2_params)

    fig = plt.figure(figsize=(6, 8.0))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.05)
    ax_main = fig.add_subplot(gs[0:3])
    ax_del = fig.add_subplot(gs[3], sharex=ax_main)

    plot = ui.get_fit_plot('SRC_1')
    x = plot.dataplot.x
    y = plot.dataplot.y
    yerr = plot.dataplot.yerr
    ax_main.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o',
        ms=1.0,
        capsize=1.0,
        color='#1f77b4',
        label=r'$\mathrm{SRC\ Data}$',
    )

    model_plot = plot.modelplot
    model_edge = np.hstack([model_plot.xlo[0], model_plot.xhi])
    model_y_extended = np.hstack([model_plot.y[0], model_plot.y])
    ax_main.step(
        model_edge,
        model_y_extended,
        where='pre',
        color='#ff7f0e',
        linewidth=2.5,
        label=r'$\mathrm{Full\ Model}$',
    )

    def _component_curve(component_expr, color, label=None):
        ui.set_source('SRC_1', component_expr)
        comp_plot = ui.get_fit_plot('SRC_1')
        comp_model = comp_plot.modelplot
        if comp_model.y.size == 0:
            ui.set_source('SRC_1', full_src_model)
            ui.get_fit_plot('SRC_1')
            return
        comp_edge = np.hstack([comp_model.xlo[0], comp_model.xhi])
        comp_y_extended = np.hstack([comp_model.y[0], comp_model.y])
        ax_main.step(
            comp_edge,
            comp_y_extended,
            where='pre',
            color=color,
            linewidth=1.0,
            label=label,
            alpha=0.6,
        )
        ui.set_source('SRC_1', full_src_model)
        ui.get_fit_plot('SRC_1')

    _component_curve(SrcAbs * SrcNEI1, 'purple', r'$\mathrm{tbabs}\times\mathrm{Vrnei}$')
    _component_curve(SrcAbs * SrcNEI2, 'teal', r'$\mathrm{tbabs}\times\mathrm{nei}$')

    gaussian_components = [
        (sky_scale_src * InstLine_1, r'$\mathrm{Instrumental\ Lines}$'),
        (sky_scale_src * InstLine_2, None),
        (Src_InstLine_3, None),
    ]

    for component_expr, lbl in gaussian_components:
        _component_curve(component_expr, '#7f7f7f', lbl)

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_ylabel(r'$\mathrm{Counts}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$')
    ax_main.set_xlim(0.5, 6.0)
    ax_main.set_ylim(1e-3, 0.3)
    ax_main.legend(loc='best', fontsize=9, frameon=False)

    delchi = ui.get_delchi_plot('SRC_1')
    ax_del.errorbar(
        delchi.x,
        delchi.y,
        yerr=delchi.yerr,
        fmt='o',
        ms=1.2,
        color='#9467bd',
        ecolor='#9467bd',
    )
    ax_del.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax_del.axhline(2, color='r', linestyle=':', linewidth=0.8)
    ax_del.axhline(-2, color='r', linestyle=':', linewidth=0.8)
    ax_del.set_xscale('log')
    ax_del.set_xlabel(r'$\mathrm{Energy\ (keV)}$')
    ax_del.set_ylabel(r'$\Delta\chi$')
    ax_del.set_ylim(-2.2, 2.2)

    plt.tight_layout()
    plt.show()

    ui.set_source('SRC_1', full_src_model)
    ui.save(filename='SRC_1_fitting_results_softp.sherpa', clobber=True)
    print("\nDone. Results saved to SRC_1_fitting_results_softp.sherpa")


if __name__ == "__main__":
    fit_and_plot_source()
