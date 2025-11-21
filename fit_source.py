#!/usr/bin/env python3

# 第二部分：源拟合
# ===============================================================
# 功能：使用第一部分拟合的背景模型，进行源光谱拟合
# 注意：需要先运行第一部分获得背景拟合结果
# ===============================================================

import csv
import numpy as np
import matplotlib.pyplot as plt
import sherpa.astro.ui as ui


def _resolve_param(model, candidate_names):
    """解析参数名称"""
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
    """设置参数值"""
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
    """解冻参数"""
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
    """拟合阶段"""
    thawed_names = _thaw_params(param_specs)
    print(f"\n--- {stage_label} ---")
    print("Thawed parameters: " + ", ".join(thawed_names))
    ui.fit('SRC_1')
    ui.show_model('SRC_1')


def _reset_source_initials():
    """重置源模型初始值"""
    _set_par_value('SrcAbs.nH', 2.0, freeze=True)

    _set_par_value('SrcNEI1.norm', 1e-4, freeze=False)
    _set_par_value('SrcNEI1.kT', 0.8, freeze=True)

    for tau_name in ['SrcNEI1.Tau', 'SrcNEI1.tau', 'SrcNEI1.Tau_u', 'SrcNEI1.Tau_l']:
        if _set_par_value(tau_name, 3e11, freeze=True):
            break

    for par_name in ['Mg', 'Si', 'S', 'Ar', 'Ca']:
        _set_par_value(f'SrcNEI1.{par_name}', 1.0, freeze=True)
    _set_par_value('SrcNEI1.Ca', 10.0, freeze=True)
    _set_par_value('SrcNEI1.Mg', 1.2, freeze=True)

    _set_par_value('SrcNEI2.norm', 1e-5, freeze=False)
    _set_par_value('SrcNEI2.kT', 0.3, freeze=True)
    for tau_name in ['SrcNEI2.Tau', 'SrcNEI2.tau', 'SrcNEI2.Tau_u', 'SrcNEI2.Tau_l']:
        if _set_par_value(tau_name, 4e12, freeze=True):
            break

    _set_par_value('Src_InstLine_3.norm', 1e-5, freeze=False)
    _set_par_value('Src_InstLine_3.LineE', 1.245, freeze=False)
    _set_par_value('InstLine_1.norm', 1e-5, freeze=False)
    _set_par_value('InstLine_2.norm', 1e-5, freeze=False)


def _calculate_parameter_errors(param_names, sigma=1.0, outfile='SRC_1_parameter_errors.csv'):
    """使用Sherpa conf计算关键参数误差并保存"""
    results = []

    if not param_names:
        print("未提供需要计算误差的参数列表。")
        return results

    for name in param_names:
        try:
            par = ui.get_par(name)
        except Exception as exc:
            print(f"警告：无法解析参数 {name} ({exc})，跳过。")
            continue

        # 确保参数处于可变化状态，便于误差评估
        par.thaw()

        best_val = par.val
        lower = None
        upper = None

        try:
            ui.conf(name, sigma=sigma)
            conf_res = ui.get_conf_results()
            if conf_res is not None and conf_res.parnames:
                try:
                    idx = conf_res.parnames.index(name)
                    best_val = conf_res.parvals[idx]
                    lower = conf_res.parmins[idx]
                    upper = conf_res.parmaxes[idx]
                except ValueError:
                    print(f"警告：Sherpa返回的误差结果中不包含参数 {name}。")
        except Exception as exc:
            print(f"警告：参数 {name} 误差计算失败（{exc}）。将lower/upper_error置为None。")

        results.append(
            {
                'parameter': name,
                'value': best_val,
                'lower_error': lower,
                'upper_error': upper,
            }
        )

    if results:
        fieldnames = ['parameter', 'value', 'lower_error', 'upper_error']
        with open(outfile, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n关键参数误差结果已保存至 {outfile}")
        for record in results:
            print(
                f"{record['parameter']}: value={record['value']}, "
                f"lower_error={record['lower_error']}, upper_error={record['upper_error']}"
            )
    else:
        print("\n未能计算任何参数的误差。")

    return results


def fit_source_spectrum(
    bkg2_backscal=None,
    error_sigma=1.0,
    error_output='SRC_1_parameter_errors.csv',
):
    """源光谱拟合部分，可以多次运行"""

    # 如果未提供bkg2_backscal，尝试从全局获取
    if bkg2_backscal is None:
        try:
            bkg2_backscal = ui.get_backscal('BKG_2')
        except Exception:
            print("警告：无法获取BKG_2的BACKSCAL，请确保已运行第一部分")
            return

    src_pi = '/home/wjl/odf/sp_resolved/sus/mos2S002-grp.pi'

    ui.load_pha('SRC_1', src_pi, use_errors=True)
    ui.subtract('SRC_1')
    ui.ignore_bad('SRC_1')
    ui.notice_id('SRC_1', 0.5, 6.0)

    src_backscal = ui.get_backscal('SRC_1')
    print(f"SRC_1 BACKSCAL = {src_backscal}")
    sky_backscal = bkg2_backscal
    scale_factor = src_backscal / sky_backscal
    print(f"scale factor (src/sky) = {scale_factor}")

    # 冻结背景模型参数
    LHB.norm.freeze()
    MWhalo.kT.freeze()
    MWhalo.Abundanc.freeze()
    MWhalo.norm.freeze()
    CXB.norm.freeze()
    SkyAbs.nH.freeze()
    sky_scale.c0.freeze()
    InstLine_1.norm.thaw()
    InstLine_2.norm.thaw()

    # 创建源模型组件
    ui.create_model_component('scale1d', 'sky_scale_src')
    sky_scale_src.c0 = scale_factor
    sky_scale_src.c0.freeze()

    ui.create_model_component('xstbabs', 'SrcAbs')
    SrcAbs.nH.set(val=2.0, min=1.0, max=4.0)
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

    SrcNEI1.Ca.set(val=10.0, min=0.5, max=10000.0)
    SrcNEI1.Mg.set(val=1.0, min=0.5, max=10000.0)
    SrcNEI1.Si.set(val=1.0, min=0.5, max=10000.0)
    SrcNEI1.S.set(val=1.0, min=0.5, max=10000.0)

    ui.create_model_component('xsnei', 'SrcNEI2')
    SrcNEI2.norm = 1e-3
    SrcNEI2.kT.set(val=0.3, min=0.1, max=0.7)
    SrcNEI2.kT.freeze()
    SrcNEI2.Tau.set(val=4e12, min=1e11, max=5e13)
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

    # 完整源模型
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

    # 多阶段拟合
    stage1_params = [
        (SrcAbs, ['nH']),
        (SrcNEI1, ['kT']),
        (SrcNEI1, ['Tau']),
        (SrcNEI1, ['Si']),
        (SrcNEI1, ['S']),
        (SrcNEI1, ['Ca']),
        (SrcNEI2, ['kT']),
        (SrcNEI2, ['Tau']),
        (SrcNEI1, ['Ar', 'Mg']),
    ]

    stage2_params = [
        (SrcNEI1, ['Ar', 'Ca']),
    ]

    _fit_stage(
        "Stage 1: thaw nH, SrcNEI1 (kT, Tau, Si, S, Ca, Ar, Mg), SrcNEI2 (kT, Tau)",
        stage1_params,
    )
    _fit_stage("Stage 2: thaw SrcNEI1 Ar, Ca", stage2_params)

    # 拟合完成后，计算关键参数误差
    key_params = [
        'SrcAbs.nH',
        'SrcNEI1.norm',
        'SrcNEI1.kT',
        'SrcNEI1.Tau',
        'SrcNEI1.Mg',
        'SrcNEI1.Si',
        'SrcNEI1.S',
        'SrcNEI1.Ar',
        'SrcNEI1.Ca',
        'SrcNEI2.norm',
        'SrcNEI2.kT',
        'SrcNEI2.Tau',
    ]
    _calculate_parameter_errors(
        param_names=key_params,
        sigma=error_sigma,
        outfile=error_output,
    )

    # 绘制源拟合结果
    fig = plt.figure(figsize=(8.0, 8.0))
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
        ms=2.0,
        capsize=2.0,
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
        alpha=1.0,
        label=r'$\mathrm{Full\ Model}$',
    )

    def _component_curve(component_expr, color, label=None, linestyle='-', alpha=0.7):
        """绘制单个模型成分"""
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
            linewidth=1.5,
            label=label,
            alpha=alpha,
            linestyle=linestyle,
        )
        ui.set_source('SRC_1', full_src_model)
        ui.get_fit_plot('SRC_1')

    _component_curve(SrcAbs * SrcNEI1, 'magenta', r'Absorbed vrnei')
    _component_curve(SrcAbs * SrcNEI2, 'green', r'Absorbed nei')
    _component_curve(
        sky_scale_src * (LHB + SkyAbs * (MWhalo + CXB)),
        'black',
        r'$\mathrm{Sky\ Background}$',
        linestyle='--',
        alpha=0.5,
    )

    gaussian_components = [
        (sky_scale_src * InstLine_1, r'$\mathrm{Instrumental\ Lines}$'),
        (sky_scale_src * InstLine_2, None),
        (Src_InstLine_3, None),
    ]

    for component_expr, lbl in gaussian_components:
        _component_curve(component_expr, '#7f7f7f', lbl, linestyle='--')

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
        ms=2.0,
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
    plt.savefig('vrnei_nei.png')
    plt.show()

    # 保存结果
    ui.set_source('SRC_1', full_src_model)
    ui.save(filename='SRC_1_fitting_results_softp.sherpa', clobber=True)
    print("\nDone. Results saved to SRC_1_fitting_results_softp.sherpa")


# 运行源拟合
if __name__ == "__main__":
    # 如果已经运行了第一部分，可以直接调用
    # 如果需要手动指定bkg2_backscal，可以传入参数
    fit_source_spectrum()

