#!/usr/bin/env python3

"""
第二部分：源拟合
===============================================================
功能：使用第一部分拟合的背景模型，进行源光谱拟合
注意：需要先运行第一部分获得背景拟合结果
===============================================================
"""

from __future__ import annotations

import os
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


def _fit_stage(stage_label, param_specs, data_ids):
    """拟合阶段"""
    thawed_names = _thaw_params(param_specs)
    print(f"\n--- {stage_label} ---")
    print("Thawed parameters: " + ", ".join(thawed_names))
    ui.fit(*data_ids)
    fit_results = ui.get_fit_results()
    print(f"Final fit statistic = {fit_results.statval}")
    print(f"Degrees of freedom = {fit_results.dof}")
    for data_id in data_ids:
        print(f"-- Model for {data_id} --")
        ui.show_model(data_id)
    return fit_results


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


def _calculate_parameter_errors(param_names, sigma=1.0):
    """使用conf计算关键参数的误差"""
    error_results = {}

    for param_name in param_names:
        par_expr = getattr(param_name, 'fullname', param_name)
        try:
            ui.conf(par_expr, sigma=sigma)
            conf_res = ui.get_conf_results()
            best_val = conf_res.parvals[0]
            lower_err = conf_res.parmins[0] if conf_res.parmins[0] is not None else 0
            upper_err = conf_res.parmaxes[0] if conf_res.parmaxes[0] is not None else 0

            error_results[par_expr] = {
                'value': best_val,
                'lower_error': lower_err,
                'upper_error': upper_err,
                'lower_bound': best_val + lower_err,
                'upper_bound': best_val + upper_err,
            }
        except Exception as e:
            print(f"Error calculating confidence interval for {par_expr}: {e}")
            try:
                par = ui.get_par(par_expr)
                error_results[par_expr] = {
                    'value': par.val,
                    'lower_error': 0,
                    'upper_error': 0,
                    'lower_bound': par.val,
                    'upper_bound': par.val,
                }
            except Exception:
                error_results[par_expr] = {
                    'value': 0,
                    'lower_error': 0,
                    'upper_error': 0,
                    'lower_bound': 0,
                    'upper_bound': 0,
                }

    return error_results


def _save_fitting_results(
    fit_results,
    error_results,
    filename='results_whole.txt',
    data_ids=None,
):
    """保存完整的拟合结果到文件"""
    with open(filename, 'w') as f:
        f.write("=== FITTING RESULTS ===\n")
        f.write(f"Final fit statistic: {fit_results.statval}\n")
        f.write(f"Degrees of freedom: {fit_results.dof}\n")
        f.write(f"Reduced statistic: {fit_results.rstat}\n")
        f.write(f"Data points: {fit_results.numpoints}\n")

        f.write("=== FLUX VALUES ===\n")
        total_flux = 0.0
        data_ids = data_ids or []
        for data_id in data_ids:
            try:
                flux = ui.calc_energy_flux(
                    lo=0.5, hi=6.0, id=data_id, bkg_id=None
                )
                total_flux += flux
                f.write(
                    f"{data_id} flux (0.5-6.0 keV): {flux:.2e} erg/cm²/s\n"
                )
            except Exception as e:
                f.write(f"Error calculating flux for {data_id}: {e}\n")
        if data_ids:
            f.write(f"Total flux (0.5-6.0 keV): {total_flux:.2e} erg/cm²/s\n\n")
        else:
            f.write("No detector flux calculated.\n\n")

        f.write("=== IMPORTANT PARAMETERS WITH ERRORS ===\n")
        f.write("Parameter\tValue\tLower_Offset\tUpper_Offset\n")

        important_params = [
            'SrcAbs.nH',
            'SrcNEI1.kT',
            'SrcNEI1.Tau',
            'SrcNEI1.Mg',
            'SrcNEI1.Si',
            'SrcNEI1.S',
            'SrcNEI1.Ar',
            'SrcNEI1.Ca',
        ]

        for param_name in important_params:
            if param_name in error_results:
                param_info = error_results[param_name]
                f.write(
                    f"{param_name}\t{param_info['value']:.3g}\t"
                    f"{param_info['lower_error']:.3g}\t{param_info['upper_error']:.3g}\n"
                )
            else:
                f.write(f"{param_name}\tNot available\t0\t0\n")

    print(f"完整的拟合结果已保存到 {filename}")


def fit_source_spectrum(
    region_name='whole',
    bkg2_backscal=None,
    error_sigma=1.0,
    detectors=(
        'mos2',
        'mos1',
    ),
):
    """源光谱拟合部分，可以多次运行

    Parameters:
    -----------
    region_name : str
        区域名称，用于构建文件路径和结果文件名
    bkg2_backscal : float, optional
        背景区域的backscal值
    error_sigma : float
        误差计算的sigma值
    detectors : tuple/list of str
        需要联合拟合的探测器名称，支持' mos1'和'mos2'
    """

    detector_templates = {
        'mos2': {
            'pha': f'/home/wjl/odf/sp_resolved/{region_name}/mos2S002-grp.pi',
            'id': 'SRC_MOS2',
            'color': '#1f77b4',
            'label': 'MOS2',
            'scale': f'sky_scale_src_{region_name}_mos2',
        },
        'mos1': {
            'pha': f'/home/wjl/odf/sp_resolved/{region_name}/mos1S001-grp.pi',
            'id': 'SRC_MOS1',
            'color': '#2ca02c',
            'label': 'MOS1',
            'scale': f'sky_scale_src_{region_name}_mos1',
        },
    }

    results_filename = f'results_{region_name}.txt'

    print(f"处理区域: {region_name}")
    print(f"结果文件: {results_filename}")

    if bkg2_backscal is None:
        try:
            bkg2_backscal = ui.get_backscal('BKG_2')
        except Exception:
            print("警告：无法获取BKG_2的BACKSCAL，请确保已运行第一部分")
            return

    active_detectors = []
    full_src_models = {}
    scale_components = {}
    data_ids = []

    for det_name in detectors:
        det_key = det_name.lower()
        if det_key not in detector_templates:
            print(f"跳过未知探测器 {det_name}")
            continue

        det_info = detector_templates[det_key]
        pha_path = det_info['pha']
        if not os.path.exists(pha_path):
            print(f"警告：{det_info['label']} 光谱 {pha_path} 不存在，跳过该探测器")
            continue

        data_id = det_info['id']
        print(f"加载 {det_info['label']} 数据: {pha_path}")
        ui.load_pha(data_id, pha_path, use_errors=True)
        ui.subtract(data_id)
        ui.ignore_bad(data_id)
        ui.notice_id(data_id, 0.5, 6.0)

        src_backscal = ui.get_backscal(data_id)
        sky_backscal = bkg2_backscal
        scale_factor = src_backscal / sky_backscal
        print(f"{data_id} BACKSCAL = {src_backscal}")
        print(f"scale factor ({data_id}/sky) = {scale_factor}")

        scale_component = ui.create_model_component('scale1d', det_info['scale'])
        scale_component.c0 = scale_factor
        scale_component.c0.freeze()

        active_detectors.append(
            {
                'name': det_info['label'],
                'id': data_id,
                'color': det_info['color'],
                'scale_component': scale_component,
            }
        )
        scale_components[data_id] = scale_component
        data_ids.append(data_id)

    if not data_ids:
        print("未能成功加载任何探测器，结束。")
        return

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

    ui.create_model_component('xstbabs', 'SrcAbs')
    SrcAbs.nH.set(val=2.0, min=1.0, max=4.0)
    SrcAbs.nH.freeze()

    ui.create_model_component('xsvnei', 'SrcNEI1')
    SrcNEI1.norm = 1e-3
    SrcNEI1.kT.set(val=0.9, min=0.3, max=3.0)
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
    SrcNEI2.norm = 1e-2
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

    background_expr = LHB + SkyAbs * (MWhalo + CXB) + InstLine_1 + InstLine_2 + SoftProton
    source_expr = SrcAbs * SrcNEI1 + Src_InstLine_3

    for det in active_detectors:
        data_id = det['id']
        full_model = det['scale_component'] * background_expr + source_expr
        ui.set_source(data_id, full_model)
        full_src_models[data_id] = full_model

    _reset_source_initials()

    ui.set_stat('chi2gehrels')
    ui.set_method('levmar')
    print("\nInitial fit with all relevant SrcAbs/SrcNEI parameters frozen...")
    ui.fit(*data_ids)
    for data_id in data_ids:
        ui.show_model(data_id)

    stage1_params = [
        (SrcAbs, ['nH']),
        (SrcNEI1, ['kT']),
        (SrcNEI1, ['Tau']),
        (SrcNEI1, ['Si']),
        (SrcNEI1, ['S']),
        (SrcNEI1, ['Ca']),
        (SrcNEI1, ['Ar', 'Mg']),
    ]

    stage2_params = [
        (SrcNEI1, ['Ar', 'Ca']),
    ]

    stage1_fit = _fit_stage(
        "Stage 1: thaw nH, SrcNEI1 (kT, Tau, Si, S, Ca, Ar, Mg)",
        stage1_params,
        data_ids,
    )
    stage2_fit = _fit_stage(
        "Stage 2: thaw SrcNEI1 Ar, Ca",
        stage2_params,
        data_ids,
    )

    print("\n--- Calculating parameter errors using conf ---")
    important_params = [
        SrcAbs.nH,
        SrcNEI1.kT,
        SrcNEI1.Tau,
        SrcNEI1.Mg,
        SrcNEI1.Si,
        SrcNEI1.S,
        SrcNEI1.Ar,
        SrcNEI1.Ca,
    ]
    error_results = _calculate_parameter_errors(important_params, sigma=error_sigma)
    _save_fitting_results(stage2_fit, error_results, filename=results_filename, data_ids=data_ids)

    fig = plt.figure(figsize=(8.0, 8.0))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.05)
    ax_main = fig.add_subplot(gs[0:3])
    ax_del = fig.add_subplot(gs[3], sharex=ax_main)

    fig.suptitle(f'{region_name.upper()}', fontsize=18, y=0.95)

    for det in active_detectors:
        data_id = det['id']
        color = det['color']
        label = det['name']
        plot = ui.get_fit_plot(data_id)
        dp = plot.dataplot
        mp = plot.modelplot
        x_err = [dp.x - dp.xlo, dp.xhi - dp.x]

        ax_main.errorbar(
            dp.x,
            dp.y,
            xerr=x_err,
            yerr=dp.yerr,
            fmt='none',
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
            label=rf'$\\mathrm{{{label}\ Data}}$',
            alpha=1.0,
        )

        ax_main.plot(
            dp.x,
            dp.y,
            'o',
            ms=0.0,
            color=color,
            alpha=0.9,
        )

        model_edge = np.hstack([mp.xlo[0], mp.xhi])
        model_y_extended = np.hstack([mp.y[0], mp.y])
        ax_main.step(
            model_edge,
            model_y_extended,
            where='pre',
            color=color,
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
            label=rf'$\\mathrm{{Full\ Model\ ({label})}}$',
        )

        delchi = ui.get_delchi_plot(data_id)
        x_err_delchi = [delchi.x - delchi.xlo, delchi.xhi - delchi.x]
        ax_del.errorbar(
            delchi.x,
            delchi.y,
            xerr=x_err_delchi,
            yerr=delchi.yerr,
            fmt='none',
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
            alpha=0.8,
        )
        ax_del.plot(
            delchi.x,
            delchi.y,
            'o',
            ms=0.0,
            color=color,
            alpha=0.9,
        )

    primary_id = data_ids[0]

    def _component_curve(component_expr, color, label=None, linestyle='-', alpha=0.8):
        """绘制单个模型成分（仅针对主探测器）"""
        if component_expr is None:
            return
        ui.set_source(primary_id, component_expr)
        comp_plot = ui.get_fit_plot(primary_id)
        comp_model = comp_plot.modelplot
        if comp_model.y.size == 0:
            ui.set_source(primary_id, full_src_models[primary_id])
            ui.get_fit_plot(primary_id)
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
        ui.set_source(primary_id, full_src_models[primary_id])
        ui.get_fit_plot(primary_id)

    _component_curve(SrcAbs * SrcNEI1, 'magenta', r'Absorbed vnei')
    if primary_id in scale_components:
        sky_component = scale_components[primary_id] * (LHB + SkyAbs * (MWhalo + CXB))
        _component_curve(
            sky_component,
            'black',
            r'$\\mathrm{Sky\ Background}$',
            linestyle='dashdot',
            alpha=0.6,
        )
        gaussian_components = [
            (scale_components[primary_id] * InstLine_1, r'$\\mathrm{Instrumental\ Lines}$'),
            (scale_components[primary_id] * InstLine_2, None),
        ]
    else:
        gaussian_components = []

    gaussian_components.append((Src_InstLine_3, None))

    for component_expr, lbl in gaussian_components:
        _component_curve(component_expr, '#7f7f7f', lbl, linestyle='--')

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_ylabel(r'$\\mathrm{Counts}\\,\\mathrm{s}^{-1}\\,\\mathrm{keV}^{-1}$')
    ax_main.set_xlim(0.5, 6.0)
    ax_main.set_ylim(1e-3, 3e-1)
    ax_main.legend(loc='best', fontsize=9, frameon=False)

    ax_del.axhline(0, color='k', linestyle='--', linewidth=0.9)
    ax_del.axhline(2, color='r', linestyle=':', linewidth=0.9)
    ax_del.axhline(-2, color='r', linestyle=':', linewidth=0.9)
    ax_del.set_xscale('log')
    ax_del.set_xlabel(r'$\\mathrm{Energy\ (keV)}$')
    ax_del.set_ylabel(r'$\\Delta\\chi$')
    ax_del.set_ylim(-2.2, 2.2)

    plt.savefig(f'vnei_{region_name}.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.show()

    for data_id, model in full_src_models.items():
        ui.set_source(data_id, model)


if __name__ == "__main__":
    fit_source_spectrum(region_name='CE')
    # fit_source_spectrum(region_name='whole')
    # fit_source_spectrum(region_name='N')
    # fit_source_spectrum(region_name='CN')
    # fit_source_spectrum(region_name='L')
