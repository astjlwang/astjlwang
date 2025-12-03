#!/usr/bin/env python3

# 第二部分：源拟合
# ===============================================================
# 功能：使用第一部分拟合的背景模型，进行源光谱拟合
# 注意：需要先运行第一部分获得背景拟合结果
# ===============================================================

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import sherpa.astro.ui as ui

# 允许用户通过编号快速选择重要参数，编号与名称的映射如下。
IMPORTANT_PARAMETER_CHOICES = [
    'SrcAbs.nH',      # 1
    'SrcNEI1.kT',     # 2
    'SrcNEI1.Mg',     # 3
    'SrcNEI1.Tau',    # 4
    'SrcNEI1.Si',     # 5
    'SrcNEI1.S',      # 6
    'SrcNEI1.Ar',     # 7
    'SrcNEI1.Ca',     # 8
]
DEFAULT_IMPORTANT_PARAM_SELECTION = [3]  # 维持原始默认行为（仅 Mg）


def _print_param_menu():
    """打印支持的参数编号，便于命令行快速查阅。"""
    print("\n可选的重要参数（编号 -> 名称）：")
    for idx, name in enumerate(IMPORTANT_PARAMETER_CHOICES, 1):
        print(f"  {idx}: {name}")


def _normalize_param_selection(selection):
    """
    将用户输入（编号或名称）统一转换为 Sherpa 参数名称列表。

    支持的形式：
    - None 或空列表：采用 DEFAULT_IMPORTANT_PARAM_SELECTION
    - 单个 int/str：自动封装为列表
    - 列表：元素可以是编号（int 或数字字符串）或参数名称字符串
    """
    if not selection:
        selection = DEFAULT_IMPORTANT_PARAM_SELECTION

    if isinstance(selection, (int, str)):
        selection = [selection]

    normalized = []
    for item in selection:
        # 字符串且不是纯数字，则视为已经是参数名
        if isinstance(item, str) and not item.isdigit():
            normalized.append(item)
            continue

        # 其余情况统一按编号解析
        try:
            idx = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"无法解析参数选择：{item!r}")

        if idx < 1 or idx > len(IMPORTANT_PARAMETER_CHOICES):
            raise ValueError(
                f"参数编号 {idx} 超出可选范围 1-{len(IMPORTANT_PARAMETER_CHOICES)}"
            )
        normalized.append(IMPORTANT_PARAMETER_CHOICES[idx - 1])

    # 去除重复但保持顺序，避免重复计算
    deduped = []
    for name in normalized:
        if name not in deduped:
            deduped.append(name)
    return deduped


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
    fit_results = ui.get_fit_results()
    print(f"Final fit statistic = {fit_results.statval}")
    print(f"Degrees of freedom = {fit_results.dof}")
    ui.show_model('SRC_1')
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
        try:
            # 使用conf计算单个参数的误差
            ui.conf(param_name)
            conf_res = ui.get_conf_results()
            best_val = conf_res.parvals[0]
            lower_err = conf_res.parmins[0] if conf_res.parmins[0] is not None else 0
            upper_err = conf_res.parmaxes[0] if conf_res.parmaxes[0] is not None else 0

            error_results[param_name] = {
                'value': best_val,
                'lower_error': lower_err,
                'upper_error': upper_err,
                'lower_bound': best_val + lower_err,
                'upper_bound': best_val + upper_err,
            }
            print(f"Warning: Could not find {param_name} in conf results")

        except Exception as e:
            print(f"Error calculating confidence interval for {param_name}: {e}")
            # 如果conf失败，尝试使用get_par获取参数值（但没有误差）
            try:
                par = ui.get_par(param_name)
                error_results[param_name] = {
                    'value': par.val,
                    'lower_error': 0,
                    'upper_error': 0,
                    'lower_bound': par.val,
                    'upper_bound': par.val,
                }
            except Exception:
                error_results[param_name] = {
                    'value': 0,
                    'lower_error': 0,
                    'upper_error': 0,
                    'lower_bound': 0,
                    'upper_bound': 0,
                }
        print(error_results)

    return error_results


def _save_fitting_results(
    fit_results,
    error_results,
    selected_params,
    filename='SRC_1_fitting_results.txt',
):
    """保存完整的拟合结果到文件"""
    with open(filename, 'w') as f:
        # 写入拟合统计信息
        f.write("=== FITTING RESULTS ===\n")
        f.write(f"Final fit statistic: {fit_results.statval}\n")
        f.write(f"Degrees of freedom: {fit_results.dof}\n")
        f.write(f"Reduced statistic: {fit_results.rstat}\n")
        f.write(f"Data points: {fit_results.numpoints}\n")
        # 写入flux信息
        f.write("=== FLUX VALUES ===\n")
        try:
            # 计算总模型在0.5-6.0 keV的flux
            total_flux = ui.calc_energy_flux(lo=0.5, hi=6.0, id='SRC_1', bkg_id=None)
            f.write(f"Total model flux (0.5-6.0 keV): {total_flux:.2e} erg/cm²/s\n\n")
        except Exception as e:
            f.write(f"Error calculating flux: {e}\n\n")

        # 写入重要参数信息（带误差）
        f.write("=== IMPORTANT PARAMETERS WITH ERRORS ===\n")
        f.write("Parameter\tValue\tLower_Offset\tUpper_Offset\n")

        # 写入每个参数的值和误差
        for param_name in selected_params:
            if param_name in error_results:
                param_info = error_results[param_name]
                f.write(
                    f"{param_name}\t"
                    f"{param_info['value']:.3g}\t"
                    f"{param_info['lower_error']:.3g}\t"
                    f"{param_info['upper_error']:.3g}\n"
                )
            else:
                f.write(f"{param_name}\tNot available\t0\t0\n")

    print(f"完整的拟合结果已保存到 {filename}")


def fit_source_spectrum(
    bkg2_backscal=None,
    error_sigma=1.0,
    error_output='SRC_1_parameter_errors.csv',
    important_param_choices=None,
):
    """源光谱拟合部分，可以多次运行"""

    try:
        selected_param_names = _normalize_param_selection(important_param_choices)
    except ValueError as exc:
        print(f"重要参数选择错误：{exc}")
        _print_param_menu()
        return

    _print_param_menu()
    print("当前将计算/输出的参数：", ", ".join(selected_param_names))

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
    ) + SrcAbs * SrcNEI1 + Src_InstLine_3

    ui.set_source('SRC_1', full_src_model)

    _reset_source_initials()

    ui.set_stat('chi2gehrels')
    ui.set_method('levmar')
    print("\nInitial fit with all relevant SrcAbs/SrcNEI parameters frozen...")
    initial_fit = ui.fit('SRC_1')
    ui.show_model('SRC_1')

    # 多阶段拟合
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
    )
    stage2_fit = _fit_stage("Stage 2: thaw SrcNEI1 Ar, Ca", stage2_params)

    # 计算关键参数的误差
    print("\n--- Calculating parameter errors using conf ---")

    # 计算参数误差
    error_results = _calculate_parameter_errors(selected_param_names)

    # 使用第二次拟合的结果保存完整信息
    _save_fitting_results(stage2_fit, error_results, selected_param_names)

    # 绘制源拟合结果
    fig = plt.figure(figsize=(8.0, 8.0))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.05)
    ax_main = fig.add_subplot(gs[0:3])
    ax_del = fig.add_subplot(gs[3], sharex=ax_main)

    plot = ui.get_fit_plot('SRC_1')
    dp = plot.dataplot
    mp = plot.modelplot

    # 获取x error bar数据（能箱上下界）
    x_err = [dp.x - dp.xlo, dp.xhi - dp.x]

    # 主要改进：使用capsize=0去掉误差棒帽子，并添加x error bar
    # 先画误差棒（十字线，不带帽子）
    ax_main.errorbar(
        dp.x,
        dp.y,
        xerr=x_err,  # 添加x误差棒
        yerr=dp.yerr,
        fmt='none',  # 不要数据点
        ecolor='#1f77b4',
        elinewidth=1.2,
        capsize=0,  # 去掉帽子
        alpha=0.8,
    )

    # 再单独画数据点
    ax_main.plot(
        dp.x,
        dp.y,
        '-',
        ms=1.2,
        color='#1f77b4',
        label=r'$\mathrm{SRC\ Data}$',
        alpha=0.9,
    )

    model_edge = np.hstack([mp.xlo[0], mp.xhi])
    model_y_extended = np.hstack([mp.y[0], mp.y])
    ax_main.step(
        model_edge,
        model_y_extended,
        where='pre',
        color='#ff7f0e',
        linewidth=2.0,
        alpha=1.0,
        zorder=5,
        label=r'$\mathrm{Full\ Model}$',
    )

    def _component_curve(component_expr, color, label=None, linestyle='-', alpha=0.8):
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
    _component_curve(
        sky_scale_src * (LHB + SkyAbs * (MWhalo + CXB)),
        'black',
        r'$\mathrm{Sky\ Background}$',
        linestyle='dashdot',
        alpha=0.6,
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
    ax_main.set_ylim(1e-3, 3e-1)
    ax_main.legend(loc='best', fontsize=9, frameon=False)

    # 残差图：同样添加x error bar并去掉帽子
    delchi = ui.get_delchi_plot('SRC_1')
    x_err_delchi = [delchi.x - delchi.xlo, delchi.xhi - delchi.x]

    # 残差图的误差棒（十字线，不带帽子）
    ax_del.errorbar(
        delchi.x,
        delchi.y,
        xerr=x_err_delchi,  # 添加x误差棒
        yerr=delchi.yerr,
        fmt='none',  # 不要数据点
        ecolor='#9467bd',
        elinewidth=1.2,
        capsize=0,  # 去掉帽子
        alpha=0.8,
    )

    # 残差图的数据点
    ax_del.plot(
        delchi.x,
        delchi.y,
        'o',
        ms=0.0,
        color='#9467bd',
        alpha=0.9,
    )

    ax_del.axhline(0, color='k', linestyle='--', linewidth=0.9)
    ax_del.axhline(2, color='r', linestyle=':', linewidth=0.9)
    ax_del.axhline(-2, color='r', linestyle=':', linewidth=0.9)

    ax_del.set_xscale('log')
    ax_del.set_xlabel(r'$\mathrm{Energy\ (keV)}$')
    ax_del.set_ylabel(r'$\Delta\chi$')
    ax_del.set_ylim(-2.2, 2.2)
    plt.savefig('vrnei_only.png')
    plt.show()

    # 保存结果
    ui.set_source('SRC_1', full_src_model)


# 运行源拟合
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="运行源光谱拟合，并可通过编号快速选择需要输出的关键参数。",
    )
    parser.add_argument(
        'params',
        nargs='*',
        help=(
            "需要计算误差/输出的参数编号或名称，例如："
            "'3' 表示 SrcNEI1.Mg，'SrcNEI1.Si' 表示直接按名称指定。"
        ),
    )
    args = parser.parse_args()
    param_choices = args.params if args.params else None
    fit_source_spectrum(important_param_choices=param_choices)
