import numpy as np
import matplotlib.pyplot as plt
from sherpa.astro import ui


def fit_and_plot_source():
    """
    Run the spectral fit workflow for SRC_1 and produce a plot showing
    the data, model, individual components, and residuals with polished styling.

    This function assumes that the Sherpa UI environment has already been
    configured and that the following symbols are defined in the global scope:
      - full_src_model: the complete source model expression for SRC_1
      - SrcAbs, SrcNEI1, SrcNEI2: Sherpa model components
      - sky_scale_src, InstLine_1, InstLine_2, Src_InstLine_3: Gaussian or line components
    """

    # ---------- 11. 绘图 ----------
    fig = plt.figure(figsize=(6, 6.8))
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
        ms=1.5,
        capsize=1.5,
        color='#1f77b4',
        label=r'$\mathrm{SRC~Data}$',
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
        label=r'$\mathrm{Full~Model}$',
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
            linewidth=0.8,
            label=label,
            alpha=0.6,
        )
        ui.set_source('SRC_1', full_src_model)
        ui.get_fit_plot('SRC_1')

    _component_curve(SrcAbs * SrcNEI1, 'purple', r'$\mathrm{tbabs}\times\mathrm{Vrnei}$')
    _component_curve(SrcAbs * SrcNEI2, 'teal', r'$\mathrm{tbabs}\times\mathrm{nei}$')

    gaussian_components = [
        (sky_scale_src * InstLine_1, r'$\mathrm{Instrumental~Lines}$'),
        (sky_scale_src * InstLine_2, None),
        (Src_InstLine_3, None),
    ]

    for component_expr, lbl in gaussian_components:
        _component_curve(component_expr, '#7f7f7f', lbl)

    ax_main.set_xscale('linear')
    ax_main.set_yscale('log')
    ax_main.set_ylabel(r'$\mathrm{Counts}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$')
    ax_main.set_xlim(0.5, 6.0)
    ax_main.set_ylim(1e-3, 0.3)
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
    ax_del.set_xlabel(r'$\mathrm{Energy\ (keV)}$')
    ax_del.set_ylabel(r'$\Delta\chi$')
    ax_del.set_ylim(-2.1, 2.1)

    plt.tight_layout()
    plt.show()

    ui.set_source('SRC_1', full_src_model)
    ui.save(filename='SRC_1_fitting_results_softp.sherpa', clobber=True)
    print("\nDone. Results saved to SRC_1_fitting_results_softp.sherpa")


if __name__ == "__main__":
    fit_and_plot_source()
