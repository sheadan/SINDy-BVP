"""Provide default plot options for plotter.py."""


class PlotOptions:
    def __init__(self):
        fontsize = 12
        self.figsize = (7, 3.5)
        self.fontsize = fontsize
        self.plot_options = dict(fontsize=fontsize)
        self.reg_opts = dict(color='black', ms=5, mec='black', mfc='white',
                             lw=2, linestyle='--')
        self.true_opts = dict(linestyle='-', linewidth=4)
        self.legend_opts = dict(loc='center left', bbox_to_anchor=(1.05, 0.5),
                                fontsize=fontsize)
        self.npts = 50
        self.ode_colors = ['#257352', '#ff6e54', '#8454ff', '#ffc354']
        self.ode_opts = dict(linewidth=2)
        self.coeff_colors = ['#8CBFB9', '#DA888E', '#D2C095', '#E8CC5D']
        self.markers = ['s', 'h', 'd', '^', 'o']
        self.dpi = 1200
        self.fig_dir = './Figs/'
