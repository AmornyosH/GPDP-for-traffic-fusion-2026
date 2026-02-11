import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
# import seaborn as sns
# sns.set_theme()

DEFAULT_ALPHA = 0.2

class ActionsPlots:
    def __init__(self):
       self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
       self.sc = None
    #    self.fig = plt.figure(figsize=(12, 6))
    #    gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    #    self.ax1 = self.fig.add_subplot(gs[0, 0])
    #    self.ax2 = self.fig.add_subplot(gs[0, 1])
    #    self.cax = self.fig.add_subplot(gs[0, 2])   # colorbar axis       

    def plot_actions(self, results, label):
        _time = np.arange(10000) / 10

        action_0 = results[:, 0]  # Acceleration action
        action_1 = results[:, 1]  # Lateral Action

        if label == 'Ours':
            _target_ax = self.ax2
            self.sc = _target_ax.scatter(action_0, action_1, c=_time, alpha=1, cmap='plasma')
        else:
            _target_ax = self.ax1
            _target_ax.scatter(action_0, action_1, c=_time, alpha=1, cmap='plasma')

    def plot_dataset_actions(self, results, label):
        _action_0 = results[:, 0]
        _action_1 = results[:, 1]
        
        # self.ax1.scatter(_action_0, _action_1, color='black', alpha=0.5, label=label)
        # self.ax2.scatter(_action_0, _action_1, color='black', alpha=0.5, label=label)

        _x1 = np.linspace(start=-1, stop=1, num=100)
        _y1 = np.repeat(0.5, 100)
        self.ax1.fill_between(_x1, _y1+0.075, _y1-0.075, color='gray', alpha=0.1)
        self.ax2.fill_between(_x1, _y1+0.075, _y1-0.075, color='gray', alpha=0.1)
        
        _x2 = np.linspace(start=-1, stop=1, num=100)
        _y2 = np.repeat(-0.5, 100)
        self.ax1.fill_between(_x2, _y2+0.075, _y2-0.075, color='gray', alpha=0.1)
        self.ax2.fill_between(_x2, _y2+0.075, _y2-0.075, color='gray', alpha=0.1)
        # self.ax1.text(-0.35, 0.6, 'Good Action Area in $T_{\\text{train}}$')
        # self.ax1.text(-0.35, -0.65, 'Poor Action Area in $T_{\\text{train}}$')
        # self.ax2.text(-0.35, 0.6, 'Good Action Area in $T_{\\text{train}}$')
        # self.ax2.text(-0.35, -0.65, 'Poor Action Area in $T_{\\text{train}}$')
        self.ax1.text(-0.25, 0.6, 'Good in $T_{\\text{train}}$', fontsize=14)
        self.ax1.text(-0.25, -0.65, 'Poor in $T_{\\text{train}}$', fontsize=14)
        self.ax2.text(-0.25, 0.6, 'Good in $T_{\\text{train}}$', fontsize=14)
        self.ax2.text(-0.25, -0.65, 'Poor in $T_{\\text{train}}$', fontsize=14)


    def show_plots(self):
        self.ax1.set_xlim(-1, 1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True)
        self.ax1.tick_params(axis='both', labelsize=14)
        # self.ax1.set_title('$\mathbf{a}\sim\\boldsymbol{\pi}_{\\text{TRPO}}$', fontsize=16)
        self.ax1.text(0.70, 0.9, 'TRPO', color='purple', fontsize=16)
        
        self.ax2.set_xlim(-1, 1)
        self.ax2.set_ylim(-1, 1)
        self.ax2.grid(True)
        self.ax2.tick_params(axis='both', labelsize=14)
        # self.ax2.set_title('$\mathbf{a}\sim\\boldsymbol{\pi}_{\\boldsymbol{\\theta}}=\\tilde{p}_{\\boldsymbol{\\theta}}(\mathbf{a}^{0:I}|\mathbf{o})$', fontsize=16)
        self.ax2.text(0.725, 0.9, 'Ours', color='blue', fontsize=16)
        
        self.ax1.set_aspect("equal")
        self.ax2.set_aspect("equal")
        
        self.fig.supxlabel('$a_0^0$', fontsize=24)
        self.fig.supylabel('$a_1^0$', fontsize=24)
        
        divider1 = make_axes_locatable(self.ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cax1.set_visible(False)   # hide it     
        
        divider = make_axes_locatable(self.ax2)
        cax = divider.append_axes("right", size="5%", pad=0.15)

        cbar = self.fig.colorbar(self.sc, cax=cax)
        # cbar = self.fig.colorbar(self.sc, ax=self.ax2, shrink=0.7)
        cbar.set_label('Time', fontsize=14, rotation=90)
        cbar.ax.hlines(200, xmin=0, xmax=1, colors='red', linewidth=5)
        cbar.ax.text(0.55, 500, "$T_{\\text{test}}\\neq T_{\\text{train}} $", color='white', ha='center', va='center', fontsize=14, rotation=90)
        # cbar.ax.text(0.55, 150, "$T_{\\text{test}} = T_{\\text{train}}$", color='white', ha='center', va='center', fontsize=14, rotation=90)
        cbar.ax.text(0.55, 100, "$T_{\\text{train}}$", color='white', ha='center', va='center', fontsize=14, rotation=90)
        
        # # Custom legend handles
        # # hollow = Line2D([0], [0], marker='o', color='w', label='Hollow Circle',
        # #                 markerfacecolor='none', markeredgecolor='blue', markersize=10)
        # filled = Circle((0,0), radius=1, facecolor='blue', edgecolor='none', alpha=DEFAULT_ALPHA, label='Poor action area in $T_{\\text{train}}$')
        # filled2 = Circle((0,0), radius=1, facecolor='red', edgecolor='none', alpha=DEFAULT_ALPHA, label='Good action area in $T_{\\text{train}}$')
        # filled3 = Circle((0,0), radius=1, facecolor='black', edgecolor='none', alpha=1, label='$\mathbf{a} \sim \mathcal{D}$')
        # # Add legend using HandlerPatch to render circle properly
        # self.ax1.legend(handles=[filled2, filled, filled3], 
        #                 # handler_map={Circle: HandlerPatch()}, 
        #                 loc='center', fontsize=16)
        # # self.ax2.legend(handles=[filled2, filled, filled3], 
        # #                 # handler_map={Circle: HandlerPatch()}, 
        # #                 loc='center', fontsize=16)
        # # self.ax2.legend(handles=[filled2, filled, filled3], 
        # #                 loc='center left', bbox_to_anchor=(0.3, -0.3), fontsize=16)
        # self.ax1.legend(handles=[filled3], loc='upper right', fontsize=16)
        # self.ax2.legend(handles=[filled3], loc='upper right', fontsize=16)
        
        # self.ax1.legend(loc='upper right', fontsize=16)
        # self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()
        plt.show()
        # plt.savefig("C:/Amornyos/PhD/Materials/action_space_shifted_fusion_1.png", dpi=300)  # 300 DPI for high quality
        ...