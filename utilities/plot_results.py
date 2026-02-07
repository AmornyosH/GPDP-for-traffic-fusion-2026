import numpy as np
import matplotlib.pyplot as plt

class ResultsPlots:
    def __init__(self):
       self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    def plot_speed_results(self, results, density, label, color):
        _veh_density = density

        results = results['arr_0'].item()
        av_speed_mean = np.squeeze(results['avg_av_speed']) * 3600 / 1000
        av_speed_std = np.squeeze(results['std_av_speed']) * 3600 / 1000
        all_speed_mean = np.squeeze(results['avg_all_speed']) * 3600 / 1000
        all_speed_std = np.squeeze(results['std_all_speed']) * 3600 / 1000

        # ========= AX1: Plot all vehicles speed (Left Figure)
        self.ax1.scatter(_veh_density, all_speed_mean, color=color)
        self.ax1.plot(_veh_density, all_speed_mean, color=color, label=label)
        self.ax1.fill_between(_veh_density, 
                            np.squeeze(all_speed_mean+all_speed_std), 
                            np.squeeze(all_speed_mean-all_speed_std), color=color, alpha=0.10)

        # self.ax1.set_ylim(0, 4*3600/1000)
        self.ax1.grid(True)
        self.ax1.legend(fontsize=16, loc='upper right')
        # self.ax1.set_title('Average All Vehicle Speed')

        # ========= AX2: Plot AV speed (Right Figure)
        self.ax2.scatter(_veh_density, av_speed_mean, color=color)
        self.ax2.plot(_veh_density, av_speed_mean, color=color, label=label)
        self.ax2.fill_between(_veh_density, 
                            np.squeeze(av_speed_mean+av_speed_std), 
                            np.squeeze(av_speed_mean-av_speed_std), color=color, alpha=0.10)
        # self.ax2.set_ylim(0, 4*3600/1000)
        self.ax2.grid(True)
        self.ax2.legend(fontsize=16, loc='upper right')
        # self.ax2.set_title('Average AV Speed')

    def show_plots(self):
        self.fig.supxlabel('Vehicle Density (veh/km)', fontsize=20)
        self.fig.supylabel('Speed (km/hr)', fontsize=20)
        self.ax1.tick_params(axis='both', labelsize=14)
        self.ax2.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        plt.show()
        ...