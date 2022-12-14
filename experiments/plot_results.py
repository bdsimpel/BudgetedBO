import os
import sys
import numpy as np
import matplotlib.pylab as pl

obj_func = "ackley"
algorithms = ["EI", "EI-PUC", "EI-PUC-CC_60", "B-MS-EI_111_1_60"]

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
results_dir = script_dir + "\\results\\" + obj_func + "\\"

def plot_func(algorithms, obj_func, func, cummul=False):
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_dir = script_dir + "\\results\\" + obj_func + "\\"
    for algo in algorithms:
        current_dir = results_dir + algo + "\\"
        current_file = current_dir + func
        # convert file to numpy array
        with open(current_file, 'r') as file:
            data = file.readlines()
        data = [line.strip() for line in data]
        data = [line.split() for line in data]
        data = [[float(x) for x in line] for line in data]
        data = np.array(data)
        # if cumulative plot, sum up all values
        if cummul == True:
            data = np.cumsum(data, axis=0)
        pl.plot(np.linspace(1, data.shape[0], data.shape[0]), data, label=algo)
    pl.legend()
    pl.show()

if __name__ == "__main__":
    pass
    
    # plot cummulative cost
    plot_func(algorithms, obj_func, "cost_X\\cost_X_1.txt", cummul=True)
    """
    # plot objective
    plot_func(algorithms, obj_func, "objective_X\\objective_X_1.txt")
    # plot cummulative running times
    plot_func(algorithms, obj_func, "running_times\\running_times_1.txt", cummul=True)
    # plot best observed values
    plot_func(algorithms, obj_func,  "best_obs_vals_1.txt")
    """
    