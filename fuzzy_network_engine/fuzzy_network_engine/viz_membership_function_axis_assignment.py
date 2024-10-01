from fuzzy_network_engine.membership_functions import DescRampMf, AscRampMf, TriMf, TrapMf

import numpy as np

import matplotlib.pyplot as plt

def viz_axis_assignment():
    par = [-7, -4]
    desc_ramp = DescRampMf(par)
    par1 = [4, 7]
    asc_ramp = AscRampMf(par1)
    par2 = [1, 4, 7]
    pn_ramp = TriMf(par2)
    par3 = [-7, -4, -1]
    nn_ramp = TriMf(par3)
    par3 = [-3, -1, 1, 3]
    danger_ramp = TrapMf(par3)

    t = np.linspace(-8, 8, 200)
    y0 = desc_ramp.eval(t)
    y1 = asc_ramp.eval(t)
    y_pn = pn_ramp.eval(t)
    y_nn = nn_ramp.eval(t)
    y_danger = danger_ramp.eval(t)
    plt.plot(t, y0)
    plt.plot(t, y1)
    plt.plot(t, y_pn)
    plt.plot(t, y_nn)
    plt.plot(t, y_danger)
    ax = plt.gca()
    ax.legend(['NF', 'PF', 'PN', 'NN', 'Danger'])
    plt.show()

def main():
    viz_axis_assignment()

if __name__ == "__main__":
    main()