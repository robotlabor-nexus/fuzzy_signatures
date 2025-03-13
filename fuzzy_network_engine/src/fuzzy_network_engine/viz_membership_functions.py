import matplotlib.pyplot as plt

import numpy as np


from fuzzy_network_engine.fuzzy.membership_functions import DescRampMf, AscRampMf, TriMf, TrapMf



def viz_desc_ramp():
    par = [0.4, 0.6]
    desc_ramp = DescRampMf(par)
    t = np.linspace(0, 1, 100)
    y = desc_ramp.eval(t)
    plt.title("Descending ramp MF")
    plt.plot(t, y)
    plt.show()


def viz_asc_ramp():
    par = [0.4, 0.6]
    asc_ramp = AscRampMf(par)
    t = np.linspace(0, 1, 100)
    y = asc_ramp.eval(t)
    plt.title("Ascending ramp MF")
    plt.plot(t, y)
    plt.show()


def viz_tri():
    par = [0.4, 0.6, 0.8]
    tri = TriMf(par)
    t = np.linspace(0, 1, 100)
    y = tri.eval(t)
    plt.title("Triangular MF")
    plt.plot(t, y)
    plt.show()


def viz_trapmf():
    par = [0.4, 0.6, 0.8, 0.9]
    trap = TrapMf(par)
    t = np.linspace(0, 1, 100)
    y = trap.eval(t)
    plt.title("Trapezoidal MF")
    plt.plot(t, y)
    plt.show()


def viz_anim_trapmf():
    par = np.array([0.4, 0.6, 0.8, 0.9])
    trap = TrapMf(par)
    t = np.linspace(0, 1, 100)
    y = trap.eval(t)
    plt.title("Anim Trapezoidal MF")
    ax = plt.gca()
    line = ax.plot(t, y)[0]
    for i in range(1, 10):
        par -= 0.1
        y = trap.eval(t)
        line.set_data(t, y)
        plt.draw()
    plt.show()



def main():
    viz_desc_ramp()
    viz_asc_ramp()
    viz_tri()
    viz_trapmf()
    viz_anim_trapmf()


if __name__ == "__main__":
    main()

