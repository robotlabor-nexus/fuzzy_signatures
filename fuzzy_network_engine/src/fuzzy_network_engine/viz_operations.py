import numpy as np

import matplotlib.pyplot as plt

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_node_elements import CylindricalExtension
from fuzzy_network_engine.fuzzy.membership_functions import TrapMf, TriMf


def viz_cylindrical_extension():
    t0 = np.linspace(0.0, 1.0, 100)
    t1 = np.linspace(0.0, 1.0, 100)
    par_0 = [0.4, 0.6, 0.8, 0.9]
    trap = TrapMf(par_0)
    par_1 = [0.4, 0.6, 0.8]
    tri = TriMf(par_1)
    y0 = trap.eval(t0)
    y1 = tri.eval(t1)
    cyl_ext = CylindricalExtension("cyl_0")
    cyl_ = cyl_ext.execute({"y0": y0, "y1": y1})
    X, Y = np.meshgrid(t0, t1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.title("Cylindrical extension (y0)")
    ax.plot_wireframe(X, Y, cyl_["y0"], rstride=10, cstride=10)
    ax.plot_wireframe(Y, X, cyl_["y1"], rstride=10, cstride=10, color='purple')
    plt.show()
    # SUM two cylindrical extension
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show()
    ax.plot_wireframe(X, Y, cyl_["y0"] + cyl_["y1"].T, rstride=10, cstride=10, color='purple')
    # PROD two cylindrical extension
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show()
    ax.plot_wireframe(X, Y, np.multiply(cyl_["y0"], cyl_["y1"].T), rstride=10, cstride=10, color='purple')


def main():
    viz_cylindrical_extension()


if __name__=="__main__":
    main()
