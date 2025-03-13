import numpy as np

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_node import FuzzyNode

import matplotlib.pyplot as plt

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_node_elements import Activation
from fuzzy_network_engine.fuzzy.membership_functions import TrapMf


def main():
    hyperparameters = {
        "mf_right": np.array([-0.1,  0.2]),
        "mf_left":  np.array([-0.2,  0.1]),
        "mf_up":    np.array([-0.1,  0.2]),
        "mf_down":  np.array([-0.2,  0.1])
    }
    node_placer = FuzzyNode(hyperparameters)
    mf_right = TrapMf(hyperparameters["mf_right"])
    mf_left  = TrapMf(hyperparameters["mf_left"])
    mf_up    = TrapMf(hyperparameters["mf_up"])
    mf_down  = TrapMf(hyperparameters["mf_down"])
    # Position placements
    x_placement = Activation("pos_x")
    y_placement = Activation("pos_y")
    #
    x_placement.add_membership("mf_right", mf_right)
    x_placement.add_membership("mf_left", mf_left)
    #
    y_placement.add_membership("mf_up", mf_up)
    y_placement.add_membership("mf_down", mf_down)
    #
    node_placer.add_activation("x", x_placement)
    node_placer.add_activation("y", y_placement)
    #
    rng = np.random.default_rng(0)
    r_points = 100 * rng.random((50, 2)) - 50
    vals = {"x": r_points[:, 0], "y": r_points[:, 1]}
    print(vals)
    plt.scatter(vals["x"], vals["y"])
    plt.axhline(y=0, color='r')
    plt.axvline(x=0, color='r')
    plt.show()
    #node_placer.add_activation()


if __name__=="__main__":
    main()
