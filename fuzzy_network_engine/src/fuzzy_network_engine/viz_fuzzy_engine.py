import matplotlib.pyplot as plt

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_node import FuzzyNode
from fuzzy_network_engine.fuzzy_quadtree.fuzzy_node_elements import Activation, ConsequentActivation, FuzzyRule
from fuzzy_network_engine.fuzzy.membership_functions import DescRampMf, TrapMf, AscRampMf, TriMf
from fuzzy_network_engine.fuzzy.norm import sum_s_norm

import numpy as np

def main():
    hyper_parameters = {
        "mf0":
            np.array([0.0, 0.1]),
        "mf1":
            np.array([0.1, 0.2, 0.3, 0.4]),
        "mf2":
            np.array([0.3, 0.4, 0.6, 0.7]),
        "mf3": np.array([0.6, 0.7]),
        "mf2_0":
            np.array([0.2, 0.3, 0.4, 0.5]),
        "mf2_1":
            np.array([0.4, 0.5, 0.6, 0.7]),
        "a_end_mf0":
            np.array([0.1, 0.2, 0.3]),
        "a_end_mf1":
            np.array([0.2, 0.4, 0.6]),
        "a_end_mf2":
            np.array([0.4, 0.6, 0.8]),
        "a_end_mf3":
            np.array([0.6, 0.9, 1.0])
    }
    engine = FuzzyNode(hyper_parameters)
    mf0 = DescRampMf(hyper_parameters["mf0"])
    mf1 = TrapMf(hyper_parameters["mf1"])
    mf2 = TrapMf(hyper_parameters["mf2"])
    mf3 = AscRampMf(hyper_parameters["mf3"])
    r1 = Activation("control_ang_0")
    r1.add_membership("small", mf0)
    r1.add_membership("moderate", mf1)
    r1.add_membership("large", mf2)
    r1.add_membership("very_large", mf3)
    engine.add_activation("lin_err", r1)
    vals = {}
    t = np.linspace(0, 1, 100)
    vals["ang_err"] = np.linspace(0, 1, 100)
    vals["lin_err"] = np.linspace(0, 1, 100)
    engine.set_s_norm(sum_s_norm)
    r2 = Activation("control_ang_1")
    mf2_0 = TrapMf(hyper_parameters["mf2_0"])
    mf2_1 = TrapMf(hyper_parameters["mf2_1"])
    r2.add_membership("normal", mf2_0)
    r2.add_membership("large", mf2_1)
    engine.add_activation("ang_err", r2)
    fuzzed = engine.fuzzify(vals)
    for v in fuzzed.items():
        for i,v0 in enumerate(v[1][1]):
            print(v0)
            plt.plot(t, v0, '^', label=v[0])
    # Add consequent activation
    a_end = ConsequentActivation("cmd")
    a_end_mf0 = TriMf(hyper_parameters["a_end_mf0"])
    a_end_mf1 = TriMf(hyper_parameters["a_end_mf1"])
    a_end_mf2 = TriMf(hyper_parameters["a_end_mf2"])
    a_end_mf3 = TriMf(hyper_parameters["a_end_mf3"])
    a_end.add_membership("small", a_end_mf0)
    a_end.add_membership("medium", a_end_mf1)
    a_end.add_membership("large", a_end_mf2)
    a_end.add_membership("very_large", a_end_mf3)
    # Act rule 1
    r0 = FuzzyRule()
    r0.add_antecedent("lin_err", "small")
    r0.add_antecedent("ang_err", "normal")
    r0.add_consequent("cmd", "small")
    # Act rule 2
    r1 = FuzzyRule()
    r1.add_antecedent("lin_err", "moderate")
    r1.add_antecedent("ang_err", "large")
    r1.add_consequent("cmd", "medium")
    # Add rules to engine
    engine.add_rule("end_control_1", r0)
    engine.add_rule("end_control_2", r1)
    engine.add_output_activation("cmd", a_end)
    infer_1 = engine.infer(fuzzed)
    plt.plot(t, infer_1[0][1])
    plt.plot(t, infer_1[1][1])
    act = engine.infer_activate(infer_1)
    plt.plot(t, act[0][2])
    plt.plot(t, act[1][2])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
