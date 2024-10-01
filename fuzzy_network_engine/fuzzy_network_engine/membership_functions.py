import numpy as np

import unittest


class Membership(object):
    def __init__(self, hy_par):
        self.par = hy_par

    def eval(self):
        raise NotImplementedError


class TriMf(Membership):
    def __init__(self, hy_par):
        Membership.__init__(self, hy_par)
        self.a = self.par[0]
        self.b = self.par[1]
        self.c = self.par[2]

    def eval(self, val):
        return np.maximum(
            0.0,
            np.minimum(
                (val - self.par[0])/(self.par[1] - self.par[0]),
                (self.par[2] - val)/(self.par[2] - self.par[1])))


class AscRampMf(Membership):
    def __init__(self, hy_par):
        Membership.__init__(self, hy_par)
        self.a = self.par[0]
        self.b = self.par[1]

    def eval(self, val):
        return np.maximum(
            np.minimum((val - self.par[0])/(self.par[1] - self.par[0]), 1.0), 0.0)


class DescRampMf(Membership):
    def __init__(self, hy_par):
        Membership.__init__(self, hy_par)
        self.a = self.par[0]
        self.b = self.par[1]

    def eval(self, val):
        return np.maximum(
            np.minimum((self.par[1] - val)/(self.par[1] - self.par[0]), 1.0), 0.0)


class TrapMf(Membership):
    def __init__(self, hy_par):
        Membership.__init__(self, hy_par)
        self.a = self.par[0]
        self.b = self.par[1]
        self.c = self.par[2]
        self.d = self.par[3]

    def eval(self, val):
        return np.maximum(
            np.minimum(
                np.minimum((val - self.par[0])/(self.par[1] - self.par[0]),
                           (self.par[3] - val)/(self.par[3] - self.par[2])), 1.0), 0.0)

