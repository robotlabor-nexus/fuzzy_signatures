import unittest

from fuzzy_network_engine.fuzzy.membership_functions import TriMf, AscRampMf, DescRampMf, TrapMf


class TestMembershipFunctions(unittest.TestCase):

    def test_trimf(self):
        par = [0.3, 0.5, 0.7]
        tri = TriMf(par)
        self.assertEqual(tri.eval(0.1), 0.0)
        self.assertEqual(tri.eval(0.5), 1.0)
        self.assertEqual(tri.eval(0.8), 0.0)

    def test_ascrampmf(self):
        par = [0.4, 0.6]
        ramp = AscRampMf(par)
        self.assertEqual(ramp.eval(0.0), 0.0)
        self.assertEqual(ramp.eval(0.5), 0.5)
        self.assertEqual(ramp.eval(0.7), 1.0)

    def test_descrampmf(self):
        par = [0.4, 0.6]
        ramp = DescRampMf(par)
        self.assertEqual(ramp.eval(0.0), 1.0)
        self.assertEqual(ramp.eval(0.5), 0.5)
        self.assertEqual(ramp.eval(0.7), 0.0)

    def test_trapmf(self):
        par = [0.3, 0.4, 0.7, 0.9]
        trapmf = TrapMf(par)
        self.assertEqual(trapmf.eval(0.5), 1.0)
        self.assertAlmostEqual(trapmf.eval(0.35), 0.5, 5)
        self.assertEqual(trapmf.eval(0.0), 0.0)
        self.assertAlmostEqual(trapmf.eval(0.8), 0.5, 5)
        self.assertEqual(trapmf.eval(1.0), 0.0)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
