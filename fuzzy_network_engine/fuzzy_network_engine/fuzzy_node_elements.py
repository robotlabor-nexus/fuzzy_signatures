import numpy as np


class FuzzyNodeElement(object):
    def __init__(self, name):
        self.name = name


class Activation(FuzzyNodeElement):

    def __init__(self, rule_name):
        FuzzyNodeElement.__init__(rule_name)
        self.activation_memberships = {}

    def add_membership(self, mf_name, mf):
        self.activation_memberships[mf_name] = mf

    def eval(self, var):
        res = np.zeros((len(self.activation_memberships), var.shape[0]))
        mfs = {}
        for i, mf in enumerate(self.activation_memberships.items()):
            res[i] = mf[1].eval(var)
            mfs[mf[0]] = i
        return mfs, res


class ConsequentActivation(Activation):
    def __init__(self, rule_name):
        Activation.__init__(self, rule_name)

    def single_eval(self, mf_name, var):
        res = self.activation_memberships[mf_name].eval(var)
        return res


class OperatorElement(FuzzyNodeElement):
    def __init__(self, name):
        FuzzyNodeElement.__init__(self, name)

    def execute(self, val):
        raise NotImplementedError


class CylindricalExtension(OperatorElement):
    def __init__(self, name):
        OperatorElement.__init__(self, name)

    def execute(self, val):
        res = {}
        for k, v in val.items():
            x = v
            for i in range(len(val.keys()) - 1):
                x = np.tile(x, (v.shape[0], 1))
            res[k] = x
        return res





class ProjectionElement(OperatorElement):
    def __init__(self, name):
        OperatorElement.__init__(self, name)

    def execute(self, val):
        res = {}
        for k, v in val.items():
            x = v
            for i in range(len(val.keys()) - 1):
                x = np.tile(x, (v.shape[0], 1))
            res[k] = x
        return res


class FuzzyRule(FuzzyNodeElement):
    def __init__(self):
        self.antecedents = {}
        self.consequent_activation = {}
        self.consequent_mf_name = None

    def add_antecedent(self, var_name, mf_name):
        self.antecedents[var_name] = mf_name

    def add_consequent(self, activation_name, mf_name):
        self.consequent_activation = activation_name
        self.consequent_mf_name = mf_name

    def eval(self, s_norm, vars):
        eval_res = []
        var_names = []
        for a in self.antecedents.items():
            named_var = vars[a[0]]
            k = named_var[0]
            val = named_var[1]
            eval_res.append(val[k[a[1]]])
            var_names.append(a[1])
        s = np.array(eval_res)
        normalized = s_norm(s)
        return (var_names, self.consequent_activation, self.consequent_mf_name), normalized
