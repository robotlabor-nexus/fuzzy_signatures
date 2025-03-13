
class FuzzyNode(object):
    def __init__(self, hyperparameters):
        self.activations = {}
        self.output_activations = {}
        self.rules = {}
        self.hyperparameters = hyperparameters

    def add_activation(self, var_name, activation):
        self.activations[var_name] = activation

    def add_output_activation(self, var_name, activation):
        self.output_activations[var_name] = activation

    def assign_hyperparameter(self, param_name, value):
        self.hyperparameters[param_name] = value

    def fuzzify(self, vars):
        res = {}
        for k, v in vars.items():
            res[k] = self.activations[k].eval(v)
        return res

    def add_rule(self, rule_name, rule):
        self.rules[rule_name] = rule

    def infer(self, fuzzed_vals):
        s = []
        for r in self.rules.values():
            s.append(r.eval(self.s_norm, fuzzed_vals))
        return s

    def infer_activate(self, s_vals):
        res = []
        for v in s_vals:
            x = self.output_activations[v[0][1]].single_eval(v[0][2], v[1])
            res.append((v[0][1], v[0][2], x))
        return res

    def set_s_norm(self, s_norm):
        self.s_norm = s_norm
