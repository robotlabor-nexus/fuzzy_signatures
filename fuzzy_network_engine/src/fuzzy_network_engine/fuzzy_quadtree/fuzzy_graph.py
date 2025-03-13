import numpy as np


class TreeNode(object):
    def __init__(self, children=[], parent=None):
        self.children = children
        self.parent = parent

    def add_children(self, child):
        self.children.append(child)


class FuzzySignature(object):
    def __init__(self, membership, parameter_name):
        self.membership = membership
        self.parameter_name = parameter_name


class FuzzySignatureTreeNode(TreeNode):
    def __init__(self, fuzzy_node, signatures=[], children=[],
                 parent=None):
        TreeNode.__init__(self, children, parent)
        self.fuzzy_node = fuzzy_node
        self.signatures = signatures

    def node_parameter_update(self):
        self.fuzzy_node.assign_hyperparameter(self.parameter_name, self.parameter_value)

    def add_signature(self, signature):
        self.signatures.append(signature)
