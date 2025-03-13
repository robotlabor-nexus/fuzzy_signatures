from collections import deque

class QuadTree(object):

    def __init__(self, node_factory):
        self.leaf_nodes = []
        self.root_node = None
        self.node_factory = node_factory

    def add_element(self, element, boundary):
        if self.root_node is None:
            #self.root_node = QuadTreeNode(boundary, label="X")
            self.root_node = self.node_factory.create_node(boundary, label="X")
            self.root_node.update_element(element)
        else:
            self.root_node.add_element(element)

    def traverse_leaf_nodes(self):
        fringe = deque()
        fringe.append(self.root_node)
        leaf_nodes = []
        intermediate_nodes = []
        while len(fringe) > 0:
            node = fringe.pop()
            for n in node.neighbors:
                if not n.is_leaf():
                    fringe.append(n)
                else:
                    leaf_nodes.append(n)
            # Plotting boundary
            intermediate_nodes.append(node)
        return intermediate_nodes, leaf_nodes
