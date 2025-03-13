import numpy as np

from fuzzy_network_engine.quadtree.quadtree import QuadTree
from fuzzy_network_engine.quadtree.quadtree_geometry import Boundary


class QuadTreeNodeFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_node(new_boundary_points, parent=None, label=""):
        return QuadTreeNode(new_boundary_points, parent, label)

def generate_quadtree(elements, node_factory=QuadTreeNodeFactory(),
                      boundary=np.array([[-12, 12], [12, 12], [12, -12], [-12, -12]])):
    q_tree = QuadTree(node_factory)
    # Add robot position
    # Construct quadtree
    for el in elements:
        q_tree.add_element(el, boundary)
    # Traverse quadtree
    intermediate_nodes, leaf_nodes = q_tree.traverse_leaf_nodes()
    return q_tree, intermediate_nodes, leaf_nodes

class QuadTreeNode(object):

    def __init__(self, boundary_points, parent=None, label=None):
        self.neighbors = []
        self.element = None
        self.boundary = Boundary(boundary_points)
        self.parent = parent
        self.label = label
        self.node_generator = QuadTreeNodeFactory()

    def update_element(self, element):
        self.element = element
        self.element.container_element = self
        self.subdivide()

    def is_leaf(self):
        return self.element is None

    def subdivide(self):
        # Add X+ sector boundary
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Parent point
        new_boundary_points[2][0] = self.element.coordinate[0]
        new_boundary_points[2][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[1][0] = self.element.coordinate[0]
        new_boundary_points[3][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(
            new_boundary_points, self, self.label+"0"))
        # Add X- sector boundary
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Parent point
        new_boundary_points[3][0] = self.element.coordinate[0]
        new_boundary_points[3][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[0][0] = self.element.coordinate[0]
        new_boundary_points[2][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(
            new_boundary_points, self, self.label+"1"))
        # Add Y+ sector
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Extreme point
        new_boundary_points[0][0] = self.element.coordinate[0]
        new_boundary_points[0][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[3][0] = self.element.coordinate[0]
        new_boundary_points[1][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(
            new_boundary_points, self, self.label + "2"))
        # Add Y- sector
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Extreme point
        new_boundary_points[1][0] = self.element.coordinate[0]
        new_boundary_points[1][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[2][0] = self.element.coordinate[0]
        new_boundary_points[0][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(
            new_boundary_points, self, self.label + "3"))

    def insert_element(self, sector_index, element):
        if self.neighbors[sector_index].is_leaf():
            self.neighbors[sector_index].update_element(element)
        else:
            self.neighbors[sector_index].add_element(element)

    def add_element(self, element):
        d = element.coordinate - self.element.coordinate
        # Keep in mind: we use SAE coordinate system! So X-axis is vertical, Y axis is horizontal direction
        if d[0] < 0.0:
            if d[1] >= 0.0:
                # Add element
                self.insert_element(0, element)
            else:
                # Add element
                self.insert_element(3, element)
        else:
            if d[1] >= 0.0:
                # Add element
                self.insert_element(1, element)
            else:
                # Add element
                self.insert_element(2, element)
