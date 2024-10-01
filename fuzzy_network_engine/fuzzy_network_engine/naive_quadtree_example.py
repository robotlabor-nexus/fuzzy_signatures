
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from graphviz import Digraph


class Boundary(object):
    def __init__(self, points):
        self.points = points

    def width(self):
        return self.points[1][0] - self.points[0][0]

    def height(self):
        return self.points[0][1] - self.points[3][1]

    def get_boundary_points(self):
        return (self.points[0], self.points[1]),\
               (self.points[3], self.points[0])

    def center_point(self):
        return ((self.points[1][0] + self.points[0][0])/2.0), ((self.points[0][1] + self.points[3][1])/2.0)


class QuadTreeElement(object):
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.container_element = None
        self.obstacle_boundary = None

    def update_obstacle_boundary(self, dimensions):
        p0 = self.coordinate[0] - dimensions[0] / 2.0, self.coordinate[1] + dimensions[1] / 2.0
        p1 = self.coordinate[0] + dimensions[0] / 2.0, self.coordinate[1] + dimensions[1] / 2.0
        p2 = self.coordinate[0] + dimensions[0] / 2.0, self.coordinate[1] - dimensions[1] / 2.0
        p3 = self.coordinate[0] - dimensions[0] / 2.0, self.coordinate[1] - dimensions[1] / 2.0
        self.obstacle_boundary = Boundary([p0, p1, p2, p3])

    def get_obstacle_boundary(self):
        if self.obstacle_boundary is None:
            return self.container_element.boundary
        elif self.container_element is None:
            return None
        return self.obstacle_boundary


class QuadTreeElementFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_element(coordinate, parent=None, label=""):
        return QuadTreeElement(coordinate)


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
        self.neighbors.append(self.node_generator.create_node(new_boundary_points, self, self.label+"0"))
        # Add X- sector boundary
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Parent point
        new_boundary_points[3][0] = self.element.coordinate[0]
        new_boundary_points[3][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[0][0] = self.element.coordinate[0]
        new_boundary_points[2][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(new_boundary_points, self, self.label+"1"))
        # Add Y+ sector
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Extreme point
        new_boundary_points[0][0] = self.element.coordinate[0]
        new_boundary_points[0][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[3][0] = self.element.coordinate[0]
        new_boundary_points[1][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(new_boundary_points, self, self.label + "2"))
        # Add Y- sector
        new_boundary_points = np.array(self.boundary.points, copy=True)
        # Extreme point
        new_boundary_points[1][0] = self.element.coordinate[0]
        new_boundary_points[1][1] = self.element.coordinate[1]
        # Migrating from parent point
        new_boundary_points[2][0] = self.element.coordinate[0]
        new_boundary_points[0][1] = self.element.coordinate[1]
        self.neighbors.append(self.node_generator.create_node(new_boundary_points, self, self.label + "3"))

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


class QuadTreeNodeFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_node(new_boundary_points, parent=None, label=""):
        return QuadTreeNode(new_boundary_points, parent, label)


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


from collections import deque


def generate_elements(element_factory, r_points):
    pos = np.array([0.0, 0.0])
    r_points = np.vstack([pos, r_points])
    elements_list = []
    for p in r_points:
        new_element = element_factory.create_element(p)
        elements_list.append(new_element)
    return elements_list


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


def visualize_graph(intermediate_nodes, leaf_nodes):
    # Graphical view
    dot = Digraph(comment="Quadtree visualization")
    # Add nodes into visualizing graph
    for i, n in enumerate(intermediate_nodes):
        dot.node("{0}".format(str(n)), n.label)
    # Add leaf nodes to graph
    for i, n in enumerate(leaf_nodes):
        dot.node("{0}".format(str(n)), n.label, color="blue")
    # Add edges into visualizing graph
    for i, n in enumerate(intermediate_nodes):
        if n.parent is not None:
            dot.edge("{0}".format(str(n.parent)), "{0}".format(str(n)))
    for i, n in enumerate(leaf_nodes):
        if n.parent is not None:
            dot.edge("{0}".format(str(n.parent)), "{0}".format(str(n)))
    dot.graph_attr["rotation"] = '180'
    dot.render('test-output/round-table.gv', view=True)


def visualize_quadtree(r_points, intermediate_nodes, leaf_nodes):
    ax = plt.gca()


    for node in intermediate_nodes:
        b = node.boundary
        anchor = node.boundary.points[3]
        c_point = b.center_point()
        #ax.scatter(c_point[0], c_point[1], color='r')
        rect_boundary = patches.Rectangle(anchor, b.width(), b.height(), edgecolor='purple', facecolor='none')
        ax.add_patch(rect_boundary)
        plt.pause(0.1)
    sector_centres = []
    for l in leaf_nodes:
        b = l.boundary
        c_point = b.center_point()
        # Draw leaf boundary
        anchor = b.points[3]
        rect_boundary = patches.Rectangle(anchor, b.width(), b.height(), edgecolor='purple', facecolor='none', alpha=0.6)
        ax.add_patch(rect_boundary)
        sector_centres.append([c_point[0], c_point[1]])
    sector_centres = np.array(sector_centres)
    ax.scatter(0, 0, color='red', linewidths=10, alpha=1.0,
               label="Starting position")
    ax.scatter(sector_centres[:, 0], sector_centres[:,1], color='orange', linewidths=1, alpha=1.0, label="Subdivision centre")
    ax.scatter(r_points[:, 0], r_points[:, 1], linewidth=7, alpha=1.0, label="Obstacles")
    ax.legend()
    plt.show()



def viz1():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0]])
    elements = generate_elements(QuadTreeElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
    visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)


def viz2():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0], [-1.0, -3.0]])
    elements = generate_elements(QuadTreeElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
    visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)


def viz3():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0],
                         [ 5.0, -5.0],
                         [ 2.0,  3.0],
                         [-6.0,  3.0],
                         [-1.0, -3.0],
                         [-2.0,  5.0]])
    elements = generate_elements(QuadTreeElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
    visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)


def viz4():
    rng = np.random.default_rng(0)
    r_points = 20 * rng.random((20, 2)) - 10
    elements = generate_elements(QuadTreeElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
    visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)


def main():
    viz1()
    viz2()
    viz3()
    #viz4()


if __name__=="__main__":
    main()
