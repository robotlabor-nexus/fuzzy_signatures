import numpy as np

from fuzzy_network_engine.quadtree.quadtree_geometry import Boundary


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


