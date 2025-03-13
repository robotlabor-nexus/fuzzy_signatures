import numpy as np


from fuzzy_network_engine.fuzzy.membership_functions import TriMf


from fuzzy_network_engine.quadtree.quadtree_elements import QuadTreeElement
from fuzzy_network_engine.quadtree.quadtree_factory import QuadTreeElementFactory


class FuzzySignatureQuadTreeElement(QuadTreeElement):
    def __init__(self, coordinate):
        QuadTreeElement.__init__(self, coordinate)
        self.axis_membership_functions = []
        self.obstacle_boundary = None

    def update_fuzzy_memberships(self):
        # Change: add fuzzy membership function according to boundary
        boundary = self.get_obstacle_boundary()
        x_bound, y_bound = boundary.get_boundary_points()
        c = boundary.center_point()
        # X axis membership
        self.axis_membership_functions.append(TriMf([x_bound[0][0],
                                                    self.coordinate[0],
                                                    x_bound[1][0]]
                                                    ))
        # Y axis membership
        self.axis_membership_functions.append(TriMf([y_bound[0][1],
                                                    self.coordinate[1],
                                                    y_bound[1][1]]))


class FuzzySignatureElementFactory(QuadTreeElementFactory):
    def __init__(self):
        QuadTreeElementFactory.__init__(self)
        pass

    @staticmethod
    def create_element(coordinate):
        return FuzzySignatureQuadTreeElement(coordinate)


class FuzzySignatureEnvironmentRepresentation(object):
    def __init__(self, q_tree):
        self.q_tree = q_tree
        self.elements = []

    def initialize_fuzzy_inference_system(self, elements):
        self.elements = elements
        # Elements
        for el in elements:
            el.update_fuzzy_memberships()

    def inference_grid(self, dt):
        intermediate_nodes, leaf_nodes = self.q_tree.traverse_leaf_nodes()
        root_b_x, root_b_y = self.q_tree.root_node.boundary.get_boundary_points()
        t_x = np.arange(root_b_x[0][0], root_b_x[1][0], dt)
        t_y = np.arange(root_b_y[0][1], root_b_y[1][1], dt)
        offset_x = -int(root_b_x[0][0]/dt)
        offset_y = -int(root_b_y[0][1]/dt)
        X, Y = np.meshgrid(t_x, t_y)
        infer_grid = np.zeros(X.shape)
        intermediate_inference_result = []
        grid_width, grid_height = infer_grid.shape
        for i, n in enumerate(intermediate_nodes[1:]):
            #x_bound, y_bound = n.boundary.get_boundary_points()
            boundary = n.element.get_obstacle_boundary()
            x_bound, y_bound = boundary.get_boundary_points()
            width_i, height_i = int(boundary.width() / dt), int(boundary.height() / dt)
            start_infer_x = max(x_bound[0][0], root_b_x[0][0])
            start_infer_y = max(y_bound[0][1], root_b_y[0][1])
            end_infer_x = min(x_bound[1][0], root_b_x[1][0])
            end_infer_y = min(y_bound[1][1], root_b_y[1][1])
            t_x = np.arange(start_infer_x, end_infer_x, dt)
            t_y = np.arange(start_infer_y, end_infer_y, dt)
            v1 = n.element.axis_membership_functions[0].eval(t_x)
            v2 = n.element.axis_membership_functions[1].eval(t_y)
            # PROD rule
            sum_v = np.multiply(np.tile(v1, (v2.shape[0], 1)).T, np.tile(v2, (v1.shape[0], 1)))
            # TODO: get offset index from starting boundary
            start_grid_point_x = int(x_bound[0][0] / dt) + offset_x
            start_grid_point_y = int(y_bound[0][1] / dt) + offset_y
            # Slice SUM
            start_sub_x = 0
            start_sub_y = 0
            if start_grid_point_x < 0:
                start_sub_x = 0 - start_grid_point_x
            if start_grid_point_y < 0:
                start_sub_y = 0 - start_grid_point_y
            cx = max(start_grid_point_x, 0), min(start_grid_point_x + width_i, grid_width)
            cy = max(start_grid_point_y, 0), min(start_grid_point_y + height_i, grid_height)
            infer_grid[cx[0]:cx[1], cy[0]:cy[1]] += sum_v
            intermediate_inference_result.append((cx, cy, sum_v))
        # Average pivot axes
        return X, Y, infer_grid, intermediate_inference_result





