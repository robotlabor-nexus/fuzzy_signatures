import unittest
import numpy as np

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_signature_quadtree import FuzzySignatureElementFactory, \
    FuzzySignatureEnvironmentRepresentation
from fuzzy_network_engine.fuzzy_quadtree.visualize import visualize_infer_grid
from fuzzy_network_engine.quadtree.quadtree_factory import generate_elements
from fuzzy_network_engine.quadtree.quadtree_node import generate_quadtree
from fuzzy_network_engine.quadtree.quadtree_visualize import visualize_quadtree


class TestFuzzySignatureQuadtree(unittest.TestCase):

    def test_fuzzy_signature_quadtree_extreme(self):
        rng = np.random.default_rng(0)
        # r_points = 100 * rng.random((10, 2)) - 50
        r_points = np.array([[-5.0, 0.0], [5.0, 0.0], [0.0, 3.0], [0.0, -2.0]])
        elements = generate_elements(FuzzySignatureElementFactory(), r_points)
        q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
        env_repr = FuzzySignatureEnvironmentRepresentation(q_tree)
        visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
        X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
        X, Y, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
        visualize_infer_grid(X, Y, infer_grid, inference_result, coarse_grid)

    def test_fuzzy_signature_quadtree_viz1(self):
        rng = np.random.default_rng(0)
        # r_points = 100 * rng.random((10, 2)) - 50
        r_points = np.array([[-5.0, -5.0], [5.0, -7.0], [2.0, 3.0], [-6.0, 2.0]])
        elements = generate_elements(FuzzySignatureElementFactory(), r_points)
        q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
        env_repr = FuzzySignatureEnvironmentRepresentation(q_tree)
        visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
        X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
        X, Y, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
        visualize_infer_grid(X, Y, infer_grid, inference_result, coarse_grid)

    def test_var_obstacle_size(self):
        rng = np.random.default_rng(0)
        r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0], [-1.0, -3.0]])
        elements = generate_elements(FuzzySignatureElementFactory(), r_points)
        q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
        # Infering on element
        env_repr = FuzzySignatureEnvironmentRepresentation(q_tree)
        obstacle_dimensions = [[5.0, 7.0],
                               [12.0, 12.0],
                               [10.0, 10.0],
                               [10.0, 10.0],
                               [ 5.0,  5.0]]
        for i,el in enumerate(elements[1:]):
            el.update_obstacle_boundary(obstacle_dimensions[i])
        env_repr.initialize_fuzzy_inference_system(elements)
        visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
        X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
        X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
        visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)

    def test_var_obstacle_size_random(self):
        rng = np.random.default_rng(0)
        r_points = 100 * rng.random((5, 2)) - 50
        elements = generate_elements(FuzzySignatureElementFactory(), r_points)
        q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements)
        # Infering on element
        env_repr = FuzzySignatureEnvironmentRepresentation(q_tree)
        obstacle_dimensions = [[5.0, 7.0],
                               [12.0, 12.0],
                               [10.0, 10.0],
                               [10.0, 10.0],
                               [ 5.0,  5.0]]
        for i,el in enumerate(elements[1:]):
            el.update_obstacle_boundary(obstacle_dimensions[i])
        env_repr.initialize_fuzzy_inference_system(elements)
        visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
        X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
        X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
        visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)
