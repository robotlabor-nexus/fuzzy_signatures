import numpy as np

from fuzzy_network_engine.fuzzy_quadtree.fuzzy_signature_quadtree import FuzzySignatureElementFactory, \
    FuzzySignatureEnvironmentRepresentation
from fuzzy_network_engine.fuzzy_quadtree.visualize import visualize_infer_grid
from fuzzy_network_engine.quadtree.quadtree_factory import generate_elements
from fuzzy_network_engine.quadtree.quadtree_node import generate_quadtree


def viz_var_obstacle_size():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
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
    #visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
    X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
    X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
    visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)


def viz_var_obstacle_size_40x40():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0], [-1.0, -3.0]])
    elements = generate_elements(FuzzySignatureElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements, boundary=np.array([[-40, 40], [40, 40], [40, -40], [-40, -40]]))
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
    #visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
    X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
    X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(1.0)
    visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)


def viz_var_obstacle_size_5x5():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0], [-1.0, -3.0]])
    elements = generate_elements(FuzzySignatureElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements, boundary=np.array([[-5, 5], [5, 5], [5, -5], [-5, -5]]))
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
    #visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
    X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
    X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(0.1)
    visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)


def viz_var_obstacle_size_5x7():
    rng = np.random.default_rng(0)
    # r_points = 100 * rng.random((10, 2)) - 50
    r_points = np.array([[-5.0, -5.0], [5.0, -5.0], [2.0, 3.0], [-6.0, 3.0], [-1.0, -3.0]])
    elements = generate_elements(FuzzySignatureElementFactory(), r_points)
    q_tree, intermediate_nodes, leaf_nodes = generate_quadtree(elements, boundary=np.array([[-5, 7], [5, 7], [5, -7], [-5, -7]]))
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
    #visualize_quadtree(r_points, intermediate_nodes, leaf_nodes)
    X, Y, infer_grid, inference_result = env_repr.inference_grid(0.1)
    X_coarse, Y_coarse, coarse_grid, coarse_inference_result = env_repr.inference_grid(0.1)
    visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid)


def main():
    #viz_var_obstacle_size()
    viz_var_obstacle_size_40x40()
    #viz_var_obstacle_size_5x5()
    #viz_var_obstacle_size_5x7()


if __name__=="__main__":
    main()
