import numpy as np

from fuzzy_network_engine.quadtree.quadtree_factory import generate_elements, QuadTreeElementFactory
from fuzzy_network_engine.quadtree.quadtree_node import generate_quadtree
from fuzzy_network_engine.quadtree.quadtree_visualize import visualize_quadtree


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
    #viz2()
    #viz3()
    #viz4()


if __name__=="__main__":
    main()
