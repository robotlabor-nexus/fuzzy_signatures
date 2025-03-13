import numpy as np
from graphviz import Digraph
from matplotlib import patches, pyplot as plt


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
        ax.scatter(c_point[0], c_point[1], color='r')
        rect_boundary = patches.Rectangle(anchor, b.width(), b.height(), edgecolor='purple', facecolor='none')
        ax.add_patch(rect_boundary)
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