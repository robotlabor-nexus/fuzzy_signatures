import numpy as np

from fuzzy_network_engine.quadtree.quadtree_elements import QuadTreeElement


class QuadTreeElementFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def create_element(coordinate, parent=None, label=""):
        return QuadTreeElement(coordinate)




def generate_elements(element_factory, r_points):
    pos = np.array([0.0, 0.0])
    r_points = np.vstack([pos, r_points])
    elements_list = []
    for p in r_points:
        new_element = element_factory.create_element(p)
        elements_list.append(new_element)
    return elements_list



