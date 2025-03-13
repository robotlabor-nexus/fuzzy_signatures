class Boundary(object):
    def __init__(self, points):
        self.points = points

    def width(self):
        return self.points[1][0] - self.points[0][0]

    def height(self):
        return self.points[0][1] - self.points[3][1]

    def get_boundary_points(self):
        return (self.points[0], self.points[1]), \
            (self.points[3], self.points[0])

    def center_point(self):
        return ((self.points[1][0] + self.points[0][0])/2.0), ((self.points[0][1] + self.points[3][1])/2.0)


