import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np

def visualize_infer_grid(X, Y, infer_grid, inference_result, X_coarse, Y_coarse, coarse_grid):
    smooth_grid = gaussian_filter(infer_grid.T, sigma=9)
    # Plot results
    fig = plt.figure()
    anim_grid = np.zeros(X.shape)
    for sub_grid in inference_result:
        ax = fig.add_subplot(111, projection='3d')
        cx, cy, sum_v = sub_grid
        anim_grid[cx[0]:cx[1], cy[0]:cy[1]] += sum_v
        ax.plot_wireframe(X, Y, anim_grid, rstride=5, cstride=5)
        plt.cla()
    ax.plot_wireframe(X_coarse, Y_coarse, coarse_grid, rstride=1, cstride=1, color="green")
    plt.show()
    plt.imshow(smooth_grid)
    plt.show()