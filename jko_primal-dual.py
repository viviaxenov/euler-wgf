# 1-d toy implememtation of methods from paper 
# <<Primal-dual methods for Wasserstein gradient flows>> Carillo et al. 2021

import numpy as np

def get_grid(x_left : float, x_right : float, N_x : int, T_max : float, N_t : int)
    assert (N_t > 0)
    assert (N_x > 0)
    x_edge = np.linspace(x_left, x_right, N_cells + 1, endpoint=True)
    x_cell = 0.5*(x_edge[1:] + x_edge[:-1])

    t = np.linspace(0, T_max, N_t + 1, endpoint=True)
    
    return x_edge, x_cell, t


