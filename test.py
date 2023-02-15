import matplotlib.pyplot as plt
import numpy as np

from jko_primal-dual import *

N_cells = 100
N_timestep = 100
x_left = 0.
x_right = 1.
T = 1.
#

x_edge, x_cell, t = get_grid(x_left, x_right, N_x, T, N_timestep) 
