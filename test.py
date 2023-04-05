import matplotlib.pyplot as plt
import numpy as np

from numpy.random import normal

from jko_primal_dual import *

N_x = 8
N_t = 8
#

probl = Problem(
    N_x,
    N_t,
    lambda _x: np.exp(-(_x ** 2) / 2.0),
    lambda _x: np.log(_x) + 1.0,
    lambda _x: -((_x - 1.0) ** 2),
    deltas=(1e-7,) * 4,
)

primal_variables = probl.get_primal_variables()
dual_variables = probl.get_dual_variables()

errors = []

for _ in range(1000):
    u = normal(size=primal_variables.shape)
    phi = normal(size=dual_variables[1].shape)

    Au = probl.apply_A_boundary_condition(u)
    At_phi = probl.apply_At_boundary_condition((phi,))

    dot_in_primal = u.ravel() @ At_phi.ravel()
    dot_in_dual= Au.ravel() @ phi.ravel()
    errors.append(np.abs(dot_in_primal - dot_in_dual))

errors = np.array(errors)

print(errors.max(), errors.mean())






