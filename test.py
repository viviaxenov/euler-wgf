import matplotlib.pyplot as plt
import numpy as np

from numpy.random import normal

from jko_primal_dual import *

N_x = 10
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
primal_variables[1, :, :] = 0.01
dual_variables = probl.get_dual_variables()

new_primal = probl.prox_in_primal(primal_variables, 1e-3)
new_dual = probl.prox_in_dual(dual_variables, 1e-3)

errors = []

for _ in range(1000):
    u = normal(size=primal_variables.shape)
    phi = tuple(normal(size=_phi.shape) for _phi in dual_variables)

    Au = probl.apply_A(u)
    At_phi = probl.apply_At(phi)

    dot_in_primal = u.ravel() @ At_phi.ravel()
    dot_in_dual= np.sum([Au[i].ravel() @ phi[i].ravel() for i in range(4)])
    errors.append(np.abs(dot_in_primal - dot_in_dual))

print(new_dual)
