import matplotlib.pyplot as plt
import numpy as np

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

u = primal_variables
Au = probl.apply_A_pde(primal_variables)

phi = dual_variables[0]
At_phi = probl.apply_At_pde(dual_variables)

dot_in_primal = u.ravel() @ At_phi.ravel()
dot_in_dual= Au.ravel() @ phi.ravel()

print(dot_in_primal, dot_in_dual, np.abs(dot_in_primal - dot_in_dual))



