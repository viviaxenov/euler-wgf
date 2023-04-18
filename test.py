import numpy as np
import matplotlib.pyplot as plt


from numpy.random import normal

from jko_primal_dual import *


N_x = 10
N_t = 8
#
# for demonstration we do WGF between 2 [clipped]  Gaussians
m_init = 1.0 / 2
m_fin = 3.0 / 4.0

sigma_init = 1.0
sigma_fin = 0.3

jko_step = JKO_step(
    N_x,
    N_t,
    lambda _x: np.exp(-(((_x - m_init) / sigma_init) ** 2) / 2.0),
    lambda _x: np.log(_x) + 1.0,
    lambda _x: 0.5 * ((_x - m_fin) / sigma_fin) ** 2,
    1e-3,
    deltas=(1e-7,) * 4,
)

primal_variables = jko_step.get_primal_variables()
primal_variables[1, :, :] = 0.01
dual_variables = jko_step.get_dual_variables()

pv, dv, N_steps_done = jko_step.minimize((1e-3, 1e-3), 30)

plt.plot(jko_step.x_cell, jko_step.rho_0_h, label="Initial density")
plt.plot(jko_step.x_cell, pv[0, 0, :], label="$\\rho_{t=0}$")
plt.plot(jko_step.x_cell, pv[0, -1, :], label="$\\rho_{t=1}$")

plt.title(f"steps done: {N_steps_done}")
plt.savefig("density.pdf")

plt.legend()
plt.grid()
plt.show()

# new_primal = jko_step.prox_in_primal(primal_variables, 1e-3)
# new_dual = jko_step.prox_in_dual(dual_variables, 1e-3)
#
# errors = []
#
# for _ in range(1000):
#    u = normal(size=primal_variables.shape)
#    phi = tuple(normal(size=_phi.shape) for _phi in dual_variables)
#
#    Au = jko_step.apply_A(u)
#    t_phi = jko_step.apply_At(phi)
#
#    dot_in_primal = u.ravel() @ At_phi.ravel()
#    dot_in_dual= np.sum([Au[i].ravel() @ phi[i].ravel() for i in range(4)])
#    errors.append(np.abs(dot_in_primal - dot_in_dual))
