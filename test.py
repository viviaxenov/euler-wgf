import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import normal

import os
from time import perf_counter

from jko_primal_dual import *

# np.seterr(all="raise")

N_x = 20
N_t = 10
delta = 1e-5
#
# for demonstration we do WGF between 2 [clipped]  Gaussians
m_init = 1.0 / 2
# m_fin = 1.0 / 6.0
m_fin = m_init

sigma_init = 1.0
sigma_fin = 0.2

tau = 0.01

# for gaussians we can get analytical solution
# this is actially a wrong solution because our Gaussians are clipped
# but maybe will work for large X
m_tau = (m_init / tau + m_fin / sigma_fin**2) / (1 / tau + 1 / sigma_fin**2)
sigma_tau = (
    (sigma_init + np.sqrt(sigma_init**2 + 4.0 * (tau + tau**2 / sigma_fin**2)))
    / (1 + tau / sigma_fin**2)
    / 2.0
)
density_one_step = lambda _x: np.exp(-(((_x - m_tau) / sigma_tau) ** 2) / 2.0)

jko_step = JKO_step(
    N_x,
    N_t,
    lambda _x: np.exp(-(((_x - m_init) / sigma_init) ** 2) / 2.0),
    lambda _x: np.log(_x) + 1.0,
    lambda _x: 0.5 *((_x - m_fin) / sigma_fin) ** 2,
    tau,
    deltas=(delta,) * 4,
    debug=True,
    newt_steps=3,
)


rho_tau = density_one_step(jko_step.x_cell)
rho_tau = rho_tau / rho_tau.sum() / jko_step.cell_vol

primal_variables = jko_step.get_primal_variables()
dual_variables = jko_step.get_dual_variables()

# test if adjoint operator really works
errors = []
for _ in range(1000):
    u = normal(size=primal_variables.shape)
    phi = tuple(normal(size=_phi.shape) for _phi in dual_variables)

    Au = jko_step.apply_A(u)
    At_phi = jko_step.apply_At(phi)

    dot_in_primal = u.ravel() @ At_phi.ravel()
    dot_in_dual = np.sum([Au[i].ravel() @ phi[i].ravel() for i in range(4)])
    errors.append(np.abs(dot_in_primal - dot_in_dual))
errors = np.array(errors)
print(errors.max())

N_steps = 200_000
stepsizes = jko_step.estimate_step_sizes(1e-5)
print(stepsizes)
# main optimization loop
t = perf_counter()
pv, dv, history = jko_step.minimize(
    stepsizes,
    N_steps,
)
t = perf_counter() - t
print(t)


# print(jko_step.apply_A_initial_condition(pv) - jko_step.rhs[-1])

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
pv = history["pv"][-1]
axs[0].plot(
    jko_step.x_cell,
    jko_step.rho_0_h,
    label="Initial density",
    color="C1",
    linestyle="--",
)
axs[0].plot(
    jko_step.x_cell, rho_tau, label="Reference solution", color="C2", linestyle="--"
)
(l_rho1,) = axs[0].plot(jko_step.x_cell, pv[0, 0, :], label="$\\rho_{t=0}$", color="C1")
(l_rho2,) = axs[0].plot(
    jko_step.x_cell, pv[0, -1, :], label="$\\rho_{t=1}$", color="C2"
)
axs[0].plot(
    jko_step.x_cell[::2],
    pv[0, -1, ::2],
    label="$\\rho_{t=1}, even$",
    linestyle=None,
    marker="*",
    color="C0",
)
axs[0].plot(
    jko_step.x_cell[1::2],
    pv[0, -1, 1::2],
    label="$\\rho_{t=1}, odd$",
    linestyle=None,
    marker="+",
    color="C0",
)
axs[0].set_title("density")

(l_flux1,) = axs[1].plot(jko_step.x_cell, pv[1, 0, :], label="t = 0")
(l_flux2,) = axs[1].plot(jko_step.x_cell, pv[1, -2, :], label="t = 1 - dt")
axs[1].set_title("flux")

for ax in axs:
    ax.legend()
    ax.grid()

fig.suptitle(f"steps done: {len(history['pv'])}")
fig.savefig("density.pdf")


def animate(idx):
    pv = history["pv"][idx]
    l_rho1.set_ydata(pv[0, 0, :])
    l_rho2.set_ydata(pv[0, -1, :])

    l_flux1.set_ydata(pv[1, 0, :])
    l_flux2.set_ydata(pv[1, -2, :])

    fig.suptitle(f"steps done: {idx + 1}")

    return (
        l_rho1,
        l_rho2,
        l_flux1,
        l_flux2,
    )


# ani = animation.FuncAnimation(fig, animate, frames=len(history['pv']), interval=1, blit=True)
# writer = animation.FFMpegWriter(fps=25, metadata=dict(artist="Me"), bitrate=1800)
# ani.save("movie.gif", writer=writer)


fig, axs = plt.subplots(nrows=1, ncols=2)
# plotting the constraint + objective history
for key in history.keys():
    if not key.startswith("violation_"):
        continue
    axs[0].plot(history[key][20:], label=key)

axs[0].axhline(delta, linestyle="--")

for key in ["F", "W", "objective"]:
    axs[1].plot(history[key][20:], label=key)

axs[0].set_yscale("log")
axs[0].grid()
axs[0].legend()

axs[1].set_yscale("log")
axs[1].grid()
axs[1].legend()

fig.savefig("history.pdf")

fig.show()
plt.show()
