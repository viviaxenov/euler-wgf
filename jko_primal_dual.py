# 1-d toy implememtation of methods from paper
# <<Primal-dual methods for Wasserstein gradient flows>> Carillo et al. 2021

import numpy as np

# from scipy.sparse.linalg import LinearOperator

from typing import Callable, Tuple

# I think it improve readability in annotations


class Problem:
    def __init__(
        self,
        N_x: int,
        N_t: int,
        rho_init: Callable,
        internal_energy_derivative_fn: Callable,
        potential_energy_fn: Callable,
        deltas: Tuple[float, float, float, float],
        spatial_dim: int = 1,
        x_left: float = 0.0,
        x_right: float = 1.0,
    ):
        # Here timestep is for discretization of  internal cycle's time, i.e. the variable that parametrizes the intermediate Wasserstein geodesic

        assert N_t > 0
        assert N_x > 0

        # TODO later
        if not spatial_dim == 1:
            raise NotImplementedError

        self.deltas = deltas

        self.N_t = N_t
        self.N_x = N_x

        x_edge = np.linspace(x_left, x_right, N_x + 1, endpoint=True)
        self.x_edge = x_edge

        self.x_cell = 0.5 * (x_edge[1:] + x_edge[:-1])

        self.t = np.linspace(0.0, 1.0, N_t + 1, endpoint=True)

        # respective cell sizes for the regular grid
        self.dx = x_edge[1]
        self.dt = self.t[1]
        # cell volume for integration
        self.cell_vol = self.dx ** spatial_dim

        self.U_prime = internal_energy_derivative_fn
        self.V = potential_energy_fn
        # Certainly won't do it for a high-dimensional example, but for 1-2d will do
        self.V_cell = potential_energy_fn(self.x_cell)

        rho_0_h = rho_init(self.x_cell)
        # normalize so that integral of discretized rho is equal to 1
        # idk if its necessary?
        self.rho_0_h = rho_0_h / np.sum(rho_0_h) * self.cell_vol

    def get_primal_variables(
        self,
    ) -> np.ndarray:

        rho = np.stack((self.rho_0_h,) * self.N_t)
        m = np.stack((np.zeros_like(self.rho_0_h),) * self.N_t)

        return np.stack((rho, m))

    def get_dual_variables(self):
        # it's stupid but I'll rewrite it later
        Au = self.apply_A(self.get_primal_variables())
        duals = [np.zeros_like(_x) for _x in Au]

        return tuple(duals)

    # b_i in constraints are used on the Prox step
    def apply_A_pde(self, primal_variables):
        """check continuity equation in interioir cells"""

        u = primal_variables
        rho = u[0, :, :]
        m = u[1, :, :]

        # use centered difference for d/dx for now, maybe use C.N. later
        Au = (rho[1:, 1:-1] - rho[:-1, 1:-1]) / self.dt + (
            m[:-1, 2:] - m[:-1, :-2]
        ) / self.dx / 2.0

        return Au

    def apply_At_pde(self, dual_variables):
        # apply the ADJOINT operator to the one that evaluates PDE constraint residual
        phi = dual_variables[0]

        rho = -phi[1:, :] + phi[:-1, :]
        rho = np.vstack(
            (
                -phi[0, :],
                rho,
                phi[-1:, :],
            )
        )

        zero_pad = np.zeros((self.N_t, 1))

        rho = np.hstack((zero_pad, rho, zero_pad)) / self.dt

        m = phi[:, :-2] - phi[:, 2:]
        m = np.hstack((-phi[:, :2], m, phi[:, -2:]))
        zero_pad = np.zeros((1, self.N_x))
        m = np.vstack((m, zero_pad)) / 2.0 / self.dx

        return np.stack((rho, m))

    def apply_A_boundary_condition(self, primal_variables):
        """m at boundary should be approx. 0"""
        m = primal_variables[1, :, :]

        # in the paper it's m_boundary*(dx)^(d-1)*dt but i think it's better to rescale delta at prox step
        # ! need to be careful with the scaling of dual variables !
        return m[:, [0, -1]]

    def apply_A_mass(self, primal_variables):
        rho = primal_variables[0, :, :]
        # in the paper it's *(dx)^(d)*dt but i think it's better to rescale delta at prox step
        # ! need to be careful with the scaling of dual variables !

        return rho.sum(axis=1)

    def apply_A_initial_condition(self, primal_variables):

        rho = primal_variables[0, :, :]
        return rho[0, :]

    def apply_A(self, primal_variables):

        return (
            self.apply_A_pde(primal_variables),
            self.apply_A_boundary_condition(primal_variables),
            self.apply_A_mass(primal_variables),
            self.apply_A_initial_condition(primal_variables),
        )
