# 1-d toy implememtation of methods from paper
# <<Primal-dual methods for Wasserstein gradient flows>> Carillo et al. 2021

import numpy as np

# from scipy.sparse.linalg import LinearOperator

from typing import Callable, Tuple

# I think it improve readability in annotations


class JKO_step:
    def __init__(
        self,
        N_x: int,
        N_t: int,
        rho_init: Callable,
        internal_energy_derivative_fn: Callable,
        potential_energy_fn: Callable,
        tau: float,  # Think of better name
        deltas: Tuple[float, float, float, float],  # Move to minimization method?
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

        self.deltas = np.array(deltas)
        self.tau = tau

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

        # Prepare r.h.s. and tols needed in proximal mapping in dual space
        # Initial density for I.C. constraint check
        rho_0_h = rho_init(self.x_cell)
        # normalize so that integral of discretized rho is equal to 1
        self.rho_0_h = rho_0_h / np.sum(rho_0_h) * self.cell_vol
        self.rhs = [0.0, 0.0, 1.0, self.rho_0_h]

        # I believe that there could be some inconsistency in the paper
        # So i define these tols for prox evaluation so that the constraints (30-31) hold for delta := self.delta (user-defined)
        # and then delta from definition of prox := self.tols
        self.tols = (
            self.deltas
            / (
                np.array(
                    [
                        self.cell_vol * self.dt,
                        self.dx * self.dt,
                        1.0,
                        self.cell_vol,
                    ]
                )
            )
            ** 0.5
        )

    def get_primal_variables(self) -> np.ndarray:

        rho = np.stack((self.rho_0_h,) * self.N_t)
        m = np.stack((np.zeros_like(self.rho_0_h),) * self.N_t)

        return np.stack((rho, m))

    def get_dual_variables(self) -> Tuple[np.ndarray]:
        # it's stupid but I'll rewrite it later
        Au = self.apply_A(self.get_primal_variables())
        duals = [np.zeros_like(_x) for _x in Au]

        return tuple(duals)

    # b_i in constraints are used on the Prox step
    def apply_A_pde(self, primal_variables: np.ndarray) -> np.ndarray:
        """check continuity equation in interioir cells"""

        u = primal_variables
        rho = u[0, :, :]
        m = u[1, :, :]

        # use centered difference for d/dx for now, maybe use C.N. later
        # TODO maybe we should make an array of cell vols? for grid with cells of diff.size (but same vol)?
        Au = (rho[1:, 1:-1] - rho[:-1, 1:-1]) / self.dt + (
            m[:-1, 2:] - m[:-1, :-2]
        ) / self.dx / 2.0

        return Au

    def apply_At_pde(self, phi: np.ndarray) -> np.ndarray:
        """
        apply the ADJOINT operator to the one that evaluates PDE constraint residual
        phi = dual_variables[0]
        """
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

    def apply_A_boundary_condition(self, primal_variables: np.ndarray):
        """m at boundary should be approx. 0"""
        m = primal_variables[1, :, :]

        # TODO would multiply by cell face area (would be different from dx in general);
        # maybe need to think/read more about this from the perspective of Finite Volume methods
        return m[:-1, [0, -1]] * self.dx * self.dt

    def apply_At_boundary_condition(self, phi: np.ndarray) -> np.ndarray:
        # phi = dual_variables[1]
        rho = np.zeros((self.N_t, self.N_x))
        m = rho.copy()
        m[:-1, [0, -1]] = phi

        return np.stack((rho, m))

    def apply_A_mass(self, primal_variables) -> np.ndarray:
        # This is actually dense, but maybe .sum is still better optimized
        rho = primal_variables[0, :, :]
        # in the paper it's *(dx)^(d)*dt but i think it's better to rescale delta at prox step
        # ! need to be careful with the scaling of dual variables !

        return rho.sum(axis=1)

    def apply_At_mass(self, phi: np.ndarray) -> np.ndarray:
        m = np.zeros((self.N_t, self.N_x))
        rho = np.broadcast_to(np.atleast_2d(phi).T, m.shape)
        return np.stack((rho, m))

    def apply_A_initial_condition(self, primal_variables) -> np.ndarray:

        rho = primal_variables[0, :, :]
        return rho[0, :]

    def apply_At_initial_condition(self, phi: np.ndarray) -> np.ndarray:
        m = np.zeros((self.N_t, self.N_x))
        rho = m.copy()
        rho[0, :] = phi
        return np.stack((rho, m))

    def apply_A(self, primal_variables: np.ndarray) -> Tuple[np.ndarray]:
        return (
            self.apply_A_pde(primal_variables),
            self.apply_A_boundary_condition(primal_variables),
            self.apply_A_mass(primal_variables),
            self.apply_A_initial_condition(primal_variables),
        )

    def apply_At(self, dual_variables: Tuple[np.ndarray]) -> np.ndarray:
        return np.sum(
            np.stack(
                (
                    self.apply_At_pde(dual_variables[0]),
                    self.apply_At_boundary_condition(dual_variables[1]),
                    self.apply_At_mass(dual_variables[2]),
                    self.apply_At_initial_condition(dual_variables[3]),
                )
            ),
            axis=0,
        )

    def prox_in_primal(
        self, primal_variables: np.ndarray, stepsize: float
    ) -> np.ndarray:
        """Prox map of functional \Varphi, which involves taking a root of a polynomial; see eq (36)"""

        rho = primal_variables[0, :, :]
        m = primal_variables[1, :, :]

        _lambda = stepsize

        # define coefs of a polynomial
        # these are all element-wise
        b = 2.0 * _lambda - rho
        c = _lambda * (_lambda - 2.0 * rho)
        d = -_lambda * (m ** 2 / 2.0 - rho * _lambda)

        # define coefs of a reduced polynomial
        p = c - b ** 2 / 3.0
        q = (2 * b ** 3 - 9 * b * c + 27 * d) / 27.0

        # TODO maybe we can't rule out the case with Q = 0
        # it's probably the case with |m| = 0=> rho = 0 => new_rho = 0
        # but i'll have to pass the IF 'to C level'
        # maybe there won't be this problem with iterative method
        Q = (p / 3.0) ** 3 + (q / 2.0) ** 2
        # print(Q[Q<0])
        # print(primal_variables[:, Q <= 0])
        # assert np.all(Q > 0)

        # find root of the reduced polynomial with explicit Cardano formula
        # wonder if all of them used it too...
        alpha = np.cbrt(np.sqrt(Q) - q / 2.0)
        beta = np.cbrt(-np.sqrt(Q) - q / 2.0)
        y = alpha + beta

        # evaluate rho, m
        rho_new = y - b / 3.0
        m_new = m * rho_new / (rho_new + _lambda)

        return np.stack((rho_new, m_new))

    def prox_in_dual(
        self, dual_variables: Tuple[np.ndarray], stepsize: float
    ) -> Tuple[np.ndarray]:
        """Prox map of functional i, which is clipping according to constraints; see eq (35)"""

        sigma = stepsize

        dual_variables_new = []

        for i in range(4):
            phi = dual_variables[i]
            phi_new = phi - sigma * ball_projection(
                phi / sigma, self.rhs[i], self.tols[i]
            )
            dual_variables_new.append(phi)

        return dual_variables_new

    def energy_gradient(self, primal_variables: np.ndarray) -> np.ndarray:
        rho_1 = primal_variables[0, -1, :]  # density if final moment
        grad = self.U_prime(rho_1) + self.V_cell  # gradient in JKO scheme
        return grad * 2.0 * self.tau  # here JKO is rescaled as W^2_2 + 2*tau*E

    def minimize(
        self,
        step_sizes: Tuple[float],
        N_max_steps: int,
        primal_init: np.ndarray = None,
        dual_init: Tuple[np.ndarray] = None,
        stopping_rtol: float = 1e-5,
    ):
        _lambda, sigma = step_sizes  # in primal and dual spaces respectively

        u = primal_init if primal_init is not None else self.get_primal_variables()
        u_bar = u.copy()
        phi = dual_init if dual_init is not None else self.get_dual_variables()

        for steps_done in range(N_max_steps):
            Au_bar = self.apply_A(u_bar)
            phi_new = tuple([phi[i] + sigma * Au_bar[i] for i in range(4)])
            phi_new = self.prox_in_dual(phi_new, sigma)

            grad_u = self.energy_gradient(u)
            u_new = self.prox_in_primal(
                u - _lambda * grad_u - _lambda * self.apply_At(phi_new),
                _lambda,
            )

            grad_u_new = self.energy_gradient(u_new)
            u_bar_new = 2.0 * u_new - u + _lambda * (grad_u - grad_u_new)

            rdiff = np.linalg.norm(u - u_new) / np.linalg.norm(u)

            phi = phi_new
            u = u_new
            u_bar = u_bar_new

            if rdiff < stopping_rtol:
                break

        return u, phi, steps_done


def ball_projection(x, b, delta):

    diff = x - b
    diff_norm = np.linalg.norm(diff)

    return x if diff_norm <= delta else diff * delta / diff_norm + b
