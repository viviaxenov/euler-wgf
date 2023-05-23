# 1-d toy implememtation of methods from paper
# <<Primal-dual methods for Wasserstein gradient flows>> Carillo et al. 2021
import numpy as np

from typing import Callable, Tuple
import traceback

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
        newt_steps=3,
        debug=False,
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
        self.dx = x_edge[1] - x_edge[0]
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
        self.rho_0_h = rho_0_h / np.sum(rho_0_h) / self.cell_vol
        self.rhs = [0.0, 0.0, 1.0, self.rho_0_h]

        # number of Newton iterations in the prox evaluation
        self.newt_steps = newt_steps
        if newt_steps is None:
            self.prox_in_primal = self.__prox_in_primal_algebraic
        else:
            self.prox_in_primal = self.__prox_in_primal_newton

        if debug:
            print(
                "Using debug mode: catching RuntimeErrors as exceptions and providing [a lot of] dumps."
            )
            np.seterr(all="raise")
            self.minimize = self.__minimize_debug

        else:
            self.minimize = self.__minimize_clean

        # I believe that there could be some inconsistency in the paper
        # So i define these tols for prox evaluation so that the constraints (30-31) hold for delta := self.delta (user-defined)
        # and then delta from definition of prox := self.tols
        self.tols = self.deltas / (
            np.array(
                [
                    1.,
                    self.dx * self.dt,
                    self.dt,
                    self.cell_vol,
                ]
            )
            ** 0.5
        )

    def get_primal_variables(self) -> np.ndarray:

        rho = np.zeros((self.N_t, self.N_x))
        rho[:, :] = self.rho_0_h
        m = np.full_like(rho, 0.0)

        return np.stack((rho, m))

    def get_dual_variables(self, fill_value=0.0) -> Tuple[np.ndarray]:
        # it's stupid but I'll rewrite it later
        Au = self.apply_A(self.get_primal_variables())
        duals = [np.full_like(_x, fill_value) for _x in Au]

        return tuple(duals)

    # b_i in constraints are used on the Prox step
    def apply_A_pde(self, primal_variables: np.ndarray) -> np.ndarray:
        """check continuity equation in interioir cells"""

        u = primal_variables
        rho = u[0, :, :]
        # default behaviour is to pad w. zeros
        m = np.pad(
            u[1, :, :],
            ((0, 0), (1, 1)),
        )

        # use centered difference for d/dx for now, maybe use C.N. later
        # TODO maybe we should make an array of cell vols? for grid with cells of diff.size (but same vol)?
        Au = (rho[1:, :] - rho[:-1, :]) / self.dt + (
            m[:-1, :-2] - m[:-1, 2:]
        ) / self.dx / 2.0

        return Au*np.sqrt(self.cell_vol*self.dt)

    def apply_At_pde(self, phi: np.ndarray) -> np.ndarray:
        """
        apply the ADJOINT operator to the one that evaluates PDE constraint residual
        phi = dual_variables[0]
        """
        phi_t_pad = np.pad(phi, ((1,1), (0,0)))

        rho = (-phi_t_pad[1:,:] + phi_t_pad[:-1, :])/self.dt

        phi_x_pad = np.pad(
            phi,
            ((0, 1), (1, 1)),
        )
        m = (-phi_x_pad[:, :-2] + phi_x_pad[:, 2:])/2./self.dx

        return np.stack((rho, m))*np.sqrt(self.cell_vol*self.dt)

    def apply_A_boundary_condition(self, primal_variables: np.ndarray):
        """m at boundary should be approx. 0"""
        m = primal_variables[1, :, :]

        # TODO would multiply by cell face area (would be different from dx in general);
        # maybe need to think/read more about this from the perspective of Finite Volume methods
        return m[:-1, [0, -1]]

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

        return rho.sum(axis=1) * self.cell_vol

    def apply_At_mass(self, phi: np.ndarray) -> np.ndarray:
        m = np.zeros((self.N_t, self.N_x))
        rho = np.broadcast_to(np.atleast_2d(phi).T, m.shape)
        return np.stack((rho, m)) * self.cell_vol

    def apply_A_initial_condition(self, primal_variables) -> np.ndarray:
        return primal_variables[0, 0, :]

    def apply_At_initial_condition(self, phi: np.ndarray) -> np.ndarray:
        pv = np.zeros((2, self.N_t, self.N_x))
        pv[0, 0, :] = phi
        return pv

    def apply_A(self, primal_variables: np.ndarray) -> Tuple[np.ndarray]:
        # TODO: make the set of costraints with variable len?
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

    def __prox_in_primal_newton(
        self, primal_variables: np.ndarray, stepsize: float
    ) -> np.ndarray:
        """
        Prox map of functional \Varphi, which involves taking a root of a polynomial; see eq (36)
        Evaluate with Newton method w. same amount of iterations for each rho
        """
        rho = primal_variables[0, :, :]
        m = primal_variables[1, :, :]
        lam = stepsize
        m_sq = m ** 2

        rholam = (rho + lam) / np.sqrt(3)
        shift = 0.5 * m_sq * lam
        drho = np.cbrt(shift + rholam ** 3) - rholam
        rho_upper_bound = rho + drho
        rho_prox = rho_upper_bound

        for _ in range(self.newt_steps):
            rho_prox -= prox_polynom(rho_prox, lam, rho, m_sq) / prox_polynom_deriv(
                rho_prox, lam, rho, m_sq
            )

        if np.any(rho_prox < rho):
            print(rho, m, lam)
            print(((rho - rho_prox)).max())
            raise RuntimeWarning("rho_prox < rho!")

        # rho_reference = np.zeros((self.N_t, self.N_x))

        # for _i in range(self.N_t):
        #     for _j in range(self.N_x):
        #         rho_reference[_i, _j] = fsolve(lambda _x: (_x - rho[_i, _j])*(_x + lam)**2 -0.5*m_sq[_i, _j]*lam, rho_upper_bound)[0]

        m_prox = m * rho_prox / (rho_prox + lam)

        return np.stack((rho_prox, m_prox))

    def __prox_in_primal_algebraic(
        self, primal_variables: np.ndarray, stepsize: float
    ) -> np.ndarray:
        """
        Prox map of functional \Varphi, which involves taking a root of a polynomial; see eq (36)
        Solution with Cardano formula; didn't work, maybe fix it later
        """
        raise (NotImplementedError)

        rho = primal_variables[0, :, :]
        m = primal_variables[1, :, :]

        _lambda = stepsize

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

        Q += 1e-13  # TODO: REMOVE THIS

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

        for _i in range(4):
            phi = dual_variables[_i]
            phi_new = phi - sigma * ball_projection(
                phi / sigma, self.rhs[_i], self.tols[_i]
            )
            dual_variables_new.append(phi_new)
        return tuple(dual_variables_new)

    def energy_gradient(self, primal_variables: np.ndarray) -> np.ndarray:
        rho_1 = primal_variables[0, -1, :]  # density in final moment
        grad = self.U_prime(rho_1) + self.V_cell  # gradient in JKO scheme

        retval = np.zeros_like(primal_variables)
        retval[0, -1, :] = (
            grad * 2.0 * self.tau
        )  # here JKO is rescaled as W^2_2 + 2*tau*Ed
        return retval

    def __minimize_clean(
        self,
        step_sizes: Tuple[float],
        N_max_steps: int,
        primal_init: np.ndarray = None,
        dual_init: Tuple[np.ndarray] = None,
        stopping_rtol: float = 1e-10,
    ):
        _lambda, sigma = step_sizes  # in primal and dual spaces respectively

        u = primal_init if primal_init is not None else self.get_primal_variables()
        u_bar = u.copy()
        phi = dual_init if dual_init is not None else self.get_dual_variables()
        grad_u = self.energy_gradient(u)

        for steps_done in range(N_max_steps):
            Au_bar = self.apply_A(u_bar)
            phi_new = tuple([phi[i] + sigma * Au_bar[i] for i in range(4)])
            phi_new = self.prox_in_dual(phi_new, sigma)

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
            grad_u = grad_u_new

            if rdiff < stopping_rtol:
                break

        return u, phi, steps_done + 1

    def __minimize_debug(
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
        grad_u = self.energy_gradient(u)

        history = {
            "violation_pde": [],
            "violation_bc": [],
            "violation_mass": [],
            "violation_ic": [],
            "objective": [],
            "rdiff_primal": [],
            "W": [],
            "F": [],
            "pv": [],
        }

        for steps_done in range(N_max_steps):
            try:
                Au_bar = self.apply_A(u_bar)
                phi_new = tuple([phi[i] + sigma * Au_bar[i] for i in range(4)])
                phi_new = self.prox_in_dual(phi_new, sigma)

                u_new = self.prox_in_primal(
                    u - _lambda * grad_u - _lambda * self.apply_At(phi_new),
                    _lambda,
                )

                grad_u_new = self.energy_gradient(u_new)
                u_bar_new = 2.0 * u_new - u + _lambda * (grad_u - grad_u_new)

                rdiff = np.linalg.norm(u - u_new) / np.linalg.norm(u)

                rho1 = u[0, -1, :]
                F = 0. # TODO: need to pass this!!!
                Wass = (
                    np.where(u[0, :, :] > 0., u[1, :, :] ** 2 / u[0, :, :], 0.0).sum()
                    * self.cell_vol
                    * self.dt
                )
                objective = 2.0 * self.tau * F + Wass

                pde_violation = (
                    np.linalg.norm(self.apply_A_pde(u) - self.rhs[0])
                    * self.cell_vol
                    * self.dt
                )
                bc_violation = (
                    np.linalg.norm(self.apply_A_boundary_condition(u) - self.rhs[1])
                    * self.dx
                    * self.dt
                )

                # print(self.apply_A_mass(u))

                mass_violation = (
                    np.linalg.norm(self.apply_A_mass(u) - self.rhs[2]) * self.dt
                )
                ic_violation = (
                    np.linalg.norm(self.apply_A_initial_condition(u) - self.rhs[3])
                    * self.dx
                )

                history["pv"].append(u)
                history["objective"].append(objective)
                history["W"].append(Wass)
                history["F"].append(F)
                history["violation_mass"].append(mass_violation)
                history["violation_pde"].append(pde_violation)
                history["violation_bc"].append(bc_violation)
                history["violation_ic"].append(ic_violation)
                history["rdiff_primal"].append(rdiff)

                phi = phi_new
                u = u_new
                u_bar = u_bar_new
                grad_u = grad_u_new

                if rdiff < stopping_rtol:
                    break

            except (RuntimeError, FloatingPointError) as e:
                print(f"{steps_done=}")
                print("".join(traceback.TracebackException.from_exception(e).format()))
                break

        return u, phi, history

    def estimate_step_sizes(
        self, beta_guess: float, n_power_iter: int = 100
    ) -> Tuple[float]:

        # estimate norm of B=AA^t with power iteraton
        phi = self.get_dual_variables(0.01)
        for _ in range(n_power_iter):
            Bphi = self.apply_A((self.apply_At(phi)))
            norm_Bphi = np.sqrt(dot_in_dual(Bphi, Bphi))
            phi = tuple((_phi / norm_Bphi) for _phi in Bphi)

        oper_norm = dot_in_dual(phi, self.apply_A(self.apply_At(phi))) / dot_in_dual(
            phi, phi
        )
        print(oper_norm)

        lam = beta_guess / self.tau
        sigma = 1.0 / oper_norm / lam

        return lam, sigma


prox_polynom = (
    lambda _x, _lam, _rho, _m_sq: (_x + _lam) ** 2 * (_x - _rho) - 0.5 * _lam * _m_sq
)
prox_polynom_deriv = (
    lambda _x, _lam, _rho, _m_sq: 3.0 * (_x + _lam) * (_x - (2 * _rho - _lam) / 3.0)
)


def dot_in_dual(dv1: Tuple[np.ndarray], dv2: Tuple[np.ndarray]) -> float:
    dot = 0.0
    for _i in range(4):
        dot += dv1[_i].ravel() @ dv2[_i].ravel()
    return dot


def ball_projection(x: np.ndarray, b: np.ndarray, delta: float) -> np.ndarray:
    diff = x - b
    diff_norm = np.linalg.norm(diff)

    return x if diff_norm <= delta else diff * delta / diff_norm + b


def dot_in_dual(dv1: Tuple[np.ndarray], dv2: Tuple[np.ndarray]) -> float:
    dot = 0.0
    for _i in range(4):
        dot += dv1[_i].ravel() @ dv2[_i].ravel()
    return dot
