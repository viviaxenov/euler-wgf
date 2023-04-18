import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import pandas as pd
from IPython.display import display


results = []

Niter = 3

for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    for m_sq in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        for rho in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            if rho == 0.0 and m_sq > 0.0:
                continue
            shift = 0.5 * lam * m_sq
            polynom = lambda _x: (_x + lam) ** 2 * (_x - rho) - 0.5 * lam * m_sq
            polynom_deriv = lambda _x: 3.0 * (_x + lam) * (_x - (2 * rho - lam) / 3.0)

            rholam = (rho + lam) / np.sqrt(3)
            drho = np.cbrt(shift + rholam**3) - rholam
            rho_upper_bound = rho + drho

            rholam = 2.0 * (rho + lam) / 3.0
            drho = np.cbrt(shift + rholam**3) - rholam
            rho_lower_bound = rho + drho

            rho_max = rho_upper_bound * (1.01)
            rho_min = (1.0 - 1e-2) * rho_lower_bound

            rho_opt = fsolve(polynom, np.array([rho]), fprime=polynom_deriv)
            rho_opt = rho_opt[0]

            starting_values = [rho, rho_min, rho_max]
            labels = ["rho", "rho_min", "rho_max"]

            res_dict = {
                "lambda": lam,
                "rho": rho,
                "m_sq": m_sq,
                "root": rho_opt,
                "step_size": rho_opt - rho,
                "upper_bound": rho_upper_bound,
                "lower_bound": rho_lower_bound,
                # "upper_rtol": (rho_upper_bound - rho_opt) / rho_opt,
                # "lower_rtol": (rho_opt - rho_lower_bound) / rho_opt,
            }

            for _i in range(3):
                rho_newt = starting_values[_i]
                for _ in range(Niter):
                    rho_newt -= polynom(rho_newt) / polynom_deriv(rho_newt)
                res_dict["root_" + labels[_i]] = rho_newt
                res_dict["rtol_" + labels[_i]] = (
                    np.abs(rho_newt - rho_opt) / rho_opt if not rho_opt == 0.0 else 0.0
                )

            results.append(res_dict)

df = pd.DataFrame.from_records(results)

# _ = None if np.all(df["upper_rtol"] >= 0.0) else print("Upper bound error")
# _ = None if np.all(df["lower_rtol"] >= 0.0) else print("Lower bound error")
# if not np.all(df["step_size"] >= 0.0):
#     print("Step size < 0")

display(df[["rtol_" + _l for _l in labels]].describe())
# display(df[["upper_rtol", "lower_rtol"]].describe())
