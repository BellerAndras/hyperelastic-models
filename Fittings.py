###Fitting experimental data to compressible hyperelastic models

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.optimize import least_squares, root_scalar

BASE_DIR = Path(__file__).resolve().parent

import lmfit
 
 
def _fit_exp_data_Ogden(
    data,
    N,
    incompressible=False,
    alpha_bounds=(0.01, 100.0),
    mu_bounds=(1e3, 1e7),
    D_bounds=(1e-12, 1e-6),
    fixed=None,
):
    """
    Fit a compressible or incompressible Ogden N-term model with lmfit.
 
    Parameters
    ----------
    data : tuple
        (lambda1, sigma1), 1D arrays; sigma1 is the nominal (1st PK) stress.
    N : int
        Number of Ogden terms (1, 2 or 3).
    incompressible : bool, default False
        True: lambda2 = lambda3 = lambda1**-0.5 (fast, fully vectorised).
        False: P22 = 0 solved per data point with a volumetric term.
    alpha_bounds, mu_bounds, D_bounds : (min, max)
        Parameter bounds. Widen alpha_bounds (e.g. (-50, 50)) to allow
        negative exponents.
    fixed : dict, optional
        Parameters to freeze, e.g. {"alpha1": 2.0} (Neo-Hooke as Ogden N=1;
        note mu = 4*C10 in the W = 2*C10*(I1-3) convention).
 
    Returns
    -------
    fit_data : tuple
        (lambda1, sigma1_fit)
    params : dict
        Fitted values (mu1, alpha1, [D1], ...), plus "lambdaT" and
        "result" (the full lmfit MinimizerResult: stderr, correlations,
        redchi, aic, bic, fit_report(), ...).
    """
 
    if N not in (1, 2, 3):
        raise ValueError("N must be 1, 2, or 3.")
 
    lambda1_exp = np.asarray(data[0], dtype=float)
    sigma1_exp = np.asarray(data[1], dtype=float)
 
    if lambda1_exp.ndim != 1 or sigma1_exp.ndim != 1:
        raise ValueError("lambda1 and sigma1 must be 1D arrays.")
    if len(lambda1_exp) != len(sigma1_exp):
        raise ValueError("lambda1 and sigma1 must have the same length.")
    if len(lambda1_exp) == 0:
        raise ValueError("Experimental data are empty.")
    if not (np.all(np.isfinite(lambda1_exp)) and np.all(np.isfinite(sigma1_exp))):
        raise ValueError("Experimental data must be finite.")
    if np.any(lambda1_exp <= 0):
        raise ValueError("lambda1 values must be positive.")
 
    # --------------------------------------------------------------
    # Symbolic derivation (unchanged)
    # --------------------------------------------------------------
 
    lam1, lam2, lam3 = sp.symbols("lam1 lam2 lam3", positive=True)
    lamT = sp.symbols("lamT", positive=True)
 
    mu = sp.symbols(f"mu1:{N + 1}", positive=True)
    alpha = sp.symbols(f"alpha1:{N + 1}", positive=True)
 
    J = lam1 * lam2 * lam3
    lam_bar = tuple(l * J ** (-sp.Rational(1, 3)) for l in (lam1, lam2, lam3))
 
    W = sum(
        2 * mu[i] / alpha[i] ** 2
        * (sum(lb ** alpha[i] for lb in lam_bar) - 3)
        for i in range(N)
    )
 
    if not incompressible:
        D = sp.symbols(f"D1:{N + 1}", positive=True)
        W = W + sum((J - 1) ** (2 * (i + 1)) / D[i] for i in range(N))
 
    P1_raw = sp.diff(W, lam1)
    P2_raw = sp.diff(W, lam2)
 
    subs = {lam2: lamT, lam3: lamT}
    P1_uniax = P1_raw.subs(subs)
    P2_uniax = P2_raw.subs(subs)
 
    if incompressible:
        # Pressure eliminated via P22 = 0 (lamT = lam1**-0.5 is imposed)
        P1_uniax = P1_uniax - (lamT / lam1) * P2_uniax
 
    # Parameter names, in the order used by the lambdified functions
    names = []
    for i in range(1, N + 1):
        names += [f"mu{i}", f"alpha{i}"]
        if not incompressible:
            names.append(f"D{i}")
 
    sym_params = []
    for i in range(N):
        sym_params += [mu[i], alpha[i]]
        if not incompressible:
            sym_params.append(D[i])
 
    args = (lam1, lamT, *sym_params)
    P1_fun = sp.lambdify(args, P1_uniax, modules="numpy")
    P2_fun = None if incompressible else sp.lambdify(args, P2_uniax, modules="numpy")
 
    # --------------------------------------------------------------
    # Transverse stretch
    # --------------------------------------------------------------
 
    def solve_transverse(lam, previous, values):
 
        def transverse_stress(x):
            return P2_fun(lam, x, *values)
 
        try:
            sol = root_scalar(
                transverse_stress,
                x0=previous,
                x1=previous * 1.001,
                method="secant",
            )
            if sol.converged and sol.root > 0:
                return sol.root
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            pass
 
        search = np.linspace(0.2, 2.0, 100)
        vals = np.array([transverse_stress(x) for x in search])
 
        for k in range(len(search) - 1):
            if (
                np.isfinite(vals[k])
                and np.isfinite(vals[k + 1])
                and vals[k] * vals[k + 1] < 0
            ):
                sol = root_scalar(
                    transverse_stress,
                    bracket=(search[k], search[k + 1]),
                    method="brentq",
                )
                if sol.converged:
                    return sol.root
 
        raise RuntimeError(
            f"Could not solve transverse stretch at lambda1 = {lam:.6g}"
        )
 
    # --------------------------------------------------------------
    # Forward model
    # --------------------------------------------------------------
 
    def predict(lambda1_values, values):
        lambda1_values = np.asarray(lambda1_values, dtype=float)
 
        if incompressible:
            # Fully vectorised: no Python loop, no root solving
            lamT_values = lambda1_values ** -0.5
            sigma = np.asarray(
                P1_fun(lambda1_values, lamT_values, *values), dtype=float
            )
            return lamT_values, np.broadcast_to(sigma, lambda1_values.shape).copy()
 
        lamT_values = np.empty_like(lambda1_values)
        sigma = np.empty_like(lambda1_values)
        previous = 1.0
 
        for i, lam in enumerate(lambda1_values):
            lt = solve_transverse(lam, previous, values)
            sigma[i] = P1_fun(lam, lt, *values)
            lamT_values[i] = lt
            previous = lt
 
        return lamT_values, sigma
 
    # --------------------------------------------------------------
    # lmfit parameters and residuals
    # --------------------------------------------------------------
 
    params = lmfit.Parameters()
    for i in range(1, N + 1):
        params.add(f"mu{i}", value=1e5, min=mu_bounds[0], max=mu_bounds[1])
        params.add(f"alpha{i}", value=float(i), min=alpha_bounds[0], max=alpha_bounds[1])
        if not incompressible:
            params.add(f"D{i}", value=1e-9, min=D_bounds[0], max=D_bounds[1])
 
    for name, value in (fixed or {}).items():
        params[name].set(value=value, vary=False)
 
    def residuals(p):
        values = tuple(p[n].value for n in names)
        return predict(lambda1_exp, values)[1] - sigma1_exp
 
    result = lmfit.minimize(
        residuals,
        params,
        method="least_squares",  # same trust-region solver as before
        x_scale="jac",
    )
 
    if not result.success:
        raise RuntimeError(f"Ogden N={N} fit failed: {result.message}")
 
    # --------------------------------------------------------------
    # Outputs
    # --------------------------------------------------------------
 
    fitted = tuple(result.params[n].value for n in names)
    lambdaT_fit, sigma_fit = predict(lambda1_exp, fitted)
 
    out = {
        "incompressible": incompressible,
        "N": N,
        "lambdaT": lambdaT_fit,
        "result": result,
    }
    out.update({n: result.params[n].value for n in names})
 
    return (lambda1_exp, sigma_fit), out
 
 
def fit_exp_data_OgdenN1(data, incompressible=False, **kw):
    """Fit a compressible or incompressible Ogden N=1 model."""
    return _fit_exp_data_Ogden(data, N=1, incompressible=incompressible, **kw)
 
 
def fit_exp_data_OgdenN2(data, incompressible=False, **kw):
    """Fit a compressible or incompressible Ogden N=2 model."""
    return _fit_exp_data_Ogden(data, N=2, incompressible=incompressible, **kw)
 
 
def fit_exp_data_OgdenN3(data, incompressible=False, **kw):
    """Fit a compressible or incompressible Ogden N=3 model."""
    return _fit_exp_data_Ogden(data, N=3, incompressible=incompressible, **kw)
 
 
# Example of use:
#
#fit_curve, fit_params = fit_exp_data_OgdenN2(exp_data_ds, incompressible=True)
#   print(fit_params["result"].fit_report())      # stderr, correlations, redchi
#   print(fit_params["result"].aic, fit_params["result"].bic)
#
# Neo-Hooke as a frozen Ogden N=1 (mu = 4*C10 in your W = 2*C10*(I1-3) convention):
#
#   fit_curve, p = fit_exp_data_OgdenN1(exp_data_ds, incompressible=True,
#                                       fixed={"alpha1": 2.0})



def read_cut_data(filename):
    """_summary_
    """
    data = np.loadtxt(filename,delimiter="\t",skiprows=1)
    L=data[:,0]
    F=data[:,1]
    
    return(L,F)





exp_data = read_cut_data(BASE_DIR /"txtfiles" / "DATA_AVG_CONC.tab")
exp_data_ds = read_cut_data(BASE_DIR /"txtfiles" / "DATA_AVG_DS_CONC.tab")



(fit_curve1,fit_params1) = fit_exp_data_OgdenN1(exp_data_ds,incompressible=True)

(fit_curve2,fit_params2) = fit_exp_data_OgdenN2(exp_data_ds,incompressible=True)

(fit_curve3,fit_params3) = fit_exp_data_OgdenN3(exp_data_ds,incompressible=True)

(fit_curve4,fit_params4) = fit_exp_data_OgdenN1(exp_data_ds,incompressible=True,fixed={"alpha1":2.0})


#print(f"fit curve: {fit_curve1}")
print("---------------------")
print(f"fit params: {fit_params3}")
print("---------------------")

fig,ax = plt.subplots()

ax.plot(exp_data[0],exp_data[1],label="Mérési adatsor",linewidth=2)

ax.plot(fit_curve1[0],fit_curve1[1],label="Illesztett görbe, Elsőfokú Ogden",linewidth=2)
ax.plot(fit_curve2[0],fit_curve2[1],label="Illesztett görbe, Másodfokú Ogden",linewidth=5,linestyle="--",color="purple")
ax.plot(fit_curve3[0],fit_curve3[1],label="Illesztett görbe, Harmadfokú Ogden",linewidth=2)
ax.plot(fit_curve4[0],fit_curve4[1],label="Illesztett görbe, Neo-Hooke",linewidth=2)

ax.set_title("Paraméterek illesztése", fontsize=24)
ax.set_xlabel(r"$\lambda$", fontsize=24)
ax.set_ylabel("$P$/Pa", fontsize=24)
ax.grid()
ax.legend(fontsize=20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


fig2,ax2 = plt.subplots()

#ax.plot(exp_data[0],exp_data[1],label="Rezidumok",linewidth=2)

ax2.plot(exp_data_ds[0],fit_params1["result"].residual,label="Elsőfokú Ogden",linewidth=2)
ax2.plot(exp_data_ds[0],fit_params2["result"].residual,label="Másodfokú Ogden",linewidth=5,linestyle="--",color="purple")
ax2.plot(exp_data_ds[0],fit_params3["result"].residual,label="Harmadfokú Ogden",linewidth=2)
ax2.plot(exp_data_ds[0],fit_params4["result"].residual,label="Neo-Hooke",linewidth=2)

ax2.set_title("Rezidumok", fontsize=24)
ax2.set_xlabel(r"$\lambda$", fontsize=24)
ax2.set_ylabel("$P$/Pa", fontsize=24)
ax2.grid()
ax2.legend(fontsize=20)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

plt.show()





