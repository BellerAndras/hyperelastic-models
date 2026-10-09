###Fitting experimental data to compressible hyperelastic models

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, root_scalar

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import sympy as sp

from scipy.optimize import least_squares, root_scalar



def _fit_exp_data_Ogden(data, N, incompressible=False):
    """
    Fit a compressible or incompressible Ogden N-term model.

    Parameters
    ----------
    data : tuple
        (lambda1, sigma1), 1D numpy arrays.

    N : int
        Number of Ogden terms.

    incompressible : bool, default=False
        If True, enforce lambda1 * lambda2 * lambda3 = 1.
        If False, solve P22 = 0 using volumetric energy terms.

    Returns
    -------
    fit_data : tuple
        (lambda1, sigma1_fit)

    params : dict
        Fitted material parameters and optimizer result.
    """

    if N not in (1, 2, 3):
        raise ValueError("N must be 1, 2, or 3.")

    lambda1_exp, sigma1_exp = data

    lambda1_exp = np.asarray(lambda1_exp, dtype=float)
    sigma1_exp = np.asarray(sigma1_exp, dtype=float)

    if lambda1_exp.ndim != 1 or sigma1_exp.ndim != 1:
        raise ValueError("lambda1 and sigma1 must be 1D arrays.")

    if len(lambda1_exp) != len(sigma1_exp):
        raise ValueError("lambda1 and sigma1 must have the same length.")

    if len(lambda1_exp) == 0:
        raise ValueError("Experimental data are empty.")

    if not np.all(np.isfinite(lambda1_exp)) or not np.all(
        np.isfinite(sigma1_exp)
    ):
        raise ValueError("Experimental data must be finite.")

    if np.any(lambda1_exp <= 0):
        raise ValueError("lambda1 values must be positive.")

    # --------------------------------------------------------------
    # Symbolic variables
    # --------------------------------------------------------------

    lam1, lam2, lam3 = sp.symbols(
        "lam1 lam2 lam3", positive=True
    )
    lamT = sp.symbols("lamT", positive=True)

    mu = sp.symbols(f"mu1:{N + 1}", positive=True)
    alpha = sp.symbols(f"alpha1:{N + 1}", positive=True)

    # --------------------------------------------------------------
    # Deformation and isochoric stretches
    # --------------------------------------------------------------

    J = lam1 * lam2 * lam3

    lam_bar = (
        lam1 * J**(-sp.Rational(1, 3)),
        lam2 * J**(-sp.Rational(1, 3)),
        lam3 * J**(-sp.Rational(1, 3)),
    )

    # --------------------------------------------------------------
    # Deviatoric Ogden energy
    # --------------------------------------------------------------

    W_dev = sum(
        2 * mu[i] / alpha[i]**2
        * (
            sum(lam_bar[j]**alpha[i] for j in range(3))
            - 3
        )
        for i in range(N)
    )

    # --------------------------------------------------------------
    # Volumetric energy (compressible only)
    # --------------------------------------------------------------

    if incompressible:
        W = W_dev
    else:
        D = sp.symbols(f"D1:{N + 1}", positive=True)

        W_vol = sum(
            (J - 1)**(2 * (i + 1)) / D[i]
            for i in range(N)
        )

        W = W_dev + W_vol

    # --------------------------------------------------------------
    # First Piola-Kirchhoff stress
    # --------------------------------------------------------------

    P1_raw = sp.diff(W, lam1)
    P2_raw = sp.diff(W, lam2)

    substitutions = {lam2: lamT, lam3: lamT}

    P1_uniax = P1_raw.subs(substitutions)
    P2_uniax = P2_raw.subs(substitutions)

    if incompressible:
        # Eliminate pressure using P22 = 0:
        #
        # P11 = dW/dlambda1 - (lambdaT/lambda1)*dW/dlambda2
        #
        # lambdaT is fixed by lambda1*lambdaT**2 = 1.

        P1_uniax = (
            P1_uniax
            - (lamT / lam1) * P2_uniax
        )

    # --------------------------------------------------------------
    # Numerical functions
    # --------------------------------------------------------------

    if incompressible:
        arguments = (
            lam1,
            lamT,
            *[
                variable
                for i in range(N)
                for variable in (mu[i], alpha[i])
            ],
        )
    else:
        arguments = (
            lam1,
            lamT,
            *[
                variable
                for i in range(N)
                for variable in (mu[i], alpha[i], D[i])
            ],
        )

    P1_fun = sp.lambdify(
        arguments, P1_uniax, modules="numpy"
    )

    if not incompressible:
        P2_fun = sp.lambdify(
            arguments, P2_uniax, modules="numpy"
        )

    # --------------------------------------------------------------
    # Parameter scaling and unpacking
    # --------------------------------------------------------------

    def unpack_params(p):
        fitted = []

        for i in range(N):
            if incompressible:
                fitted.extend([
                    p[2*i],
                    p[2*i + 1],
                ])
            else:
                fitted.extend([
                    p[3*i],
                    p[3*i + 1],
                    p[3*i + 2] * 1e-9,
                ])

        return tuple(fitted)

    # --------------------------------------------------------------
    # Solve transverse stretch
    # --------------------------------------------------------------

    def solve_transverse(lam, previous, params):

        if incompressible:
            return 1.0 / np.sqrt(lam)

        def transverse_stress(lamT_value):
            return P2_fun(lam, lamT_value, *params)

        try:
            solution = root_scalar(
                transverse_stress,
                x0=previous,
                x1=previous * 1.001,
                method="secant",
            )

            if solution.converged and solution.root > 0:
                return solution.root

        except (ValueError, RuntimeError, OverflowError):
            pass

        search = np.linspace(0.2, 2.0, 100)
        values = np.array([
            transverse_stress(x) for x in search
        ])

        for i in range(len(search) - 1):
            if (
                np.isfinite(values[i])
                and np.isfinite(values[i + 1])
                and values[i] * values[i + 1] < 0
            ):
                solution = root_scalar(
                    transverse_stress,
                    bracket=(search[i], search[i + 1]),
                    method="brentq",
                )

                if solution.converged:
                    return solution.root

        raise RuntimeError(
            f"Could not solve transverse stretch at "
            f"lambda1 = {lam:.6g}"
        )

    # --------------------------------------------------------------
    # Forward model
    # --------------------------------------------------------------

    def predict(lambda1_values, params):

        lambda1_values = np.asarray(
            lambda1_values, dtype=float
        )

        lambdaT_values = np.empty_like(lambda1_values)
        sigma_values = np.empty_like(lambda1_values)

        previous = 1.0

        for i, lam in enumerate(lambda1_values):
            lambdaT = solve_transverse(
                lam, previous, params
            )

            sigma_values[i] = P1_fun(
                lam, lambdaT, *params
            )

            lambdaT_values[i] = lambdaT
            previous = lambdaT

        return lambdaT_values, sigma_values

    # --------------------------------------------------------------
    # Residuals
    # --------------------------------------------------------------

    def residuals(p):
        params = unpack_params(p)
        _, sigma_pred = predict(lambda1_exp, params)
        return sigma_pred - sigma1_exp

    # --------------------------------------------------------------
    # Initial guesses and bounds
    # --------------------------------------------------------------

    if incompressible:
        p0 = np.array([
            value
            for i in range(N)
            for value in (1e5, float(i + 1))
        ])

        lower = np.array([
            value
            for i in range(N)
            for value in (1e3, 0.01)
        ])

        upper = np.array([
            value
            for i in range(N)
            for value in (1e7, 100.0)
        ])

    else:
        p0 = np.array([
            value
            for i in range(N)
            for value in (1e5, float(i + 1), 1.0)
        ])

        lower = np.array([
            value
            for i in range(N)
            for value in (1e3, 0.01, 1e-3)
        ])

        upper = np.array([
            value
            for i in range(N)
            for value in (1e7, 100.0, 1e3)
        ])

    # --------------------------------------------------------------
    # Least-squares fit
    # --------------------------------------------------------------

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        x_scale="jac",
    )

    if not result.success:
        raise RuntimeError(
            f"Ogden N={N} fit failed: {result.message}"
        )

    # --------------------------------------------------------------
    # Extract fitted parameters
    # --------------------------------------------------------------

    fitted = unpack_params(result.x)

    params = {
        "incompressible": incompressible,
        "N": N,
        "result": result,
    }

    for i in range(N):
        if incompressible:
            params[f"mu{i+1}"] = fitted[2*i]
            params[f"alpha{i+1}"] = fitted[2*i + 1]
        else:
            params[f"mu{i+1}"] = fitted[3*i]
            params[f"alpha{i+1}"] = fitted[3*i + 1]
            params[f"D{i+1}"] = fitted[3*i + 2]

    # --------------------------------------------------------------
    # Fitted curve
    # --------------------------------------------------------------

    _, sigma_fit = predict(lambda1_exp, fitted)

    fit_data = (lambda1_exp, sigma_fit)

    return fit_data, params


def fit_exp_data_OgdenN1(data, incompressible=False):
    """Fit a compressible or incompressible Ogden N=1 model."""
    return _fit_exp_data_Ogden(data, N=1, incompressible=incompressible)


def fit_exp_data_OgdenN2(data, incompressible=False):
    """Fit a compressible or incompressible Ogden N=2 model."""
    return _fit_exp_data_Ogden(data, N=2, incompressible=incompressible)


def fit_exp_data_OgdenN3(data, incompressible=False):
    """Fit a compressible or incompressible Ogden N=3 model."""
    return _fit_exp_data_Ogden(data, N=3, incompressible=incompressible)



def fit_exp_data_OgdenN1(data):
    """Fit a compressible Ogden N=1 model to uniaxial data."""
    return _fit_exp_data_Ogden(data, N=1)


def fit_exp_data_OgdenN2(data):
    """Fit a compressible Ogden N=2 model to uniaxial data."""
    return _fit_exp_data_Ogden(data, N=2)



def fit_exp_data_NH(data):
    """
    Fit a compressible Neo-Hookean model to uniaxial experimental data.

    Parameters
    ----------
    data : tuple
        Experimental data:

            (lambda1, sigma1)

        where lambda1 is the stretch and sigma1 is the nominal
        (first Piola-Kirchhoff) stress.

    Returns
    -------
    fit_data : tuple
        Fitted data:

            (lambda1, sigma1_fit)

    params : dict
        Fitted Neo-Hookean parameters.
    """

    import numpy as np
    import sympy as sp

    from scipy.optimize import least_squares, root_scalar

    # ------------------------------------------------------------------
    # Experimental data
    # ------------------------------------------------------------------

    lambda1_exp, sigma1_exp = data

    lambda1_exp = np.asarray(lambda1_exp, dtype=float)
    sigma1_exp = np.asarray(sigma1_exp, dtype=float)

    if lambda1_exp.ndim != 1 or sigma1_exp.ndim != 1:
        raise ValueError(
            "lambda1 and sigma1 must be 1D arrays.")

    if len(lambda1_exp) != len(sigma1_exp):
        raise ValueError(
            "lambda1 and sigma1 must have the same length.")

    if len(lambda1_exp) == 0:
        raise ValueError(
            "Experimental data are empty.")

    # ------------------------------------------------------------------
    # Symbolic model
    # ------------------------------------------------------------------

    lam1, lam2, lam3 = sp.symbols("lam1 lam2 lam3",positive=True,)

    lamT = sp.symbols("lamT",positive=True,)

    C10 = sp.symbols("C10",positive=True,)

    D1 = sp.symbols("D1",positive=True,)

    J = lam1 * lam2 * lam3

    # ------------------------------------------------------------------
    # Isochoric principal stretches
    # ------------------------------------------------------------------

    lam1_bar = lam1 * J**(-sp.Rational(1, 3))
    lam2_bar = lam2 * J**(-sp.Rational(1, 3))
    lam3_bar = lam3 * J**(-sp.Rational(1, 3))

    # ------------------------------------------------------------------
    # Neo-Hookean strain energy
    # ------------------------------------------------------------------

    W_dev = 2 * C10 * (lam1_bar**2+ lam2_bar**2+ lam3_bar**2- 3)

    W_vol = (J - 1)**2 / D1

    W = sp.simplify(W_dev + W_vol)

    # ------------------------------------------------------------------
    # First Piola-Kirchhoff stresses
    # ------------------------------------------------------------------

    P1 = sp.diff(W, lam1)
    P2 = sp.diff(W, lam2)

    # ------------------------------------------------------------------
    # Uniaxial boundary conditions
    #
    # lambda2 = lambda3 = lambdaT
    # P22 = 0
    # ------------------------------------------------------------------

    uniaxial_subs = {lam2: lamT,lam3: lamT,}

    P1_uniax = sp.simplify(P1.subs(uniaxial_subs))

    P2_uniax = sp.simplify(P2.subs(uniaxial_subs))

    # ------------------------------------------------------------------
    # Numerical functions
    # ------------------------------------------------------------------

    P1_fun = sp.lambdify((lam1,lamT,C10,D1,),P1_uniax,modules="numpy",)

    P2_fun = sp.lambdify((lam1,lamT,C10,D1,),P2_uniax,modules="numpy",)

    # ------------------------------------------------------------------
    # Parameter scaling
    #
    # The optimizer uses D1_scaled = D1 / 1e-9.
    # ------------------------------------------------------------------

    def unpack_params(p):
        C10_value = p[0]
        D1_value = p[1] * 1e-9

        return C10_value, D1_value

    # ------------------------------------------------------------------
    # Solve traction-free transverse direction
    # ------------------------------------------------------------------

    def solve_transverse(lam,previous,C10_value,D1_value,):

        def transverse_stress(lamT_value):
            return P2_fun(lam,lamT_value,C10_value,D1_value,)

        # First attempt: secant method using the previous solution.
        try:
            solution = root_scalar(
                transverse_stress,
                x0=previous,
                x1=previous * 1.001,
                method="secant",)

            if solution.converged and solution.root > 0:
                return solution.root

        except (ValueError, RuntimeError):
            pass

        # Fallback: search for a bracketed root.
        search = np.linspace(
            0.2,
            2.0,
            100,)

        values = np.array([
            transverse_stress(x)
            for x in search])

        for i in range(len(search) - 1):

            if values[i] * values[i + 1] < 0:

                solution = root_scalar(
                    transverse_stress,
                    bracket=(
                        search[i],
                        search[i + 1],
                    ),
                    method="brentq",)

                if solution.converged:
                    return solution.root

        raise RuntimeError(
            f"Could not solve transverse stretch at "
            f"lambda1 = {lam:.6g}")

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def predict(lambda1_values, C10_value, D1_value):

        lambda1_values = np.asarray(
            lambda1_values,
            dtype=float,)

        lambdaT_values = np.empty_like(
            lambda1_values)

        sigma_values = np.empty_like(
            lambda1_values)

        previous = 1.0

        for i, lam in enumerate(lambda1_values):

            lambdaT = solve_transverse(
                lam,
                previous,
                C10_value,
                D1_value,)

            # Nominal / first Piola-Kirchhoff stress.
            P1_value = P1_fun(
                lam,
                lambdaT,
                C10_value,
                D1_value,)

            sigma_values[i] = P1_value

            lambdaT_values[i] = lambdaT
            previous = lambdaT

        return lambdaT_values, sigma_values

    # ------------------------------------------------------------------
    # Residual function
    # ------------------------------------------------------------------

    def residuals(p):

        C10_value, D1_value = unpack_params(p)

        _, sigma_pred = predict(
            lambda1_exp,
            C10_value,
            D1_value,
        )

        return sigma_pred - sigma1_exp

    # ------------------------------------------------------------------
    # Initial guess
    #
    # D1 is represented in units of 1e-9.
    # ------------------------------------------------------------------

    p0 = np.array([
        2e5,    # C10
        1.0,    # D1 / 1e-9
    ])

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    lower = np.array([
        1e3,    # C10
        1e-3,   # D1 / 1e-9
    ])

    upper = np.array([
        1e7,    # C10
        1e3,    # D1 / 1e-9
    ])

    # ------------------------------------------------------------------
    # Least-squares fit
    # ------------------------------------------------------------------

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        x_scale="jac",
    )

    if not result.success:
        raise RuntimeError(
            f"Neo-Hookean fit failed: {result.message}"
        )

    # ------------------------------------------------------------------
    # Fitted parameters
    # ------------------------------------------------------------------

    C10_fit, D1_fit = unpack_params(
        result.x)

    params = {
        "C10": C10_fit,
        "D1": D1_fit,
        "result": result,}

    # ------------------------------------------------------------------
    # Fitted curve
    # ------------------------------------------------------------------

    _, sigma_fit = predict(lambda1_exp,C10_fit,D1_fit,)

    fit_data = (lambda1_exp,sigma_fit,)

    return fit_data, params



def plot_data(data_exp, data_fit=None,exp_name="exp"):
    """
    Plot experimental and fitted uniaxial data.

    Parameters
    ----------
    data_exp : tuple
        (lambda1, sigma1) experimental data.

    data_fit : tuple
        (lambda1, sigma1) fitted data.
    """

    fig, ax = plt.subplots()

    ax.plot(data_exp[0],data_exp[1],"o",label=exp_name,)
    if(data_fit!=None):
        ax.plot(data_fit[0],data_fit[1],"-",linewidth=3,label="Fit",)

    ax.set_xlabel(r"$\lambda_1$")
    ax.set_ylabel(r"$\sigma_1$")
    ax.legend()

    plt.show()

def read_exp_data_compr(filename):
    data = np.loadtxt(filename,delimiter="\t",skiprows=3,
        converters=lambda x: float(x.replace(",", ".")))
    
    L=data[:,1]
    F=data[:,0]
    T=data[:,2]
    
    l_min_pos = np.where(L>=5.0)[0][0]
    l_max_pos = np.where(L==max(L))[0][0]

    print(l_min_pos)
    print(l_max_pos)

    Lcut = L[l_min_pos:l_max_pos]
    Fcut = F[l_min_pos:l_max_pos]
    Tcut = T[l_min_pos:l_max_pos]

    


    Lcut -= Lcut[0]
    Lcut /= 1000#mm to m
    
    Fcut -= Fcut[0]
    Tcut -= Tcut[0]

    Lcut = -Lcut
    Fcut = -Fcut

    #return np.column_stack((Lcut, Fcut, Tcut))
    return (Lcut,Fcut,Tcut)

def read_exp_data_tens(filename):
    data = np.loadtxt(filename,delimiter="\t",skiprows=3,
            converters=lambda x: float(x.replace(",", ".")))
    L=data[:,1]
    F=data[:,0]
    T=data[:,2]

    L = -L

    print(L[np.where(L==max(L))[0][0]])
    l_max_pos = np.where(L==max(L))[0][0]

    Lcut = L[0:l_max_pos]
    Fcut = F[0:l_max_pos]
    Tcut = T[0:l_max_pos]

    print(Lcut[0])
    print(Fcut[0])
    Lcut -= Lcut[0]
    Lcut /= 1000#mm to m
    Fcut -= Fcut[0]
    Tcut -= Tcut[0]

    

    return (Lcut,Fcut,Tcut)


def dimensioned_to_dimensionless(data,A0,L0):
    # L to lambda, lambda=L/L0
    # F to sigma, sigma=F/A0
    return (data[0]/L0+1,data[1]/A0)

def downsample(data, n):
    idx = np.linspace(0, len(data) - 1, n).astype(int)
    return data[idx]

A0CYLMEAN=686.61e-6
L0CYLMEAN=30.23e-3
A0RECTMEAN=30.62e-6
L0RECTMEAN=25.00e-3

filename1 = BASE_DIR / "txtfiles" / "COMPR1.tab"
exp_data1 = read_exp_data_compr(filename1)

filename2 = BASE_DIR / "txtfiles" / "TENS1.tab"
exp_data2 = read_exp_data_tens(filename2)

#plot_data(exp_data,exp_data,xlabel=r"L,mm",ylabel=r"F,N")

exp_data1_dimless = dimensioned_to_dimensionless(exp_data1,A0CYLMEAN,L0CYLMEAN)

exp_data2_dimless = dimensioned_to_dimensionless(exp_data2,A0RECTMEAN,L0RECTMEAN)

#downsampled data
exp_data1_ds = (downsample(exp_data1_dimless[0],20),downsample(exp_data1_dimless[1],20))
exp_data2_ds = (downsample(exp_data2_dimless[0],20),downsample(exp_data2_dimless[1],20))

#plot_data(exp_data_dimless,exp_data_dimless,xlabel=r"lam,1",ylabel=r"Sigma,Pa")

(fit_curve1,fit_params1) = fit_exp_data_OgdenN3(exp_data1_ds)

(fit_curve2,fit_params2) = fit_exp_data_OgdenN3(exp_data2_ds)
#print(f"fit curve: {fit_curve1}")
print("---------------------")
print(f"fit params1: {fit_params1}")
print("---------------------")
print(f"fit params2: {fit_params2}")
print("---------------------")

plot_data(exp_data1_ds,fit_curve1)

plot_data(exp_data2_ds,fit_curve2)



