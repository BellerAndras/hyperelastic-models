###Fitting experimental data to compressible hyperelastic models

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, root_scalar

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def fit_exp_data_OgdenN3(data):
    """
    Fit a compressible Ogden N=3 model to uniaxial experimental data.

    Parameters
    ----------
    data : tuple
        Experimental data:

            (lambda1, sigma1)

        where lambda1 and sigma1 are 1D numpy arrays.

    Returns
    -------
    fit_data : tuple
        Fitted data:

            (lambda1, sigma1_fit)

    params : dict
        Fitted Ogden parameters.
    """

    lambda1_exp, sigma1_exp = data

    lambda1_exp = np.asarray(lambda1_exp, dtype=float)
    sigma1_exp = np.asarray(sigma1_exp, dtype=float)

    if lambda1_exp.ndim != 1 or sigma1_exp.ndim != 1:
        raise ValueError("lambda1 and sigma1 must be 1D arrays.")

    if len(lambda1_exp) != len(sigma1_exp):
        raise ValueError("lambda1 and sigma1 must have the same length.")

    if len(lambda1_exp) == 0:
        raise ValueError("Experimental data are empty.")

    # ------------------------------------------------------------------
    # Symbolic variables
    # ------------------------------------------------------------------

    lam1, lam2, lam3 = sp.symbols("lam1 lam2 lam3",positive=True,)

    lamT = sp.symbols("lamT",positive=True,)

    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3",positive=True,)

    alpha1, alpha2, alpha3 = sp.symbols("alpha1 alpha2 alpha3",positive=True,)

    D1, D2, D3 = sp.symbols("D1 D2 D3",positive=True,)

    # ------------------------------------------------------------------
    # Deformation
    # ------------------------------------------------------------------

    J = lam1 * lam2 * lam3

    lam1_bar = lam1 * J**(-sp.Rational(1, 3))
    lam2_bar = lam2 * J**(-sp.Rational(1, 3))
    lam3_bar = lam3 * J**(-sp.Rational(1, 3))

    # ------------------------------------------------------------------
    # Ogden deviatoric energy
    # ------------------------------------------------------------------

    W_dev = (
         2 * mu1 / alpha1**2* (lam1_bar**alpha1+ lam2_bar**alpha1+ lam3_bar**alpha1- 3 )
        +2 * mu2 / alpha2**2* (lam1_bar**alpha2+ lam2_bar**alpha2+ lam3_bar**alpha2- 3)
        +2 * mu3 / alpha3**2* ( lam1_bar**alpha3+ lam2_bar**alpha3+ lam3_bar**alpha3- 3))

    # ------------------------------------------------------------------
    # Volumetric energy
    # ------------------------------------------------------------------

    W_vol = ((J - 1)**2 / D1+ (J - 1)**4 / D2+ (J - 1)**6 / D3)

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

    P1_uniax = sp.simplify(P1.subs({lam2: lamT,lam3: lamT,}))

    P2_uniax = sp.simplify(P2.subs({lam2: lamT,lam3: lamT,}))

    # ------------------------------------------------------------------
    # Numerical functions
    # ------------------------------------------------------------------

    arguments = (lam1,lamT,mu1,alpha1,D1,mu2,alpha2,D2,mu3,alpha3,D3,)

    P1_fun = sp.lambdify(arguments,P1_uniax,modules="numpy",)

    P2_fun = sp.lambdify(arguments,P2_uniax,modules="numpy",)

    # ------------------------------------------------------------------
    # Parameter scaling
    #
    # The optimizer works with D / 1e-9 rather than D directly.
    # ------------------------------------------------------------------

    def unpack_params(p):
        return (
            p[0], p[1], p[2] * 1e-9,
            p[3], p[4], p[5] * 1e-9,
            p[6], p[7], p[8] * 1e-9,)

    # ------------------------------------------------------------------
    # Solve transverse stretch
    # ------------------------------------------------------------------

    def solve_transverse(lam, previous, params):

        def transverse_stress(lamT_value):
            return P2_fun(lam,lamT_value,*params,)

        # Normally the previous solution is an excellent starting point.
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

        # Robust fallback: search for a sign change.
        search = np.linspace(0.2, 2.0, 100)
        values = np.array([
            transverse_stress(x)
            for x in search])

        for i in range(len(search) - 1):

            if values[i] * values[i + 1] < 0:

                solution = root_scalar(
                    transverse_stress,
                    bracket=(search[i], search[i + 1]),
                    method="brentq",
                )

                if solution.converged:
                    return solution.root

        raise RuntimeError(
            f"Could not solve transverse stretch at "
            f"lambda1 = {lam:.6g}")

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def predict(lambda1_values, params):

        lambda1_values = np.asarray(
            lambda1_values,
            dtype=float,
        )

        lambdaT_values = np.empty_like(lambda1_values)
        sigma_values = np.empty_like(lambda1_values)

        previous = 1.0

        for i, lam in enumerate(lambda1_values):

            lambdaT = solve_transverse(
                lam,
                previous,
                params,)

            # First Piola-Kirchhoff stress
            P1_value = P1_fun(
                lam,
                lambdaT,
                *params,)

            # Experimental stress is assumed to be nominal stress,
            # so compare directly with P1.
            sigma_values[i] = P1_value

            lambdaT_values[i] = lambdaT
            previous = lambdaT

        return lambdaT_values, sigma_values

    # ------------------------------------------------------------------
    # Residual function
    # ------------------------------------------------------------------

    def residuals(p):

        params = unpack_params(p)

        _, sigma_pred = predict(lambda1_exp,params,)

        return sigma_pred - sigma1_exp

    # ------------------------------------------------------------------
    # Initial guess
    #
    # D values are represented in units of 1e-9.
    # ------------------------------------------------------------------

    p0 = np.array([
        1e5, 1.0, 1.0,
        1e5, 2.0, 1.0,
        1e5, 3.0, 1.0,
    ])

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    lower = np.array([
        1e3,  0.01, 1e-3,
        1e3,  0.01, 1e-3,
        1e3,  0.01, 1e-3,
    ])

    upper = np.array([
        1e7, 100.0, 1e3,
        1e7, 100.0, 1e3,
        1e7, 100.0, 1e3,
    ])

    # ------------------------------------------------------------------
    # Least-squares fit
    # ------------------------------------------------------------------

    result = least_squares(residuals,p0,bounds=(lower, upper),x_scale="jac",)

    if not result.success:
        raise RuntimeError(
            f"Ogden N=3 fit failed: {result.message}"
        )

    # ------------------------------------------------------------------
    # Extract fitted parameters
    # ------------------------------------------------------------------

    fitted = unpack_params(result.x)

    (   mu1_fit, alpha1_fit, D1_fit,
        mu2_fit, alpha2_fit, D2_fit,
        mu3_fit, alpha3_fit, D3_fit,
    ) = fitted

    params = {
        "mu1": mu1_fit,
        "alpha1": alpha1_fit,
        "D1": D1_fit,

        "mu2": mu2_fit,
        "alpha2": alpha2_fit,
        "D2": D2_fit,

        "mu3": mu3_fit,
        "alpha3": alpha3_fit,
        "D3": D3_fit,

        "result": result,
    }

    # ------------------------------------------------------------------
    # Calculate fitted curve
    # ------------------------------------------------------------------

    _, sigma_fit = predict(
        lambda1_exp,
        fitted,)

    fit_data = (
        lambda1_exp,
        sigma_fit,)

    return fit_data, params

def fit_exp_data_OgdenN1(data):
    """
    Fit a compressible Ogden N=1 model to uniaxial experimental data.

    Parameters
    ----------
    data : tuple
        Experimental data:

            (lambda1, sigma1)

        where lambda1 and sigma1 are 1D numpy arrays.

    Returns
    -------
    fit_data : tuple
        Fitted data:

            (lambda1, sigma1_fit)

    params : dict
        Fitted Ogden parameters.
    """

    lambda1_exp, sigma1_exp = data

    lambda1_exp = np.asarray(lambda1_exp, dtype=float)
    sigma1_exp = np.asarray(sigma1_exp, dtype=float)

    if lambda1_exp.ndim != 1 or sigma1_exp.ndim != 1:
        raise ValueError("lambda1 and sigma1 must be 1D arrays.")

    if len(lambda1_exp) != len(sigma1_exp):
        raise ValueError("lambda1 and sigma1 must have the same length.")

    if len(lambda1_exp) == 0:
        raise ValueError("Experimental data are empty.")

    # ------------------------------------------------------------------
    # Symbolic variables
    # ------------------------------------------------------------------

    lam1, lam2, lam3 = sp.symbols("lam1 lam2 lam3",positive=True,)

    lamT = sp.symbols("lamT",positive=True,)

    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3",positive=True,)

    alpha1, alpha2, alpha3 = sp.symbols("alpha1 alpha2 alpha3",positive=True,)

    D1, D2, D3 = sp.symbols("D1 D2 D3",positive=True,)

    # ------------------------------------------------------------------
    # Deformation
    # ------------------------------------------------------------------

    J = lam1 * lam2 * lam3

    lam1_bar = lam1 * J**(-sp.Rational(1, 3))
    lam2_bar = lam2 * J**(-sp.Rational(1, 3))
    lam3_bar = lam3 * J**(-sp.Rational(1, 3))

    # ------------------------------------------------------------------
    # Ogden deviatoric energy
    # ------------------------------------------------------------------

    W_dev = (
         2 * mu1 / alpha1**2* (lam1_bar**alpha1+ lam2_bar**alpha1+ lam3_bar**alpha1- 3 )
        +2 * mu2 / alpha2**2* (lam1_bar**alpha2+ lam2_bar**alpha2+ lam3_bar**alpha2- 3)
        +2 * mu3 / alpha3**2* ( lam1_bar**alpha3+ lam2_bar**alpha3+ lam3_bar**alpha3- 3))

    # ------------------------------------------------------------------
    # Volumetric energy
    # ------------------------------------------------------------------

    W_vol = ((J - 1)**2 / D1+ (J - 1)**4 / D2+ (J - 1)**6 / D3)

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

    P1_uniax = sp.simplify(P1.subs({lam2: lamT,lam3: lamT,}))

    P2_uniax = sp.simplify(P2.subs({lam2: lamT,lam3: lamT,}))

    # ------------------------------------------------------------------
    # Numerical functions
    # ------------------------------------------------------------------

    arguments = (lam1,lamT,mu1,alpha1,D1,mu2,alpha2,D2,mu3,alpha3,D3,)

    P1_fun = sp.lambdify(arguments,P1_uniax,modules="numpy",)

    P2_fun = sp.lambdify(arguments,P2_uniax,modules="numpy",)

    # ------------------------------------------------------------------
    # Parameter scaling
    #
    # The optimizer works with D / 1e-9 rather than D directly.
    # ------------------------------------------------------------------

    def unpack_params(p):
        return (
            p[0], p[1], p[2] * 1e-9,
            p[3], p[4], p[5] * 1e-9,
            p[6], p[7], p[8] * 1e-9,)

    # ------------------------------------------------------------------
    # Solve transverse stretch
    # ------------------------------------------------------------------

    def solve_transverse(lam, previous, params):

        def transverse_stress(lamT_value):
            return P2_fun(lam,lamT_value,*params,)

        # Normally the previous solution is an excellent starting point.
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

        # Robust fallback: search for a sign change.
        search = np.linspace(0.2, 2.0, 100)
        values = np.array([
            transverse_stress(x)
            for x in search])

        for i in range(len(search) - 1):

            if values[i] * values[i + 1] < 0:

                solution = root_scalar(
                    transverse_stress,
                    bracket=(search[i], search[i + 1]),
                    method="brentq",
                )

                if solution.converged:
                    return solution.root

        raise RuntimeError(
            f"Could not solve transverse stretch at "
            f"lambda1 = {lam:.6g}")

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def predict(lambda1_values, params):

        lambda1_values = np.asarray(
            lambda1_values,
            dtype=float,
        )

        lambdaT_values = np.empty_like(lambda1_values)
        sigma_values = np.empty_like(lambda1_values)

        previous = 1.0

        for i, lam in enumerate(lambda1_values):

            lambdaT = solve_transverse(
                lam,
                previous,
                params,)

            # First Piola-Kirchhoff stress
            P1_value = P1_fun(
                lam,
                lambdaT,
                *params,)

            # Experimental stress is assumed to be nominal stress,
            # so compare directly with P1.
            sigma_values[i] = P1_value

            lambdaT_values[i] = lambdaT
            previous = lambdaT

        return lambdaT_values, sigma_values

    # ------------------------------------------------------------------
    # Residual function
    # ------------------------------------------------------------------

    def residuals(p):

        params = unpack_params(p)

        _, sigma_pred = predict(lambda1_exp,params,)

        return sigma_pred - sigma1_exp

    # ------------------------------------------------------------------
    # Initial guess
    #
    # D values are represented in units of 1e-9.
    # ------------------------------------------------------------------

    p0 = np.array([
        1e5, 1.0, 1.0,
        1e5, 2.0, 1.0,
        1e5, 3.0, 1.0,
    ])

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    lower = np.array([
        1e3,  0.01, 1e-3,
        1e3,  0.01, 1e-3,
        1e3,  0.01, 1e-3,
    ])

    upper = np.array([
        1e7, 100.0, 1e3,
        1e7, 100.0, 1e3,
        1e7, 100.0, 1e3,
    ])

    # ------------------------------------------------------------------
    # Least-squares fit
    # ------------------------------------------------------------------

    result = least_squares(residuals,p0,bounds=(lower, upper),x_scale="jac",)

    if not result.success:
        raise RuntimeError(
            f"Ogden N=3 fit failed: {result.message}"
        )

    # ------------------------------------------------------------------
    # Extract fitted parameters
    # ------------------------------------------------------------------

    fitted = unpack_params(result.x)

    (   mu1_fit, alpha1_fit, D1_fit,
        mu2_fit, alpha2_fit, D2_fit,
        mu3_fit, alpha3_fit, D3_fit,
    ) = fitted

    params = {
        "mu1": mu1_fit,
        "alpha1": alpha1_fit,
        "D1": D1_fit,

        "mu2": mu2_fit,
        "alpha2": alpha2_fit,
        "D2": D2_fit,

        "mu3": mu3_fit,
        "alpha3": alpha3_fit,
        "D3": D3_fit,

        "result": result,
    }

    # ------------------------------------------------------------------
    # Calculate fitted curve
    # ------------------------------------------------------------------

    _, sigma_fit = predict(
        lambda1_exp,
        fitted,)

    fit_data = (
        lambda1_exp,
        sigma_fit,)

    return fit_data, params

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

(fit_curve1,fit_params1) = fit_exp_data_NH(exp_data1_ds)

(fit_curve2,fit_params2) = fit_exp_data_NH(exp_data2_ds)
#print(f"fit curve: {fit_curve1}")
print("---------------------")
print(f"fit params1: {fit_params1}")
print("---------------------")
print(f"fit params2: {fit_params2}")
print("---------------------")

plot_data(exp_data1_ds,fit_curve1)

plot_data(exp_data2_ds,fit_curve2)



