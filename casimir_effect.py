#!/usr/bin/env python3
"""
Casimir force at finite temperature

Natural units:
    hbar = c = k_B = 1

Geometry:
    medium 1 | medium 3 (gap a) | medium 2


For a nondispersive choice of epsilons, the ratio F(a,T)/F(a,0)
depends only on the dimensionless combination

    z = a T

in natural units.

For physical plotting with a in micrometres and T in kelvin,

    z = a*k_B*T/(hbar*c).

The code produces:
1) F(a,T)/F(a,0) versus a [micrometres] for fixed T [K].
2) F(a,T)/F(a,0) versus dimensionless z=aT, with 0<z<3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

PI = np.pi

# Physical constants for converting a [um], T [K] to z=a*kBT/(hbar*c)
HBAR = 1.054_571_817e-34       # J s
C = 299_792_458.0             # m/s
KB = 1.380_649e-23            # J/K


PERFECT_CONDUCTORS = True

# Dielectric example (used if PERFECT_CONDUCTORS=False)
EPS1 = 4.0
EPS2 = 2.25
EPS3 = 1.0

# Fixed physical temperature 
T_FIXED_K = 300.0

# Separation range: approximately 0 < a < 3 um
A_MIN_UM = 0.01
A_MAX_UM = 3.0
N_A = 70

# Dimensionless range
Z_MIN = 1.0e-3
Z_MAX = 3.0
N_Z = 100

# Numerical cutoffs
Y_CUT = 60.0
X_CUT = 60.0


# ---------------------------------------------------------------------
# Dielectric / reflection coefficients
# ---------------------------------------------------------------------

def fresnel(y, x):
    """
    y = 2 a kappa_3
    x = 2 a zeta_n = 4 pi a T n

    For a vacuum gap eps3=1:
        q3 = y
        qj = sqrt(y^2 + (eps_j-eps3)*x^2)
    """
    q3 = np.sqrt(y*y)

    q1 = np.sqrt(y*y + (EPS1 - EPS3)*x*x)
    q2 = np.sqrt(y*y + (EPS2 - EPS3)*x*x)

    rTE1 = (q3 - q1)/(q3 + q1)
    rTE2 = (q3 - q2)/(q3 + q2)

    rTM1 = (EPS1*q3 - EPS3*q1)/(EPS1*q3 + EPS3*q1)
    rTM2 = (EPS2*q3 - EPS3*q2)/(EPS2*q3 + EPS3*q2)

    return rTE1, rTE2, rTM1, rTM2


def B_dielectric(y, x):
    rTE1, rTE2, rTM1, rTM2 = fresnel(y, x)

    e = np.exp(-y)

    zTE = rTE1*rTE2*e
    zTM = rTM1*rTM2*e

    return zTE/(1.0-zTE) + zTM/(1.0-zTM)


def B_perfect_metal(y, x):
    """
    Ideal conducting plates:
        r_TE(1) r_TE(2) = 1
        r_TM(1) r_TM(2) = 1
    """
    z = np.exp(-y)
    return 2.0*z/(1.0-z)


def B(y, x):
    if PERFECT_CONDUCTORS:
        return B_perfect_metal(y, x)
    return B_dielectric(y, x)


# ---------------------------------------------------------------------
# Zero-temperature pressure in dimensionless form
#
# P(a,0) = p0/a^4
#
# p0 = -1/(32 pi^2) int dx int_x^inf dy y^2 B(y,x)
# ---------------------------------------------------------------------

def dimensionless_P0():
    def inner(x):
        if x >= X_CUT:
            return 0.0

        val, _ = quad(
            lambda y: y*y*B(y, x),
            x, Y_CUT,
            epsabs=1e-10,
            epsrel=3e-7,
            limit=200,
        )
        return val

    val, _ = quad(
        inner,
        0.0, X_CUT,
        epsabs=1e-9,
        epsrel=2e-6,
        limit=200,
    )

    return -val/(32.0*PI**2)


# ---------------------------------------------------------------------
# Finite-temperature pressure in dimensionless form
#
# P(a,T) = -(T/(8 pi a^3)) sum_n' int_xn^inf dy y^2 B
#
# Define z=aT:
#
# P(a,T) = -(z/(8 pi a^4)) sum_n' ...
#
# Thus the ratio P(a,T)/P(a,0) is a function of z alone for
# nondispersive epsilons.
# ---------------------------------------------------------------------

def dimensionless_PT(z, tol=1e-10, nmax=100000):
    if z <= 0:
        raise ValueError("z must be positive.")

    dx = 4.0*PI*z

    def term(n):
        x = dx*n
        if x >= Y_CUT:
            return 0.0

        val, _ = quad(
            lambda y: y*y*B(y, x),
            x, Y_CUT,
            epsabs=1e-11,
            epsrel=2e-8,
            limit=200,
        )
        return val

    total = 0.5*term(0)

    n = 1
    while n < nmax:
        tn = term(n)
        total += tn

        if n*dx > 5.0 and abs(tn) < tol*max(1.0, abs(total)):
            tn1 = term(n+1)
            if abs(tn1) < tol*max(1.0, abs(total)):
                break

        n += 1

    if n == nmax:
        raise RuntimeError("Matsubara sum did not converge.")

    return -z*total/(8.0*PI)


# ---------------------------------------------------------------------
# Ratio R(z)=P(T)/P(0)
# ---------------------------------------------------------------------

print("Computing zero-temperature dimensionless pressure...")
P0_DIM = dimensionless_P0()
print(f"P0 * a^4 = {P0_DIM:.12e}")


def ratio_from_z(z):
    return dimensionless_PT(z)/P0_DIM


# ---------------------------------------------------------------------
# Fixed physical temperature, vary a from 0 to 3 micrometres.
# ---------------------------------------------------------------------

def plot_vs_distance():
    a_um = np.geomspace(A_MIN_UM, A_MAX_UM, N_A)

    # Physical conversion:
    # z = a*kB*T/(hbar*c)
    a_m = a_um*1e-6
    z = a_m*KB*T_FIXED_K/(HBAR*C)

    R = np.array([ratio_from_z(zi) for zi in z])

    np.savetxt(
        "ratio_vs_distance.txt",
        np.column_stack((a_um, R)),
        header="a_um   F(a,T)/F(a,0)"
    )

    plt.figure(figsize=(8, 5.5))
    plt.semilogx(a_um, R, "o", ms=4.5, color="purple")
    plt.xlabel(
        r"Distancia entre placas $a\,[\mu{\rm m}]$",
        fontsize=16
        )
    plt.ylabel(
        r"$F(a,T)/F(a,0)$",
        fontsize=16
    )   

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)    
    plt.xlim(A_MIN_UM, A_MAX_UM)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Dimensionless plot around z=1, 0<z<3.
# ---------------------------------------------------------------------

def plot_vs_z():
    z_values = np.linspace(Z_MIN, Z_MAX, N_Z)
    R = np.array([ratio_from_z(zi) for zi in z_values])

    np.savetxt(
        "ratio_vs_z.txt",
        np.column_stack((z_values, R)),
        header="z=a*T   F(a,T)/F(a,0)"
    )

    plt.figure(figsize=(8, 5.5))
    plt.plot(z_values, R, "-", lw=2)
    plt.xlabel(r"Variable adimensional $z=aT$", fontsize=16)
    plt.ylabel(r"$F(a,T)/F(a,0)$", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(Z_MIN, Z_MAX)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\nModel:")
    if PERFECT_CONDUCTORS:
        print("  Perfect conductors")
    else:
        print(f"  Dielectric: eps1={EPS1}, eps2={EPS2}, eps3={EPS3}")

    print(f"\nTask 1: fixed T = {T_FIXED_K:g} K, 0 < a < 3 micrometres")
    plot_vs_distance()

    print("\nTask 2: dimensionless 0 < z=aT < 3")
    plot_vs_z()
