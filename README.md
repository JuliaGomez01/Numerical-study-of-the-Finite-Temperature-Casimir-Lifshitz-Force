# Numerical Study of the Finite-Temperature Casimir–Lifshitz Force

This repository contains Python code for the numerical evaluation and
visualization of the Casimir–Lifshitz force at finite temperature.

The calculation is based on the Matsubara-frequency formulation of the
finite-temperature Casimir force and includes:

- Finite-temperature Casimir–Lifshitz force calculations.
- The primed Matsubara sum, including the 1/2 weight of the n=0 term.
- The general dielectric-medium formulation.
- The ideal perfectly conducting limit.
- The dependence of the force on the plate separation `a`.
- The dimensionless thermal parameter `z = aT`.
- The ratio F(a,T)/F(a,0) to quantify thermal corrections.
- Numerical analysis of the low-temperature `1 + C z^4` regime.
- The high-temperature/classical linear regime.
- Visualization of the thermal correction for different temperatures.

All calculations are performed in natural units (ℏ = c = k_B = 1),
with conversion to SI units when plotting the physical separation in
micrometres and temperature in kelvin.
