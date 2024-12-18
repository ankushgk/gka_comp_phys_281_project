# Non-markovian stochastic dynamics of centrosome polarization in immune cells
Author: Ankush G. K.
Course: PHYS 181/281 - Computational Physics
---------------------------------------------
Contains numerical simulations of a simplified 1D model of centrosome polarization towards Immune Synapse in immune cells, specifically analyzing the stochastic dynamics using different noise models. The project implements both white noise and colored noise simulations.
The current setup consists of Euler-Maruyama numerical scheme to solve the Stochastic Differential Equations (SDEs). The codes for testing strong and weak convergence in `test_convergence.py` were adapted from [SDE-higham GitHub Repo](https://github.com/alu042/SDE-higham/tree/master). A few other references include [^1] [^2] [^3] [^4]

The SDEs consist of:
- Linear Drift term
- Position-dependent Noise amplitude
- Optional Colored Noise for finite correlation times
- Reflecting Boundary at origina and Absorbing Boundary at the Immune Synapse.

Files:
- `sdeSolver.py `: contains the implementation of the stochastic differential equation solver
- `test_convergence.py`: contains strong and weak convergence analysis for Geometric Brownian Motion SDE.
- `deterministic_limit.py`: contains the test for decreasing noise in SDE with linear drift and constant diffusion terms.
- `fpt_test.py`: contains first passage time (fpt) analysis and comparison with a known analytical fpt distribution (from Ornstein-Uhlenbeck process).

Requirements:
- Python
- NumPy
- SciPy
- Matplotlib

[^1]: [Higham, D. J. (2001). An algorithmic introduction to numerical simulation of stochastic differential equations. SIAM review, 43(3), 525-546.](https://epubs.siam.org/doi/10.1137/S0036144500378302)
[^2]: [Kloeden, Peter E., et al. Stochastic differential equations. Springer Berlin Heidelberg, 1992.](https://link.springer.com/chapter/10.1007/978-3-662-12616-5_4)
[^3]: [https://github.com/ImperialCollegeLondon/ReCoDe-Euler-Maruyama/tree/main](https://github.com/ImperialCollegeLondon/ReCoDe-Euler-Maruyama/tree/main)
[^4]: [Computational Stochastic Processes Course - Urbain Vaes](https://urbain.vaes.uk/teaching/2020-csp/)
