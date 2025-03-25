import numpy as np
import matplotlib.pyplot as plt
from wave_solver import wave_solve

# --------------------------
# Problem Setup
# --------------------------
fexact = "exact.dat"  # File to store the exact solution

c = 1        # Advective speed
L = 2 * np.pi  # Computational domain [0, L]
T = 2 * 2 * np.pi  # End time
M = 0       # Number of intermediate solutions

sigma = 1.25  # Courant number
n = 25  # Number of interior points

# method = "forward-upwind"  
# method = "implicit-central"
#method = "beam-warming"
method = "lax-wendroff"

# Initial condition function
u0 = lambda x: np.sin(x)

# --------------------------
# Solve the PDE
# --------------------------
out = wave_solve(c, L, n, sigma, T, M, u0, method)

# --------------------------
# Plot the results
# --------------------------
xx = np.linspace(0, L, 1000)  # Fine grid for exact solution
exact = np.array([u0(xx - t) for t in out['TT']]).T  # Compute exact solutions

for i in range(out['U'].shape[1]):  # Loop over stored time snapshots
    plt.plot(out['x'], out['U'][:, i], 'ko-', label="Numerical Solution")
    plt.plot(xx, u0(xx - out['TT'][i]), 'r-', label="Exact Solution")
    plt.axis([0, L, -1.1, 1.1])
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Time = {out['TT'][i]:.6f}")
    plt.legend()
    plt.show()  # Display the plot
    #input("Press Enter to continue...")  # Pause for user interaction

# --------------------------
# Save results to .dat files
# --------------------------
fout = f"Choi_{method}_n{n}_sigma{sigma:.6f}.dat"
np.savetxt(fout, np.column_stack([out['x'], out['U'][:, 0], out['U'][:, -1]]),
           delimiter=" ", fmt="%.6e")

np.savetxt(fexact, np.column_stack([xx, exact]), delimiter=" ", fmt="%.6e")
