#!/Users/Luke/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
from amplification_factor import amplification_factor
from lax_friedrichs import lax_friedrichs
from wave_solve import wave_solve
from intro_CFD.hw6.Luke_wave_driver import wave_driver


def relative_errors():
    '''
    Plots the relative magnitude and phase errors based on
    the type of method
    '''
    Gh = lax_friedrichs
    ss = [0.25, 0.5, 0.75, 1.0]
    title = 'Lax-Friedrichs'
    fig = [1, 2, 3, 4]
    amplification_factor(fig, title, Gh, ss)
    
def solve_equation():
    method = 'lax-wendroff'
    M = 1  # Intermediate solutions (number of time steps)
    sigma = 0.25  # Courant number
    n = 30  # Number of interior points
    
    c = 1  # Advective speed
    L = 4 * np.pi  # Computational domain [0,L]
    T = 2 * 2 * np.pi  # End time

    # Initial conditions
    def u0(x):
        return np.sin(x)  # Anonymous function

    # Solve
    out = wave_solve(c, L, n, sigma, T, M, u0, method)

    # Plot
    xx = np.linspace(0, L, 1000)
    print(out['U'].shape[1])
    for i in range(out['U'].shape[1]):
        exact = u0(xx - out['TT'][i])
        plt.plot(out['x'], out['U'][:, i], 'ko-', xx, exact, 'r-')
        plt.axis([0, L, -1.1, 1.1])
        plt.xlabel('x')
        plt.ylabel('u(x) and numerical solution')
        plt.title(f'Time is {out["TT"][i]}')
        plt.savefig('plots/PLOT.png')

    fout = f'{method}_n{n}_sigma{sigma}.dat'
    np.savetxt(fout, np.column_stack([out['x'], out['U']]), delimiter=' ', fmt='%.6e')

    fexact = 'exact.dat'
    np.savetxt(fexact, np.column_stack([xx, exact]), delimiter=' ', fmt='%.6e')

if __name__ == '__main__':
    solve_equation()