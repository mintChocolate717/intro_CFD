import numpy as np
import matplotlib.pyplot as plt


def compute_fluxes(flux, sol: np.ndarray, kh: float, uL:float, uR:float, method:str) -> np.ndarray:
    """
    Computes numerical fluxes at cell interfaces for 1D conservation laws using a specified method.

    Args:
        flux (function): Flux function f(u).
        sol (np.ndarray): Array of cell-averaged solution values.
        kh (float): Ratio of time step to spatial step (k/h).
        uL (float): Left boundary condition (u at x = -L).
        uR (float): Right boundary condition (u at x = +L).
        method (str): Numerical scheme to use. One of {'FOU', 'LW', 'RM', 'MC'}.

    Returns:
        np.ndarray: Flux values at each cell interface (length N+1 for N cells).

    Raises:
        ValueError: If an unsupported method is specified.
    """
    # number of cells
    N = len(sol)
    # initialize fluxes array, there are N+1 fluxes for N cells
    fluxes = np.zeros(N+1, dtype = float)
    
    match method.upper():
        # MacCormack
        case 'MC':
            # interior fluxes
            for i in range(1, N):
                sol_star = sol[i-1] - kh * (flux(sol[i]) - flux(sol[i-1]))
                fluxes[i] = 0.5 * (flux(sol[i]) + flux(sol_star))
            # Boundary Conditions
            # Left Boundary
            sol_star_left = uL - kh * (flux(sol[0]) - flux(uL))
            fluxes[0] = 0.5 * (flux(sol[0]) + flux(sol_star_left))
            # Right Boundary
            sol_star_right = sol[-1] - kh * (flux(uR) - flux(sol[-1]))
            fluxes[-1] = 0.5 * (flux(sol[-1]) + flux(sol_star_right))

        # Richtmyer
        case 'RM':
            for i in range(1, N):
                fluxes[i] = flux(0.5 * (sol[i] + sol[i-1]) - 0.5 * kh * (flux(sol[i]) - flux(sol[i-1])))
            # Boundary Conditions
            # Left Boundary
            U_star_left = 0.5*(uL + sol[0]) - 0.5*kh*(flux(sol[0]) - flux(uL))
            fluxes[0] = flux(U_star_left)
            # Right Boundary
            U_star_right  = 0.5*(sol[-1] + uR) - 0.5*kh*(flux(uR) - flux(sol[-1]))
            fluxes[-1] = flux(U_star_right)

        # Lax-Wendroff
        case 'LW':
            # flux computation
            for i in range(1, N): # for INTERIOR fluxes
                fluxes[i] = 0.5 * (flux(sol[i-1]) + flux(sol[i])) - 0.5 * kh * (0.5 * (sol[i] + sol[i-1])) * (flux(sol[i]) - flux(sol[i-1]))
            # Boundary Conditions
            # Left Boundary
            fluxes[0] = flux(uL)
            # Right Boundary
            fluxes[-1] = flux(uR)

        # First-Order-Upwind
        case 'FOU':
            # flux computation
            for i in range(1, N): # for INTERIOR fluxes
                # flow is moving to the right --> upwind state is U(i-1) (cell to the left)
                if sol[i] > 0:
                    # flux computation - we use everything to the left of each interface
                    fluxes[i] = flux(sol[i-1])
                # flow is moving to the left --> upwind state is U(i) (cell to the right)
                elif sol[i] < 0:
                    # flux computation - we use everything to the right of each interface
                    fluxes[i] = flux(sol[i]) # not it+1 because an interface is inbetween two solutions
            # Boundary Conditions
            # Left Boundary
            fluxes[0] = flux(uL) if sol[0] > 0 else flux(sol[0]) if sol[0] < 0 else 0
            # Right Boundary
            fluxes[-1] = flux(sol[-1]) if sol[-1] > 0 else flux(uR) if sol[-1] < 0 else 0
        
        case _:
            raise ValueError(f"Unknown method: {method}. Supported methods are 'FOU', 'LW', 'RM', and 'MCK'")

    return fluxes


def update_sol(sol: np.ndarray, fluxes: np.ndarray, kh: float) -> np.ndarray:
    """
    Advances the solution by one time step using the finite volume update formula.

    Computes:
        U_i^{n+1} = U_i^n - (k/h) * (F_{i+1/2} - F_{i-1/2})

    Args:
        sol (np.ndarray): Cell-averaged solution at current time step (length N).
        fluxes (np.ndarray): Numerical fluxes at cell interfaces (length N+1).
        kh (float): Ratio of time step to spatial step (Courant number, k/h).

    Returns:
        np.ndarray: Updated solution array at the next time step (length N).
    """
    return sol - kh * (fluxes[1:] - fluxes[0:-1])


def solve_pde(u0, flux_func, Courant: float, T: float, uL: float, uR: float, xL: float, xR: float, ncells: int, method: str) -> np.ndarray:
    """
    Solves a 1D scalar conservation law using a specified flux function and numerical method.

    Args:
        u0 (callable): Initial condition function u(x).
        flux_func (callable): Flux function f(u).
        Courant (float): Courant number (CFL condition), typically < 1.
        T (float): Final simulation time.
        uL (float): Left boundary condition (u at x = xL).
        uR (float): Right boundary condition (u at x = xR).
        xL (float): Left boundary of the spatial domain.
        xR (float): Right boundary of the spatial domain.
        ncells (int): Number of spatial cells.
        method (str): Numerical scheme to use. One of {'FOU', 'LW', 'RM', 'MC'}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - x: Cell-centered spatial grid (length ncells).
            - sol: Solution at final time T (length ncells).
    """
    # spatial discretization
    h = (xR - xL) / ncells # get spatial resolution
    x = np.linspace(xL, xR, ncells) # discretize space
    x = xL + (np.arange(ncells) + 0.5) * h

    t = 0.0
    sol = u0(x) # start with initial condition

    while t < T:
        # time step size
        k = Courant * h / np.max(np.abs(sol))
        fluxes = compute_fluxes(flux=flux_func, sol=sol, kh=k/h, uL=uL, uR=uR, method=method)
        sol = update_sol(sol, fluxes, k/h)
        t += k

    return x, sol


def main_ivp_a():
    # Fixed parameters
    xL, xR, Courant, T = -2.0, 2.0, 0.75, 2/3
    methods = ["FOU", "LW", "RM", "MC"]
    full_names = ['First-Order-Upwind', 'Lax-Wendroff', 'Richtmyer', 'MacCormack']
    markers = ["o--", "s--", "^--", "d--"]
    N_values = [50, 200, 800]

    # Flux and initial condition
    flux = lambda u: 0.5 * u**2
    u0 = lambda x: np.where(x < 0.0, 2.0, 1.0)
    uL, uR = 2.0, 1.0
    Ts = 0.5 * (uL + uR) * T
    u_exact = lambda x: np.where(x < Ts, 2.0, 1.0)

    # Loop over spatial resolutions
    for N in N_values:
        x_fine = np.linspace(xL, xR, 2000)
        plt.figure(figsize=(6, 4))
        plt.plot(x_fine, u_exact(x_fine), 'k-', lw=2, label="Exact")

        for method, mkr, name in zip(methods, markers, full_names):
            x_num, u_num = solve_pde(u0, flux, Courant, T, uL, uR, xL, xR, N, method)
            plt.plot(x_num, u_num, mkr, label=name, markersize=3)

        plt.title(f"Burgers IVP (a): Right-Travelling Step at T={T:.3f}, N={N}")
        plt.xlabel("x")
        plt.ylabel(r"$u(x)\quad \text{or} \quad U_i^n$")
        plt.ylim(0.8, 2.5)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main_ivp_b():
    # Fixed parameters
    xL, xR, Courant, uL, uR = -2.0, 2.0, 0.75, 1.0, 1.0
    methods = ["FOU", "LW", "RM", "MC"]
    full_names = ['First-Order Upwind', 'Lax–Wendroff', 'Richtmyer', 'MacCormack']
    markers = ["o--", "s--", "^--", "d--"]
    N_values = [50, 200, 800]
    T = 1/np.pi - 1e-5

    # Flux and initial condition
    flux = lambda u: 0.5 * u**2
    u0 = lambda x: np.where(np.abs(x) <= 0.5, 0.5 * (np.cos(2*np.pi*x) + 1) + 1.0, 1.0)

    # Load analytical solution (adjust path as needed)
    x_dat, u_dat = np.loadtxt('/Users/kis/Desktop/COE347/intro_CFD/hw8/hw8.material/hump_analytical.dat', unpack=True)

    # Loop over spatial resolutions
    for N in N_values:
        plt.figure(figsize=(6, 4))
        plt.plot(x_dat, u_dat, 'k-', lw=2, label='Exact')

        for method, mkr, name in zip(methods, markers, full_names):
            x_num, u_num = solve_pde(u0, flux, Courant, T, uL, uR, xL, xR, N, method)
            plt.plot(x_num, u_num, mkr, ms=3, label=name)

        plt.title(f"Burgers IVP (b): Wave Breaking at T={T:.5f}, N={N}")
        plt.xlabel("x")
        plt.ylabel(r"$u(x)\;\text{and}\;U_i^n$")
        plt.ylim(0.8, 2.5)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main_ivp_b()