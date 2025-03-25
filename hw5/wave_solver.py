import numpy as np
from sys import exit


# Adjust print options
np.set_printoptions(linewidth=250)  # Adjust the width limit

def wave_solve(c: float, L:float, n:int, Courant:float, T:float, M:int, u0, method:str) -> dict:
    """
    Parameters:
        c : float
            Advective speed (assumed positive).
        L : float
            Domain size [0, L].
        n : int
            Number of interior grid points.
        Courant : float
            Courant number.
        T : float
            Final time.
        M : int
            Number of solution snapshots between [0, T]
        u0 : callable
            Initial condition function: u0(x_array) -> array of same length.
        method : str
            One of {'forward-upwind', 'implicit-central', 'beam-warming', 'lax-wendroff'}.
`
    Returns
        dict:
            A dictionary with:
                out['h']   : float, spatial grid spacing
                out['k']   : float, base time step size
                out['l']   : int,   total number of time steps taken
                out['x']   : 1D array, spatial grid from 0 to L
                out['TT']  : 1D array, times at which solutions are recorded
                out['U']   : 2D array, solution snapshots; out['U'][:, j] is the solution at time out['TT'][j]
    """
    # Initialize output
    result = {}
    
    ###############################################
    # 1. Spatial grid
    h = L / (n + 1)  # get the space-step-size
    result['h'] = h 
    
    # generate spatial points to compute solutions about:
    spatial_grid = np.linspace(0, L, n+2)  # n interior points => n+2 total
    result['x'] = spatial_grid
    N = len(spatial_grid)  # number of total points, should be equal to n+2
    exit(f"Total Number of Points (N = {N}) doesn't equal Number of Interior Points (n) + 2 = {n+2}") if N != n+2 else None
    ###############################################
    # 2. Times at which we store solutions
    TT = np.linspace(0, T, M+2)  # from 0 to T, M+2 points
    result['TT'] = TT
    ###############################################
    # 3. ideal Time-step size from user-defined Courant number
    k = Courant * h / c # biggest time-step-size that can be taken while preserving stability
    result['k'] = k
    ###############################################
    # 4. Build/update coeff matrix based on method
    match method.lower():
        case'forward-upwind':
            if c < 0:
                raise ValueError("Please specify a positive advective speed.")
            # Create an N×N matrix A for delta U, for (-U^n_i + U^n_i-1)
            A = -np.diag(np.ones(N)) + np.diag(np.ones(N-1), k=-1)
            # Periodic boundary condition on U(1) = U0 equals U_N
            A[0, n] = 1.0

        case 'implicit-central':
            Courant_stable = min(TT[1] - TT[0], k) * c / h
            # need ot create a Coeff * U_new = U_old system.
            A = np.zeros(shape = (N, N))
            # main diagonals are 2s:
            np.fill_diagonal(A, 2)
            # sub-diagonals are -Courant
            np.fill_diagonal(A[1:], -Courant_stable)
            # super-diags are +Courant
            np.fill_diagonal(A[:-1, 1:], Courant_stable)
            # apply bounary conditions
            A[0, n] = -Courant_stable # for left boundary, U_-1  = U_n
            A[-1,1] = Courant_stable # for right boundary, U_n+2 = U_1

        case 'beam-warming':
            # create N by N matrix
            A = np.zeros(shape=(N, N))
            alpha = (Courant**2 - Courant) / 2 # coeff in front of U_i-2 term
            beta = 2*Courant - Courant**2 # coeff in front of U_i-1 term
            gamma = (Courant ** 2 - 3*Courant) / 2 # coeff. in front of U_i term:
            # main diagonals
            np.fill_diagonal(A, val = gamma)
            # sub diagonals
            np.fill_diagonal(A[1:], beta)
            # sub-sub diagonal
            np.fill_diagonal(A[2:], alpha)
            # apply Periodic BCs:
            # for the i = 0 or the 1st row:
            ## i = -2 -> n - 1
            A[0, n-1] = alpha
            ## i = -1 -> n
            A[0, n] = beta
            # for the i = 1 or the 2nd row:
            ## i = -1 -> i = n
            A[1, n] = alpha

        case 'lax-wendroff':
            # create N by N matrix
            A = np.zeros(shape=(N, N))
            alpha = (Courant**2 + Courant) / 2 # coeff in front of U_i-1 term
            beta = -Courant ** 2 # coeff in front of U_i term
            gamma = (Courant ** 2 - Courant) / 2 # coeff. in front of U_i+1 term:
            # main diagonals
            np.fill_diagonal(A, val = beta)
            # sub diagonals
            np.fill_diagonal(A[1:], alpha)
            # super diagonal
            np.fill_diagonal(A[:-1, 1:], gamma)
            # apply Periodic BCs:
            # for the i = 0 or the 1st row:
            ## i = -1 -> n
            A[0, n] = alpha
            # for the last row:
            ## i = n+2 -> i = 1
            A[-1, 1] = gamma

        case _:
            raise ValueError("Unknown method.")
    ###############################################
    # 5. Solution Calculations
    # Prepare storage for solutions at each time in TT
    U = np.zeros((N, M+2)) # for N = n+2 points and M + 2 time snap-shots
    ## indices for use:
    num_time_steps = 0   # total number of time steps
    j = 0  # accessing solution matrix column
    t = 0.0 # time
    ## compute inital condition on spatial grid:
    U_temp = u0(spatial_grid)  # 1D NumPy array for t = 0, j = 0
    exit("u0(x) must return an array of length N = n+2.") if len(U_temp) != N else None
    U[:, j] = U_temp # initial condition at t = 0 or j = 0
    j += 1 # advance to next column.

    ## Time integration
    while (t < TT[-1]): # run until the end time
        # we already computed max 'k' that can be used for stable solution
        # however, user might want to store the sol at smaller increment that is still stable
        # we only use user defined time step size from spatial_grid if it's stable (smaller than theoretical k)
        k_stable = min(TT[j] - t, k)
        Courant_stable = k_stable * c / h 
        print(f"Time: {t:.6f}; Courant = {Courant_stable:.6f}; Time step = {k_stable:.6f}")

        # Zero the update for a j-th column of U matrix.
        dU = np.zeros_like(U_temp) # 1d array

        # Method-specific update
        match method.lower():
            case 'forward-upwind':
                # Simple explicit Euler + forward-upwind operator
                dU = Courant_stable * (A @ U_temp)
                # Update solution
                U_temp += dU

            case 'implicit-central':
                # solve the new U vector using current U_temp and A matrix
                U_temp = np.linalg.solve(A, 2 * U_temp)

            case 'beam-warming':
                # apply delta U to current U
                dU = A @ U_temp
                # update
                U_temp += dU

            case 'lax-wendroff':
                # apply delta U to current U
                dU = A @ U_temp
                # update
                U_temp += dU
            case _:
                exit("Method is unknwon!")
        
        # Advance indices
        num_time_steps += 1 
        t += k_stable # advance time

        # If we've just hit the next snapshot time that user defined, store the solution.
        if np.isclose(t, TT[j]):
            # Record the solution
            TT[j] = t  # ensure no floating rounding error
            U[:, j] = U_temp
            j += 1 # move to next time step / col in sol mx

    # 7. Finalize outputs
    result['U'] = U
    result['l'] = num_time_steps

    return result