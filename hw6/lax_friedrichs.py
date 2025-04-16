import numpy as np

def lax_friedrichs(x, s):
    G = np.cos(x) - 1j * s * np.sin(x)
    Ge = np.exp(-1j * s * x)
    
    # Magnitude and phase angle
    mG = np.abs(G)
    phase_Ge = np.angle(Ge)
    
    # Avoid division by zero by replacing zeros in phase_Ge with a small value (e.g., 1e-10)
    phase_Ge_safe = np.where(phase_Ge == 0, 1e-10, phase_Ge)
    
    pG = np.angle(G) / phase_Ge_safe  # Now division by zero is avoided

    return mG, pG, G
