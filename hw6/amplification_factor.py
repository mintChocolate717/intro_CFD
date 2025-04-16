import numpy as np
import matplotlib.pyplot as plt

def amplification_factor(fig, tit, Gh, ss):
    '''
    
    '''
    fs = 15  # Font size
    beta = np.linspace(0, np.pi, 180)  # [0, pi]
    colors = ['b-', 'g-', 'r-', 'm-']
    
    for i, s in enumerate(ss):
        mG, pG, G = Gh(beta, s)
        x = mG * np.cos(beta)
        y = mG * np.sin(beta)
        
        # Relative Magnitude Error in the Complex Plane
        plt.figure(fig[0])
        plt.plot(x, y, colors[i], label=f'Sigma={s}')
        
        # Relative Magnitude Error vs Beta
        plt.figure(fig[1])
        plt.plot(beta, mG, colors[i], label=f'Sigma={s}')
        plt.xlabel('Phase angle, beta')
        plt.ylabel('Magnitude of Solution, mG')
        plt.title('Relative Magnitude Error vs Beta')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{tit}-Magnitude-vs-Beta.png')
        
        x = pG * np.cos(beta)
        y = pG * np.sin(beta)
        
        # Relative Phase Error in the Complex Plane
        plt.figure(fig[2])
        plt.plot(x, y, colors[i], label=f'Sigma={s}')
        plt.xlabel('Real Part of Solution Times pG')
        plt.ylabel('Imaginary Part of Solution Times pG')
        plt.title('Relative Phase Error in the Complex Plane')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{tit}-Phase-Complex.png')
        
        # Relative Phase Error vs Beta
        plt.figure(fig[3])
        plt.plot(beta, pG, colors[i], label=f'Sigma={s}')
        plt.xlabel('Phase angle, beta')
        plt.ylabel('Phase Difference, pG')
        plt.title('Relative Phase Error vs Beta')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{tit}-Phase-vs-Beta.png')
    
    # Add unit circle to first plot
    plt.figure(fig[0])
    x = np.cos(beta)
    y = np.sin(beta)
    plt.plot(x, y, 'k--', label='Unit Circle')
    plt.savefig(f'plots/{tit}-Magnitude-Complex.png')
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([0, 1.5])
    plt.xticks(np.arange(-1.5, 1.6, 0.5), ['1.5', '1', '0.5', '0', '0.5', '1', '1.5'])
    plt.xlabel('Real Part of Solution Times mG')
    plt.ylabel('Imaginary Part of Solution Times mG')
    plt.title('Relative Magnitude Error in the Complex Plane')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{tit}-Magnitude-Complex.png')

