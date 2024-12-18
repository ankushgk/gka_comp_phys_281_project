import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sdeSolver import simulate_multiple_trajectories

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 14,
    'axes.titlesize': 8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'text.usetex': False,  # Set to True if using LaTeX
    'font.family': 'serif',
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.markersize': 6,
})

def analytical_fpt_density(t, x0, L, a, sigma):
    """
    Compute analytical FPT density for Ornstein-Uhlenbeck process
    with absorbing boundary.
    
    Parameters are same as before.
    """
    # Avoid t=0 to prevent division by zero
    t = np.maximum(t, 1e-10)
    
    # Compute mean and variance of the process
    mean = L + (x0 - L) * np.exp(-a * t)
    var = (sigma**2 / (2*a)) * (1 - np.exp(-2*a * t))
    
    # to prevent division by zero
    var = np.maximum(var, 1e-10)
    
    # Known FPT
    prefactor = np.abs(L - x0) / np.sqrt(2*np.pi*var*t**3)
    exponent = -(L - mean)**2 / (2*var)
    
    # Clip extremely large negative exponents to prevent underflow 
    # Learned this trick from StackExchange
    exponent = np.maximum(exponent, -700)
    
    density = prefactor * np.exp(exponent)
    
    return density

def validate_fpt_distribution(x0, L, a, sigma, T, n_steps, n_trajectories):
    """
    Compare numerical and analytical FPT distributions.
    Parameters are same as before.
    """
    # Define drift and diffusion functions for OU process
    drift_fn = lambda x, t: -a * (x - L)
    diffusion_fn = lambda x, t: sigma
    
    # Numerical FPTs
    _, _, _, fpts, _, _, _, _ = simulate_multiple_trajectories(
        n_trajectories=n_trajectories,
        x0=x0, T=T, n_steps=n_steps,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        L=L,
        noise_type='white',
        use_reflecting=False,
        use_absorbing=True
    )
    
    # Remove NaN values (trajectories that didn't reach boundary)
    valid_fpts = fpts[~np.isnan(fpts)]
    
    t_max = min(T, np.percentile(valid_fpts, 99))
    t_grid = np.linspace(0, t_max, 100)
    
    # Analytical density
    analytical = analytical_fpt_density(t_grid, x0, L, a, sigma)
    
    # Numerical density using histogram
    hist, edges = np.histogram(valid_fpts, bins='auto', density=True, 
                             range=(0, t_max))
    t_centers = (edges[1:] + edges[:-1])/2
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, analytical, 'r-', label='Analytical', linewidth=2)
    plt.plot(t_centers, hist, 'b.', label='Numerical', markersize=8)
    plt.xlabel('First Passage Time')
    plt.ylabel('Probability Density')
    plt.title('FPT Distribution: Analytical vs Numerical')
    plt.legend()
    plt.grid(True)
    
    # Add summary statistics
    dt = t_grid[1] - t_grid[0]
    norm_factor = np.sum(analytical) * dt  # For normalization
    mean_analytical = np.sum(t_grid * analytical) * dt / norm_factor
    mean_numerical = np.mean(valid_fpts)
    # plt.text(0.98, 0.95, f'Mean FPT (Analytical): {mean_analytical:.3f}\n' +
    #                     f'Mean FPT (Numerical): {mean_numerical:.3f}',
    #          transform=plt.gca().transAxes, ha='right', va='bottom',
    #          bbox=dict(facecolor='white', alpha=0.8), position(4,0.4))
    
    plt.show()
    print(mean_analytical, mean_numerical)

    
    return t_grid, analytical, hist


if __name__ == "__main__":
    # Parameters 
    x0 = 0.0    # Start at origin
    L = 1.0     # Absorbing boundary
    a = 1.0     # Drift
    sigma = 0.5  # Noise intensity
    
    # Simulation parameters
    T = 5.0
    n_steps = 2000 
    n_trajectories = 50000
    
    # Run validation
    validate_fpt_distribution(x0, L, a, sigma, T, n_steps, n_trajectories)