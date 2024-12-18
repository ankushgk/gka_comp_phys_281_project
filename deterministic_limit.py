import numpy as np
import matplotlib.pyplot as plt

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

def simulate_deterministic(x0, T, n_steps, drift_fn):
    """
    Simulate purely deterministic trajectory (no noise).
    
    Parameters:
    -----------
    x0 : float
        Initial position
    T : float 
        Total simulation time
    n_steps : int
        Number of time steps
    drift_fn : callable
        Drift function of form f(x, t)
    
    Returns:
    --------
    tuple
        (time points, positions)
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    # Forward Euler for the deterministic ODE
    for i in range(n_steps):
        x[i + 1] = x[i] + drift_fn(x[i], t[i]) * dt
        
    return t, x

def test_deterministic_limit(x0, T, n_steps, drift_fn, diffusion_fn, D0_values):
    """
    Test convergence to deterministic limit with decreasing noise.
    
    Parameters:
    -----------
    x0 : float
        Initial position
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    drift_fn : callable
        Drift function
    diffusion_fn : callable
        Diffusion function
    D0_values : array-like
        Different noise amplitudes to test
    """
    # Get deterministic trajectory
    t_det, x_det = simulate_deterministic(x0, T, n_steps, drift_fn)
    
    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.plot(t_det, x_det, 'k--', linewidth=2, label='Deterministic')
    
    # For different noise levels
    for D0 in D0_values:
        # Simulate multiple trajectories
        t, trajectories, _, _, mean_traj, std_traj, _, _ = simulate_multiple_trajectories(
            n_trajectories=100,
            x0=x0,
            T=T,
            n_steps=n_steps,
            drift_fn=drift_fn,
            diffusion_fn=lambda x, t: diffusion_fn(x, t, D0=D0),
            L=float('inf'),  # No absorbing boundary
            use_reflecting=False,
            use_absorbing=False
        )
        
        # Plot mean trajectory with standard deviation band
        plt.plot(t, mean_traj, '-', label=f'D₀ = {D0}')
        plt.fill_between(t, mean_traj - std_traj, mean_traj + std_traj, alpha=0.2)
    
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Convergence to Deterministic Limit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters
    x0 = 0.0
    T = 1.0
    n_steps = 1000
    L = 1.0
    
    # Define drift and diffusion functions
    def centrosome_drift(x, t, a=2.0, L=1.0):
        return -a * (x - L)
    
    def constant_diffusion(x, t, D0=0.5):
        return np.sqrt(D0)
    
    # Test different noise amplitudes
    D0_values = [0.5, 0.1, 0.01, 0.001]
    
    # Run test
    test_deterministic_limit(
        x0=x0,
        T=T,
        n_steps=n_steps,
        drift_fn=lambda x, t: centrosome_drift(x, t, L=L),
        diffusion_fn=constant_diffusion,
        D0_values=D0_values
    )