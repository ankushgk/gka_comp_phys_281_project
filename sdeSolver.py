import numpy as np
import matplotlib.pyplot as plt

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

def simulate_trajectory(x0, T, n_steps, drift_fn, diffusion_fn, L, noise_type='white', 
                       tau_c=None, use_reflecting=True, use_absorbing=True, seed=None):
    """
    Simulate a single trajectory with custom drift and diffusion terms.
    
    Parameters:
    -----------
    x0 : float
        Initial position
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    drift_fn : callable
        Function of form f(x, t, **params) returning drift term
    diffusion_fn : callable
        Function of form f(x, t, **params) returning diffusion term
    L : float
        System length (position of immune synapse)
    noise_type : str
        'white' for base model or 'colored' for colored noise model
    tau_c : float
        Correlation time for colored noise (only used if noise_type='colored')
    use_reflecting : bool
        Whether to use reflecting boundary at x=0
    use_absorbing : bool
        Whether to use absorbing boundary at x=L
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time grid
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # Initialize arrays
    x = np.zeros(n_steps + 1)
    eta = np.zeros(n_steps + 1)
    x[0] = x0
    
    # First passage time (initialize to nan if not reached)
    fpt = np.nan
    
    # Simulate trajectory
    for i in range(n_steps):
        x_current = x[i]
        eta_current = eta[i]
        t_current = t[i]
        
        # Calculate drift and diffusion terms
        drift = drift_fn(x_current, t_current)
        diffusion = diffusion_fn(x_current, t_current)
        
        # Generate noise
        if noise_type == 'white':
            # White noise
            eta_next = diffusion/np.sqrt(dt) * np.random.randn()
            dx = drift * dt + diffusion * np.sqrt(dt) * np.random.randn()
        
        else:  # colored noise
            # Update colored noise using exact solution for OU process
            D = diffusion**2 
            eta_next = eta_current * np.exp(-dt/tau_c) + \
                      np.sqrt(2*D * (1 - np.exp(-2*dt/tau_c))) * np.random.randn()
            dx = drift * dt + eta_current * dt
        
        x_next = x_current + dx
        
        # Apply reflecting boundary at x=0
        if use_reflecting and x_next < 0:
            x_next = -x_next
            
        # Apply absorbing boundary at x=L
        if use_absorbing and x_next >= L:
            x_next = L
            fpt = t[i+1]
            x[i+1:] = L
            eta[i+1:] = 0
            break
            
        x[i+1] = x_next
        eta[i+1] = eta_next
        
    return t, x, eta, fpt

def simulate_multiple_trajectories(n_trajectories, x0, T, n_steps, drift_fn, diffusion_fn, L, 
                                 noise_type='white', tau_c=None, 
                                 use_reflecting=True, use_absorbing=True, seed=None):
    """
    Simulate multiple trajectories and compute statistics.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize arrays
    t = np.linspace(0, T, n_steps + 1)
    trajectories = np.zeros((n_trajectories, n_steps + 1))
    noise_trajectories = np.zeros((n_trajectories, n_steps + 1))
    fpts = np.zeros(n_trajectories)
    
    # Simulate trajectories
    for i in range(n_trajectories):
        _, traj, noise_traj, fpt = simulate_trajectory(
            x0, T, n_steps, drift_fn, diffusion_fn, L, 
            noise_type, tau_c, use_reflecting, use_absorbing
        )
        trajectories[i] = traj
        noise_trajectories[i] = noise_traj
        fpts[i] = fpt
    
    # Compute statistics
    mean_traj = np.mean(trajectories, axis=0)
    std_traj = np.std(trajectories, axis=0)
    mean_noise = np.mean(noise_trajectories, axis=0)
    std_noise = np.std(noise_trajectories, axis=0)
    
    return t, trajectories, noise_trajectories, fpts, mean_traj, std_traj, mean_noise, std_noise

def plot_results(t, trajectories, noise_trajectories, fpts, mean_traj, std_traj, 
                mean_noise, std_noise, L, noise_type, use_reflecting, use_absorbing, 
                max_display=100):
    """Plot trajectories, noise, and statistics"""
    
    boundary_text = []
    if use_reflecting:
        boundary_text.append("reflecting at x=0")
    if use_absorbing:
        boundary_text.append("absorbing at x=L")
    boundary_str = f" ({', '.join(boundary_text)})" if boundary_text else " (no boundaries)"
    
    # Plot trajectories
    plt.figure(figsize=(15, 8))
    for i in range(min(max_display, len(trajectories))):
        plt.plot(t, trajectories[i], alpha=0.3, label='_nolegend_')
    plt.plot(t, mean_traj, 'r-', label='Mean', linewidth=2)
    plt.fill_between(t, mean_traj - std_traj, mean_traj + std_traj, 
                    color='r', alpha=0.2, label='±1 std')
    if use_absorbing:
        plt.axhline(y=L, color='k', linestyle='--', label='Immune Synapse')
    if use_reflecting:
        plt.axhline(y=0, color='g', linestyle='--', label='Reflecting Boundary')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.title(f'Sample Trajectories ({noise_type} noise)' + boundary_str)
    
    # Plot noise trajectories
    plt.figure(figsize=(15, 8))
    for i in range(min(max_display, len(noise_trajectories))):
        plt.plot(t, noise_trajectories[i], alpha=0.3, label='_nolegend_')
    plt.plot(t, mean_noise, 'r-', label='Mean', linewidth=2)
    plt.fill_between(t, mean_noise - std_noise, mean_noise + std_noise, 
                    color='r', alpha=0.2, label='±1 std')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Noise')
    plt.legend()
    plt.title(f'{noise_type.capitalize()} Noise Trajectories' + boundary_str)
    
    # Plot first passage time histogram (only if using absorbing boundary)
    if use_absorbing:
        plt.figure(figsize=(15, 8))
        plt.hist(fpts, bins=50, density=True, alpha=0.7)
        plt.axvline(np.mean(fpts), color='r', linestyle='--', 
                    label=f'Mean = {np.nanmean(fpts):.2f}')
        plt.grid(True)
        plt.xlabel('First Passage Time')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'First Passage Time Distribution ({noise_type} noise)' + boundary_str)
    
    # Plot position distribution at different times
    plt.figure(figsize=(15, 8))
    sample_times = [0.2, 0.5, 0.8]
    for t_sample in sample_times:
        idx = int(t_sample * len(t))
        positions = trajectories[:, idx]
        plt.hist(positions, bins=50, density=True, alpha=0.5, 
                label=f't = {t_sample:.1f}')
    plt.grid(True)
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Position Distribution ({noise_type} noise)' + boundary_str)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define drift and diffusion functions
    def linear_drift(x, t, a=2.0):
        """Linear Drift Term"""
        return a * t

    def centrosome_drift(x, t, a=2.0, L=1.0):
        """Default drift function"""
        return -a * (x - L)
    
    def constant_diffusion(x, t, D0=0.5):
        """Constant diffusion term"""
        return np.sqrt(D0)
    
    def centrosome_diffusion(x, t, D0=0.1, L=1.0, D_min=0.1):
        """Default diffusion function with minimum noise level"""
        return np.sqrt(D0 * (L - x) + D_min)


    
    # Set parameters
    x0 = 0.0      # Start position
    T = 5.0       # Total time
    n_steps = 1000 # Number of steps
    L = 1.0       # System length
    tau_c = 1e-6   # Noise correlation time
    
    # Test different drift and diffusion combinations
    model_configs = [
        {
            'drift': centrosome_drift,
            'diffusion': centrosome_diffusion,
            'desc': 'Default linear model'
        },
        # {
        #     'drift': centrosome_drift,
        #     'diffusion': constant_diffusion,
        #     'desc': 'Constant diffusion model'
        # }
    ]
    
    for config in model_configs:
        print(f"\nSimulating {config['desc']}...")
        noise_type = 'colored'
        # Simulate trajectories
        results = simulate_multiple_trajectories(
            n_trajectories=10000,
            x0=x0,
            T=T,
            n_steps=n_steps,
            drift_fn=config['drift'],
            diffusion_fn=config['diffusion'],
            L=L,
            noise_type=noise_type,
            tau_c=tau_c
        )
        
        # Unpack and plot results
        t, trajectories, noise_trajectories, fpts, mean_traj, std_traj, mean_noise, std_noise = results
        
        if noise_type=='colored':
            model_title = f"Colored noise - {config['desc']}"
        elif noise_type=='white':
            model_title = f"White noise - {config['desc']}"
        
        plot_results(t, trajectories, noise_trajectories, fpts,
                    mean_traj, std_traj, mean_noise, std_noise, L,
                    model_title, True, True)