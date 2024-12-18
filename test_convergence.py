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

def exact_solution(t, W, gamma, mu, x0):
    """
    Compute exact solution of the test SDE: dX = gamma*X*dt + mu*X*dW
    Solution: X(t) = X(0)*exp((gamma - 0.5*mu^2)t + mu*W(t))
    """
    return x0 * np.exp((gamma - 0.5*mu**2)*t + mu*W)

def exact_expectation(t, gamma, mu, x0):
    """Compute exact expectation E[X(t)] = X(0)*exp(gamma*t)"""
    return x0 * np.exp(gamma * t)

def euler_maruyama_step(dt, dW, X_prev, gamma, mu):
    """Single step of Euler-Maruyama method for test equation"""
    return X_prev + dt*gamma*X_prev + mu*np.multiply(X_prev, dW)

def test_strong_convergence(gamma, mu, x0, T, n_paths=1000, base_steps=2**8, 
                          n_refinements=5, seed=101):
    """
    Run strong convergence analysis

    Adapted from SDE-higham GitHub Repo
    
    Parameters:
    -----------
    gamma : float
        Drift coefficient
    mu : float
        Diffusion coefficient
    x0 : float
        Initial condition
    T : float
        Final time
    n_paths : int
        Number of paths
    base_steps : int
        Base number of steps
    n_refinements : int
        Number of refinement levels
    seed : int
        Random seed
    
    Returns:
    --------
    tuple
        (dt values, errors, convergence rate)
    """
    np.random.seed(seed)
    
    dt_values = np.zeros(n_refinements)
    errors = np.zeros((n_paths, n_refinements))
    
    dt_base = T/base_steps
    
    for p in range(n_refinements):
        R = 2**p
        dt = R * dt_base
        dt_values[p] = dt
        L = int(base_steps/R)
        
        for s in range(n_paths):
            dW = np.sqrt(dt_base) * np.random.randn(base_steps)
            W = np.cumsum(dW)
            
            X_true = exact_solution(T, W[-1], gamma, mu, x0)
            
            X_em = x0
            for j in range(L):
                dW_coarse = np.sum(dW[R*j:R*(j+1)])
                X_em = euler_maruyama_step(dt, dW_coarse, np.array([X_em]), gamma, mu)[0]
            
            errors[s,p] = abs(X_em - X_true)
    
    mean_errors = np.mean(errors, axis=0)
    
    # Convergence rate
    A = np.column_stack((np.ones(n_refinements), np.log(dt_values)))
    rate = np.linalg.lstsq(A, np.log(mean_errors), rcond=None)[0][1]
    
    return dt_values, mean_errors, rate

def test_weak_convergence(gamma, mu, x0, T, n_paths=50000, n_refinements=5, 
                         min_power=-10, seed=102):
    """
    Run weak convergence analysis

    Adapted from SDE-higham GitHub Repo
    
    Parameters:
    -----------
    gamma : float
        Drift coefficient
    mu : float
        Diffusion coefficient
    x0 : float
        Initial condition
    T : float
        Final time
    n_paths : int
        Number of paths
    n_refinements : int
        Number of refinement levels
    min_power : int
        Minimum power for dt refinement
    seed : int
        Random seed
    
    Returns:
    --------
    tuple
        (dt values, errors, convergence rate)
    """
    np.random.seed(seed)
    
    dt_values = np.power(2.0, [p + min_power for p in range(n_refinements)])
    X_means = np.zeros(n_refinements)
    
    # Exact expected value
    X_true = exact_expectation(T, gamma, mu, x0)
    
    for p in range(n_refinements):
        dt = dt_values[p]
        n_steps = int(T / dt)
        
        X = x0 * np.ones(n_paths)
        
        for _ in range(n_steps):
            dW = np.sqrt(dt) * np.random.randn(n_paths)
            X = euler_maruyama_step(dt, dW, X, gamma, mu)
        
        X_means[p] = np.mean(X)
    
    errors = np.abs(X_means - X_true)
    
    # Convergence rate
    A = np.column_stack((np.ones(n_refinements), np.log(dt_values)))
    rate = np.linalg.lstsq(A, np.log(errors), rcond=None)[0][1]
    
    return dt_values, errors, rate

def plot_convergence(dt_values, errors, rate, convergence_type="strong"):
    """Plot convergence analysis results"""
    plt.figure(figsize=(10, 6))
    plt.loglog(dt_values, errors, 'b*-', label='Numerical Error')
    
    reference_rate = 0.5 if convergence_type == "strong" else 1.0
    plt.loglog(dt_values, dt_values**reference_rate, 'r--', 
               label=f'O(Δt^{{{reference_rate}}})')
    
    plt.grid(True)
    plt.xlabel('Δt')
    
    if convergence_type == "strong":
        plt.ylabel('Average |X(T) - X_L|')
        title = "Strong Convergence Analysis"
    else:
        plt.ylabel('|E(X(T)) - Sample average of X_L|')
        title = "Weak Convergence Analysis"
        
    plt.title(f'{title}\nEstimated Rate = {rate:.3f}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Parameters
    gamma = 2.0
    mu = 1.0
    x0 = 1.0
    T = 1.0
    
    print("Testing strong convergence...")
    dt_vals_strong, errors_strong, rate_strong = test_strong_convergence(
        gamma=gamma, mu=mu, x0=x0, T=T
    )
    plot_convergence(dt_vals_strong, errors_strong, rate_strong, "strong")
    
    print("\nTesting weak convergence...")
    dt_vals_weak, errors_weak, rate_weak = test_weak_convergence(
        gamma=gamma, mu=mu, x0=x0, T=T
    )
    plot_convergence(dt_vals_weak, errors_weak, rate_weak, "weak")