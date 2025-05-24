"""ODEfit Template - A user-friendly Python package for parameter estimation and system identification of nonlinear ODE models."""

from .parameters import Parameter
from .core import ODESystemFitter

__all__ = ['Parameter', 'ODESystemFitter', 'main']


def main() -> None:
    """Demo function showing basic usage of the library."""
    print("Hello from odefit-template!")
    print("Running simple ODE parameter estimation demo...")
    
    # Import here to avoid issues if dependencies aren't installed
    import numpy as np
    
    # Define a simple second-order ODE system: A -> B -> C
    # States: x[0] = concentration of A, x[1] = concentration of B  
    # Parameters: k1 (A->B rate), k2 (B->C rate)
    def system_dynamics(t, x, u, p):
        """Simple two-state system: A -> B -> C"""
        dA_dt = -p['k1'] * x[0]        # A decreases with rate k1
        dB_dt = p['k1'] * x[0] - p['k2'] * x[1]  # B increases from A, decreases with rate k2
        return [dA_dt, dB_dt]
    
    def measurement_function(t, x, u, p):
        """Measure concentration of A (first state)"""
        return [x[0]]  # We measure the concentration of A
    
    # True parameter values (to generate synthetic data)
    true_k1 = 1.5
    true_k2 = 0.8
    true_params = {'k1': true_k1, 'k2': true_k2}
    
    # Time points for simulation and measurement
    time_points = np.linspace(0, 5, 21)  # 0 to 5 seconds, 21 points
    
    # Generate synthetic data with the true parameters
    from scipy.integrate import solve_ivp
    
    initial_conditions = np.array([1.0, 0.0])  # Start with A=1, B=0
    
    def true_ode(t, x):
        return system_dynamics(t, x, np.array([]), true_params)
    
    # Solve the true system
    sol = solve_ivp(
        true_ode,
        [time_points[0], time_points[-1]], 
        initial_conditions,
        t_eval=time_points,
        method='RK45'
    )
    
    # Generate measurements with noise
    np.random.seed(42)  # For reproducible results
    noise_level = 0.02
    true_measurements = sol.y[0, :].reshape(-1, 1)  # Measure concentration of A
    noisy_measurements = true_measurements + np.random.normal(0, noise_level, true_measurements.shape)
    
    # Define parameters to fit (with initial guesses)
    parameters_to_fit = [
        Parameter(name='k1', value=1.0, min=0.1, max=5.0),  # Initial guess: 1.0, true: 1.5
        Parameter(name='k2', value=0.5, min=0.1, max=3.0),  # Initial guess: 0.5, true: 0.8
    ]
    
    # Create model and data objects
    model = ODESystemFitter.Model(
        sys_func=system_dynamics,
        output_func=measurement_function,
        initial_conditions=initial_conditions,
        parameters=parameters_to_fit
    )
    
    data = ODESystemFitter.Data(
        time=time_points,
        measurements=noisy_measurements
    )
    
    # Create fitter and run estimation
    fitter = ODESystemFitter.Fitter(model, data)
    print("Running parameter estimation...")
    result = fitter.fit()
    
    # Extract fitted parameters
    fitted_params = result.params.valuesdict()
    
    # Print results
    print("\n" + "="*50)
    print("PARAMETER ESTIMATION RESULTS")
    print("="*50)
    print(f"True k1: {true_k1:.4f}")
    print(f"Fitted k1: {fitted_params['k1']:.4f}")
    print(f"Error k1: {abs(fitted_params['k1'] - true_k1):.4f}")
    print()
    print(f"True k2: {true_k2:.4f}")
    print(f"Fitted k2: {fitted_params['k2']:.4f}")
    print(f"Error k2: {abs(fitted_params['k2'] - true_k2):.4f}")
    print()
    
    # Calculate total parameter discrepancy
    total_error = np.sqrt((fitted_params['k1'] - true_k1)**2 + (fitted_params['k2'] - true_k2)**2)
    print(f"Total parameter discrepancy (L2 norm): {total_error:.4f}")
    print(f"Fit success: {result.success}")
    print(f"Number of function evaluations: {result.nfev}")
    print("="*50)
    
    try:
        fitter.plot()
        print("Plot displayed showing data vs fitted model.")
    except Exception as e:
        print(f"Could not display plot: {e}")
        print("This is normal in non-interactive environments.")
    
    return result
