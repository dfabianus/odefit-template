#!/usr/bin/env python3
"""
Very simple ODE parameter estimation example with one state and one parameter.

This example demonstrates:
1. A first-order exponential decay ODE: dx/dt = -k*x
2. Generate synthetic measurement data with noise
3. Identify the decay rate parameter k
4. Compare true vs identified parameter

This is the simplest possible case for ODE parameter estimation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.odefit_template import Parameter, ODESystemFitter


def exponential_decay_example():
    """Run a simple exponential decay parameter estimation example."""
    
    print("="*60)
    print("EXPONENTIAL DECAY PARAMETER ESTIMATION EXAMPLE")
    print("="*60)
    print("Model: dx/dt = -k*x  (exponential decay)")
    print("Parameter to identify: k (decay rate)")
    print("Measurement: x(t) (the state itself)")
    
    # Step 1: Define the ODE system
    # Simple first-order decay: dx/dt = -k*x
    # State: x[0] = concentration or amount
    # Parameter: k = decay rate constant
    
    def system_dynamics(t, x, u, p):
        """
        Define the exponential decay ODE: dx/dt = -k*x
        
        Args:
            t: time
            x: state vector [concentration]
            u: inputs (none in this example)
            p: parameters dict {'k': decay_rate}
        
        Returns:
            dx/dt: [dx/dt]
        """
        dx_dt = -p['k'] * x[0]  # Exponential decay
        return [dx_dt]
    
    # Step 2: Define the measurement function
    def measurement_function(t, x, u, p):
        """
        Define what we can measure.
        In this case, we measure the state directly.
        
        Returns:
            y: [concentration]
        """
        return [x[0]]  # We measure the state directly
    
    # Step 3: Set up true parameters and generate synthetic data
    print("\nGenerating synthetic data...")
    
    # True parameter value (this is what we want to recover)
    true_k = 0.5  # True decay rate
    true_params = {'k': true_k}
    
    # Initial condition: start with concentration = 2.0
    initial_conditions = np.array([2.0])
    
    # Time points for measurement
    time_points = np.linspace(0, 8, 30)  # 0 to 8 seconds, 30 measurements
    
    # Generate true data by solving the ODE with true parameters
    from scipy.integrate import solve_ivp
    
    def true_ode_system(t, x):
        return system_dynamics(t, x, np.array([]), true_params)
    
    # Solve the true system
    true_solution = solve_ivp(
        true_ode_system,
        [time_points[0], time_points[-1]],
        initial_conditions,
        t_eval=time_points,
        method='RK45',
        rtol=1e-9
    )
    
    # Create measurements with realistic noise
    np.random.seed(42)  # For reproducible results
    noise_level = 0.05  # 5% noise
    true_measurements = true_solution.y[0, :].reshape(-1, 1)  # Direct measurement of state
    noisy_measurements = true_measurements + np.random.normal(0, noise_level, true_measurements.shape)
    
    print(f"Generated {len(time_points)} data points with {noise_level*100:.1f}% noise")
    print(f"True decay rate k = {true_k}")
    print(f"Initial condition x(0) = {initial_conditions[0]}")
    print(f"Expected final value x({time_points[-1]:.1f}) ≈ {initial_conditions[0] * np.exp(-true_k * time_points[-1]):.3f}")
    
    # Step 4: Set up parameter estimation
    print("\nSetting up parameter estimation...")
    
    # Define parameter to estimate with initial guess
    parameters_to_fit = [
        Parameter(name='k', value=0.3, min=0.01, max=2.0),   # Initial guess: 0.3, true: 0.5
    ]
    
    print(f"Initial guess: k = {parameters_to_fit[0].value}")
    
    # Create the model
    model = ODESystemFitter.Model(
        sys_func=system_dynamics,
        output_func=measurement_function,
        initial_conditions=initial_conditions,
        parameters=parameters_to_fit
    )
    
    # Create the data object
    data = ODESystemFitter.Data(
        time=time_points,
        measurements=noisy_measurements
    )
    
    # Step 5: Run parameter identification
    print("\nRunning parameter identification...")
    
    fitter = ODESystemFitter.Fitter(model, data)
    result = fitter.fit()
    
    # Step 6: Analyze results
    fitted_params = result.params.valuesdict()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("Parameter Comparison:")
    print(f"  k: True = {true_k:.4f}")
    print(f"  k: Fitted = {fitted_params['k']:.4f}")
    print(f"  k: Error = {abs(fitted_params['k'] - true_k):.4f}")
    print(f"  k: Relative Error = {abs(fitted_params['k'] - true_k)/true_k*100:.2f}%")
    
    # Calculate parameter discrepancy
    parameter_error = abs(fitted_params['k'] - true_k)
    relative_error = parameter_error / true_k * 100
    
    print(f"\nParameter Discrepancy: {parameter_error:.4f}")
    print(f"Relative Error: {relative_error:.2f}%")
    print(f"Optimization Success: {result.success}")
    print(f"Number of Function Evaluations: {result.nfev}")
    print(f"Final Residual Sum of Squares: {result.chisqr:.6f}")
    
    # Check if the identification was successful
    if relative_error < 5.0:  # Less than 5% error
        print("\n✅ Parameter identification was EXCELLENT!")
    elif relative_error < 10.0:  # Less than 10% error
        print("\n✅ Parameter identification was GOOD!")
    else:
        print("\n⚠️  Parameter identification had significant errors.")
    
    # Show analytical solution comparison
    analytical_final = initial_conditions[0] * np.exp(-fitted_params['k'] * time_points[-1])
    print(f"\nAnalytical check:")
    print(f"  With fitted k={fitted_params['k']:.4f}: x({time_points[-1]:.1f}) = {analytical_final:.3f}")
    print(f"  Measured final value: {noisy_measurements[-1, 0]:.3f}")
    
    # Plot results if possible
    print("\nGenerating plots...")
    try:
        fitter.plot()
        print("✅ Plot displayed successfully.")
    except Exception as e:
        print(f"⚠️  Could not display plot: {e}")
        print("   This is normal when running in non-interactive environments.")
    
    print("="*60)
    
    return result, fitted_params


if __name__ == "__main__":
    # Run the example
    result, fitted_params = exponential_decay_example()
    
    print("\nExample completed successfully!")
    print("The library demonstrated:")
    print("✓ Simple one-state ODE system definition")
    print("✓ Synthetic data generation with noise")
    print("✓ Single parameter estimation")
    print("✓ Results analysis with analytical validation")
    
    # Show the mathematical solution
    print("\nMathematical Note:")
    print("For exponential decay dx/dt = -k*x with x(0) = x₀:")
    print("The analytical solution is: x(t) = x₀ * exp(-k*t)")
    print("This makes it easy to verify our parameter estimation results!") 