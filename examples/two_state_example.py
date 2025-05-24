#!/usr/bin/env python3
"""
Simple example demonstrating the ODEfit Template library.

This example implements the minimal viable product:
1. Define a simple ODE system with two states and two parameters
2. Generate synthetic measurement data with noise
3. Identify the parameters using the library
4. Compare true vs identified parameters
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.odefit_template import Parameter, ODESystemFitter


def simple_ode_example():
    """Run a simple ODE parameter estimation example."""
    
    print("="*60)
    print("SIMPLE ODE PARAMETER ESTIMATION EXAMPLE")
    print("="*60)
    
    # Step 1: Define the ODE system
    # Simple second-order system: A -> B -> C (two reactions)
    # States: x[0] = concentration of A, x[1] = concentration of B
    # Parameters: k1 (A->B rate), k2 (B->C rate)
    
    def system_dynamics(t, x, u, p):
        """
        Define the ODE system: A -> B -> C
        
        Args:
            t: time
            x: state vector [A, B]
            u: inputs (none in this example)
            p: parameters dict {'k1': rate1, 'k2': rate2}
        
        Returns:
            dx/dt: [dA/dt, dB/dt]
        """
        dA_dt = -p['k1'] * x[0]                    # A decreases
        dB_dt = p['k1'] * x[0] - p['k2'] * x[1]   # B increases from A, decreases to C
        return [dA_dt, dB_dt]
    
    # Step 2: Define the measurement function
    def measurement_function(t, x, u, p):
        """
        Define what we can measure.
        In this case, we measure the concentration of A (first state).
        
        Returns:
            y: [concentration_of_A]
        """
        return [x[0]]  # We can only measure A
    
    # Step 3: Set up true parameters and generate synthetic data
    print("Generating synthetic data...")
    
    # True parameter values (these are what we want to recover)
    true_k1 = 1.5
    true_k2 = 0.8
    true_params = {'k1': true_k1, 'k2': true_k2}
    
    # Initial conditions: start with A=1.0, B=0.0
    initial_conditions = np.array([1.0, 0.0])
    
    # Time points for measurement
    time_points = np.linspace(0, 4, 25)  # 0 to 4 seconds, 25 measurements
    
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
    
    # Create measurements (concentration of A) with realistic noise
    np.random.seed(42)  # For reproducible results
    noise_level = 0.03  # 3% noise
    true_measurements = true_solution.y[0, :].reshape(-1, 1)  # Measure A
    noisy_measurements = true_measurements + np.random.normal(0, noise_level, true_measurements.shape)
    
    print(f"Generated {len(time_points)} data points with {noise_level*100:.1f}% noise")
    print(f"True parameters: k1={true_k1}, k2={true_k2}")
    
    # Step 4: Set up parameter estimation
    print("\nSetting up parameter estimation...")
    
    # Define parameters to estimate with initial guesses
    parameters_to_fit = [
        Parameter(name='k1', value=1.0, min=0.1, max=5.0),   # Initial guess: 1.0
        Parameter(name='k2', value=0.5, min=0.1, max=3.0),   # Initial guess: 0.5
    ]
    
    print(f"Initial guesses: k1={parameters_to_fit[0].value}, k2={parameters_to_fit[1].value}")
    
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
    print(f"  k1: True={true_k1:.4f}, Fitted={fitted_params['k1']:.4f}, Error={abs(fitted_params['k1'] - true_k1):.4f}")
    print(f"  k2: True={true_k2:.4f}, Fitted={fitted_params['k2']:.4f}, Error={abs(fitted_params['k2'] - true_k2):.4f}")
    
    # Calculate overall discrepancy
    total_discrepancy = np.sqrt((fitted_params['k1'] - true_k1)**2 + (fitted_params['k2'] - true_k2)**2)
    
    print(f"\nOverall Parameter Discrepancy (L2 norm): {total_discrepancy:.4f}")
    print(f"Optimization Success: {result.success}")
    print(f"Number of Function Evaluations: {result.nfev}")
    print(f"Final Residual Sum of Squares: {result.chisqr:.6f}")
    
    # Check if the identification was successful
    if total_discrepancy < 0.1:
        print("\n✅ Parameter identification was SUCCESSFUL!")
    else:
        print("\n⚠️  Parameter identification had significant errors.")
    
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
    result, fitted_params = simple_ode_example()
    
    print("\nExample completed successfully!")
    print("The library demonstrated:")
    print("✓ ODE system definition")
    print("✓ Synthetic data generation with noise")
    print("✓ Parameter estimation using optimization")
    print("✓ Results comparison and analysis") 