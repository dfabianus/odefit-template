#!/usr/bin/env python3
"""
Bioprocess ODE parameter estimation example with competitive inhibition kinetics.

This example demonstrates:
1. A three-state bioprocess model: biomass growth with sequential substrate utilization
2. Competitive inhibition kinetics (glucose inhibits galactose uptake)
3. Cell death kinetics to model biomass decay
4. Generate synthetic measurement data with realistic bioprocess noise
5. Identify kinetic parameters (half-saturation constants, inhibition constant, death rate)
6. Low-frequency measurements (hourly) as typical in bioprocesses

Model equations with competitive inhibition and cell death:
- μ_glc = S_glc / (Ks_glc + S_glc)  [glucose utilization]
- μ_gal = S_gal / (Ks_gal + S_gal) * Ki_glc / (Ki_glc + S_glc)  [galactose utilization with glucose inhibition]
- μ_total = μ_max * (μ_glc + μ_gal)  [total growth rate]

- dX/dt = μ_total * X - k_d * X  [biomass growth minus death]
- dS_glc/dt = -1/Y_glc * μ_max * μ_glc * X  
- dS_gal/dt = -1/Y_gal * μ_max * μ_gal * X

Where:
- X = biomass concentration [g/L]
- S_glc = glucose concentration [g/L] 
- S_gal = galactose concentration [g/L]
- μ_max = maximum specific growth rate [1/h]
- Ks_glc, Ks_gal = half-saturation constants [g/L]
- Ki_glc = inhibition constant for glucose on galactose uptake [g/L]
- k_d = cell death rate constant [1/h]
- Y_glc, Y_gal = yield coefficients [g_biomass/g_substrate]
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.odefit_template import Parameter, ODESystemFitter


def bioprocess_example():
    """Run a bioprocess parameter estimation example with competitive inhibition kinetics."""
    
    print("="*70)
    print("BIOPROCESS PARAMETER ESTIMATION EXAMPLE")
    print("="*70)
    print("Model: Competitive inhibition kinetics (glucose inhibits galactose)")
    print("States: Biomass, Glucose, Galactose")
    print("Parameters to identify: Ks_glucose, Ks_galactose, Ki_glucose, k_death")
    print("Measurements: All three states (hourly sampling)")
    
    # Step 1: Define the bioprocess ODE system with competitive inhibition
    def bioprocess_dynamics(t, x, u, p):
        """
        Define the competitive inhibition bioprocess model with cell death.
        
        Args:
            t: time [h]
            x: state vector [biomass, glucose, galactose] [g/L]
            u: inputs (none in this example)
            p: parameters dict {
                'mu_max': maximum growth rate [1/h],
                'Ks_glc': glucose half-saturation constant [g/L],
                'Ks_gal': galactose half-saturation constant [g/L],
                'Ki_glc': glucose inhibition constant for galactose uptake [g/L],
                'k_d': cell death rate constant [1/h],
                'Y_glc': glucose yield coefficient [g_biomass/g_glucose],
                'Y_gal': galactose yield coefficient [g_biomass/g_galactose]
            }
        
        Returns:
            dx/dt: [dX/dt, dS_glc/dt, dS_gal/dt]
        """
        # Extract states
        X = x[0]        # Biomass [g/L]
        S_glc = x[1]    # Glucose [g/L] 
        S_gal = x[2]    # Galactose [g/L]
        
        # Extract parameters
        mu_max = p['mu_max']
        Ks_glc = p['Ks_glc']
        Ks_gal = p['Ks_gal']
        Ki_glc = p['Ki_glc']
        k_d = p['k_d']  # Cell death rate
        Y_glc = p['Y_glc']
        Y_gal = p['Y_gal']
        
        # Competitive inhibition kinetics
        mu_glc = S_glc / (Ks_glc + S_glc)  # Glucose uptake (normal Monod)
        
        # Galactose uptake inhibited by glucose
        mu_gal_base = S_gal / (Ks_gal + S_gal)  # Base galactose Monod term
        inhibition_factor = Ki_glc / (Ki_glc + S_glc)  # Glucose inhibition factor
        mu_gal = mu_gal_base * inhibition_factor  # Inhibited galactose uptake
        
        # Total specific growth rate (additive utilization)
        mu_total = mu_max * (mu_glc + mu_gal)
        
        # Individual substrate utilization rates
        mu_glc_actual = mu_max * mu_glc
        mu_gal_actual = mu_max * mu_gal
        
        # ODE equations with cell death
        dX_dt = mu_total * X - k_d * X                     # Biomass growth minus death
        dS_glc_dt = -(1/Y_glc) * mu_glc_actual * X         # Glucose consumption
        dS_gal_dt = -(1/Y_gal) * mu_gal_actual * X         # Galactose consumption
        
        return [dX_dt, dS_glc_dt, dS_gal_dt]
    
    # Step 2: Define the measurement function
    def measurement_function(t, x, u, p):
        """
        Define what we can measure in the bioprocess.
        We measure all three states: biomass, glucose, galactose.
        
        Returns:
            y: [biomass, glucose, galactose] [g/L]
        """
        return [x[0], x[1], x[2]]  # Measure all three states
    
    # Step 3: Set up true parameters and generate synthetic data
    print("\nGenerating synthetic bioprocess data...")
    
    # True parameter values (typical for E. coli or yeast with catabolite repression)
    true_mu_max = 0.5     # [1/h] - reasonable growth rate
    true_Ks_glc = 0.1     # [g/L] - glucose half-saturation
    true_Ks_gal = 4.0     # [g/L] - galactose half-saturation (higher than glucose)
    true_Ki_glc = 0.2     # [g/L] - glucose inhibition constant (low value means strong inhibition)
    true_k_d = 0.05       # [1/h] - cell death rate (small but realistic)
    true_Y_glc = 0.5      # [g/g] - glucose yield coefficient  
    true_Y_gal = 0.4      # [g/g] - galactose yield coefficient (lower than glucose)
    
    true_params = {
        'mu_max': true_mu_max,
        'Ks_glc': true_Ks_glc, 
        'Ks_gal': true_Ks_gal,
        'Ki_glc': true_Ki_glc,
        'k_d': true_k_d,
        'Y_glc': true_Y_glc,
        'Y_gal': true_Y_gal
    }
    
    # Initial conditions: small biomass, abundant substrates
    initial_conditions = np.array([0.1, 10.0, 5.0])  # [X, S_glc, S_gal] in g/L
    
    # Time points for measurement (hourly for 24 hours)
    time_points = np.arange(0, 25, 1.0)  # Every hour for 24 hours
    
    # Generate true data by solving the ODE with true parameters
    from scipy.integrate import solve_ivp
    
    def true_bioprocess_system(t, x):
        return bioprocess_dynamics(t, x, np.array([]), true_params)
    
    # Solve the true system
    true_solution = solve_ivp(
        true_bioprocess_system,
        [time_points[0], time_points[-1]],
        initial_conditions,
        t_eval=time_points,
        method='RK45',
        rtol=1e-9
    )
    
    # Create measurements with realistic bioprocess noise
    np.random.seed(42)  # For reproducible results
    
    # Different noise levels for different measurements (realistic)
    biomass_noise = 0.05    # 5% noise for biomass (harder to measure accurately)
    substrate_noise = 0.02  # 2% noise for substrates (more accurate analytical methods)
    
    true_measurements = true_solution.y.T  # Shape: [time_points, states]
    noisy_measurements = true_measurements.copy()
    
    # Add noise to each measurement type
    noisy_measurements[:, 0] += np.random.normal(0, biomass_noise, len(time_points))    # Biomass
    noisy_measurements[:, 1] += np.random.normal(0, substrate_noise, len(time_points))  # Glucose
    noisy_measurements[:, 2] += np.random.normal(0, substrate_noise, len(time_points))  # Galactose
    
    # Ensure no negative concentrations
    noisy_measurements = np.maximum(noisy_measurements, 0.01)
    
    print(f"Generated {len(time_points)} hourly data points")
    print(f"True parameters:")
    print(f"  μ_max = {true_mu_max:.3f} [1/h]")
    print(f"  Ks_glucose = {true_Ks_glc:.3f} [g/L]")  
    print(f"  Ks_galactose = {true_Ks_gal:.3f} [g/L]")
    print(f"  Ki_glucose = {true_Ki_glc:.3f} [g/L] (inhibition constant)")
    print(f"  k_d = {true_k_d:.3f} [1/h] (death rate)")
    print(f"  Y_glucose = {true_Y_glc:.3f} [g/g]")
    print(f"  Y_galactose = {true_Y_gal:.3f} [g/g]")
    print(f"Initial conditions: X={initial_conditions[0]:.1f}, Glc={initial_conditions[1]:.1f}, Gal={initial_conditions[2]:.1f} [g/L]")
    
    # Step 4: Set up parameter estimation
    print("\nSetting up parameter estimation...")
    print("Focus: Estimating the half-saturation constants and inhibition constant")
    
    # Define parameters to estimate - focus on the kinetic constants
    # Keep other parameters fixed for this example
    parameters_to_fit = [
        Parameter(name='mu_max', value=0.4, min=0.1, max=1.0, vary=True),      # Estimate max growth rate
        Parameter(name='Ks_glc', value=0.2, min=0.01, max=2.0, vary=True),     # Estimate glucose Ks
        Parameter(name='Ks_gal', value=0.8, min=0.01, max=3.0, vary=True),     # Estimate galactose Ks  
        Parameter(name='Ki_glc', value=0.5, min=0.01, max=1.0, vary=True),     # Estimate glucose inhibition constant
        Parameter(name='k_d', value=0.03, min=0.001, max=0.2, vary=True),      # Estimate death rate
        Parameter(name='Y_glc', value=true_Y_glc, vary=False),                  # Fix glucose yield
        Parameter(name='Y_gal', value=true_Y_gal, vary=False),                  # Fix galactose yield
    ]
    
    print("Parameters to estimate:")
    for param in parameters_to_fit:
        if param.vary:
            print(f"  {param.name}: initial guess = {param.value:.3f} (true = {true_params[param.name]:.3f})")
        else:
            print(f"  {param.name}: fixed at {param.value:.3f}")
    
    # Create the model
    model = ODESystemFitter.Model(
        sys_func=bioprocess_dynamics,
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
    print("\nRunning bioprocess parameter identification...")
    print("This may take a moment due to the complexity of the model...")
    
    fitter = ODESystemFitter.Fitter(model, data)
    result = fitter.fit()
    
    # Step 6: Analyze results
    fitted_params = result.params.valuesdict()
    
    print("\n" + "="*70)
    print("BIOPROCESS PARAMETER ESTIMATION RESULTS")
    print("="*70)
    
    print("Parameter Comparison:")
    param_errors = {}
    for param_name in ['mu_max', 'Ks_glc', 'Ks_gal', 'Ki_glc', 'k_d']:
        true_val = true_params[param_name]
        fitted_val = fitted_params[param_name]
        error = abs(fitted_val - true_val)
        rel_error = error / true_val * 100
        param_errors[param_name] = rel_error
        
        print(f"  {param_name:8}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Error={error:.4f} ({rel_error:.1f}%)")
    
    # Calculate overall parameter discrepancy
    total_discrepancy = np.sqrt(sum((fitted_params[name] - true_params[name])**2 
                                   for name in ['mu_max', 'Ks_glc', 'Ks_gal', 'Ki_glc', 'k_d']))
    
    print(f"\nOverall Parameter Discrepancy (L2 norm): {total_discrepancy:.4f}")
    print(f"Optimization Success: {result.success}")
    print(f"Number of Function Evaluations: {result.nfev}")
    print(f"Final Residual Sum of Squares: {result.chisqr:.6f}")
    
    # Assess estimation quality
    avg_rel_error = np.mean(list(param_errors.values()))
    
    if avg_rel_error < 10.0:
        print("\n✅ Bioprocess parameter identification was EXCELLENT!")
    elif avg_rel_error < 20.0:
        print("\n✅ Bioprocess parameter identification was GOOD!")
    else:
        print("\n⚠️  Bioprocess parameter identification had significant errors.")
        print("    This is common with complex bioprocess models and limited data.")
    
    # Show final biomass and substrate predictions
    print(f"\nFinal state predictions (t = {time_points[-1]:.0f}h):")
    final_sim = fitter._simulate(fitted_params)[-1, :]
    final_measured = noisy_measurements[-1, :]
    
    print(f"  Biomass:   Predicted={final_sim[0]:.2f}, Measured={final_measured[0]:.2f} [g/L]")
    print(f"  Glucose:   Predicted={final_sim[1]:.2f}, Measured={final_measured[1]:.2f} [g/L]")
    print(f"  Galactose: Predicted={final_sim[2]:.2f}, Measured={final_measured[2]:.2f} [g/L]")
    
    # Custom plotting function to show all states in one plot
    def plot_bioprocess_results(fitter, fitted_params):
        """Custom plotting function to show all three states in one plot."""
        import matplotlib.pyplot as plt
        
        # Simulate with fitted parameters
        y_fitted = fitter._simulate(fitted_params)
        time = fitter.data.time
        measurements = fitter.data.measurements
        
        # Create single plot with all states
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Define colors and labels for each state
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        labels = ['Biomass [g/L]', 'Glucose [g/L]', 'Galactose [g/L]']
        markers = ['o', 's', '^']
        
        # Plot measured data and fitted curves
        for i in range(measurements.shape[1]):
            # Plot measurements
            ax.plot(time, measurements[:, i], markers[i], 
                   color=colors[i], markersize=6, alpha=0.7,
                   label=f'{labels[i]} (Data)', markerfacecolor='white', 
                   markeredgecolor=colors[i], markeredgewidth=2)
            
            # Plot fitted curves
            ax.plot(time, y_fitted[:, i], '-', 
                   color=colors[i], linewidth=2.5, alpha=0.9,
                   label=f'{labels[i]} (Fitted)')
        
        # Customize plot
        ax.set_xlabel('Time [h]', fontsize=14)
        ax.set_ylabel('Concentration [g/L]', fontsize=14)
        ax.set_title('Bioprocess Model: Competitive Inhibition with Cell Death', fontsize=16, fontweight='bold')
        ax.legend(loc='center right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # Add parameter information as text box
        param_text = f"""Fitted Parameters:
μ_max = {fitted_params['mu_max']:.3f} [1/h]
Ks_glc = {fitted_params['Ks_glc']:.3f} [g/L]
Ks_gal = {fitted_params['Ks_gal']:.3f} [g/L]
Ki_glc = {fitted_params['Ki_glc']:.3f} [g/L]
k_d = {fitted_params['k_d']:.3f} [1/h]"""
        
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure to PNG in figures subfolder
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        output_path = os.path.join(figures_dir, 'bioprocess_example_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Figure saved to: {output_path}")
        
        plt.show()
        
        return fig, ax

    # Plot results if possible
    print("\nGenerating bioprocess plots...")
    try:
        plot_bioprocess_results(fitter, fitted_params)
        print("✅ Bioprocess plots displayed successfully.")
    except Exception as e:
        print(f"⚠️  Could not display plots: {e}")
        print("   This is normal when running in non-interactive environments.")
    
    print("="*70)
    
    return result, fitted_params


if __name__ == "__main__":
    # Run the bioprocess example
    result, fitted_params = bioprocess_example()
    
    print("\nBioprocess example completed successfully!")
    print("The library demonstrated:")
    print("✓ Complex multi-state bioprocess model")
    print("✓ Competitive inhibition kinetics")
    print("✓ Realistic hourly measurement frequency")
    print("✓ Multiple parameter estimation with constraints")
    print("✓ Bioprocess-specific noise modeling")
    
    print("\nBioprocess Insights:")
    print("- Glucose inhibits galactose uptake, creating sequential utilization")
    print("- Lower Ki_glc values mean stronger inhibition (more pronounced diauxic growth)")
    print("- Cell death rate k_d affects biomass dynamics and yield estimation")
    print("- Competitive inhibition models capture catabolite repression effects")  
    print("- Low measurement frequency challenges parameter identifiability")
    print("- Yield coefficients are often easier to estimate than kinetic constants")
    print("- This model shows diauxic growth patterns common in microbiology")
    print("- Including cell death makes the model more realistic for longer fermentations")