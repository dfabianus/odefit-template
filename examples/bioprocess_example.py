#!/usr/bin/env python3
"""
Bioprocess ODE parameter estimation example with competitive inhibition kinetics and acetate production.

This example demonstrates:
1. A four-state bioprocess model: biomass growth with sequential substrate utilization and acetate production
2. Competitive inhibition kinetics (glucose inhibits galactose uptake)
3. Growth-coupled acetate production (overflow metabolism)
4. Cell death kinetics to model biomass decay
5. Generate synthetic measurement data with realistic bioprocess noise
6. Identify kinetic parameters (half-saturation constants, inhibition constant, death rate, acetate yield)
7. Low-frequency measurements (hourly) as typical in bioprocesses

Model equations with competitive inhibition, maintenance deficit-based death, and acetate production:
- μ_glc = S_glc / (Ks_glc + S_glc)  [glucose utilization]
- μ_gal = S_gal / (Ks_gal + S_gal) * Ki_glc / (Ki_glc + S_glc)  [galactose utilization with glucose inhibition]
- μ_total = μ_max * (μ_glc + μ_gal)  [total growth rate]
- maintenance_shortfall = max(0, μ_maintenance - μ_total)  [growth deficit]
- k_d_effective = k_d_aging * Deficit / (deficit_half + Deficit)  [aging-based death]

- dX/dt = μ_total * X - k_d_effective * X  [biomass growth minus aging-based death]
- dS_glc/dt = -1/Y_glc * μ_max * μ_glc * X  
- dS_gal/dt = -1/Y_gal * μ_max * μ_gal * X
- dAcetate/dt = Y_acetate * μ_total * X  [growth-coupled acetate production]
- dDeficit/dt = maintenance_shortfall  [accumulation of maintenance deficit]

Where:
- X = biomass concentration [g/L]
- S_glc = glucose concentration [g/L] 
- S_gal = galactose concentration [g/L]
- Acetate = acetate concentration [g/L]
- Deficit = accumulated maintenance deficit [dimensionless]
- μ_max = maximum specific growth rate [1/h]
- Ks_glc, Ks_gal = half-saturation constants [g/L]
- Ki_glc = inhibition constant for glucose on galactose uptake [g/L]
- μ_maintenance = minimum growth rate for cell maintenance [1/h]
- k_d_aging = death rate coefficient due to maintenance deficit [1/h]
- deficit_half = half-saturation for maintenance deficit effects [dimensionless]
- Y_glc, Y_gal = yield coefficients [g_biomass/g_substrate]
- Y_acetate = acetate yield coefficient [g_acetate/g_biomass] (overflow metabolism)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.odefit_template import Parameter, ODESystemFitter

# ============================================================================
# BIOPROCESS MODEL CONFIGURATION
# ============================================================================
# Modify these parameters to customize the bioprocess model

# Initial conditions [X, S_glc, S_gal, Acetate, Maintenance_deficit] in g/L and dimensionless
INITIAL_CONDITIONS = np.array([0.1, 10.0, 5.0, 0.0, 0.0])

# True parameter values (for synthetic data generation)
TRUE_PARAMETERS = {
    'mu_max': 0.5,        # [1/h] - maximum specific growth rate
    'Ks_glc': 0.3,        # [g/L] - glucose half-saturation constant
    'Ks_gal': 4.0,        # [g/L] - galactose half-saturation constant (higher than glucose)
    'Ki_glc': 0.2,        # [g/L] - glucose inhibition constant (low = strong inhibition)
    'mu_maintenance': 0.05, # [1/h] - minimum growth rate required for cell maintenance
    'k_d_aging': 0.13,    # [1/h] - death rate coefficient due to maintenance deficit
    'deficit_half': 2.0,  # [dimensionless] - half-saturation for maintenance deficit effects
    'Y_glc': 0.5,         # [g_biomass/g_glucose] - glucose yield coefficient
    'Y_gal': 0.4,         # [g_biomass/g_galactose] - galactose yield coefficient
    'Y_acetate': 0.15     # [g_acetate/g_biomass] - acetate yield (overflow metabolism)
}

# Parameter estimation setup
PARAMETERS_TO_FIT = [
    # Parameter(name, initial_guess, min_bound, max_bound, vary)
    ('mu_max', 0.4, 0.1, 1.0, True),
    ('Ks_glc', 0.2, 0.01, 2.0, True),
    ('Ks_gal', 0.8, 0.01, 3.0, True),
    ('Ki_glc', 0.5, 0.01, 1.0, True),
    ('mu_maintenance', 0.015, 0.005, 0.05, True),         # Estimate maintenance threshold
    ('k_d_aging', 0.06, 0.02, 0.15, True),               # Estimate aging death coefficient  
    ('deficit_half', 1.5, 0.5, 5.0, True),               # Estimate deficit half-saturation
    ('Y_glc', TRUE_PARAMETERS['Y_glc'], 0.1, 1.0, False),  # Fixed
    ('Y_gal', TRUE_PARAMETERS['Y_gal'], 0.1, 1.0, False),  # Fixed
    ('Y_acetate', 0.1, 0.05, 0.3, True)  # Estimate acetate yield
]

# Time configuration
TIME_START = 0.0          # [h] - start time
TIME_END = 24.0           # [h] - end time  
TIME_STEP = 1.0           # [h] - measurement interval (hourly sampling)

# Noise configuration (realistic bioprocess measurement uncertainties)
NOISE_LEVELS = {
    'biomass': 0.05,      # 5% noise for biomass (optical density, dry weight)
    'glucose': 0.02,      # 2% noise for glucose (HPLC, enzymatic assays)
    'galactose': 0.02,    # 2% noise for galactose (HPLC, enzymatic assays)
    'acetate': 0.03       # 3% noise for acetate (HPLC, GC-MS)
}

# Random seed for reproducible results
RANDOM_SEED = 42

# ============================================================================
# BIOPROCESS MODEL IMPLEMENTATION
# ============================================================================

def bioprocess_example():
    """Run a bioprocess parameter estimation example with competitive inhibition kinetics and acetate production."""
    
    print("="*80)
    print("BIOPROCESS PARAMETER ESTIMATION EXAMPLE WITH ACETATE PRODUCTION")
    print("="*80)
    print("Model: Competitive inhibition kinetics + growth-coupled acetate production + maintenance deficit-based death")
    print("States: Biomass, Glucose, Galactose, Acetate")
    print("Parameters to identify: Ks_glucose, Ks_galactose, Ki_glucose, death kinetics, Y_acetate")
    print("Measurements: All four states (hourly sampling)")
    
    # Step 1: Define the bioprocess ODE system with competitive inhibition and acetate production
    def bioprocess_dynamics(t, x, u, p):
        """
        Define the competitive inhibition bioprocess model with maintenance deficit-based death and acetate production.
        
        Args:
            t: time [h]
            x: state vector [biomass, glucose, galactose, acetate, maintenance_deficit] [g/L, dimensionless]
            u: inputs (none in this example)
            p: parameters dict {
                'mu_max': maximum growth rate [1/h],
                'Ks_glc': glucose half-saturation constant [g/L],
                'Ks_gal': galactose half-saturation constant [g/L],
                'Ki_glc': glucose inhibition constant for galactose uptake [g/L],
                'mu_maintenance': minimum growth rate for cell maintenance [1/h],
                'k_d_aging': death rate coefficient due to maintenance deficit [1/h],
                'deficit_half': half-saturation for maintenance deficit effects [dimensionless],
                'Y_glc': glucose yield coefficient [g_biomass/g_glucose],
                'Y_gal': galactose yield coefficient [g_biomass/g_galactose],
                'Y_acetate': acetate yield coefficient [g_acetate/g_biomass]
            }
        
        Returns:
            dx/dt: [dX/dt, dS_glc/dt, dS_gal/dt, dAcetate/dt, dDeficit/dt]
        """
        # Extract states
        X = x[0]        # Biomass [g/L]
        S_glc = x[1]    # Glucose [g/L] 
        S_gal = x[2]    # Galactose [g/L]
        Acetate = x[3]  # Acetate [g/L]
        Deficit = x[4]  # Accumulated maintenance deficit [dimensionless]
        
        # Extract parameters
        mu_max = p['mu_max']
        Ks_glc = p['Ks_glc']
        Ks_gal = p['Ks_gal']
        Ki_glc = p['Ki_glc']
        mu_maintenance = p['mu_maintenance']
        k_d_aging = p['k_d_aging']
        deficit_half = p['deficit_half']
        Y_glc = p['Y_glc']
        Y_gal = p['Y_gal']
        Y_acetate = p['Y_acetate']
        
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
        
        # Maintenance deficit dynamics
        # When growth rate < maintenance: accumulate deficit
        # When growth rate >= maintenance: no additional deficit (but existing deficit persists)
        maintenance_shortfall = max(0, mu_maintenance - mu_total)
        
        # Death rate based on accumulated maintenance deficit
        # Uses Monod-like kinetics: k_d = k_d_aging * Deficit / (deficit_half + Deficit)
        k_d_effective = k_d_aging * Deficit / (deficit_half + Deficit)
        
        # ODE equations with maintenance deficit-based death and acetate production
        dX_dt = mu_total * X - k_d_effective * X               # Biomass growth minus aging-based death
        dS_glc_dt = -(1/Y_glc) * mu_glc_actual * X             # Glucose consumption
        dS_gal_dt = -(1/Y_gal) * mu_gal_actual * X             # Galactose consumption
        dAcetate_dt = Y_acetate * mu_total * X                 # Growth-coupled acetate production
        dDeficit_dt = maintenance_shortfall                    # Accumulation of maintenance deficit
        
        return [dX_dt, dS_glc_dt, dS_gal_dt, dAcetate_dt, dDeficit_dt]
    
    # Step 2: Define the measurement function
    def measurement_function(t, x, u, p):
        """
        Define what we can measure in the bioprocess.
        We measure the four observable states: biomass, glucose, galactose, acetate.
        The maintenance deficit is an internal state variable (not directly measurable).
        
        Returns:
            y: [biomass, glucose, galactose, acetate] [g/L]
        """
        return [x[0], x[1], x[2], x[3]]  # Measure first four states (not the deficit)
    
    # Step 3: Set up time points and generate synthetic data
    print("\nGenerating synthetic bioprocess data...")
    print(f"Configuration:")
    print(f"  Time range: {TIME_START:.0f} - {TIME_END:.0f} h (every {TIME_STEP:.1f} h)")
    print(f"  Initial conditions: X={INITIAL_CONDITIONS[0]:.1f}, Glc={INITIAL_CONDITIONS[1]:.1f}, Gal={INITIAL_CONDITIONS[2]:.1f}, Ace={INITIAL_CONDITIONS[3]:.1f}, Def={INITIAL_CONDITIONS[4]:.1f}")
    
    # Time points for measurement
    time_points = np.arange(TIME_START, TIME_END + TIME_STEP, TIME_STEP)
    
    # Generate true data by solving the ODE with true parameters
    from scipy.integrate import solve_ivp
    
    def true_bioprocess_system(t, x):
        return bioprocess_dynamics(t, x, np.array([]), TRUE_PARAMETERS)
    
    # Solve the true system
    true_solution = solve_ivp(
        true_bioprocess_system,
        [time_points[0], time_points[-1]],
        INITIAL_CONDITIONS,
        t_eval=time_points,
        method='RK45',
        rtol=1e-9
    )
    
    # Create measurements with realistic bioprocess noise
    np.random.seed(RANDOM_SEED)
    
    # Extract only the measurable states (first 4: biomass, glucose, galactose, acetate)
    # The 5th state (maintenance deficit) is internal and not directly measurable
    true_measurements = true_solution.y[:4, :].T  # Shape: [time_points, 4_measurable_states]
    noisy_measurements = true_measurements.copy()
    
    # Add noise to each measurement type
    noisy_measurements[:, 0] += np.random.normal(0, NOISE_LEVELS['biomass'], len(time_points))     # Biomass
    noisy_measurements[:, 1] += np.random.normal(0, NOISE_LEVELS['glucose'], len(time_points))     # Glucose
    noisy_measurements[:, 2] += np.random.normal(0, NOISE_LEVELS['galactose'], len(time_points))   # Galactose
    noisy_measurements[:, 3] += np.random.normal(0, NOISE_LEVELS['acetate'], len(time_points))     # Acetate
    
    # Ensure no negative concentrations
    noisy_measurements = np.maximum(noisy_measurements, 0.01)
    
    print(f"Generated {len(time_points)} hourly data points")
    print(f"True parameters:")
    for param_name, value in TRUE_PARAMETERS.items():
        print(f"  {param_name} = {value:.3f}")
    
    # Step 4: Set up parameter estimation
    print("\nSetting up parameter estimation...")
    print("Focus: Estimating kinetic constants and acetate yield coefficient")
    
    # Define parameters to estimate
    parameters_to_fit = []
    for param_name, initial_guess, min_bound, max_bound, vary in PARAMETERS_TO_FIT:
        parameters_to_fit.append(
            Parameter(name=param_name, value=initial_guess, min=min_bound, max=max_bound, vary=vary)
        )
    
    print("Parameters to estimate:")
    for param in parameters_to_fit:
        if param.vary:
            true_val = TRUE_PARAMETERS.get(param.name, 'N/A')
            print(f"  {param.name}: initial guess = {param.value:.3f} (true = {true_val:.3f})")
        else:
            print(f"  {param.name}: fixed at {param.value:.3f}")
    
    # Create the model
    model = ODESystemFitter.Model(
        sys_func=bioprocess_dynamics,
        output_func=measurement_function,
        initial_conditions=INITIAL_CONDITIONS,
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
    
    print("\n" + "="*80)
    print("BIOPROCESS PARAMETER ESTIMATION RESULTS")
    print("="*80)
    
    print("Parameter Comparison:")
    param_errors = {}
    for param_name in ['mu_max', 'Ks_glc', 'Ks_gal', 'Ki_glc', 'k_d_aging', 'deficit_half', 'Y_acetate']:
        if param_name in fitted_params and param_name in TRUE_PARAMETERS:
            true_val = TRUE_PARAMETERS[param_name]
            fitted_val = fitted_params[param_name]
            error = abs(fitted_val - true_val)
            rel_error = error / true_val * 100
            param_errors[param_name] = rel_error
            
            print(f"  {param_name:10}: True={true_val:.4f}, Fitted={fitted_val:.4f}, Error={error:.4f} ({rel_error:.1f}%)")
    
    # Calculate overall parameter discrepancy
    total_discrepancy = np.sqrt(sum((fitted_params[name] - TRUE_PARAMETERS[name])**2 
                                   for name in ['mu_max', 'Ks_glc', 'Ks_gal', 'Ki_glc', 'k_d_aging', 'deficit_half', 'Y_acetate']
                                   if name in fitted_params))
    
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
    final_sim = fitter._simulate(fitted_params)[-1, :]  # This returns 4 measurable states
    final_measured = noisy_measurements[-1, :]          # This has 4 measurable states
    
    print(f"  Biomass:   Predicted={final_sim[0]:.2f}, Measured={final_measured[0]:.2f} [g/L]")
    print(f"  Glucose:   Predicted={final_sim[1]:.2f}, Measured={final_measured[1]:.2f} [g/L]")
    print(f"  Galactose: Predicted={final_sim[2]:.2f}, Measured={final_measured[2]:.2f} [g/L]")
    print(f"  Acetate:   Predicted={final_sim[3]:.2f}, Measured={final_measured[3]:.2f} [g/L]")
    
    # Custom plotting function to show all states in one plot
    def plot_bioprocess_results(fitter, fitted_params):
        """Custom plotting function to show all four states in one plot."""
        import matplotlib.pyplot as plt
        
        # Simulate with fitted parameters
        y_fitted = fitter._simulate(fitted_params)
        time = fitter.data.time
        measurements = fitter.data.measurements
        
        # Create single plot with all states
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Define colors and labels for each state
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        labels = ['Biomass [g/L]', 'Glucose [g/L]', 'Galactose [g/L]', 'Acetate [g/L]']
        markers = ['o', 's', '^', 'x']
        
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
        ax.set_title('Bioprocess Model: Competitive Inhibition with Cell Death and Acetate Production', fontsize=16, fontweight='bold')
        ax.legend(loc='center right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # Add parameter information as text box
        param_text = f"""Fitted Parameters:
μ_max = {fitted_params['mu_max']:.3f} [1/h]
Ks_glc = {fitted_params['Ks_glc']:.3f} [g/L]
Ks_gal = {fitted_params['Ks_gal']:.3f} [g/L]
Ki_glc = {fitted_params['Ki_glc']:.3f} [g/L]
k_d_aging = {fitted_params['k_d_aging']:.3f} [1/h]
deficit_half = {fitted_params['deficit_half']:.3f} [dimensionless]
Y_acetate = {fitted_params['Y_acetate']:.3f} [g_acetate/g_biomass]"""
        
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
    
    print("="*80)
    
    return result, fitted_params


if __name__ == "__main__":
    # Run the bioprocess example
    result, fitted_params = bioprocess_example()
    
    print("\nBioprocess example with acetate production completed successfully!")
    print("The library demonstrated:")
    print("✓ Complex multi-state bioprocess model (4 states)")
    print("✓ Competitive inhibition kinetics")
    print("✓ Growth-coupled acetate production (overflow metabolism)")
    print("✓ Realistic hourly measurement frequency")
    print("✓ Multiple parameter estimation with constraints")
    print("✓ Bioprocess-specific noise modeling")
    print("✓ Configuration-driven parameter setup")
    
    print("\nBioprocess Insights:")
    print("- Glucose inhibits galactose uptake, creating sequential utilization")
    print("- Lower Ki_glc values mean stronger inhibition (more pronounced diauxic growth)")
    print("- Maintenance deficit death: cells accumulate damage when μ < μ_maintenance")
    print("- deficit_half defines the deficit level where death rate becomes significant")
    print("- k_d_aging controls the maximum death rate due to accumulated damage")
    print("- Death kinetics: k_d = k_d_aging * Deficit / (deficit_half + Deficit)")
    print("- μ_maintenance represents minimum growth needed for cellular maintenance")
    print("- Deficit accumulates over time when cells can't meet maintenance requirements")
    print("- Competitive inhibition models capture catabolite repression effects")  
    print("- Acetate production is growth-coupled (overflow metabolism)")
    print("- Y_acetate represents the fraction of growth that produces acetate")
    print("- Low measurement frequency challenges parameter identifiability")
    print("- Yield coefficients are often easier to estimate than kinetic constants")
    print("- This model shows diauxic growth patterns common in microbiology")
    print("- Maintenance deficit creates realistic aging-based death progression")
    print("- Acetate accumulation is typical in high-glucose, aerobic fermentations")
    print("- Death kinetics: growth → stationary → progressive cellular aging death")