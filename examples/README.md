# ODEfit Template Examples

This folder contains examples demonstrating the use of the ODEfit Template library for parameter estimation in ODE systems, progressing from simple to complex scenarios.

## Examples (in order of complexity)

### 1. `one_state_example.py` - Exponential Decay ⭐ START HERE
**Simplest possible case - Perfect for beginners**
- **Model**: `dx/dt = -k*x` (exponential decay)
- **States**: 1 (concentration/amount)
- **Parameters**: 1 (decay rate `k`)
- **Measurements**: Direct measurement of the state
- **Features**: 
  - Easy to understand and verify analytically
  - Perfect for learning the library basics
  - Excellent parameter identifiability
  - Fast execution

Run with:
```bash
python examples/one_state_example.py
```

### 2. `two_state_example.py` - Sequential Reactions
**Introduces identifiability challenges**
- **Model**: `A → B → C` (sequential reactions)
- **States**: 2 (concentrations of A and B)
- **Parameters**: 2 (reaction rates `k1` and `k2`)
- **Measurements**: Concentration of A only
- **Features**:
  - Demonstrates parameter identifiability issues
  - Shows how measurement choice affects estimation quality
  - More realistic example of practical challenges

Run with:
```bash
python examples/two_state_example.py
```

### 3. `bioprocess_example.py` - Dual Substrate Monod Kinetics
**Complex bioprocess model with realistic challenges**
- **Model**: Dual substrate Monod growth kinetics (glucose + galactose)
- **States**: 3 (biomass, glucose, galactose concentrations)
- **Parameters**: 3 (μ_max, Ks_glucose, Ks_galactose) + 2 fixed yields
- **Measurements**: All three states (hourly sampling)
- **Features**:
  - Realistic bioprocess complexity
  - Multiple substrate competition
  - Low-frequency measurements (hourly)
  - Different noise levels for different measurements
  - Parameter correlation challenges

Run with:
```bash
python examples/bioprocess_example.py
```

## Key Learning Points

### Parameter Identifiability
1. **Start Simple**: The one-state example has perfect identifiability
2. **Measurement Matters**: The two-state example shows limited measurements can cause identifiability issues
3. **Complex Systems**: The bioprocess example demonstrates how parameter correlations affect estimation

### Measurement Strategy
- **What you measure greatly affects which parameters can be estimated**
- **Measurement frequency matters**: More frequent = better, but realistic constraints exist
- **Different measurements have different noise characteristics**

### Library Workflow
All examples follow the same pattern:
1. Define ODE system function (`sys_func`)
2. Define measurement function (`output_func`)
3. Generate/provide experimental data
4. Set up parameters with initial guesses and bounds
5. Run fitting and analyze results

## Bioprocess Insights (from example 3)

- **Glucose vs Galactose**: Glucose typically has lower Ks (higher affinity)
- **Parameter Correlations**: μ_max and Ks values can be correlated
- **Yield Coefficients**: Often easier to estimate than kinetic constants
- **Measurement Noise**: Biomass measurements typically have higher noise than substrate measurements

## Tips for Success

1. **Start with the simple example** to understand the workflow
2. **Pay attention to parameter bounds** - realistic bounds improve convergence
3. **Consider measurement frequency** - balance information content vs. realism
4. **Multiple outputs help identifiability** - measure what you can!
5. **Check parameter correlations** - some parameters may not be independently identifiable

## Next Steps

- Try modifying the measurement functions
- Experiment with different noise levels
- Add more measurement points or change the time horizon
- Create your own ODE system!
- Try fixing some parameters while estimating others 