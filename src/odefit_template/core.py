"""Core ODE system fitting functionality."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from lmfit import Parameters, minimize
from typing import Callable, List, Optional, Dict, Any
import matplotlib.pyplot as plt

from .parameters import Parameter


class ODESystemFitter:
    """Main class for ODE system fitting."""
    
    class Model:
        """Encapsulates an ODE model definition."""
        
        def __init__(
            self,
            sys_func: Callable[[float, np.ndarray, np.ndarray, Dict[str, float]], List[float]],
            output_func: Callable[[float, np.ndarray, np.ndarray, Dict[str, float]], List[float]],
            initial_conditions: np.ndarray,
            parameters: List[Parameter]
        ):
            """Initialize an ODE model.
            
            Args:
                sys_func: Function defining the ODE system (t, x, u, p) -> dx/dt
                output_func: Function defining measurements (t, x, u, p) -> y
                initial_conditions: Initial state values
                parameters: List of Parameter objects to fit
            """
            self.sys_func = sys_func
            self.output_func = output_func
            self.initial_conditions = initial_conditions
            self.parameters = parameters
            
    class Data:
        """Encapsulates experimental data."""
        
        def __init__(
            self,
            time: np.ndarray,
            measurements: np.ndarray,
            inputs_time: Optional[np.ndarray] = None,
            inputs_values: Optional[np.ndarray] = None
        ):
            """Initialize experimental data.
            
            Args:
                time: Time points of measurements
                measurements: Measured values (shape: [n_time, n_outputs])
                inputs_time: Time points for inputs (optional)
                inputs_values: Input values (optional, shape: [n_input_time, n_inputs])
            """
            self.time = time
            self.measurements = measurements
            self.inputs_time = inputs_time
            self.inputs_values = inputs_values
            
            # Create input interpolation function if inputs are provided
            if inputs_time is not None and inputs_values is not None:
                self.input_func = interp1d(
                    inputs_time, inputs_values.T, 
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
            else:
                self.input_func = None
                
    class Fitter:
        """Handles the optimization process."""
        
        def __init__(self, model: 'ODESystemFitter.Model', data: 'ODESystemFitter.Data'):
            """Initialize the fitter.
            
            Args:
                model: ODE model to fit
                data: Experimental data to fit to
            """
            self.model = model
            self.data = data
            self.result = None
            
        def _simulate(self, params_dict: Dict[str, float]) -> np.ndarray:
            """Simulate the ODE model with given parameters."""
            
            def ode_func(t, x):
                # Get inputs at time t
                if self.data.input_func is not None:
                    u = self.data.input_func(t)
                    if u.ndim == 0:
                        u = np.array([u])
                else:
                    u = np.array([])
                
                return self.model.sys_func(t, x, u, params_dict)
            
            # Solve ODE
            sol = solve_ivp(
                ode_func,
                [self.data.time[0], self.data.time[-1]],
                self.model.initial_conditions,
                t_eval=self.data.time,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if not sol.success:
                return np.full_like(self.data.measurements, np.inf)
            
            # Calculate outputs
            outputs = []
            for i, t in enumerate(self.data.time):
                x = sol.y[:, i]
                if self.data.input_func is not None:
                    u = self.data.input_func(t)
                    if u.ndim == 0:
                        u = np.array([u])
                else:
                    u = np.array([])
                
                y = self.model.output_func(t, x, u, params_dict)
                outputs.append(y)
            
            return np.array(outputs)
        
        def _objective(self, params):
            """Objective function for optimization."""
            # Convert lmfit Parameters to dict
            params_dict = params.valuesdict()
            
            # Simulate model
            y_model = self._simulate(params_dict)
            
            # Calculate residuals
            residuals = (y_model - self.data.measurements).flatten()
            
            return residuals
        
        def fit(self) -> Any:
            """Run the parameter estimation."""
            # Convert our Parameter objects to lmfit Parameters
            lm_params = Parameters()
            for param in self.model.parameters:
                lm_params.add(
                    param.name,
                    value=param.value,
                    min=param.min,
                    max=param.max,
                    vary=param.vary
                )
            
            # Run optimization
            self.result = minimize(self._objective, lm_params, method='leastsq')
            
            return self.result
        
        def plot(self):
            """Plot the fit results."""
            if self.result is None:
                raise ValueError("No fit result available. Run fit() first.")
            
            # Simulate with fitted parameters
            y_fitted = self._simulate(self.result.params.valuesdict())
            
            fig, axes = plt.subplots(self.data.measurements.shape[1], 1, figsize=(10, 6))
            if self.data.measurements.shape[1] == 1:
                axes = [axes]
            
            for i in range(self.data.measurements.shape[1]):
                axes[i].plot(self.data.time, self.data.measurements[:, i], 'o', label='Data')
                axes[i].plot(self.data.time, y_fitted[:, i], '-', label='Fitted')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(f'Output {i+1}')
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.show()
            
            return fig, axes 