"""Parameter handling for ODE model fitting."""

from typing import Optional


class Parameter:
    """A parameter for ODE model fitting.
    
    Args:
        name: Name of the parameter
        value: Initial value/guess for the parameter
        min: Minimum bound (optional)
        max: Maximum bound (optional)
        vary: Whether to vary this parameter during fitting
    """
    
    def __init__(
        self, 
        name: str, 
        value: float, 
        min: Optional[float] = None, 
        max: Optional[float] = None, 
        vary: bool = True
    ):
        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
    
    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', value={self.value}, min={self.min}, max={self.max}, vary={self.vary})" 