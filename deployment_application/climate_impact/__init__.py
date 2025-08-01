"""Climate Impact Assessment Module.

This module provides functionality for assessing climate change impacts
on glacier dynamics and related systems.

Components:
- Climate scenario analysis
- Temperature and precipitation projections
- Glacier response modeling
- Impact assessment and adaptation planning
"""

__version__ = "1.0.0"
__author__ = "Tibetan Glacier PINNs Project Team"

# Import main classes and functions
try:
    from .climate_scenarios import (
        ClimateScenario,
        ClimateProjection,
        ScenarioGenerator,
        create_climate_scenario
    )
except ImportError:
    pass

try:
    from .temperature_projections import (
        TemperatureModel,
        TemperatureProjector,
        create_temperature_projector
    )
except ImportError:
    pass

try:
    from .precipitation_analysis import (
        PrecipitationModel,
        PrecipitationAnalyzer,
        create_precipitation_analyzer
    )
except ImportError:
    pass

try:
    from .glacier_response import (
        GlacierResponseModel,
        ClimateImpactAssessor,
        create_glacier_response_model
    )
except ImportError:
    pass

# Define what gets imported with "from climate_impact import *"
__all__ = [
    # Climate scenarios
    'ClimateScenario',
    'ClimateProjection', 
    'ScenarioGenerator',
    'create_climate_scenario',
    
    # Temperature projections
    'TemperatureModel',
    'TemperatureProjector',
    'create_temperature_projector',
    
    # Precipitation analysis
    'PrecipitationModel',
    'PrecipitationAnalyzer',
    'create_precipitation_analyzer',
    
    # Glacier response
    'GlacierResponseModel',
    'ClimateImpactAssessor',
    'create_glacier_response_model'
]