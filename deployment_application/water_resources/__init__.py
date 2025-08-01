"""Water Resources Module.

This module provides functionality for water resource assessment and management
based on glacier dynamics predictions, including runoff estimation, water
availability analysis, and hydrological modeling.

Components:
- runoff_estimation: Glacier melt runoff estimation
- water_availability: Water resource availability analysis
- hydrological_modeling: Hydrological process modeling
- seasonal_analysis: Seasonal water resource analysis
"""

__version__ = "1.0.0"
__author__ = "Glacier Dynamics Research Team"

from .runoff_estimation import (
    RunoffEstimator,
    GlacierMeltModel,
    SnowMeltModel,
    create_runoff_estimator
)

from .water_availability import (
    WaterAvailabilityAnalyzer,
    SeasonalWaterBalance,
    WaterDemandModel,
    create_water_analyzer
)

from .hydrological_modeling import (
    HydrologicalModel,
    CatchmentModel,
    StreamflowModel,
    create_hydrological_model
)

from .seasonal_analysis import (
    SeasonalAnalyzer,
    ClimateImpactAnalyzer,
    WaterResourceProjector,
    create_seasonal_analyzer
)

__all__ = [
    # Runoff estimation
    "RunoffEstimator",
    "GlacierMeltModel",
    "SnowMeltModel",
    "create_runoff_estimator",
    
    # Water availability
    "WaterAvailabilityAnalyzer",
    "SeasonalWaterBalance",
    "WaterDemandModel",
    "create_water_analyzer",
    
    # Hydrological modeling
    "HydrologicalModel",
    "CatchmentModel",
    "StreamflowModel",
    "create_hydrological_model",
    
    # Seasonal analysis
    "SeasonalAnalyzer",
    "ClimateImpactAnalyzer",
    "WaterResourceProjector",
    "create_seasonal_analyzer"
]