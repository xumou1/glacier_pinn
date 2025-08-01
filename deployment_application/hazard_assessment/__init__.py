"""Hazard Assessment Module.

This module provides functionality for assessing glacier-related hazards
and risks based on glacier dynamics predictions.

Components:
- Glacial Lake Outburst Flood (GLOF) assessment
- Ice avalanche risk evaluation
- Glacier surge detection and prediction
- Infrastructure vulnerability assessment
"""

__version__ = "1.0.0"
__author__ = "Tibetan Glacier PINNs Project Team"

# Import main classes and functions
from .glof_assessment import (
    GLOFRiskAssessment,
    GlacialLakeMonitor,
    create_glof_assessor
)

from .ice_avalanche import (
    IceAvalanchePredictor,
    AvalancheRiskCalculator,
    create_avalanche_predictor
)

from .glacier_surge import (
    GlacierSurgeDetector,
    SurgePredictor,
    create_surge_detector
)

from .infrastructure_vulnerability import (
    InfrastructureVulnerabilityAssessment,
    RiskMitigationPlanner,
    create_vulnerability_assessor
)

__all__ = [
    # GLOF Assessment
    'GLOFRiskAssessment',
    'GlacialLakeMonitor',
    'create_glof_assessor',
    
    # Ice Avalanche
    'IceAvalanchePredictor',
    'AvalancheRiskCalculator',
    'create_avalanche_predictor',
    
    # Glacier Surge
    'GlacierSurgeDetector',
    'SurgePredictor',
    'create_surge_detector',
    
    # Infrastructure Vulnerability
    'InfrastructureVulnerabilityAssessment',
    'RiskMitigationPlanner',
    'create_vulnerability_assessor'
]