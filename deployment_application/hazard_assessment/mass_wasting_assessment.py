"""Infrastructure Vulnerability Assessment Module.

This module provides functionality for assessing the vulnerability
of infrastructure to glacier-related hazards.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy.spatial.distance import cdist
from scipy import interpolate
import json


class InfrastructureType(Enum):
    """Types of infrastructure."""
    ROAD = "road"
    BRIDGE = "bridge"
    BUILDING = "building"
    DAM = "dam"
    POWER_LINE = "power_line"
    PIPELINE = "pipeline"
    RAILWAY = "railway"
    AIRPORT = "airport"
    SETTLEMENT = "settlement"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"
    TOURISM = "tourism"


class HazardType(Enum):
    """Types of glacier-related hazards."""
    GLOF = "glof"
    ICE_AVALANCHE = "ice_avalanche"
    GLACIER_SURGE = "glacier_surge"
    DEBRIS_FLOW = "debris_flow"
    FLOODING = "flooding"
    EROSION = "erosion"
    GROUND_INSTABILITY = "ground_instability"


class VulnerabilityLevel(Enum):
    """Vulnerability level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class InfrastructureAsset:
    """Infrastructure asset characteristics."""
    asset_id: str
    name: str
    infrastructure_type: InfrastructureType
    
    # Location
    latitude: float
    longitude: float
    elevation: float  # m above sea level
    
    # Physical characteristics
    construction_year: int
    design_life: int  # years
    replacement_cost: float  # USD
    
    # Structural properties
    material_type: str  # concrete, steel, wood, etc.
    structural_condition: float  # 0-1 scale (1 = excellent)
    foundation_type: str  # shallow, deep, rock, etc.
    
    # Operational characteristics
    capacity: float  # varies by type (people, vehicles/day, MW, etc.)
    criticality: float  # 0-1 scale (1 = critical)
    redundancy: float  # 0-1 scale (1 = high redundancy)
    
    # Exposure characteristics
    exposure_to_water: float  # 0-1 scale
    exposure_to_debris: float  # 0-1 scale
    exposure_to_ice: float  # 0-1 scale
    
    # Economic characteristics
    annual_revenue: Optional[float] = None  # USD/year
    maintenance_cost: Optional[float] = None  # USD/year
    
    # Social characteristics
    population_served: Optional[int] = None
    essential_services: bool = False  # hospital, emergency services, etc.


@dataclass
class HazardScenario:
    """Hazard scenario characteristics."""
    scenario_id: str
    hazard_type: HazardType
    
    # Spatial characteristics
    source_latitude: float
    source_longitude: float
    affected_area_polygon: List[Tuple[float, float]]  # lat, lon pairs
    
    # Intensity characteristics
    magnitude: float  # hazard-specific magnitude
    probability: float  # annual probability (0-1)
    duration: float  # hours
    
    # Physical parameters
    flow_velocity: Optional[float] = None  # m/s
    flow_depth: Optional[float] = None  # m
    debris_concentration: Optional[float] = None  # kg/m³
    impact_pressure: Optional[float] = None  # Pa
    
    # Temporal characteristics
    warning_time: float = 0.0  # hours
    return_period: Optional[float] = None  # years


@dataclass
class VulnerabilityConfig:
    """Configuration for vulnerability assessment."""
    # Distance thresholds (km)
    direct_impact_distance: float = 1.0
    indirect_impact_distance: float = 5.0
    regional_impact_distance: float = 20.0
    
    # Vulnerability weights by infrastructure type
    infrastructure_weights: Dict[str, Dict[str, float]] = None
    
    # Hazard-specific parameters
    hazard_parameters: Dict[str, Dict[str, float]] = None
    
    # Economic parameters
    discount_rate: float = 0.03  # annual
    analysis_period: int = 50  # years
    
    # Fragility curve parameters
    fragility_curves: Dict[str, Dict[str, List[float]]] = None
    
    def __post_init__(self):
        """Initialize default parameters."""
        if self.infrastructure_weights is None:
            self.infrastructure_weights = {
                'road': {'structural': 0.3, 'operational': 0.4, 'economic': 0.3},
                'bridge': {'structural': 0.5, 'operational': 0.3, 'economic': 0.2},
                'building': {'structural': 0.4, 'operational': 0.3, 'economic': 0.3},
                'dam': {'structural': 0.6, 'operational': 0.3, 'economic': 0.1},
                'power_line': {'structural': 0.4, 'operational': 0.5, 'economic': 0.1},
                'default': {'structural': 0.4, 'operational': 0.3, 'economic': 0.3}
            }
        
        if self.hazard_parameters is None:
            self.hazard_parameters = {
                'glof': {'velocity_threshold': 2.0, 'depth_threshold': 1.0},
                'ice_avalanche': {'pressure_threshold': 10000, 'debris_threshold': 100},
                'glacier_surge': {'advance_threshold': 100, 'velocity_threshold': 200},
                'debris_flow': {'velocity_threshold': 5.0, 'debris_threshold': 500},
                'flooding': {'depth_threshold': 0.5, 'velocity_threshold': 1.0}
            }
        
        if self.fragility_curves is None:
            # Simplified fragility curves: [no_damage, minor, moderate, major, complete]
            self.fragility_curves = {
                'road': {
                    'depth': [0.0, 0.3, 0.8, 1.5, 3.0],
                    'velocity': [0.0, 1.0, 2.5, 5.0, 10.0]
                },
                'bridge': {
                    'depth': [0.0, 0.5, 1.2, 2.5, 5.0],
                    'velocity': [0.0, 1.5, 3.0, 6.0, 12.0]
                },
                'building': {
                    'depth': [0.0, 0.2, 0.6, 1.2, 2.5],
                    'pressure': [0.0, 1000, 5000, 15000, 50000]
                }
            }


class SpatialAnalysis:
    """Spatial analysis for infrastructure vulnerability."""
    
    def __init__(self, config: VulnerabilityConfig = None):
        """
        Initialize spatial analysis.
        
        Args:
            config: Vulnerability assessment configuration
        """
        self.config = config or VulnerabilityConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_exposure(
        self,
        assets: List[InfrastructureAsset],
        hazard_scenario: HazardScenario
    ) -> Dict[str, Any]:
        """
        Calculate exposure of infrastructure assets to hazard scenario.
        
        Args:
            assets: List of infrastructure assets
            hazard_scenario: Hazard scenario
            
        Returns:
            Exposure analysis results
        """
        exposure_results = {}
        
        for asset in assets:
            # Calculate distance to hazard source
            distance = self._calculate_distance(
                asset.latitude, asset.longitude,
                hazard_scenario.source_latitude, hazard_scenario.source_longitude
            )
            
            # Check if asset is within affected area
            within_affected_area = self._point_in_polygon(
                asset.latitude, asset.longitude,
                hazard_scenario.affected_area_polygon
            )
            
            # Determine exposure level
            exposure_level = self._determine_exposure_level(
                distance, within_affected_area, hazard_scenario
            )
            
            # Calculate hazard intensity at asset location
            hazard_intensity = self._calculate_hazard_intensity(
                asset, hazard_scenario, distance
            )
            
            exposure_results[asset.asset_id] = {
                'distance_to_source_km': float(distance),
                'within_affected_area': within_affected_area,
                'exposure_level': exposure_level,
                'hazard_intensity': hazard_intensity,
                'asset_info': {
                    'name': asset.name,
                    'type': asset.infrastructure_type.value,
                    'criticality': asset.criticality
                }
            }
        
        return exposure_results
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in km."""
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2) * np.sin(dlat/2) + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2) * np.sin(dlon/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _point_in_polygon(
        self,
        lat: float, lon: float,
        polygon: List[Tuple[float, float]]
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        if not polygon:
            return False
        
        x, y = lon, lat
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _determine_exposure_level(
        self,
        distance: float,
        within_affected_area: bool,
        hazard_scenario: HazardScenario
    ) -> str:
        """Determine exposure level based on distance and affected area."""
        if within_affected_area or distance <= self.config.direct_impact_distance:
            return "direct"
        elif distance <= self.config.indirect_impact_distance:
            return "indirect"
        elif distance <= self.config.regional_impact_distance:
            return "regional"
        else:
            return "none"
    
    def _calculate_hazard_intensity(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        distance: float
    ) -> Dict[str, float]:
        """Calculate hazard intensity at asset location."""
        # Distance attenuation factor
        attenuation = max(0.1, 1.0 / (1.0 + distance))
        
        intensity = {
            'magnitude': hazard_scenario.magnitude * attenuation,
            'attenuation_factor': attenuation
        }
        
        # Add hazard-specific parameters
        if hazard_scenario.flow_velocity is not None:
            intensity['flow_velocity'] = hazard_scenario.flow_velocity * attenuation
        
        if hazard_scenario.flow_depth is not None:
            intensity['flow_depth'] = hazard_scenario.flow_depth * attenuation
        
        if hazard_scenario.impact_pressure is not None:
            intensity['impact_pressure'] = hazard_scenario.impact_pressure * attenuation
        
        if hazard_scenario.debris_concentration is not None:
            intensity['debris_concentration'] = hazard_scenario.debris_concentration * attenuation
        
        return intensity


class VulnerabilityAnalysis:
    """Analyze infrastructure vulnerability to hazards."""
    
    def __init__(self, config: VulnerabilityConfig = None):
        """
        Initialize vulnerability analysis.
        
        Args:
            config: Vulnerability assessment configuration
        """
        self.config = config or VulnerabilityConfig()
        self.logger = logging.getLogger(__name__)
    
    def assess_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assess vulnerability of infrastructure asset to hazard.
        
        Args:
            asset: Infrastructure asset
            hazard_scenario: Hazard scenario
            hazard_intensity: Hazard intensity at asset location
            
        Returns:
            Vulnerability assessment
        """
        # Structural vulnerability
        structural_vulnerability = self._assess_structural_vulnerability(
            asset, hazard_scenario, hazard_intensity
        )
        
        # Operational vulnerability
        operational_vulnerability = self._assess_operational_vulnerability(
            asset, hazard_scenario, hazard_intensity
        )
        
        # Economic vulnerability
        economic_vulnerability = self._assess_economic_vulnerability(
            asset, hazard_scenario, hazard_intensity
        )
        
        # Social vulnerability
        social_vulnerability = self._assess_social_vulnerability(
            asset, hazard_scenario, hazard_intensity
        )
        
        # Overall vulnerability
        overall_vulnerability = self._calculate_overall_vulnerability(
            asset, structural_vulnerability, operational_vulnerability, 
            economic_vulnerability, social_vulnerability
        )
        
        return {
            'asset_id': asset.asset_id,
            'overall_vulnerability': overall_vulnerability,
            'structural_vulnerability': structural_vulnerability,
            'operational_vulnerability': operational_vulnerability,
            'economic_vulnerability': economic_vulnerability,
            'social_vulnerability': social_vulnerability,
            'hazard_scenario_id': hazard_scenario.scenario_id
        }
    
    def _assess_structural_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess structural vulnerability."""
        # Base vulnerability from asset characteristics
        age_factor = min((datetime.now().year - asset.construction_year) / asset.design_life, 1.0)
        condition_factor = 1.0 - asset.structural_condition
        
        base_vulnerability = (age_factor + condition_factor) / 2.0
        
        # Hazard-specific vulnerability
        hazard_vulnerability = self._calculate_hazard_specific_vulnerability(
            asset, hazard_scenario, hazard_intensity
        )
        
        # Material-specific factors
        material_factors = {
            'concrete': 0.8,
            'steel': 0.6,
            'wood': 1.2,
            'masonry': 1.0,
            'earth': 1.5
        }
        material_factor = material_factors.get(asset.material_type.lower(), 1.0)
        
        # Foundation-specific factors
        foundation_factors = {
            'rock': 0.7,
            'deep': 0.8,
            'shallow': 1.0,
            'pile': 0.9,
            'slab': 1.1
        }
        foundation_factor = foundation_factors.get(asset.foundation_type.lower(), 1.0)
        
        # Combined structural vulnerability
        structural_vuln = min(
            base_vulnerability * hazard_vulnerability * material_factor * foundation_factor,
            1.0
        )
        
        return {
            'vulnerability_score': float(structural_vuln),
            'age_factor': float(age_factor),
            'condition_factor': float(condition_factor),
            'hazard_factor': float(hazard_vulnerability),
            'material_factor': float(material_factor),
            'foundation_factor': float(foundation_factor)
        }
    
    def _assess_operational_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess operational vulnerability."""
        # Redundancy factor (lower redundancy = higher vulnerability)
        redundancy_factor = 1.0 - asset.redundancy
        
        # Criticality factor
        criticality_factor = asset.criticality
        
        # Hazard duration impact
        duration_factor = min(hazard_scenario.duration / 24.0, 1.0)  # Normalize to 24 hours
        
        # Warning time factor (more warning = lower vulnerability)
        warning_factor = max(0.2, 1.0 - hazard_scenario.warning_time / 24.0)
        
        # Exposure-specific factors
        exposure_factor = max(
            asset.exposure_to_water,
            asset.exposure_to_debris,
            asset.exposure_to_ice
        )
        
        # Combined operational vulnerability
        operational_vuln = min(
            (redundancy_factor + criticality_factor + duration_factor + 
             warning_factor + exposure_factor) / 5.0,
            1.0
        )
        
        return {
            'vulnerability_score': float(operational_vuln),
            'redundancy_factor': float(redundancy_factor),
            'criticality_factor': float(criticality_factor),
            'duration_factor': float(duration_factor),
            'warning_factor': float(warning_factor),
            'exposure_factor': float(exposure_factor)
        }
    
    def _assess_economic_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess economic vulnerability."""
        # Replacement cost factor (higher cost = higher vulnerability)
        if asset.replacement_cost > 0:
            cost_factor = min(asset.replacement_cost / 10000000, 1.0)  # Normalize to $10M
        else:
            cost_factor = 0.5  # Default
        
        # Revenue loss factor
        if asset.annual_revenue is not None and asset.annual_revenue > 0:
            revenue_factor = min(asset.annual_revenue / 1000000, 1.0)  # Normalize to $1M
        else:
            revenue_factor = 0.3  # Default
        
        # Maintenance cost factor
        if asset.maintenance_cost is not None and asset.maintenance_cost > 0:
            maintenance_factor = min(asset.maintenance_cost / 100000, 1.0)  # Normalize to $100K
        else:
            maintenance_factor = 0.2  # Default
        
        # Recovery time factor (based on hazard type and asset type)
        recovery_factors = {
            HazardType.GLOF: {'road': 0.3, 'bridge': 0.8, 'building': 0.6},
            HazardType.ICE_AVALANCHE: {'road': 0.4, 'bridge': 0.7, 'building': 0.9},
            HazardType.GLACIER_SURGE: {'road': 0.2, 'bridge': 0.5, 'building': 0.3}
        }
        
        recovery_factor = recovery_factors.get(
            hazard_scenario.hazard_type, {}
        ).get(asset.infrastructure_type.value, 0.5)
        
        # Combined economic vulnerability
        economic_vuln = min(
            (cost_factor + revenue_factor + maintenance_factor + recovery_factor) / 4.0,
            1.0
        )
        
        return {
            'vulnerability_score': float(economic_vuln),
            'cost_factor': float(cost_factor),
            'revenue_factor': float(revenue_factor),
            'maintenance_factor': float(maintenance_factor),
            'recovery_factor': float(recovery_factor)
        }
    
    def _assess_social_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess social vulnerability."""
        # Population served factor
        if asset.population_served is not None and asset.population_served > 0:
            population_factor = min(asset.population_served / 100000, 1.0)  # Normalize to 100K
        else:
            population_factor = 0.3  # Default
        
        # Essential services factor
        essential_factor = 0.9 if asset.essential_services else 0.3
        
        # Infrastructure type social importance
        social_importance = {
            InfrastructureType.ROAD: 0.8,
            InfrastructureType.BRIDGE: 0.7,
            InfrastructureType.BUILDING: 0.6,
            InfrastructureType.POWER_LINE: 0.9,
            InfrastructureType.SETTLEMENT: 1.0,
            InfrastructureType.AIRPORT: 0.7,
            InfrastructureType.RAILWAY: 0.6
        }
        
        importance_factor = social_importance.get(asset.infrastructure_type, 0.5)
        
        # Warning time factor (less warning = higher social vulnerability)
        warning_factor = max(0.3, 1.0 - hazard_scenario.warning_time / 12.0)
        
        # Combined social vulnerability
        social_vuln = min(
            (population_factor + essential_factor + importance_factor + warning_factor) / 4.0,
            1.0
        )
        
        return {
            'vulnerability_score': float(social_vuln),
            'population_factor': float(population_factor),
            'essential_factor': float(essential_factor),
            'importance_factor': float(importance_factor),
            'warning_factor': float(warning_factor)
        }
    
    def _calculate_hazard_specific_vulnerability(
        self,
        asset: InfrastructureAsset,
        hazard_scenario: HazardScenario,
        hazard_intensity: Dict[str, float]
    ) -> float:
        """Calculate hazard-specific vulnerability using fragility curves."""
        hazard_type = hazard_scenario.hazard_type.value
        asset_type = asset.infrastructure_type.value
        
        # Get fragility curve for asset type
        fragility_curve = self.config.fragility_curves.get(
            asset_type, self.config.fragility_curves.get('road', {})
        )
        
        vulnerability = 0.0
        
        # Check different hazard parameters
        if 'flow_depth' in hazard_intensity and 'depth' in fragility_curve:
            depth = hazard_intensity['flow_depth']
            depth_thresholds = fragility_curve['depth']
            depth_vuln = self._interpolate_fragility(depth, depth_thresholds)
            vulnerability = max(vulnerability, depth_vuln)
        
        if 'flow_velocity' in hazard_intensity and 'velocity' in fragility_curve:
            velocity = hazard_intensity['flow_velocity']
            velocity_thresholds = fragility_curve['velocity']
            velocity_vuln = self._interpolate_fragility(velocity, velocity_thresholds)
            vulnerability = max(vulnerability, velocity_vuln)
        
        if 'impact_pressure' in hazard_intensity and 'pressure' in fragility_curve:
            pressure = hazard_intensity['impact_pressure']
            pressure_thresholds = fragility_curve['pressure']
            pressure_vuln = self._interpolate_fragility(pressure, pressure_thresholds)
            vulnerability = max(vulnerability, pressure_vuln)
        
        # Use magnitude as fallback
        if vulnerability == 0.0:
            magnitude = hazard_intensity.get('magnitude', 0)
            vulnerability = min(magnitude / 10.0, 1.0)  # Normalize to scale 0-10
        
        return vulnerability
    
    def _interpolate_fragility(
        self,
        value: float,
        thresholds: List[float]
    ) -> float:
        """Interpolate vulnerability from fragility curve thresholds."""
        if value <= thresholds[0]:
            return 0.0
        elif value >= thresholds[-1]:
            return 1.0
        else:
            # Linear interpolation between thresholds
            for i in range(len(thresholds) - 1):
                if thresholds[i] <= value <= thresholds[i + 1]:
                    # Interpolate between damage states
                    ratio = (value - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                    vuln_low = i / (len(thresholds) - 1)
                    vuln_high = (i + 1) / (len(thresholds) - 1)
                    return vuln_low + ratio * (vuln_high - vuln_low)
        
        return 0.5  # Default
    
    def _calculate_overall_vulnerability(
        self,
        asset: InfrastructureAsset,
        structural: Dict[str, float],
        operational: Dict[str, float],
        economic: Dict[str, float],
        social: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate overall vulnerability score."""
        # Get weights for asset type
        asset_type = asset.infrastructure_type.value
        weights = self.config.infrastructure_weights.get(
            asset_type, self.config.infrastructure_weights['default']
        )
        
        # Calculate weighted vulnerability
        overall_score = (
            weights['structural'] * structural['vulnerability_score'] +
            weights['operational'] * operational['vulnerability_score'] +
            weights['economic'] * economic['vulnerability_score']
        )
        
        # Add social component (not in weights as it's cross-cutting)
        overall_score = (overall_score + social['vulnerability_score']) / 2.0
        
        # Determine vulnerability level
        vulnerability_level = self._determine_vulnerability_level(overall_score)
        
        return {
            'vulnerability_score': float(overall_score),
            'vulnerability_level': vulnerability_level.value,
            'component_scores': {
                'structural': structural['vulnerability_score'],
                'operational': operational['vulnerability_score'],
                'economic': economic['vulnerability_score'],
                'social': social['vulnerability_score']
            },
            'weights_used': weights
        }
    
    def _determine_vulnerability_level(self, score: float) -> VulnerabilityLevel:
        """Determine vulnerability level from score."""
        if score >= 0.9:
            return VulnerabilityLevel.EXTREME
        elif score >= 0.7:
            return VulnerabilityLevel.VERY_HIGH
        elif score >= 0.5:
            return VulnerabilityLevel.HIGH
        elif score >= 0.3:
            return VulnerabilityLevel.MODERATE
        elif score >= 0.1:
            return VulnerabilityLevel.LOW
        else:
            return VulnerabilityLevel.VERY_LOW


class RiskAssessment:
    """Assess risk by combining hazard, exposure, and vulnerability."""
    
    def __init__(self, config: VulnerabilityConfig = None):
        """
        Initialize risk assessment.
        
        Args:
            config: Vulnerability assessment configuration
        """
        self.config = config or VulnerabilityConfig()
        self.spatial_analyzer = SpatialAnalysis(self.config)
        self.vulnerability_analyzer = VulnerabilityAnalysis(self.config)
        self.logger = logging.getLogger(__name__)
    
    def assess_infrastructure_risk(
        self,
        assets: List[InfrastructureAsset],
        hazard_scenarios: List[HazardScenario]
    ) -> Dict[str, Any]:
        """
        Assess infrastructure risk for multiple assets and hazard scenarios.
        
        Args:
            assets: List of infrastructure assets
            hazard_scenarios: List of hazard scenarios
            
        Returns:
            Comprehensive risk assessment
        """
        risk_results = {
            'asset_risks': {},
            'scenario_risks': {},
            'summary_statistics': {},
            'high_risk_assets': [],
            'critical_scenarios': []
        }
        
        # Assess risk for each asset-scenario combination
        for scenario in hazard_scenarios:
            scenario_risks = []
            
            # Calculate exposure for all assets
            exposure_results = self.spatial_analyzer.calculate_exposure(assets, scenario)
            
            for asset in assets:
                if asset.asset_id in exposure_results:
                    exposure = exposure_results[asset.asset_id]
                    
                    # Skip if no exposure
                    if exposure['exposure_level'] == 'none':
                        continue
                    
                    # Assess vulnerability
                    vulnerability = self.vulnerability_analyzer.assess_vulnerability(
                        asset, scenario, exposure['hazard_intensity']
                    )
                    
                    # Calculate risk
                    risk_assessment = self._calculate_risk(
                        asset, scenario, exposure, vulnerability
                    )
                    
                    # Store results
                    asset_scenario_key = f"{asset.asset_id}_{scenario.scenario_id}"
                    risk_results['asset_risks'][asset_scenario_key] = {
                        'asset_id': asset.asset_id,
                        'scenario_id': scenario.scenario_id,
                        'exposure': exposure,
                        'vulnerability': vulnerability,
                        'risk': risk_assessment
                    }
                    
                    scenario_risks.append(risk_assessment['risk_score'])
                    
                    # Check if high risk
                    if risk_assessment['risk_level'] in ['high', 'very_high', 'extreme']:
                        risk_results['high_risk_assets'].append({
                            'asset_id': asset.asset_id,
                            'asset_name': asset.name,
                            'scenario_id': scenario.scenario_id,
                            'risk_score': risk_assessment['risk_score'],
                            'risk_level': risk_assessment['risk_level']
                        })
            
            # Scenario-level statistics
            if scenario_risks:
                risk_results['scenario_risks'][scenario.scenario_id] = {
                    'mean_risk': float(np.mean(scenario_risks)),
                    'max_risk': float(np.max(scenario_risks)),
                    'affected_assets': len(scenario_risks),
                    'high_risk_assets': sum(1 for r in scenario_risks if r > 0.7)
                }
                
                # Check if critical scenario
                if np.mean(scenario_risks) > 0.5 or np.max(scenario_risks) > 0.8:
                    risk_results['critical_scenarios'].append({
                        'scenario_id': scenario.scenario_id,
                        'hazard_type': scenario.hazard_type.value,
                        'mean_risk': float(np.mean(scenario_risks)),
                        'max_risk': float(np.max(scenario_risks)),
                        'affected_assets': len(scenario_risks)
                    })
        
        # Overall summary statistics
        all_risks = []
        for asset_risk in risk_results['asset_risks'].values():
            all_risks.append(asset_risk['risk']['risk_score'])
        
        if all_risks:
            risk_results['summary_statistics'] = {
                'total_assessments': len(all_risks),
                'mean_risk': float(np.mean(all_risks)),
                'median_risk': float(np.median(all_risks)),
                'std_risk': float(np.std(all_risks)),
                'max_risk': float(np.max(all_risks)),
                'min_risk': float(np.min(all_risks)),
                'high_risk_count': sum(1 for r in all_risks if r > 0.7),
                'moderate_risk_count': sum(1 for r in all_risks if 0.3 <= r <= 0.7),
                'low_risk_count': sum(1 for r in all_risks if r < 0.3)
            }
        
        return risk_results
    
    def _calculate_risk(
        self,
        asset: InfrastructureAsset,
        scenario: HazardScenario,
        exposure: Dict[str, Any],
        vulnerability: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk as function of hazard, exposure, and vulnerability."""
        # Basic risk calculation: Risk = Hazard × Exposure × Vulnerability
        hazard_probability = scenario.probability
        exposure_factor = self._get_exposure_factor(exposure['exposure_level'])
        vulnerability_score = vulnerability['overall_vulnerability']['vulnerability_score']
        
        # Calculate risk score
        risk_score = hazard_probability * exposure_factor * vulnerability_score
        
        # Calculate expected annual loss
        expected_loss = self._calculate_expected_loss(
            asset, scenario, risk_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            asset, scenario, risk_score, expected_loss
        )
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level.value,
            'expected_annual_loss_usd': expected_loss,
            'risk_metrics': risk_metrics,
            'components': {
                'hazard_probability': float(hazard_probability),
                'exposure_factor': float(exposure_factor),
                'vulnerability_score': float(vulnerability_score)
            }
        }
    
    def _get_exposure_factor(self, exposure_level: str) -> float:
        """Get exposure factor based on exposure level."""
        exposure_factors = {
            'direct': 1.0,
            'indirect': 0.6,
            'regional': 0.3,
            'none': 0.0
        }
        return exposure_factors.get(exposure_level, 0.0)
    
    def _calculate_expected_loss(
        self,
        asset: InfrastructureAsset,
        scenario: HazardScenario,
        risk_score: float
    ) -> float:
        """Calculate expected annual loss in USD."""
        # Base loss is replacement cost weighted by risk
        base_loss = asset.replacement_cost * risk_score
        
        # Add operational losses if applicable
        operational_loss = 0.0
        if asset.annual_revenue is not None:
            # Assume disruption duration based on hazard type and asset type
            disruption_days = self._estimate_disruption_duration(
                asset.infrastructure_type, scenario.hazard_type
            )
            daily_revenue = asset.annual_revenue / 365
            operational_loss = daily_revenue * disruption_days * risk_score
        
        # Total expected annual loss
        total_loss = base_loss + operational_loss
        
        return float(total_loss)
    
    def _estimate_disruption_duration(
        self,
        infrastructure_type: InfrastructureType,
        hazard_type: HazardType
    ) -> float:
        """Estimate disruption duration in days."""
        # Disruption duration matrix (days)
        disruption_matrix = {
            InfrastructureType.ROAD: {
                HazardType.GLOF: 30,
                HazardType.ICE_AVALANCHE: 14,
                HazardType.GLACIER_SURGE: 7,
                HazardType.DEBRIS_FLOW: 21,
                HazardType.FLOODING: 10
            },
            InfrastructureType.BRIDGE: {
                HazardType.GLOF: 180,
                HazardType.ICE_AVALANCHE: 90,
                HazardType.GLACIER_SURGE: 30,
                HazardType.DEBRIS_FLOW: 120,
                HazardType.FLOODING: 60
            },
            InfrastructureType.BUILDING: {
                HazardType.GLOF: 90,
                HazardType.ICE_AVALANCHE: 120,
                HazardType.GLACIER_SURGE: 14,
                HazardType.DEBRIS_FLOW: 60,
                HazardType.FLOODING: 30
            }
        }
        
        return disruption_matrix.get(infrastructure_type, {}).get(hazard_type, 30)
    
    def _calculate_risk_metrics(
        self,
        asset: InfrastructureAsset,
        scenario: HazardScenario,
        risk_score: float,
        expected_loss: float
    ) -> Dict[str, float]:
        """Calculate additional risk metrics."""
        # Risk per unit cost
        risk_per_cost = risk_score / max(asset.replacement_cost, 1) * 1000000  # per million USD
        
        # Risk per person served
        if asset.population_served and asset.population_served > 0:
            risk_per_person = risk_score / asset.population_served * 1000  # per thousand people
        else:
            risk_per_person = 0.0
        
        # Annual risk (considering return period)
        if scenario.return_period and scenario.return_period > 0:
            annual_risk = risk_score / scenario.return_period
        else:
            annual_risk = risk_score * scenario.probability
        
        # Risk reduction potential (based on criticality and vulnerability)
        risk_reduction_potential = risk_score * asset.criticality * 0.5  # Assume 50% max reduction
        
        return {
            'risk_per_million_usd': float(risk_per_cost),
            'risk_per_thousand_people': float(risk_per_person),
            'annual_risk': float(annual_risk),
            'risk_reduction_potential': float(risk_reduction_potential),
            'cost_benefit_ratio': float(expected_loss / max(asset.replacement_cost * 0.1, 1))
        }
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return RiskLevel.EXTREME
        elif risk_score >= 0.6:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.4:
            return RiskLevel.HIGH
        elif risk_score >= 0.2:
            return RiskLevel.MODERATE
        elif risk_score >= 0.05:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW


def create_vulnerability_assessor(
    direct_impact_distance: float = 1.0,
    **kwargs
) -> RiskAssessment:
    """
    Create an infrastructure vulnerability assessor with specified configuration.
    
    Args:
        direct_impact_distance: Direct impact distance threshold (km)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured vulnerability assessor
    """
    config = VulnerabilityConfig(
        direct_impact_distance=direct_impact_distance,
        **kwargs
    )
    return RiskAssessment(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create example infrastructure assets
    assets = [
        InfrastructureAsset(
            asset_id="ROAD_001",
            name="Mountain Highway",
            infrastructure_type=InfrastructureType.ROAD,
            latitude=35.5,
            longitude=76.2,
            elevation=3500,
            construction_year=1995,
            design_life=50,
            replacement_cost=5000000,
            material_type="concrete",
            structural_condition=0.7,
            foundation_type="shallow",
            capacity=5000,  # vehicles/day
            criticality=0.8,
            redundancy=0.3,
            exposure_to_water=0.6,
            exposure_to_debris=0.7,
            exposure_to_ice=0.4,
            annual_revenue=500000,
            population_served=25000
        ),
        InfrastructureAsset(
            asset_id="BRIDGE_001",
            name="Glacier Valley Bridge",
            infrastructure_type=InfrastructureType.BRIDGE,
            latitude=35.52,
            longitude=76.22,
            elevation=3450,
            construction_year=1985,
            design_life=75,
            replacement_cost=15000000,
            material_type="steel",
            structural_condition=0.6,
            foundation_type="deep",
            capacity=2000,  # vehicles/day
            criticality=0.9,
            redundancy=0.1,
            exposure_to_water=0.9,
            exposure_to_debris=0.8,
            exposure_to_ice=0.6,
            population_served=25000,
            essential_services=True
        ),
        InfrastructureAsset(
            asset_id="BUILDING_001",
            name="Emergency Response Center",
            infrastructure_type=InfrastructureType.BUILDING,
            latitude=35.48,
            longitude=76.18,
            elevation=3600,
            construction_year=2010,
            design_life=50,
            replacement_cost=8000000,
            material_type="concrete",
            structural_condition=0.9,
            foundation_type="rock",
            capacity=200,  # people
            criticality=1.0,
            redundancy=0.2,
            exposure_to_water=0.3,
            exposure_to_debris=0.4,
            exposure_to_ice=0.2,
            population_served=50000,
            essential_services=True
        )
    ]
    
    # Create example hazard scenarios
    hazard_scenarios = [
        HazardScenario(
            scenario_id="GLOF_001",
            hazard_type=HazardType.GLOF,
            source_latitude=35.55,
            source_longitude=76.25,
            affected_area_polygon=[
                (35.45, 76.15), (35.45, 76.25), (35.55, 76.25), 
                (35.55, 76.15), (35.45, 76.15)
            ],
            magnitude=8.0,
            probability=0.02,  # 2% annual probability
            duration=6.0,  # hours
            flow_velocity=5.0,  # m/s
            flow_depth=3.0,  # m
            debris_concentration=200,  # kg/m³
            warning_time=2.0,  # hours
            return_period=50  # years
        ),
        HazardScenario(
            scenario_id="ICE_AVALANCHE_001",
            hazard_type=HazardType.ICE_AVALANCHE,
            source_latitude=35.58,
            source_longitude=76.28,
            affected_area_polygon=[
                (35.48, 76.18), (35.48, 76.28), (35.58, 76.28), 
                (35.58, 76.18), (35.48, 76.18)
            ],
            magnitude=6.0,
            probability=0.05,  # 5% annual probability
            duration=0.5,  # hours
            impact_pressure=25000,  # Pa
            debris_concentration=500,  # kg/m³
            warning_time=0.5,  # hours
            return_period=20  # years
        )
    ]
    
    # Create vulnerability assessor
    assessor = create_vulnerability_assessor(
        direct_impact_distance=2.0,
        indirect_impact_distance=10.0
    )
    
    # Perform risk assessment
    print("Performing infrastructure vulnerability assessment...")
    risk_results = assessor.assess_infrastructure_risk(assets, hazard_scenarios)
    
    print("\nInfrastructure Risk Assessment Results:")
    print("=" * 50)
    
    # Summary statistics
    stats = risk_results['summary_statistics']
    if stats:
        print(f"Total Assessments: {stats['total_assessments']}")
        print(f"Mean Risk Score: {stats['mean_risk']:.3f}")
        print(f"Maximum Risk Score: {stats['max_risk']:.3f}")
        print(f"High Risk Assets: {stats['high_risk_count']}")
        print(f"Moderate Risk Assets: {stats['moderate_risk_count']}")
        print(f"Low Risk Assets: {stats['low_risk_count']}")
    
    # High risk assets
    print(f"\nHigh Risk Assets ({len(risk_results['high_risk_assets'])}):")
    for asset_risk in risk_results['high_risk_assets']:
        print(f"  {asset_risk['asset_name']} ({asset_risk['asset_id']})")
        print(f"    Scenario: {asset_risk['scenario_id']}")
        print(f"    Risk Score: {asset_risk['risk_score']:.3f}")
        print(f"    Risk Level: {asset_risk['risk_level'].upper()}")
    
    # Critical scenarios
    print(f"\nCritical Scenarios ({len(risk_results['critical_scenarios'])}):")
    for scenario in risk_results['critical_scenarios']:
        print(f"  {scenario['scenario_id']} ({scenario['hazard_type'].upper()})")
        print(f"    Mean Risk: {scenario['mean_risk']:.3f}")
        print(f"    Max Risk: {scenario['max_risk']:.3f}")
        print(f"    Affected Assets: {scenario['affected_assets']}")
    
    # Detailed results for each asset-scenario combination
    print(f"\nDetailed Risk Assessment:")
    for key, result in risk_results['asset_risks'].items():
        asset_id = result['asset_id']
        scenario_id = result['scenario_id']
        risk = result['risk']
        vulnerability = result['vulnerability']
        exposure = result['exposure']
        
        asset_name = next(a.name for a in assets if a.asset_id == asset_id)
        
        print(f"\n{asset_name} ({asset_id}) - {scenario_id}:")
        print(f"  Risk Score: {risk['risk_score']:.3f} ({risk['risk_level'].upper()})")
        print(f"  Expected Annual Loss: ${risk['expected_annual_loss_usd']:,.0f}")
        print(f"  Distance to Source: {exposure['distance_to_source_km']:.1f} km")
        print(f"  Exposure Level: {exposure['exposure_level'].title()}")
        print(f"  Vulnerability Score: {vulnerability['overall_vulnerability']['vulnerability_score']:.3f}")
        print(f"  Vulnerability Level: {vulnerability['overall_vulnerability']['vulnerability_level'].upper()}")
        
        # Component scores
        components = vulnerability['overall_vulnerability']['component_scores']
        print(f"  Vulnerability Components:")
        print(f"    Structural: {components['structural']:.3f}")
        print(f"    Operational: {components['operational']:.3f}")
        print(f"    Economic: {components['economic']:.3f}")
        print(f"    Social: {components['social']:.3f}")
    
    print(f"\nInfrastructure vulnerability assessment completed for {len(assets)} assets and {len(hazard_scenarios)} scenarios")