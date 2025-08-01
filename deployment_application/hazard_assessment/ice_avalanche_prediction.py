"""Ice Avalanche Risk Assessment Module.

This module provides functionality for predicting and assessing
ice avalanche risks in glaciated mountain regions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import interpolate, ndimage
from scipy.spatial.distance import cdist


class AvalancheType(Enum):
    """Types of ice avalanches."""
    SERAC_FALL = "serac_fall"
    ICE_CLIFF_COLLAPSE = "ice_cliff_collapse"
    HANGING_GLACIER_BREAK = "hanging_glacier_break"
    CREVASSE_INDUCED = "crevasse_induced"
    THERMAL_INDUCED = "thermal_induced"
    SEISMIC_TRIGGERED = "seismic_triggered"


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class TerrainCharacteristics:
    """Terrain characteristics for avalanche assessment."""
    location_id: str
    elevation: float  # m above sea level
    slope_angle: float  # degrees
    aspect: float  # degrees (0-360, 0=North)
    slope_length: float  # m
    slope_width: float  # m
    
    # Ice characteristics
    ice_thickness: float  # m
    ice_temperature: float  # °C
    crevasse_density: float  # crevasses per km²
    serac_presence: bool
    
    # Geographic information
    latitude: float
    longitude: float
    
    # Exposure information
    solar_exposure: float  # hours per day
    wind_exposure: float  # 0-1 scale


@dataclass
class AvalancheConfig:
    """Configuration for ice avalanche assessment."""
    # Critical slope parameters
    critical_slope_angle: float = 30.0  # degrees
    maximum_stable_angle: float = 60.0  # degrees
    
    # Ice stability parameters
    critical_ice_thickness: float = 10.0  # m
    critical_temperature: float = -2.0  # °C
    critical_crevasse_density: float = 5.0  # per km²
    
    # Environmental thresholds
    temperature_change_threshold: float = 5.0  # °C/day
    wind_speed_threshold: float = 15.0  # m/s
    precipitation_threshold: float = 20.0  # mm/day
    
    # Seismic parameters
    seismic_magnitude_threshold: float = 4.0
    seismic_distance_threshold: float = 50.0  # km
    
    # Modeling parameters
    runout_angle: float = 20.0  # degrees
    friction_coefficient: float = 0.3
    air_resistance_factor: float = 0.001
    
    # Risk assessment weights
    terrain_weight: float = 0.3
    ice_condition_weight: float = 0.25
    weather_weight: float = 0.2
    seismic_weight: float = 0.15
    historical_weight: float = 0.1


class IceStabilityAnalysis:
    """Analyze ice stability conditions."""
    
    def __init__(self, config: AvalancheConfig = None):
        """
        Initialize ice stability analysis.
        
        Args:
            config: Avalanche assessment configuration
        """
        self.config = config or AvalancheConfig()
        self.logger = logging.getLogger(__name__)
    
    def assess_ice_stability(
        self,
        terrain: TerrainCharacteristics,
        ice_conditions: Dict[str, Any],
        weather_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Assess ice stability based on multiple factors.
        
        Args:
            terrain: Terrain characteristics
            ice_conditions: Current ice conditions
            weather_data: Weather conditions
            
        Returns:
            Ice stability assessment
        """
        # Temperature stability
        temp_stability = self._assess_temperature_stability(
            terrain.ice_temperature,
            weather_data.get('temperature', terrain.ice_temperature),
            weather_data.get('temperature_trend', 0)
        )
        
        # Structural stability
        structural_stability = self._assess_structural_stability(
            terrain.ice_thickness,
            terrain.crevasse_density,
            terrain.slope_angle
        )
        
        # Stress analysis
        stress_analysis = self._assess_stress_conditions(
            terrain,
            ice_conditions,
            weather_data
        )
        
        # Thermal effects
        thermal_effects = self._assess_thermal_effects(
            terrain,
            weather_data
        )
        
        # Overall stability score
        stability_score = np.mean([
            temp_stability['stability_score'],
            structural_stability['stability_score'],
            stress_analysis['stability_score'],
            thermal_effects['stability_score']
        ])
        
        return {
            'overall_stability_score': float(stability_score),
            'temperature_stability': temp_stability,
            'structural_stability': structural_stability,
            'stress_analysis': stress_analysis,
            'thermal_effects': thermal_effects
        }
    
    def _assess_temperature_stability(
        self,
        current_temp: float,
        ambient_temp: float,
        temp_trend: float
    ) -> Dict[str, float]:
        """Assess temperature-related stability."""
        # Temperature difference from critical threshold
        temp_margin = current_temp - self.config.critical_temperature
        temp_risk = max(0, 1 - temp_margin / 5.0)  # Risk increases as temp approaches critical
        
        # Ambient temperature effect
        ambient_risk = max(0, (ambient_temp - self.config.critical_temperature) / 10.0)
        
        # Temperature trend risk
        trend_risk = max(0, temp_trend / self.config.temperature_change_threshold)
        
        # Combined temperature stability
        temp_stability = 1 - np.mean([temp_risk, ambient_risk, trend_risk])
        
        return {
            'stability_score': float(max(0, temp_stability)),
            'temperature_risk': float(temp_risk),
            'ambient_risk': float(ambient_risk),
            'trend_risk': float(trend_risk),
            'current_temperature': float(current_temp),
            'critical_temperature': float(self.config.critical_temperature)
        }
    
    def _assess_structural_stability(
        self,
        ice_thickness: float,
        crevasse_density: float,
        slope_angle: float
    ) -> Dict[str, float]:
        """Assess structural stability of ice."""
        # Thickness stability (thicker ice generally more stable)
        thickness_stability = min(ice_thickness / self.config.critical_ice_thickness, 1.0)
        
        # Crevasse density risk
        crevasse_risk = min(crevasse_density / self.config.critical_crevasse_density, 1.0)
        
        # Slope angle stability
        if slope_angle < self.config.critical_slope_angle:
            slope_stability = 1.0
        elif slope_angle > self.config.maximum_stable_angle:
            slope_stability = 0.0
        else:
            # Linear decrease between critical and maximum angles
            angle_range = self.config.maximum_stable_angle - self.config.critical_slope_angle
            slope_stability = 1.0 - (slope_angle - self.config.critical_slope_angle) / angle_range
        
        # Combined structural stability
        structural_score = thickness_stability * (1 - crevasse_risk) * slope_stability
        
        return {
            'stability_score': float(structural_score),
            'thickness_stability': float(thickness_stability),
            'crevasse_risk': float(crevasse_risk),
            'slope_stability': float(slope_stability),
            'ice_thickness': float(ice_thickness),
            'slope_angle': float(slope_angle)
        }
    
    def _assess_stress_conditions(
        self,
        terrain: TerrainCharacteristics,
        ice_conditions: Dict[str, Any],
        weather_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess stress conditions affecting ice stability."""
        # Gravitational stress (function of slope and ice thickness)
        gravity_stress = np.sin(np.radians(terrain.slope_angle)) * terrain.ice_thickness
        gravity_risk = min(gravity_stress / 100.0, 1.0)  # Normalize
        
        # Wind loading stress
        wind_speed = weather_data.get('wind_speed', 0)
        wind_stress = wind_speed * terrain.wind_exposure
        wind_risk = min(wind_stress / 50.0, 1.0)  # Normalize
        
        # Thermal stress from temperature gradients
        temp_gradient = abs(weather_data.get('temperature', 0) - terrain.ice_temperature)
        thermal_stress_risk = min(temp_gradient / 10.0, 1.0)
        
        # Seismic stress
        seismic_magnitude = weather_data.get('seismic_magnitude', 0)
        seismic_distance = weather_data.get('seismic_distance', 1000)
        
        if seismic_distance > 0:
            seismic_intensity = seismic_magnitude / (1 + seismic_distance / 10.0)
            seismic_risk = min(seismic_intensity / 5.0, 1.0)
        else:
            seismic_risk = 0.0
        
        # Combined stress score
        total_stress_risk = np.mean([gravity_risk, wind_risk, thermal_stress_risk, seismic_risk])
        stress_stability = 1 - total_stress_risk
        
        return {
            'stability_score': float(max(0, stress_stability)),
            'gravity_risk': float(gravity_risk),
            'wind_risk': float(wind_risk),
            'thermal_stress_risk': float(thermal_stress_risk),
            'seismic_risk': float(seismic_risk),
            'total_stress_risk': float(total_stress_risk)
        }
    
    def _assess_thermal_effects(
        self,
        terrain: TerrainCharacteristics,
        weather_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess thermal effects on ice stability."""
        # Solar radiation effect
        solar_hours = terrain.solar_exposure
        solar_intensity = weather_data.get('solar_radiation', 0.5)  # 0-1 scale
        solar_effect = solar_hours * solar_intensity / 12.0  # Normalize to 12 hours
        
        # Aspect-related heating (south-facing slopes get more sun)
        aspect_factor = 0.5 * (1 + np.cos(np.radians(terrain.aspect - 180)))  # Peak at south (180°)
        
        # Air temperature effect
        air_temp = weather_data.get('temperature', terrain.ice_temperature)
        temp_effect = max(0, air_temp / 10.0)  # Positive temperatures increase risk
        
        # Precipitation effect (rain can accelerate melting)
        precipitation = weather_data.get('precipitation', 0)
        precip_temp = weather_data.get('precipitation_temperature', 0)
        rain_effect = 0
        if precip_temp > 0:  # Rain, not snow
            rain_effect = min(precipitation / 20.0, 1.0)
        
        # Combined thermal risk
        thermal_risk = np.mean([
            solar_effect * aspect_factor,
            temp_effect,
            rain_effect
        ])
        
        thermal_stability = 1 - thermal_risk
        
        return {
            'stability_score': float(max(0, thermal_stability)),
            'solar_effect': float(solar_effect),
            'aspect_factor': float(aspect_factor),
            'temperature_effect': float(temp_effect),
            'rain_effect': float(rain_effect),
            'thermal_risk': float(thermal_risk)
        }


class AvalancheRunoutModel:
    """Model avalanche runout and impact zones."""
    
    def __init__(self, config: AvalancheConfig = None):
        """
        Initialize avalanche runout model.
        
        Args:
            config: Avalanche assessment configuration
        """
        self.config = config or AvalancheConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_runout_distance(
        self,
        terrain: TerrainCharacteristics,
        avalanche_volume: float,
        avalanche_type: AvalancheType
    ) -> Dict[str, float]:
        """
        Calculate avalanche runout distance.
        
        Args:
            terrain: Terrain characteristics
            avalanche_volume: Estimated avalanche volume (m³)
            avalanche_type: Type of avalanche
            
        Returns:
            Runout calculations
        """
        # Adjust parameters based on avalanche type
        type_factors = {
            AvalancheType.SERAC_FALL: {'friction': 0.4, 'air_resistance': 0.002},
            AvalancheType.ICE_CLIFF_COLLAPSE: {'friction': 0.35, 'air_resistance': 0.0015},
            AvalancheType.HANGING_GLACIER_BREAK: {'friction': 0.3, 'air_resistance': 0.001},
            AvalancheType.CREVASSE_INDUCED: {'friction': 0.45, 'air_resistance': 0.0025},
            AvalancheType.THERMAL_INDUCED: {'friction': 0.25, 'air_resistance': 0.0008},
            AvalancheType.SEISMIC_TRIGGERED: {'friction': 0.3, 'air_resistance': 0.001}
        }
        
        factors = type_factors.get(avalanche_type, {'friction': 0.3, 'air_resistance': 0.001})
        
        # Calculate initial velocity from potential energy
        # Assuming avalanche starts from rest and accelerates down slope
        height_drop = terrain.slope_length * np.sin(np.radians(terrain.slope_angle))
        
        # Energy-based approach
        g = 9.81  # m/s²
        
        # Initial velocity at bottom of slope (ignoring friction for now)
        v_initial = np.sqrt(2 * g * height_drop)
        
        # Runout on flat terrain using energy balance
        # Kinetic energy = Work done against friction
        # 0.5 * m * v² = μ * m * g * d
        # d = v² / (2 * μ * g)
        
        friction_coeff = factors['friction']
        runout_distance = (v_initial ** 2) / (2 * friction_coeff * g)
        
        # Apply runout angle constraint (empirical)
        max_runout_by_angle = height_drop / np.tan(np.radians(self.config.runout_angle))
        runout_distance = min(runout_distance, max_runout_by_angle)
        
        # Volume effect (larger avalanches travel farther)
        volume_factor = 1 + 0.1 * np.log10(max(avalanche_volume / 1000, 1))  # Normalize to 1000 m³
        runout_distance *= volume_factor
        
        # Calculate impact area (simplified)
        # Assume triangular deposition zone
        impact_width = min(terrain.slope_width, runout_distance * 0.3)
        impact_area = 0.5 * runout_distance * impact_width
        
        # Calculate average velocity during runout
        avg_velocity = v_initial * 0.6  # Rough approximation
        
        # Calculate impact pressure (simplified)
        # P = 0.5 * ρ * v²
        ice_density = 900  # kg/m³
        impact_pressure = 0.5 * ice_density * (avg_velocity ** 2)  # Pa
        
        return {
            'runout_distance_m': float(runout_distance),
            'impact_area_m2': float(impact_area),
            'impact_width_m': float(impact_width),
            'initial_velocity_ms': float(v_initial),
            'average_velocity_ms': float(avg_velocity),
            'impact_pressure_pa': float(impact_pressure),
            'height_drop_m': float(height_drop),
            'volume_factor': float(volume_factor),
            'friction_coefficient': float(friction_coeff),
            'avalanche_type': avalanche_type.value
        }
    
    def estimate_avalanche_volume(
        self,
        terrain: TerrainCharacteristics,
        avalanche_type: AvalancheType,
        trigger_intensity: float = 1.0
    ) -> Dict[str, float]:
        """
        Estimate avalanche volume based on terrain and trigger.
        
        Args:
            terrain: Terrain characteristics
            avalanche_type: Type of avalanche
            trigger_intensity: Intensity of trigger event (0-1)
            
        Returns:
            Volume estimates
        """
        # Base volume calculation
        slope_area = terrain.slope_length * terrain.slope_width
        
        # Type-specific volume factors
        type_volume_factors = {
            AvalancheType.SERAC_FALL: 0.1,  # Small localized failures
            AvalancheType.ICE_CLIFF_COLLAPSE: 0.3,  # Medium-sized events
            AvalancheType.HANGING_GLACIER_BREAK: 0.8,  # Large events
            AvalancheType.CREVASSE_INDUCED: 0.15,  # Small to medium
            AvalancheType.THERMAL_INDUCED: 0.4,  # Medium events
            AvalancheType.SEISMIC_TRIGGERED: 0.6   # Large events
        }
        
        volume_factor = type_volume_factors.get(avalanche_type, 0.3)
        
        # Calculate volume based on affected ice thickness
        affected_thickness = terrain.ice_thickness * volume_factor * trigger_intensity
        estimated_volume = slope_area * affected_thickness
        
        # Apply constraints
        min_volume = 100  # m³
        max_volume = slope_area * terrain.ice_thickness  # Cannot exceed total ice
        
        estimated_volume = max(min_volume, min(estimated_volume, max_volume))
        
        # Calculate mass
        ice_density = 900  # kg/m³
        estimated_mass = estimated_volume * ice_density
        
        return {
            'estimated_volume_m3': float(estimated_volume),
            'estimated_mass_kg': float(estimated_mass),
            'affected_thickness_m': float(affected_thickness),
            'volume_factor': float(volume_factor),
            'trigger_intensity': float(trigger_intensity),
            'slope_area_m2': float(slope_area)
        }


class IceAvalanchePredictor:
    """Main class for ice avalanche prediction."""
    
    def __init__(self, config: AvalancheConfig = None):
        """
        Initialize ice avalanche predictor.
        
        Args:
            config: Avalanche assessment configuration
        """
        self.config = config or AvalancheConfig()
        self.stability_analyzer = IceStabilityAnalysis(self.config)
        self.runout_model = AvalancheRunoutModel(self.config)
        self.logger = logging.getLogger(__name__)
    
    def assess_avalanche_risk(
        self,
        terrain: TerrainCharacteristics,
        ice_conditions: Dict[str, Any],
        weather_data: Dict[str, Any],
        historical_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive avalanche risk assessment.
        
        Args:
            terrain: Terrain characteristics
            ice_conditions: Current ice conditions
            weather_data: Weather conditions
            historical_data: Historical avalanche data
            
        Returns:
            Risk assessment results
        """
        # Ice stability analysis
        stability_assessment = self.stability_analyzer.assess_ice_stability(
            terrain, ice_conditions, weather_data
        )
        
        # Terrain risk factors
        terrain_risk = self._assess_terrain_risk(terrain)
        
        # Weather risk factors
        weather_risk = self._assess_weather_risk(weather_data)
        
        # Historical risk factors
        historical_risk = self._assess_historical_risk(historical_data or {})
        
        # Trigger probability assessment
        trigger_probabilities = self._assess_trigger_probabilities(
            terrain, ice_conditions, weather_data
        )
        
        # Overall risk calculation
        risk_components = {
            'terrain': terrain_risk['risk_score'],
            'ice_stability': 1 - stability_assessment['overall_stability_score'],
            'weather': weather_risk['risk_score'],
            'historical': historical_risk['risk_score']
        }
        
        overall_risk = (
            self.config.terrain_weight * risk_components['terrain'] +
            self.config.ice_condition_weight * risk_components['ice_stability'] +
            self.config.weather_weight * risk_components['weather'] +
            self.config.historical_weight * risk_components['historical']
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk)
        
        # Most likely avalanche type
        most_likely_type = self._determine_most_likely_type(
            trigger_probabilities, terrain, weather_data
        )
        
        return {
            'overall_risk_score': float(overall_risk),
            'risk_level': risk_level,
            'risk_components': risk_components,
            'stability_assessment': stability_assessment,
            'terrain_assessment': terrain_risk,
            'weather_assessment': weather_risk,
            'historical_assessment': historical_risk,
            'trigger_probabilities': trigger_probabilities,
            'most_likely_avalanche_type': most_likely_type,
            'location_id': terrain.location_id
        }
    
    def _assess_terrain_risk(self, terrain: TerrainCharacteristics) -> Dict[str, float]:
        """Assess terrain-related risk factors."""
        # Slope angle risk
        if terrain.slope_angle < self.config.critical_slope_angle:
            slope_risk = 0.1
        elif terrain.slope_angle > self.config.maximum_stable_angle:
            slope_risk = 1.0
        else:
            # Linear increase between critical and maximum
            angle_range = self.config.maximum_stable_angle - self.config.critical_slope_angle
            slope_risk = (terrain.slope_angle - self.config.critical_slope_angle) / angle_range
        
        # Elevation risk (higher elevations generally more unstable)
        elevation_risk = min(terrain.elevation / 6000.0, 1.0)  # Normalize to 6000m
        
        # Ice thickness risk
        thickness_risk = min(terrain.ice_thickness / 50.0, 1.0)  # Normalize to 50m
        
        # Crevasse density risk
        crevasse_risk = min(terrain.crevasse_density / self.config.critical_crevasse_density, 1.0)
        
        # Serac presence risk
        serac_risk = 0.8 if terrain.serac_presence else 0.1
        
        # Exposure risk
        exposure_risk = (terrain.solar_exposure / 12.0 + terrain.wind_exposure) / 2.0
        
        # Combined terrain risk
        terrain_score = np.mean([
            slope_risk, elevation_risk, thickness_risk, 
            crevasse_risk, serac_risk, exposure_risk
        ])
        
        return {
            'risk_score': float(terrain_score),
            'slope_risk': float(slope_risk),
            'elevation_risk': float(elevation_risk),
            'thickness_risk': float(thickness_risk),
            'crevasse_risk': float(crevasse_risk),
            'serac_risk': float(serac_risk),
            'exposure_risk': float(exposure_risk)
        }
    
    def _assess_weather_risk(self, weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess weather-related risk factors."""
        # Temperature risk
        temperature = weather_data.get('temperature', -10)
        temp_risk = max(0, temperature / 5.0)  # Risk increases with positive temps
        
        # Temperature change risk
        temp_change = abs(weather_data.get('temperature_change_24h', 0))
        temp_change_risk = min(temp_change / self.config.temperature_change_threshold, 1.0)
        
        # Wind risk
        wind_speed = weather_data.get('wind_speed', 0)
        wind_risk = min(wind_speed / self.config.wind_speed_threshold, 1.0)
        
        # Precipitation risk
        precipitation = weather_data.get('precipitation', 0)
        precip_risk = min(precipitation / self.config.precipitation_threshold, 1.0)
        
        # Solar radiation risk
        solar_radiation = weather_data.get('solar_radiation', 0.5)
        solar_risk = solar_radiation  # Direct relationship
        
        # Seismic activity risk
        seismic_magnitude = weather_data.get('seismic_magnitude', 0)
        seismic_risk = min(seismic_magnitude / self.config.seismic_magnitude_threshold, 1.0)
        
        # Combined weather risk
        weather_score = np.mean([
            temp_risk, temp_change_risk, wind_risk, 
            precip_risk, solar_risk, seismic_risk
        ])
        
        return {
            'risk_score': float(weather_score),
            'temperature_risk': float(temp_risk),
            'temperature_change_risk': float(temp_change_risk),
            'wind_risk': float(wind_risk),
            'precipitation_risk': float(precip_risk),
            'solar_risk': float(solar_risk),
            'seismic_risk': float(seismic_risk)
        }
    
    def _assess_historical_risk(self, historical_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk based on historical avalanche activity."""
        # Frequency of past events
        past_events = historical_data.get('avalanche_count_10_years', 0)
        frequency_risk = min(past_events / 10.0, 1.0)  # Normalize to 10 events
        
        # Recency of last event
        years_since_last = historical_data.get('years_since_last_avalanche', 100)
        recency_risk = max(0, 1 - years_since_last / 20.0)  # Higher risk if recent
        
        # Magnitude of past events
        max_past_volume = historical_data.get('max_avalanche_volume_m3', 0)
        magnitude_risk = min(max_past_volume / 100000.0, 1.0)  # Normalize to 100,000 m³
        
        # Seasonal pattern
        current_month = historical_data.get('current_month', 6)
        seasonal_pattern = historical_data.get('monthly_avalanche_frequency', [1]*12)
        if len(seasonal_pattern) >= current_month:
            seasonal_risk = seasonal_pattern[current_month - 1] / max(seasonal_pattern)
        else:
            seasonal_risk = 0.5
        
        # Combined historical risk
        if past_events == 0:
            historical_score = 0.3  # Default moderate risk for unknown areas
        else:
            historical_score = np.mean([
                frequency_risk, recency_risk, magnitude_risk, seasonal_risk
            ])
        
        return {
            'risk_score': float(historical_score),
            'frequency_risk': float(frequency_risk),
            'recency_risk': float(recency_risk),
            'magnitude_risk': float(magnitude_risk),
            'seasonal_risk': float(seasonal_risk),
            'past_events': int(past_events)
        }
    
    def _assess_trigger_probabilities(
        self,
        terrain: TerrainCharacteristics,
        ice_conditions: Dict[str, Any],
        weather_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess probabilities for different trigger mechanisms."""
        probabilities = {}
        
        # Serac fall probability
        serac_prob = 0.1 if terrain.serac_presence else 0.01
        serac_prob *= (1 + weather_data.get('temperature', -10) / 10.0)  # Higher with warming
        probabilities[AvalancheType.SERAC_FALL.value] = min(serac_prob, 1.0)
        
        # Ice cliff collapse probability
        cliff_prob = 0.05 * (terrain.slope_angle / 45.0)  # Steeper = higher probability
        cliff_prob *= (1 + abs(weather_data.get('temperature_change_24h', 0)) / 5.0)
        probabilities[AvalancheType.ICE_CLIFF_COLLAPSE.value] = min(cliff_prob, 1.0)
        
        # Hanging glacier break probability
        hanging_prob = 0.02
        if terrain.slope_angle > 45:
            hanging_prob *= 2.0
        if terrain.ice_thickness > 20:
            hanging_prob *= 1.5
        probabilities[AvalancheType.HANGING_GLACIER_BREAK.value] = min(hanging_prob, 1.0)
        
        # Crevasse-induced probability
        crevasse_prob = 0.03 * (terrain.crevasse_density / 5.0)
        crevasse_prob *= (1 + weather_data.get('wind_speed', 0) / 20.0)
        probabilities[AvalancheType.CREVASSE_INDUCED.value] = min(crevasse_prob, 1.0)
        
        # Thermal-induced probability
        thermal_prob = 0.02
        if weather_data.get('temperature', -10) > 0:
            thermal_prob *= 3.0
        thermal_prob *= weather_data.get('solar_radiation', 0.5)
        probabilities[AvalancheType.THERMAL_INDUCED.value] = min(thermal_prob, 1.0)
        
        # Seismic-triggered probability
        seismic_mag = weather_data.get('seismic_magnitude', 0)
        seismic_prob = 0.01 * (seismic_mag / 5.0) ** 2
        probabilities[AvalancheType.SEISMIC_TRIGGERED.value] = min(seismic_prob, 1.0)
        
        return probabilities
    
    def _determine_most_likely_type(
        self,
        trigger_probabilities: Dict[str, float],
        terrain: TerrainCharacteristics,
        weather_data: Dict[str, Any]
    ) -> AvalancheType:
        """Determine the most likely avalanche type."""
        # Find type with highest probability
        max_prob = 0
        most_likely = AvalancheType.SERAC_FALL
        
        for type_name, prob in trigger_probabilities.items():
            if prob > max_prob:
                max_prob = prob
                most_likely = AvalancheType(type_name)
        
        return most_likely
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on overall score."""
        if risk_score >= 0.9:
            return RiskLevel.EXTREME
        elif risk_score >= 0.7:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MODERATE
        elif risk_score >= 0.1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def predict_avalanche_scenario(
        self,
        terrain: TerrainCharacteristics,
        avalanche_type: AvalancheType,
        trigger_intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Predict avalanche scenario including runout and impact.
        
        Args:
            terrain: Terrain characteristics
            avalanche_type: Type of avalanche
            trigger_intensity: Intensity of trigger (0-1)
            
        Returns:
            Avalanche scenario prediction
        """
        # Estimate avalanche volume
        volume_estimate = self.runout_model.estimate_avalanche_volume(
            terrain, avalanche_type, trigger_intensity
        )
        
        # Calculate runout
        runout_results = self.runout_model.calculate_runout_distance(
            terrain, volume_estimate['estimated_volume_m3'], avalanche_type
        )
        
        return {
            'avalanche_type': avalanche_type.value,
            'trigger_intensity': trigger_intensity,
            'volume_estimate': volume_estimate,
            'runout_prediction': runout_results,
            'terrain_id': terrain.location_id,
            'prediction_timestamp': datetime.now().isoformat()
        }


class AvalancheRiskCalculator:
    """Calculate and manage avalanche risk for multiple locations."""
    
    def __init__(self, config: AvalancheConfig = None):
        """
        Initialize avalanche risk calculator.
        
        Args:
            config: Avalanche assessment configuration
        """
        self.config = config or AvalancheConfig()
        self.predictor = IceAvalanchePredictor(self.config)
        self.logger = logging.getLogger(__name__)
    
    def calculate_regional_risk(
        self,
        terrain_locations: List[TerrainCharacteristics],
        regional_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate avalanche risk for multiple locations in a region.
        
        Args:
            terrain_locations: List of terrain characteristics
            regional_conditions: Regional weather and ice conditions
            
        Returns:
            Regional risk assessment
        """
        location_risks = {}
        risk_levels = []
        
        for terrain in terrain_locations:
            # Assess risk for each location
            risk_assessment = self.predictor.assess_avalanche_risk(
                terrain,
                regional_conditions.get('ice_conditions', {}),
                regional_conditions.get('weather_data', {}),
                regional_conditions.get('historical_data', {})
            )
            
            location_risks[terrain.location_id] = risk_assessment
            risk_levels.append(risk_assessment['overall_risk_score'])
        
        # Regional statistics
        if risk_levels:
            regional_stats = {
                'mean_risk': float(np.mean(risk_levels)),
                'max_risk': float(np.max(risk_levels)),
                'min_risk': float(np.min(risk_levels)),
                'std_risk': float(np.std(risk_levels)),
                'high_risk_locations': sum(1 for r in risk_levels if r > 0.7),
                'total_locations': len(risk_levels)
            }
        else:
            regional_stats = {'total_locations': 0}
        
        return {
            'location_risks': location_risks,
            'regional_statistics': regional_stats,
            'assessment_timestamp': datetime.now().isoformat()
        }


def create_avalanche_predictor(
    critical_slope_angle: float = 30.0,
    **kwargs
) -> IceAvalanchePredictor:
    """
    Create an ice avalanche predictor with specified configuration.
    
    Args:
        critical_slope_angle: Critical slope angle for avalanche risk (degrees)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured avalanche predictor
    """
    config = AvalancheConfig(
        critical_slope_angle=critical_slope_angle,
        **kwargs
    )
    return IceAvalanchePredictor(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create example terrain
    terrain = TerrainCharacteristics(
        location_id="AVALANCHE_ZONE_001",
        elevation=5200,  # m
        slope_angle=45,  # degrees
        aspect=180,  # south-facing
        slope_length=800,  # m
        slope_width=300,  # m
        ice_thickness=25,  # m
        ice_temperature=-5,  # °C
        crevasse_density=8,  # per km²
        serac_presence=True,
        latitude=28.5,
        longitude=86.8,
        solar_exposure=8,  # hours/day
        wind_exposure=0.7  # 0-1 scale
    )
    
    # Create predictor
    predictor = create_avalanche_predictor(
        critical_slope_angle=30.0,
        maximum_stable_angle=60.0
    )
    
    # Example conditions
    ice_conditions = {
        'stability_index': 0.6,
        'recent_changes': 'moderate_warming'
    }
    
    weather_data = {
        'temperature': 2.0,  # °C
        'temperature_change_24h': 8.0,  # °C
        'wind_speed': 12.0,  # m/s
        'precipitation': 15.0,  # mm
        'solar_radiation': 0.8,  # 0-1 scale
        'seismic_magnitude': 0.0
    }
    
    historical_data = {
        'avalanche_count_10_years': 3,
        'years_since_last_avalanche': 2,
        'max_avalanche_volume_m3': 50000,
        'current_month': 6,
        'monthly_avalanche_frequency': [1, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    }
    
    # Perform risk assessment
    print("Performing ice avalanche risk assessment...")
    risk_assessment = predictor.assess_avalanche_risk(
        terrain, ice_conditions, weather_data, historical_data
    )
    
    print("\nIce Avalanche Risk Assessment Results:")
    print("=" * 45)
    print(f"Location ID: {terrain.location_id}")
    print(f"Overall Risk Score: {risk_assessment['overall_risk_score']:.3f}")
    print(f"Risk Level: {risk_assessment['risk_level'].value.upper()}")
    print(f"Most Likely Type: {risk_assessment['most_likely_avalanche_type'].value.replace('_', ' ').title()}")
    
    print("\nRisk Components:")
    for component, score in risk_assessment['risk_components'].items():
        print(f"  {component.replace('_', ' ').title()}: {score:.3f}")
    
    print("\nTrigger Probabilities:")
    for trigger, prob in risk_assessment['trigger_probabilities'].items():
        print(f"  {trigger.replace('_', ' ').title()}: {prob:.3f}")
    
    # Predict avalanche scenario
    print("\nPredicting avalanche scenario...")
    scenario = predictor.predict_avalanche_scenario(
        terrain,
        risk_assessment['most_likely_avalanche_type'],
        trigger_intensity=0.8
    )
    
    volume = scenario['volume_estimate']
    runout = scenario['runout_prediction']
    
    print(f"\nAvalanche Scenario Prediction:")
    print(f"Avalanche Type: {scenario['avalanche_type'].replace('_', ' ').title()}")
    print(f"Estimated Volume: {volume['estimated_volume_m3']:.0f} m³")
    print(f"Estimated Mass: {volume['estimated_mass_kg']:.0f} kg")
    print(f"Runout Distance: {runout['runout_distance_m']:.0f} m")
    print(f"Impact Area: {runout['impact_area_m2']:.0f} m²")
    print(f"Impact Pressure: {runout['impact_pressure_pa']:.0f} Pa")
    print(f"Average Velocity: {runout['average_velocity_ms']:.1f} m/s")
    
    # Stability analysis details
    stability = risk_assessment['stability_assessment']
    print(f"\nIce Stability Analysis:")
    print(f"Overall Stability: {stability['overall_stability_score']:.3f}")
    print(f"Temperature Stability: {stability['temperature_stability']['stability_score']:.3f}")
    print(f"Structural Stability: {stability['structural_stability']['stability_score']:.3f}")
    
    # Create risk calculator for regional analysis
    print("\nPerforming regional risk analysis...")
    calculator = AvalancheRiskCalculator()
    
    # Create multiple terrain locations
    terrain_locations = [terrain]
    for i in range(3):
        new_terrain = TerrainCharacteristics(
            location_id=f"AVALANCHE_ZONE_{i+2:03d}",
            elevation=terrain.elevation + i * 200,
            slope_angle=terrain.slope_angle + i * 5,
            aspect=terrain.aspect + i * 30,
            slope_length=terrain.slope_length - i * 100,
            slope_width=terrain.slope_width + i * 50,
            ice_thickness=terrain.ice_thickness - i * 3,
            ice_temperature=terrain.ice_temperature - i * 1,
            crevasse_density=terrain.crevasse_density + i * 2,
            serac_presence=i % 2 == 0,
            latitude=terrain.latitude + i * 0.1,
            longitude=terrain.longitude + i * 0.1,
            solar_exposure=terrain.solar_exposure - i * 1,
            wind_exposure=terrain.wind_exposure + i * 0.1
        )
        terrain_locations.append(new_terrain)
    
    regional_conditions = {
        'ice_conditions': ice_conditions,
        'weather_data': weather_data,
        'historical_data': historical_data
    }
    
    regional_risk = calculator.calculate_regional_risk(
        terrain_locations, regional_conditions
    )
    
    stats = regional_risk['regional_statistics']
    print(f"\nRegional Risk Statistics:")
    print(f"Total Locations: {stats['total_locations']}")
    print(f"Mean Risk: {stats['mean_risk']:.3f}")
    print(f"Max Risk: {stats['max_risk']:.3f}")
    print(f"High Risk Locations: {stats['high_risk_locations']}")
    
    print(f"\nIce avalanche assessment completed for {len(terrain_locations)} locations")