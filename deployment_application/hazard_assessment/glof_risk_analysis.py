"""Glacial Lake Outburst Flood (GLOF) Assessment Module.

This module provides functionality for assessing the risk of glacial lake
outburst floods based on glacier dynamics, lake characteristics, and
environmental conditions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import interpolate, optimize
from scipy.spatial.distance import cdist


class GLOFTriggerType(Enum):
    """Types of GLOF triggers."""
    ICE_DAM_FAILURE = "ice_dam_failure"
    MORAINE_DAM_FAILURE = "moraine_dam_failure"
    LANDSLIDE_DISPLACEMENT = "landslide_displacement"
    ICE_AVALANCHE = "ice_avalanche"
    SEISMIC_ACTIVITY = "seismic_activity"
    EXTREME_PRECIPITATION = "extreme_precipitation"
    RAPID_GLACIER_RETREAT = "rapid_glacier_retreat"


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class LakeCharacteristics:
    """Characteristics of a glacial lake."""
    lake_id: str
    area: float  # km²
    volume: float  # million m³
    max_depth: float  # m
    elevation: float  # m above sea level
    dam_type: str  # 'moraine', 'ice', 'bedrock'
    dam_height: float  # m
    dam_width: float  # m
    freeboard: float  # m (height above water level)
    
    # Geographic information
    latitude: float
    longitude: float
    
    # Temporal information
    formation_year: Optional[int] = None
    last_survey_date: Optional[datetime] = None


@dataclass
class GLOFConfig:
    """Configuration for GLOF assessment."""
    # Risk thresholds
    critical_volume_threshold: float = 1.0  # million m³
    high_risk_volume_threshold: float = 0.5  # million m³
    critical_area_threshold: float = 0.1  # km²
    
    # Dam stability parameters
    moraine_stability_factor: float = 1.5
    ice_dam_stability_factor: float = 1.0
    critical_freeboard: float = 5.0  # m
    
    # Environmental thresholds
    extreme_temp_threshold: float = 5.0  # °C above normal
    extreme_precip_threshold: float = 50.0  # mm/day
    seismic_magnitude_threshold: float = 5.0
    
    # Modeling parameters
    breach_width_factor: float = 2.0  # times dam height
    breach_time_hours: float = 1.0
    manning_coefficient: float = 0.035
    
    # Monitoring parameters
    monitoring_frequency_days: int = 30
    alert_threshold_change: float = 0.1  # fractional change


class GLOFHydrodynamics:
    """Hydrodynamic modeling for GLOF scenarios."""
    
    def __init__(self, config: GLOFConfig = None):
        """
        Initialize GLOF hydrodynamics model.
        
        Args:
            config: GLOF assessment configuration
        """
        self.config = config or GLOFConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_breach_parameters(
        self,
        lake_chars: LakeCharacteristics,
        trigger_type: GLOFTriggerType
    ) -> Dict[str, float]:
        """
        Calculate dam breach parameters.
        
        Args:
            lake_chars: Lake characteristics
            trigger_type: Type of GLOF trigger
            
        Returns:
            Breach parameters
        """
        # Base breach width (empirical relationship)
        if trigger_type == GLOFTriggerType.ICE_DAM_FAILURE:
            breach_width = min(
                self.config.breach_width_factor * lake_chars.dam_height,
                lake_chars.dam_width
            )
            breach_time = self.config.breach_time_hours * 0.5  # Faster for ice
        elif trigger_type == GLOFTriggerType.MORAINE_DAM_FAILURE:
            breach_width = min(
                self.config.breach_width_factor * lake_chars.dam_height * 0.8,
                lake_chars.dam_width * 0.8
            )
            breach_time = self.config.breach_time_hours
        else:
            # Conservative estimate for other triggers
            breach_width = min(
                self.config.breach_width_factor * lake_chars.dam_height * 0.6,
                lake_chars.dam_width * 0.6
            )
            breach_time = self.config.breach_time_hours * 1.5
        
        # Breach depth (typically full dam height for catastrophic failure)
        breach_depth = lake_chars.dam_height - lake_chars.freeboard
        
        return {
            'breach_width': float(breach_width),
            'breach_depth': float(breach_depth),
            'breach_time_hours': float(breach_time),
            'breach_area': float(breach_width * breach_depth)
        }
    
    def calculate_peak_discharge(
        self,
        lake_chars: LakeCharacteristics,
        breach_params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate peak discharge using empirical relationships.
        
        Args:
            lake_chars: Lake characteristics
            breach_params: Breach parameters
            
        Returns:
            Discharge calculations
        """
        # Convert volume to m³
        volume_m3 = lake_chars.volume * 1e6
        
        # Empirical peak discharge formulas
        
        # Method 1: Clague-Mathews relationship
        q_peak_cm = 75 * (volume_m3 ** 0.67)  # m³/s
        
        # Method 2: Breach-based calculation
        # Assuming broad-crested weir flow
        g = 9.81  # m/s²
        head = lake_chars.max_depth * 0.8  # Effective head
        q_peak_breach = 1.7 * breach_params['breach_width'] * (head ** 1.5)
        
        # Method 3: Volume-based empirical relationship
        q_peak_volume = 1.3 * (volume_m3 ** 0.72)
        
        # Take conservative (higher) estimate
        q_peak = max(q_peak_cm, q_peak_breach, q_peak_volume)
        
        # Calculate total outflow volume (typically 70-90% of lake volume)
        outflow_fraction = 0.8
        total_outflow = volume_m3 * outflow_fraction
        
        # Estimate flood duration
        flood_duration_hours = total_outflow / (q_peak * 3600) * 2  # Factor for hydrograph shape
        
        return {
            'peak_discharge_m3s': float(q_peak),
            'peak_discharge_clague_mathews': float(q_peak_cm),
            'peak_discharge_breach': float(q_peak_breach),
            'peak_discharge_volume': float(q_peak_volume),
            'total_outflow_m3': float(total_outflow),
            'flood_duration_hours': float(flood_duration_hours),
            'outflow_fraction': float(outflow_fraction)
        }
    
    def model_flood_routing(
        self,
        peak_discharge: float,
        valley_profile: np.ndarray,
        distances: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Model flood wave routing downstream.
        
        Args:
            peak_discharge: Peak discharge (m³/s)
            valley_profile: Valley cross-sectional areas (m²) at different points
            distances: Distances downstream (km)
            
        Returns:
            Flood routing results
        """
        # Simplified flood routing using Muskingum method
        n_points = len(distances)
        
        # Estimate travel times based on valley characteristics
        # Assume average velocity based on Manning's equation
        manning_n = self.config.manning_coefficient
        
        # Estimate hydraulic radius and slope
        avg_area = np.mean(valley_profile)
        hydraulic_radius = np.sqrt(avg_area / np.pi)  # Approximate for circular channel
        slope = 0.05  # Typical mountain valley slope
        
        # Manning velocity
        velocity = (1 / manning_n) * (hydraulic_radius ** (2/3)) * (slope ** 0.5)
        
        # Travel times
        travel_times = distances / (velocity * 3.6)  # Convert km/h to hours
        
        # Attenuation factors (empirical)
        attenuation = np.exp(-0.1 * distances)  # Exponential decay
        
        # Peak discharges at each point
        peak_discharges = peak_discharge * attenuation
        
        # Estimate flood depths (simplified)
        flood_depths = np.sqrt(peak_discharges / (velocity * 10))  # Rough approximation
        
        return {
            'distances_km': distances,
            'travel_times_hours': travel_times,
            'peak_discharges_m3s': peak_discharges,
            'flood_depths_m': flood_depths,
            'velocities_ms': np.full_like(distances, velocity),
            'attenuation_factors': attenuation
        }


class GLOFRiskAssessment:
    """Main class for GLOF risk assessment."""
    
    def __init__(self, config: GLOFConfig = None):
        """
        Initialize GLOF risk assessment.
        
        Args:
            config: GLOF assessment configuration
        """
        self.config = config or GLOFConfig()
        self.hydrodynamics = GLOFHydrodynamics(self.config)
        self.logger = logging.getLogger(__name__)
    
    def assess_lake_susceptibility(
        self,
        lake_chars: LakeCharacteristics,
        glacier_data: Dict[str, Any] = None,
        environmental_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Assess susceptibility of a glacial lake to outburst flooding.
        
        Args:
            lake_chars: Lake characteristics
            glacier_data: Glacier dynamics data
            environmental_data: Environmental conditions
            
        Returns:
            Susceptibility assessment
        """
        # Initialize assessment components
        geometric_risk = self._assess_geometric_factors(lake_chars)
        dam_stability = self._assess_dam_stability(lake_chars)
        environmental_risk = self._assess_environmental_factors(
            environmental_data or {}
        )
        glacier_influence = self._assess_glacier_influence(
            glacier_data or {}, lake_chars
        )
        
        # Calculate trigger probabilities
        trigger_probabilities = self._calculate_trigger_probabilities(
            lake_chars, glacier_data, environmental_data
        )
        
        # Overall susceptibility score (0-1)
        weights = {
            'geometric': 0.25,
            'dam_stability': 0.30,
            'environmental': 0.20,
            'glacier_influence': 0.25
        }
        
        overall_score = (
            weights['geometric'] * geometric_risk['risk_score'] +
            weights['dam_stability'] * dam_stability['risk_score'] +
            weights['environmental'] * environmental_risk['risk_score'] +
            weights['glacier_influence'] * glacier_influence['risk_score']
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        return {
            'overall_risk_score': float(overall_score),
            'risk_level': risk_level,
            'geometric_assessment': geometric_risk,
            'dam_stability_assessment': dam_stability,
            'environmental_assessment': environmental_risk,
            'glacier_influence_assessment': glacier_influence,
            'trigger_probabilities': trigger_probabilities,
            'assessment_weights': weights,
            'lake_id': lake_chars.lake_id
        }
    
    def _assess_geometric_factors(self, lake_chars: LakeCharacteristics) -> Dict[str, Any]:
        """Assess geometric risk factors."""
        # Volume-based risk
        volume_risk = min(lake_chars.volume / self.config.critical_volume_threshold, 1.0)
        
        # Area-based risk
        area_risk = min(lake_chars.area / self.config.critical_area_threshold, 1.0)
        
        # Depth-based risk (deeper lakes are generally more dangerous)
        depth_risk = min(lake_chars.max_depth / 100.0, 1.0)  # Normalize to 100m
        
        # Freeboard risk (lower freeboard = higher risk)
        freeboard_risk = max(0, 1 - lake_chars.freeboard / self.config.critical_freeboard)
        
        # Dam geometry risk
        dam_height_risk = min(lake_chars.dam_height / 50.0, 1.0)  # Normalize to 50m
        
        # Combined geometric risk
        geometric_score = np.mean([
            volume_risk, area_risk, depth_risk, freeboard_risk, dam_height_risk
        ])
        
        return {
            'risk_score': float(geometric_score),
            'volume_risk': float(volume_risk),
            'area_risk': float(area_risk),
            'depth_risk': float(depth_risk),
            'freeboard_risk': float(freeboard_risk),
            'dam_height_risk': float(dam_height_risk)
        }
    
    def _assess_dam_stability(self, lake_chars: LakeCharacteristics) -> Dict[str, Any]:
        """Assess dam stability factors."""
        # Dam type risk factors
        dam_type_risks = {
            'ice': 0.9,      # Very high risk
            'moraine': 0.6,  # Moderate-high risk
            'bedrock': 0.1   # Low risk
        }
        
        dam_type_risk = dam_type_risks.get(lake_chars.dam_type.lower(), 0.5)
        
        # Dam geometry stability
        # Width-to-height ratio (higher ratio = more stable)
        width_height_ratio = lake_chars.dam_width / max(lake_chars.dam_height, 1)
        geometry_stability = max(0, 1 - width_height_ratio / 10.0)  # Normalize
        
        # Freeboard adequacy
        freeboard_adequacy = max(0, 1 - lake_chars.freeboard / self.config.critical_freeboard)
        
        # Overall dam stability risk
        stability_score = np.mean([
            dam_type_risk,
            geometry_stability,
            freeboard_adequacy
        ])
        
        return {
            'risk_score': float(stability_score),
            'dam_type_risk': float(dam_type_risk),
            'geometry_stability_risk': float(geometry_stability),
            'freeboard_adequacy_risk': float(freeboard_adequacy),
            'dam_type': lake_chars.dam_type
        }
    
    def _assess_environmental_factors(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess environmental risk factors."""
        # Temperature anomaly risk
        temp_anomaly = env_data.get('temperature_anomaly', 0)
        temp_risk = min(abs(temp_anomaly) / self.config.extreme_temp_threshold, 1.0)
        
        # Precipitation risk
        extreme_precip = env_data.get('extreme_precipitation', 0)
        precip_risk = min(extreme_precip / self.config.extreme_precip_threshold, 1.0)
        
        # Seismic activity risk
        seismic_magnitude = env_data.get('seismic_magnitude', 0)
        seismic_risk = min(seismic_magnitude / self.config.seismic_magnitude_threshold, 1.0)
        
        # Seasonal factors
        season = env_data.get('season', 'unknown')
        seasonal_risks = {
            'spring': 0.7,  # High melt season
            'summer': 0.8,  # Peak melt
            'autumn': 0.4,  # Moderate
            'winter': 0.2   # Low
        }
        seasonal_risk = seasonal_risks.get(season, 0.5)
        
        # Combined environmental risk
        env_score = np.mean([temp_risk, precip_risk, seismic_risk, seasonal_risk])
        
        return {
            'risk_score': float(env_score),
            'temperature_risk': float(temp_risk),
            'precipitation_risk': float(precip_risk),
            'seismic_risk': float(seismic_risk),
            'seasonal_risk': float(seasonal_risk),
            'season': season
        }
    
    def _assess_glacier_influence(self, glacier_data: Dict[str, Any], lake_chars: LakeCharacteristics) -> Dict[str, Any]:
        """Assess glacier influence on GLOF risk."""
        # Glacier retreat rate
        retreat_rate = glacier_data.get('retreat_rate_m_per_year', 0)
        retreat_risk = min(retreat_rate / 50.0, 1.0)  # Normalize to 50 m/year
        
        # Glacier velocity changes
        velocity_change = glacier_data.get('velocity_change_percent', 0)
        velocity_risk = min(abs(velocity_change) / 100.0, 1.0)  # Normalize to 100%
        
        # Ice thickness changes
        thickness_change = glacier_data.get('thickness_change_m_per_year', 0)
        thickness_risk = min(abs(thickness_change) / 5.0, 1.0)  # Normalize to 5 m/year
        
        # Distance to glacier terminus
        distance_to_glacier = glacier_data.get('distance_to_terminus_km', 10)
        proximity_risk = max(0, 1 - distance_to_glacier / 5.0)  # Higher risk if < 5 km
        
        # Calving activity
        calving_rate = glacier_data.get('calving_rate', 0)
        calving_risk = min(calving_rate / 10.0, 1.0)  # Normalize
        
        # Combined glacier influence
        glacier_score = np.mean([
            retreat_risk, velocity_risk, thickness_risk, proximity_risk, calving_risk
        ])
        
        return {
            'risk_score': float(glacier_score),
            'retreat_risk': float(retreat_risk),
            'velocity_risk': float(velocity_risk),
            'thickness_risk': float(thickness_risk),
            'proximity_risk': float(proximity_risk),
            'calving_risk': float(calving_risk)
        }
    
    def _calculate_trigger_probabilities(
        self,
        lake_chars: LakeCharacteristics,
        glacier_data: Dict[str, Any],
        env_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate probabilities for different GLOF triggers."""
        probabilities = {}
        
        # Ice dam failure probability
        if lake_chars.dam_type.lower() == 'ice':
            ice_dam_prob = 0.1 + 0.3 * glacier_data.get('temperature_anomaly', 0) / 5.0
        else:
            ice_dam_prob = 0.0
        
        # Moraine dam failure probability
        if lake_chars.dam_type.lower() == 'moraine':
            moraine_prob = 0.02 + 0.05 * (1 - lake_chars.freeboard / self.config.critical_freeboard)
        else:
            moraine_prob = 0.0
        
        # Landslide displacement probability
        seismic_mag = env_data.get('seismic_magnitude', 0)
        landslide_prob = 0.01 + 0.1 * min(seismic_mag / 7.0, 1.0)
        
        # Ice avalanche probability
        slope_angle = glacier_data.get('slope_angle', 0)
        avalanche_prob = 0.005 + 0.02 * min(slope_angle / 45.0, 1.0)
        
        # Extreme precipitation probability
        precip_intensity = env_data.get('extreme_precipitation', 0)
        precip_prob = 0.01 + 0.05 * min(precip_intensity / 100.0, 1.0)
        
        return {
            GLOFTriggerType.ICE_DAM_FAILURE.value: min(ice_dam_prob, 1.0),
            GLOFTriggerType.MORAINE_DAM_FAILURE.value: min(moraine_prob, 1.0),
            GLOFTriggerType.LANDSLIDE_DISPLACEMENT.value: min(landslide_prob, 1.0),
            GLOFTriggerType.ICE_AVALANCHE.value: min(avalanche_prob, 1.0),
            GLOFTriggerType.EXTREME_PRECIPITATION.value: min(precip_prob, 1.0)
        }
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on overall score."""
        if risk_score >= 0.9:
            return RiskLevel.CRITICAL
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
    
    def simulate_glof_scenario(
        self,
        lake_chars: LakeCharacteristics,
        trigger_type: GLOFTriggerType,
        valley_profile: Optional[np.ndarray] = None,
        downstream_distances: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Simulate a GLOF scenario.
        
        Args:
            lake_chars: Lake characteristics
            trigger_type: Type of trigger event
            valley_profile: Valley cross-sectional areas downstream
            downstream_distances: Distances downstream (km)
            
        Returns:
            GLOF simulation results
        """
        # Calculate breach parameters
        breach_params = self.hydrodynamics.calculate_breach_parameters(
            lake_chars, trigger_type
        )
        
        # Calculate discharge
        discharge_results = self.hydrodynamics.calculate_peak_discharge(
            lake_chars, breach_params
        )
        
        # Flood routing (if valley profile provided)
        routing_results = None
        if valley_profile is not None and downstream_distances is not None:
            routing_results = self.hydrodynamics.model_flood_routing(
                discharge_results['peak_discharge_m3s'],
                valley_profile,
                downstream_distances
            )
        
        return {
            'trigger_type': trigger_type.value,
            'lake_characteristics': {
                'lake_id': lake_chars.lake_id,
                'volume_million_m3': lake_chars.volume,
                'area_km2': lake_chars.area,
                'dam_type': lake_chars.dam_type
            },
            'breach_parameters': breach_params,
            'discharge_results': discharge_results,
            'flood_routing': routing_results,
            'simulation_timestamp': datetime.now().isoformat()
        }


class GlacialLakeMonitor:
    """Monitor glacial lakes for changes that affect GLOF risk."""
    
    def __init__(self, config: GLOFConfig = None):
        """
        Initialize glacial lake monitor.
        
        Args:
            config: GLOF assessment configuration
        """
        self.config = config or GLOFConfig()
        self.logger = logging.getLogger(__name__)
        self.lake_database = {}  # Store lake monitoring data
    
    def add_lake_monitoring(
        self,
        lake_chars: LakeCharacteristics,
        monitoring_data: Dict[str, Any]
    ) -> None:
        """
        Add lake to monitoring system.
        
        Args:
            lake_chars: Lake characteristics
            monitoring_data: Initial monitoring data
        """
        self.lake_database[lake_chars.lake_id] = {
            'characteristics': lake_chars,
            'monitoring_history': [{
                'timestamp': datetime.now(),
                'data': monitoring_data
            }],
            'last_assessment': None,
            'alert_status': 'normal'
        }
        
        self.logger.info(f"Added lake {lake_chars.lake_id} to monitoring system")
    
    def update_lake_data(
        self,
        lake_id: str,
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update monitoring data for a lake.
        
        Args:
            lake_id: Lake identifier
            new_data: New monitoring data
            
        Returns:
            Change analysis results
        """
        if lake_id not in self.lake_database:
            raise ValueError(f"Lake {lake_id} not found in monitoring system")
        
        # Add new data point
        self.lake_database[lake_id]['monitoring_history'].append({
            'timestamp': datetime.now(),
            'data': new_data
        })
        
        # Analyze changes
        change_analysis = self._analyze_changes(lake_id)
        
        # Update alert status
        self._update_alert_status(lake_id, change_analysis)
        
        return change_analysis
    
    def _analyze_changes(self, lake_id: str) -> Dict[str, Any]:
        """Analyze changes in lake monitoring data."""
        history = self.lake_database[lake_id]['monitoring_history']
        
        if len(history) < 2:
            return {'status': 'insufficient_data'}
        
        # Get latest and previous data
        latest = history[-1]['data']
        previous = history[-2]['data']
        
        changes = {}
        alerts = []
        
        # Analyze key parameters
        key_params = ['area', 'volume', 'water_level', 'freeboard', 'dam_stability']
        
        for param in key_params:
            if param in latest and param in previous:
                current_val = latest[param]
                prev_val = previous[param]
                
                if prev_val != 0:
                    change_fraction = (current_val - prev_val) / prev_val
                    changes[f'{param}_change_fraction'] = change_fraction
                    
                    # Check for significant changes
                    if abs(change_fraction) > self.config.alert_threshold_change:
                        alerts.append({
                            'parameter': param,
                            'change_fraction': change_fraction,
                            'current_value': current_val,
                            'previous_value': prev_val,
                            'severity': 'high' if abs(change_fraction) > 0.2 else 'moderate'
                        })
        
        # Calculate trend over longer period
        if len(history) >= 5:
            trends = self._calculate_trends(lake_id)
            changes['trends'] = trends
        
        return {
            'status': 'analyzed',
            'timestamp': datetime.now().isoformat(),
            'changes': changes,
            'alerts': alerts,
            'data_points_analyzed': len(history)
        }
    
    def _calculate_trends(self, lake_id: str) -> Dict[str, float]:
        """Calculate trends in monitoring parameters."""
        history = self.lake_database[lake_id]['monitoring_history']
        
        # Extract time series for key parameters
        timestamps = [entry['timestamp'] for entry in history]
        time_days = [(ts - timestamps[0]).days for ts in timestamps]
        
        trends = {}
        
        for param in ['area', 'volume', 'water_level']:
            values = []
            for entry in history:
                if param in entry['data']:
                    values.append(entry['data'][param])
                else:
                    values.append(np.nan)
            
            if len(values) >= 3 and not all(np.isnan(values)):
                # Simple linear trend
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) >= 3:
                    valid_times = np.array(time_days)[valid_mask]
                    valid_values = np.array(values)[valid_mask]
                    
                    # Linear regression
                    slope, intercept = np.polyfit(valid_times, valid_values, 1)
                    trends[f'{param}_trend_per_day'] = slope
        
        return trends
    
    def _update_alert_status(self, lake_id: str, change_analysis: Dict[str, Any]) -> None:
        """Update alert status based on change analysis."""
        alerts = change_analysis.get('alerts', [])
        
        if not alerts:
            status = 'normal'
        else:
            # Determine highest severity
            severities = [alert['severity'] for alert in alerts]
            if 'high' in severities:
                status = 'high_alert'
            else:
                status = 'moderate_alert'
        
        self.lake_database[lake_id]['alert_status'] = status
        
        if status != 'normal':
            self.logger.warning(
                f"Alert status updated for lake {lake_id}: {status}"
            )
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored lakes."""
        summary = {
            'total_lakes': len(self.lake_database),
            'alert_counts': {'normal': 0, 'moderate_alert': 0, 'high_alert': 0},
            'lakes_by_status': {'normal': [], 'moderate_alert': [], 'high_alert': []}
        }
        
        for lake_id, lake_data in self.lake_database.items():
            status = lake_data['alert_status']
            summary['alert_counts'][status] += 1
            summary['lakes_by_status'][status].append(lake_id)
        
        return summary


def create_glof_assessor(
    critical_volume_threshold: float = 1.0,
    **kwargs
) -> GLOFRiskAssessment:
    """
    Create a GLOF risk assessor with specified configuration.
    
    Args:
        critical_volume_threshold: Volume threshold for critical risk (million m³)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GLOF risk assessor
    """
    config = GLOFConfig(
        critical_volume_threshold=critical_volume_threshold,
        **kwargs
    )
    return GLOFRiskAssessment(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create example lake
    lake = LakeCharacteristics(
        lake_id="TEST_LAKE_001",
        area=0.15,  # km²
        volume=2.5,  # million m³
        max_depth=45,  # m
        elevation=4200,  # m
        dam_type="moraine",
        dam_height=25,  # m
        dam_width=150,  # m
        freeboard=3,  # m
        latitude=28.5,
        longitude=86.8
    )
    
    # Create assessor
    assessor = create_glof_assessor(
        critical_volume_threshold=1.0,
        high_risk_volume_threshold=0.5
    )
    
    # Example environmental data
    env_data = {
        'temperature_anomaly': 3.0,  # °C above normal
        'extreme_precipitation': 25.0,  # mm/day
        'seismic_magnitude': 4.5,
        'season': 'summer'
    }
    
    # Example glacier data
    glacier_data = {
        'retreat_rate_m_per_year': 15.0,
        'velocity_change_percent': 25.0,
        'thickness_change_m_per_year': -2.0,
        'distance_to_terminus_km': 2.5,
        'calving_rate': 3.0
    }
    
    # Perform risk assessment
    print("Performing GLOF risk assessment...")
    risk_assessment = assessor.assess_lake_susceptibility(
        lake, glacier_data, env_data
    )
    
    print("\nGLOF Risk Assessment Results:")
    print("=" * 40)
    print(f"Lake ID: {lake.lake_id}")
    print(f"Overall Risk Score: {risk_assessment['overall_risk_score']:.3f}")
    print(f"Risk Level: {risk_assessment['risk_level'].value.upper()}")
    
    print("\nDetailed Assessment:")
    for component, results in risk_assessment.items():
        if isinstance(results, dict) and 'risk_score' in results:
            print(f"  {component.replace('_', ' ').title()}: {results['risk_score']:.3f}")
    
    print("\nTrigger Probabilities:")
    for trigger, prob in risk_assessment['trigger_probabilities'].items():
        print(f"  {trigger.replace('_', ' ').title()}: {prob:.3f}")
    
    # Simulate GLOF scenario
    print("\nSimulating GLOF scenario...")
    scenario = assessor.simulate_glof_scenario(
        lake,
        GLOFTriggerType.MORAINE_DAM_FAILURE
    )
    
    discharge = scenario['discharge_results']
    print(f"\nGLOF Simulation Results:")
    print(f"Peak Discharge: {discharge['peak_discharge_m3s']:.0f} m³/s")
    print(f"Total Outflow: {discharge['total_outflow_m3']:.0f} m³")
    print(f"Flood Duration: {discharge['flood_duration_hours']:.1f} hours")
    
    breach = scenario['breach_parameters']
    print(f"\nBreach Parameters:")
    print(f"Breach Width: {breach['breach_width']:.1f} m")
    print(f"Breach Depth: {breach['breach_depth']:.1f} m")
    print(f"Breach Time: {breach['breach_time_hours']:.1f} hours")
    
    # Create monitoring system
    print("\nSetting up monitoring system...")
    monitor = GlacialLakeMonitor()
    
    # Add lake to monitoring
    initial_data = {
        'area': lake.area,
        'volume': lake.volume,
        'water_level': lake.max_depth - 5,
        'freeboard': lake.freeboard,
        'dam_stability': 0.7
    }
    
    monitor.add_lake_monitoring(lake, initial_data)
    
    # Simulate monitoring updates
    for i in range(3):
        updated_data = {
            'area': lake.area * (1 + 0.02 * i),  # Gradual increase
            'volume': lake.volume * (1 + 0.03 * i),
            'water_level': (lake.max_depth - 5) + 0.5 * i,
            'freeboard': lake.freeboard - 0.2 * i,
            'dam_stability': 0.7 - 0.05 * i
        }
        
        change_analysis = monitor.update_lake_data(lake.lake_id, updated_data)
        print(f"\nMonitoring Update {i+1}:")
        print(f"  Status: {change_analysis['status']}")
        if 'alerts' in change_analysis:
            print(f"  Alerts: {len(change_analysis['alerts'])}")
    
    # Get monitoring summary
    summary = monitor.get_monitoring_summary()
    print(f"\nMonitoring Summary:")
    print(f"Total Lakes: {summary['total_lakes']}")
    print(f"Alert Status: {summary['alert_counts']}")
    
    print(f"\nGLOF assessment completed for lake {lake.lake_id}")