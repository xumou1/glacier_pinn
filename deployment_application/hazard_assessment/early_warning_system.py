"""Glacier Surge Detection and Analysis Module.

This module provides functionality for detecting and analyzing
glacier surge events and their associated risks.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import signal, interpolate, ndimage
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class SurgePhase(Enum):
    """Phases of glacier surge cycle."""
    QUIESCENT = "quiescent"
    INITIATION = "initiation"
    ACTIVE = "active"
    TERMINATION = "termination"
    POST_SURGE = "post_surge"


class SurgeType(Enum):
    """Types of glacier surges."""
    THERMAL = "thermal"
    HYDROLOGICAL = "hydrological"
    STRUCTURAL = "structural"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class GlacierCharacteristics:
    """Glacier characteristics for surge analysis."""
    glacier_id: str
    glacier_name: str
    
    # Geometric properties
    length: float  # km
    width: float  # km
    area: float  # km²
    volume: float  # km³
    
    # Elevation information
    terminus_elevation: float  # m
    accumulation_area_elevation: float  # m
    equilibrium_line_altitude: float  # m
    
    # Thermal properties
    mean_temperature: float  # °C
    temperature_gradient: float  # °C/km
    
    # Hydrological properties
    drainage_area: float  # km²
    annual_precipitation: float  # mm
    melt_rate: float  # mm/day
    
    # Structural properties
    bed_slope: float  # degrees
    surface_slope: float  # degrees
    ice_thickness: float  # m
    
    # Geographic information
    latitude: float
    longitude: float
    
    # Historical surge information
    last_surge_year: Optional[int] = None
    surge_frequency: Optional[float] = None  # years between surges
    surge_history: List[int] = None  # years of known surges


@dataclass
class SurgeConfig:
    """Configuration for glacier surge detection and analysis."""
    # Velocity thresholds
    surge_velocity_threshold: float = 100.0  # m/year
    normal_velocity_threshold: float = 50.0  # m/year
    
    # Acceleration thresholds
    surge_acceleration_threshold: float = 50.0  # m/year²
    
    # Temporal parameters
    surge_duration_min: float = 0.5  # years
    surge_duration_max: float = 10.0  # years
    quiescent_period_min: float = 10.0  # years
    
    # Geometric thresholds
    terminus_advance_threshold: float = 100.0  # m/year
    surface_elevation_change_threshold: float = 10.0  # m/year
    
    # Thermal thresholds
    temperature_anomaly_threshold: float = 2.0  # °C
    
    # Hydrological thresholds
    discharge_anomaly_threshold: float = 2.0  # standard deviations
    precipitation_anomaly_threshold: float = 1.5  # standard deviations
    
    # Detection parameters
    smoothing_window: int = 3  # years
    anomaly_detection_window: int = 10  # years
    confidence_threshold: float = 0.7  # 0-1
    
    # Risk assessment weights
    velocity_weight: float = 0.3
    geometric_weight: float = 0.25
    thermal_weight: float = 0.2
    hydrological_weight: float = 0.15
    historical_weight: float = 0.1


class VelocityAnalysis:
    """Analyze glacier velocity patterns for surge detection."""
    
    def __init__(self, config: SurgeConfig = None):
        """
        Initialize velocity analysis.
        
        Args:
            config: Surge detection configuration
        """
        self.config = config or SurgeConfig()
        self.logger = logging.getLogger(__name__)
    
    def detect_velocity_anomalies(
        self,
        velocity_data: Dict[str, np.ndarray],
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect velocity anomalies indicating potential surge activity.
        
        Args:
            velocity_data: Dictionary with velocity measurements
            time_series: Time series (years)
            
        Returns:
            Velocity anomaly analysis
        """
        velocities = velocity_data.get('velocity', np.array([]))
        
        if len(velocities) == 0:
            return {'anomalies_detected': False, 'reason': 'No velocity data'}
        
        # Smooth velocity data
        smoothed_velocities = self._smooth_data(
            velocities, self.config.smoothing_window
        )
        
        # Calculate velocity statistics
        velocity_stats = self._calculate_velocity_statistics(
            velocities, smoothed_velocities
        )
        
        # Detect surge events
        surge_events = self._detect_surge_events(
            smoothed_velocities, time_series
        )
        
        # Calculate acceleration
        acceleration = self._calculate_acceleration(
            smoothed_velocities, time_series
        )
        
        # Identify anomalous periods
        anomalous_periods = self._identify_anomalous_periods(
            smoothed_velocities, acceleration, time_series
        )
        
        # Classify current state
        current_state = self._classify_current_state(
            velocities, acceleration
        )
        
        return {
            'anomalies_detected': len(surge_events) > 0,
            'velocity_statistics': velocity_stats,
            'surge_events': surge_events,
            'acceleration': acceleration.tolist() if len(acceleration) > 0 else [],
            'anomalous_periods': anomalous_periods,
            'current_state': current_state,
            'smoothed_velocities': smoothed_velocities.tolist()
        }
    
    def _smooth_data(self, data: np.ndarray, window: int) -> np.ndarray:
        """Apply smoothing to data."""
        if len(data) < window:
            return data
        
        # Use moving average
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='same')
        
        # Handle edges
        for i in range(window // 2):
            smoothed[i] = np.mean(data[:i+window//2+1])
            smoothed[-(i+1)] = np.mean(data[-(i+window//2+1):])
        
        return smoothed
    
    def _calculate_velocity_statistics(
        self,
        velocities: np.ndarray,
        smoothed_velocities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate velocity statistics."""
        return {
            'mean_velocity': float(np.mean(velocities)),
            'median_velocity': float(np.median(velocities)),
            'std_velocity': float(np.std(velocities)),
            'max_velocity': float(np.max(velocities)),
            'min_velocity': float(np.min(velocities)),
            'velocity_range': float(np.max(velocities) - np.min(velocities)),
            'smoothed_mean': float(np.mean(smoothed_velocities)),
            'smoothed_std': float(np.std(smoothed_velocities)),
            'coefficient_of_variation': float(np.std(velocities) / np.mean(velocities)) if np.mean(velocities) > 0 else 0
        }
    
    def _detect_surge_events(
        self,
        velocities: np.ndarray,
        time_series: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect surge events in velocity data."""
        surge_events = []
        
        # Find periods where velocity exceeds threshold
        surge_mask = velocities > self.config.surge_velocity_threshold
        
        if not np.any(surge_mask):
            return surge_events
        
        # Find continuous surge periods
        surge_periods = self._find_continuous_periods(surge_mask, time_series)
        
        for start_idx, end_idx in surge_periods:
            duration = time_series[end_idx] - time_series[start_idx]
            
            # Filter by duration
            if (duration >= self.config.surge_duration_min and 
                duration <= self.config.surge_duration_max):
                
                event = {
                    'start_time': float(time_series[start_idx]),
                    'end_time': float(time_series[end_idx]),
                    'duration_years': float(duration),
                    'peak_velocity': float(np.max(velocities[start_idx:end_idx+1])),
                    'mean_velocity': float(np.mean(velocities[start_idx:end_idx+1])),
                    'start_index': int(start_idx),
                    'end_index': int(end_idx)
                }
                surge_events.append(event)
        
        return surge_events
    
    def _find_continuous_periods(
        self,
        mask: np.ndarray,
        time_series: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find continuous periods where mask is True."""
        periods = []
        
        if not np.any(mask):
            return periods
        
        # Find start and end indices of continuous periods
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        for start, end in zip(starts, ends):
            periods.append((start, end))
        
        return periods
    
    def _calculate_acceleration(
        self,
        velocities: np.ndarray,
        time_series: np.ndarray
    ) -> np.ndarray:
        """Calculate acceleration from velocity data."""
        if len(velocities) < 2:
            return np.array([])
        
        # Calculate time differences
        dt = np.diff(time_series)
        
        # Calculate velocity differences
        dv = np.diff(velocities)
        
        # Calculate acceleration
        acceleration = dv / dt
        
        # Pad to match original length
        acceleration = np.concatenate(([acceleration[0]], acceleration))
        
        return acceleration
    
    def _identify_anomalous_periods(
        self,
        velocities: np.ndarray,
        acceleration: np.ndarray,
        time_series: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify anomalous periods in velocity and acceleration."""
        anomalous_periods = []
        
        if len(velocities) == 0:
            return anomalous_periods
        
        # Z-score based anomaly detection
        velocity_z = np.abs(zscore(velocities))
        
        if len(acceleration) > 0:
            accel_z = np.abs(zscore(acceleration))
        else:
            accel_z = np.zeros_like(velocity_z)
        
        # Combined anomaly score
        anomaly_score = (velocity_z + accel_z) / 2
        
        # Find anomalous points (z-score > 2)
        anomalous_mask = anomaly_score > 2.0
        
        if not np.any(anomalous_mask):
            return anomalous_periods
        
        # Find continuous anomalous periods
        periods = self._find_continuous_periods(anomalous_mask, time_series)
        
        for start_idx, end_idx in periods:
            period = {
                'start_time': float(time_series[start_idx]),
                'end_time': float(time_series[end_idx]),
                'duration_years': float(time_series[end_idx] - time_series[start_idx]),
                'max_anomaly_score': float(np.max(anomaly_score[start_idx:end_idx+1])),
                'mean_velocity': float(np.mean(velocities[start_idx:end_idx+1])),
                'max_velocity': float(np.max(velocities[start_idx:end_idx+1]))
            }
            
            if len(acceleration) > 0:
                period['mean_acceleration'] = float(np.mean(acceleration[start_idx:end_idx+1]))
                period['max_acceleration'] = float(np.max(acceleration[start_idx:end_idx+1]))
            
            anomalous_periods.append(period)
        
        return anomalous_periods
    
    def _classify_current_state(
        self,
        velocities: np.ndarray,
        acceleration: np.ndarray
    ) -> Dict[str, Any]:
        """Classify current glacier state based on recent data."""
        if len(velocities) == 0:
            return {'phase': SurgePhase.UNKNOWN.value, 'confidence': 0.0}
        
        # Use last few measurements
        recent_window = min(3, len(velocities))
        recent_velocities = velocities[-recent_window:]
        recent_velocity = np.mean(recent_velocities)
        
        if len(acceleration) >= recent_window:
            recent_acceleration = np.mean(acceleration[-recent_window:])
        else:
            recent_acceleration = 0.0
        
        # Classification logic
        if recent_velocity > self.config.surge_velocity_threshold:
            if recent_acceleration > self.config.surge_acceleration_threshold:
                phase = SurgePhase.INITIATION
                confidence = 0.9
            elif recent_acceleration < -self.config.surge_acceleration_threshold:
                phase = SurgePhase.TERMINATION
                confidence = 0.8
            else:
                phase = SurgePhase.ACTIVE
                confidence = 0.85
        elif recent_velocity > self.config.normal_velocity_threshold:
            if recent_acceleration > self.config.surge_acceleration_threshold / 2:
                phase = SurgePhase.INITIATION
                confidence = 0.6
            else:
                phase = SurgePhase.POST_SURGE
                confidence = 0.7
        else:
            phase = SurgePhase.QUIESCENT
            confidence = 0.8
        
        return {
            'phase': phase.value,
            'confidence': float(confidence),
            'recent_velocity': float(recent_velocity),
            'recent_acceleration': float(recent_acceleration),
            'velocity_trend': 'increasing' if recent_acceleration > 0 else 'decreasing' if recent_acceleration < 0 else 'stable'
        }


class GeometricAnalysis:
    """Analyze geometric changes for surge detection."""
    
    def __init__(self, config: SurgeConfig = None):
        """
        Initialize geometric analysis.
        
        Args:
            config: Surge detection configuration
        """
        self.config = config or SurgeConfig()
        self.logger = logging.getLogger(__name__)
    
    def analyze_geometric_changes(
        self,
        geometric_data: Dict[str, np.ndarray],
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze geometric changes for surge indicators.
        
        Args:
            geometric_data: Dictionary with geometric measurements
            time_series: Time series (years)
            
        Returns:
            Geometric change analysis
        """
        # Terminus position analysis
        terminus_analysis = self._analyze_terminus_changes(
            geometric_data.get('terminus_position', np.array([])),
            time_series
        )
        
        # Surface elevation analysis
        elevation_analysis = self._analyze_elevation_changes(
            geometric_data.get('surface_elevation', np.array([])),
            time_series
        )
        
        # Length and area analysis
        morphology_analysis = self._analyze_morphology_changes(
            geometric_data, time_series
        )
        
        # Combined geometric indicators
        geometric_indicators = self._calculate_geometric_indicators(
            terminus_analysis, elevation_analysis, morphology_analysis
        )
        
        return {
            'terminus_analysis': terminus_analysis,
            'elevation_analysis': elevation_analysis,
            'morphology_analysis': morphology_analysis,
            'geometric_indicators': geometric_indicators
        }
    
    def _analyze_terminus_changes(
        self,
        terminus_positions: np.ndarray,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze terminus position changes."""
        if len(terminus_positions) < 2:
            return {'insufficient_data': True}
        
        # Calculate terminus advance/retreat rates
        dt = np.diff(time_series)
        dx = np.diff(terminus_positions)
        rates = dx / dt  # m/year
        
        # Statistics
        stats = {
            'mean_rate': float(np.mean(rates)),
            'std_rate': float(np.std(rates)),
            'max_advance_rate': float(np.max(rates)),
            'max_retreat_rate': float(np.min(rates)),
            'total_change': float(terminus_positions[-1] - terminus_positions[0]),
            'net_rate': float((terminus_positions[-1] - terminus_positions[0]) / (time_series[-1] - time_series[0]))
        }
        
        # Detect rapid advance events
        advance_events = []
        surge_mask = rates > self.config.terminus_advance_threshold
        
        if np.any(surge_mask):
            periods = self._find_continuous_periods(surge_mask, time_series[1:])
            
            for start_idx, end_idx in periods:
                event = {
                    'start_time': float(time_series[start_idx + 1]),
                    'end_time': float(time_series[end_idx + 1]),
                    'duration': float(time_series[end_idx + 1] - time_series[start_idx + 1]),
                    'advance_distance': float(np.sum(dx[start_idx:end_idx + 1])),
                    'mean_rate': float(np.mean(rates[start_idx:end_idx + 1])),
                    'max_rate': float(np.max(rates[start_idx:end_idx + 1]))
                }
                advance_events.append(event)
        
        return {
            'statistics': stats,
            'advance_events': advance_events,
            'rates': rates.tolist(),
            'surge_indicator': len(advance_events) > 0
        }
    
    def _analyze_elevation_changes(
        self,
        elevations: np.ndarray,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze surface elevation changes."""
        if len(elevations) < 2:
            return {'insufficient_data': True}
        
        # Calculate elevation change rates
        dt = np.diff(time_series)
        dh = np.diff(elevations)
        rates = dh / dt  # m/year
        
        # Statistics
        stats = {
            'mean_rate': float(np.mean(rates)),
            'std_rate': float(np.std(rates)),
            'max_thickening_rate': float(np.max(rates)),
            'max_thinning_rate': float(np.min(rates)),
            'total_change': float(elevations[-1] - elevations[0]),
            'net_rate': float((elevations[-1] - elevations[0]) / (time_series[-1] - time_series[0]))
        }
        
        # Detect rapid elevation changes
        rapid_changes = []
        change_mask = np.abs(rates) > self.config.surface_elevation_change_threshold
        
        if np.any(change_mask):
            periods = self._find_continuous_periods(change_mask, time_series[1:])
            
            for start_idx, end_idx in periods:
                change = {
                    'start_time': float(time_series[start_idx + 1]),
                    'end_time': float(time_series[end_idx + 1]),
                    'duration': float(time_series[end_idx + 1] - time_series[start_idx + 1]),
                    'elevation_change': float(np.sum(dh[start_idx:end_idx + 1])),
                    'mean_rate': float(np.mean(rates[start_idx:end_idx + 1])),
                    'change_type': 'thickening' if np.mean(rates[start_idx:end_idx + 1]) > 0 else 'thinning'
                }
                rapid_changes.append(change)
        
        return {
            'statistics': stats,
            'rapid_changes': rapid_changes,
            'rates': rates.tolist(),
            'surge_indicator': any(change['change_type'] == 'thickening' for change in rapid_changes)
        }
    
    def _analyze_morphology_changes(
        self,
        geometric_data: Dict[str, np.ndarray],
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze glacier morphology changes."""
        morphology_analysis = {}
        
        # Analyze length changes
        if 'length' in geometric_data and len(geometric_data['length']) > 1:
            length_data = geometric_data['length']
            dt = np.diff(time_series)
            dl = np.diff(length_data)
            length_rates = dl / dt
            
            morphology_analysis['length'] = {
                'mean_rate': float(np.mean(length_rates)),
                'total_change': float(length_data[-1] - length_data[0]),
                'rates': length_rates.tolist()
            }
        
        # Analyze area changes
        if 'area' in geometric_data and len(geometric_data['area']) > 1:
            area_data = geometric_data['area']
            dt = np.diff(time_series)
            da = np.diff(area_data)
            area_rates = da / dt
            
            morphology_analysis['area'] = {
                'mean_rate': float(np.mean(area_rates)),
                'total_change': float(area_data[-1] - area_data[0]),
                'rates': area_rates.tolist()
            }
        
        # Analyze width changes
        if 'width' in geometric_data and len(geometric_data['width']) > 1:
            width_data = geometric_data['width']
            dt = np.diff(time_series)
            dw = np.diff(width_data)
            width_rates = dw / dt
            
            morphology_analysis['width'] = {
                'mean_rate': float(np.mean(width_rates)),
                'total_change': float(width_data[-1] - width_data[0]),
                'rates': width_rates.tolist()
            }
        
        return morphology_analysis
    
    def _find_continuous_periods(
        self,
        mask: np.ndarray,
        time_series: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Find continuous periods where mask is True."""
        periods = []
        
        if not np.any(mask):
            return periods
        
        # Find start and end indices of continuous periods
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        for start, end in zip(starts, ends):
            periods.append((start, end))
        
        return periods
    
    def _calculate_geometric_indicators(
        self,
        terminus_analysis: Dict[str, Any],
        elevation_analysis: Dict[str, Any],
        morphology_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate combined geometric surge indicators."""
        indicators = {
            'terminus_surge_indicator': terminus_analysis.get('surge_indicator', False),
            'elevation_surge_indicator': elevation_analysis.get('surge_indicator', False)
        }
        
        # Count positive indicators
        positive_indicators = sum(indicators.values())
        
        # Overall geometric surge probability
        geometric_surge_probability = positive_indicators / len(indicators)
        
        indicators.update({
            'positive_indicators': positive_indicators,
            'total_indicators': len(indicators),
            'geometric_surge_probability': geometric_surge_probability,
            'surge_likely': geometric_surge_probability > 0.5
        })
        
        return indicators


class SurgePredictor:
    """Main class for glacier surge prediction and analysis."""
    
    def __init__(self, config: SurgeConfig = None):
        """
        Initialize surge predictor.
        
        Args:
            config: Surge detection configuration
        """
        self.config = config or SurgeConfig()
        self.velocity_analyzer = VelocityAnalysis(self.config)
        self.geometric_analyzer = GeometricAnalysis(self.config)
        self.logger = logging.getLogger(__name__)
    
    def predict_surge_probability(
        self,
        glacier: GlacierCharacteristics,
        velocity_data: Dict[str, np.ndarray],
        geometric_data: Dict[str, np.ndarray],
        environmental_data: Dict[str, np.ndarray],
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict glacier surge probability based on multiple indicators.
        
        Args:
            glacier: Glacier characteristics
            velocity_data: Velocity measurements
            geometric_data: Geometric measurements
            environmental_data: Environmental conditions
            time_series: Time series (years)
            
        Returns:
            Surge probability prediction
        """
        # Velocity analysis
        velocity_analysis = self.velocity_analyzer.detect_velocity_anomalies(
            velocity_data, time_series
        )
        
        # Geometric analysis
        geometric_analysis = self.geometric_analyzer.analyze_geometric_changes(
            geometric_data, time_series
        )
        
        # Environmental analysis
        environmental_analysis = self._analyze_environmental_factors(
            environmental_data, time_series
        )
        
        # Historical analysis
        historical_analysis = self._analyze_historical_patterns(glacier)
        
        # Calculate surge probability
        surge_probability = self._calculate_surge_probability(
            velocity_analysis,
            geometric_analysis,
            environmental_analysis,
            historical_analysis
        )
        
        # Determine surge type
        surge_type = self._determine_surge_type(
            velocity_analysis,
            environmental_analysis,
            glacier
        )
        
        # Risk assessment
        risk_assessment = self._assess_surge_risk(
            surge_probability, glacier, velocity_analysis
        )
        
        return {
            'glacier_id': glacier.glacier_id,
            'surge_probability': surge_probability,
            'surge_type': surge_type,
            'risk_assessment': risk_assessment,
            'velocity_analysis': velocity_analysis,
            'geometric_analysis': geometric_analysis,
            'environmental_analysis': environmental_analysis,
            'historical_analysis': historical_analysis,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_environmental_factors(
        self,
        environmental_data: Dict[str, np.ndarray],
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze environmental factors affecting surge probability."""
        analysis = {}
        
        # Temperature analysis
        if 'temperature' in environmental_data:
            temp_data = environmental_data['temperature']
            temp_analysis = self._analyze_temperature_trends(temp_data, time_series)
            analysis['temperature'] = temp_analysis
        
        # Precipitation analysis
        if 'precipitation' in environmental_data:
            precip_data = environmental_data['precipitation']
            precip_analysis = self._analyze_precipitation_patterns(precip_data, time_series)
            analysis['precipitation'] = precip_analysis
        
        # Discharge analysis
        if 'discharge' in environmental_data:
            discharge_data = environmental_data['discharge']
            discharge_analysis = self._analyze_discharge_patterns(discharge_data, time_series)
            analysis['discharge'] = discharge_analysis
        
        return analysis
    
    def _analyze_temperature_trends(
        self,
        temperature_data: np.ndarray,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze temperature trends."""
        if len(temperature_data) < 2:
            return {'insufficient_data': True}
        
        # Calculate temperature statistics
        temp_stats = {
            'mean_temperature': float(np.mean(temperature_data)),
            'temperature_trend': float(np.polyfit(time_series, temperature_data, 1)[0]),
            'temperature_variability': float(np.std(temperature_data))
        }
        
        # Detect temperature anomalies
        temp_anomalies = np.abs(temperature_data - np.mean(temperature_data)) > self.config.temperature_anomaly_threshold
        
        # Recent warming events
        recent_warming = []
        if len(temperature_data) >= 3:
            recent_trend = np.polyfit(time_series[-3:], temperature_data[-3:], 1)[0]
            if recent_trend > 0.5:  # Significant warming
                recent_warming.append({
                    'trend': float(recent_trend),
                    'period': f"{time_series[-3]:.1f}-{time_series[-1]:.1f}"
                })
        
        return {
            'statistics': temp_stats,
            'anomaly_count': int(np.sum(temp_anomalies)),
            'recent_warming_events': recent_warming,
            'surge_indicator': len(recent_warming) > 0 or temp_stats['temperature_trend'] > 0.3
        }
    
    def _analyze_precipitation_patterns(
        self,
        precipitation_data: np.ndarray,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze precipitation patterns."""
        if len(precipitation_data) < 2:
            return {'insufficient_data': True}
        
        # Calculate precipitation statistics
        precip_stats = {
            'mean_precipitation': float(np.mean(precipitation_data)),
            'precipitation_trend': float(np.polyfit(time_series, precipitation_data, 1)[0]),
            'precipitation_variability': float(np.std(precipitation_data))
        }
        
        # Detect precipitation anomalies
        z_scores = np.abs(zscore(precipitation_data))
        precip_anomalies = z_scores > self.config.precipitation_anomaly_threshold
        
        return {
            'statistics': precip_stats,
            'anomaly_count': int(np.sum(precip_anomalies)),
            'surge_indicator': precip_stats['precipitation_trend'] > 50  # Increasing precipitation
        }
    
    def _analyze_discharge_patterns(
        self,
        discharge_data: np.ndarray,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze discharge patterns."""
        if len(discharge_data) < 2:
            return {'insufficient_data': True}
        
        # Calculate discharge statistics
        discharge_stats = {
            'mean_discharge': float(np.mean(discharge_data)),
            'discharge_trend': float(np.polyfit(time_series, discharge_data, 1)[0]),
            'discharge_variability': float(np.std(discharge_data))
        }
        
        # Detect discharge anomalies
        z_scores = np.abs(zscore(discharge_data))
        discharge_anomalies = z_scores > self.config.discharge_anomaly_threshold
        
        return {
            'statistics': discharge_stats,
            'anomaly_count': int(np.sum(discharge_anomalies)),
            'surge_indicator': discharge_stats['discharge_variability'] > np.mean(discharge_data) * 0.5
        }
    
    def _analyze_historical_patterns(self, glacier: GlacierCharacteristics) -> Dict[str, Any]:
        """Analyze historical surge patterns."""
        analysis = {
            'has_surge_history': glacier.last_surge_year is not None,
            'last_surge_year': glacier.last_surge_year,
            'surge_frequency': glacier.surge_frequency
        }
        
        if glacier.last_surge_year is not None:
            current_year = datetime.now().year
            years_since_last = current_year - glacier.last_surge_year
            analysis['years_since_last_surge'] = years_since_last
            
            if glacier.surge_frequency is not None:
                expected_next_surge = glacier.last_surge_year + glacier.surge_frequency
                analysis['expected_next_surge_year'] = expected_next_surge
                analysis['overdue_years'] = max(0, current_year - expected_next_surge)
                analysis['surge_cycle_position'] = years_since_last / glacier.surge_frequency
            
        if glacier.surge_history:
            analysis['total_known_surges'] = len(glacier.surge_history)
            if len(glacier.surge_history) > 1:
                intervals = np.diff(glacier.surge_history)
                analysis['mean_surge_interval'] = float(np.mean(intervals))
                analysis['surge_interval_variability'] = float(np.std(intervals))
        
        return analysis
    
    def _calculate_surge_probability(
        self,
        velocity_analysis: Dict[str, Any],
        geometric_analysis: Dict[str, Any],
        environmental_analysis: Dict[str, Any],
        historical_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate overall surge probability."""
        # Velocity component
        velocity_score = 0.0
        if velocity_analysis.get('anomalies_detected', False):
            velocity_score += 0.5
        
        current_state = velocity_analysis.get('current_state', {})
        if current_state.get('phase') in ['initiation', 'active']:
            velocity_score += 0.3 * current_state.get('confidence', 0)
        
        # Geometric component
        geometric_score = 0.0
        geometric_indicators = geometric_analysis.get('geometric_indicators', {})
        geometric_score = geometric_indicators.get('geometric_surge_probability', 0.0)
        
        # Environmental component
        environmental_score = 0.0
        for factor_analysis in environmental_analysis.values():
            if isinstance(factor_analysis, dict) and factor_analysis.get('surge_indicator', False):
                environmental_score += 0.2
        environmental_score = min(environmental_score, 1.0)
        
        # Historical component
        historical_score = 0.0
        if historical_analysis.get('has_surge_history', False):
            if 'surge_cycle_position' in historical_analysis:
                cycle_pos = historical_analysis['surge_cycle_position']
                # Higher probability as we approach expected surge time
                if cycle_pos > 0.8:
                    historical_score = 0.8
                elif cycle_pos > 0.6:
                    historical_score = 0.5
                else:
                    historical_score = 0.2
            else:
                historical_score = 0.3  # Default for glaciers with surge history
        
        # Weighted combination
        overall_probability = (
            self.config.velocity_weight * velocity_score +
            self.config.geometric_weight * geometric_score +
            self.config.thermal_weight * environmental_score +
            self.config.hydrological_weight * environmental_score +
            self.config.historical_weight * historical_score
        )
        
        return {
            'overall_probability': float(min(overall_probability, 1.0)),
            'velocity_component': float(velocity_score),
            'geometric_component': float(geometric_score),
            'environmental_component': float(environmental_score),
            'historical_component': float(historical_score)
        }
    
    def _determine_surge_type(self, velocity_analysis, environmental_analysis, glacier) -> SurgeType:
        """Determine the most likely surge type."""
        # Check for thermal indicators
        thermal_indicators = 0
        if 'temperature' in environmental_analysis:
            temp_analysis = environmental_analysis['temperature']
            if temp_analysis.get('surge_indicator', False):
                thermal_indicators += 1
        
        # Check for hydrological indicators
        hydro_indicators = 0
        for factor in ['precipitation', 'discharge']:
            if factor in environmental_analysis:
                if environmental_analysis[factor].get('surge_indicator', False):
                    hydro_indicators += 1
        
        # Determine type based on indicators
        if thermal_indicators > 0 and hydro_indicators > 0:
            return SurgeType.MIXED
        elif thermal_indicators > 0:
            return SurgeType.THERMAL
        elif hydro_indicators > 0:
            return SurgeType.HYDROLOGICAL
        elif glacier.bed_slope > 10:  # Steep bed suggests structural control
            return SurgeType.STRUCTURAL
        else:
            return SurgeType.UNKNOWN
    
    def _assess_surge_risk(self, surge_probability, glacier, velocity_analysis) -> Dict[str, Any]:
        """Assess risk level and potential impacts."""
        prob = surge_probability['overall_probability']
        
        # Determine risk level
        if prob >= 0.8:
            risk_level = RiskLevel.EXTREME
        elif prob >= 0.6:
            risk_level = RiskLevel.VERY_HIGH
        elif prob >= 0.4:
            risk_level = RiskLevel.HIGH
        elif prob >= 0.2:
            risk_level = RiskLevel.MODERATE
        elif prob >= 0.1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW
        
        # Estimate potential impacts
        current_state = velocity_analysis.get('current_state', {})
        recent_velocity = current_state.get('recent_velocity', 0)
        
        # Estimate surge magnitude based on glacier size and current velocity
        potential_advance = glacier.length * 0.1 * prob  # Rough estimate
        potential_velocity_increase = recent_velocity * (1 + 2 * prob)
        
        return {
            'risk_level': risk_level.value,
            'risk_score': float(prob),
            'potential_terminus_advance_km': float(potential_advance),
            'potential_peak_velocity_m_per_year': float(potential_velocity_increase),
            'confidence': float(min(prob + 0.2, 1.0)),
            'monitoring_priority': 'high' if prob > 0.4 else 'medium' if prob > 0.2 else 'low'
        }


def create_surge_predictor(
    surge_velocity_threshold: float = 100.0,
    **kwargs
) -> SurgePredictor:
    """
    Create a glacier surge predictor with specified configuration.
    
    Args:
        surge_velocity_threshold: Velocity threshold for surge detection (m/year)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured surge predictor
    """
    config = SurgeConfig(
        surge_velocity_threshold=surge_velocity_threshold,
        **kwargs
    )
    return SurgePredictor(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create example glacier
    glacier = GlacierCharacteristics(
        glacier_id="SURGE_GLACIER_001",
        glacier_name="Example Surge Glacier",
        length=15.0,  # km
        width=2.5,  # km
        area=37.5,  # km²
        volume=3.75,  # km³
        terminus_elevation=4200,  # m
        accumulation_area_elevation=5800,  # m
        equilibrium_line_altitude=5200,  # m
        mean_temperature=-8.0,  # °C
        temperature_gradient=6.5,  # °C/km
        drainage_area=150.0,  # km²
        annual_precipitation=800,  # mm
        melt_rate=5.0,  # mm/day
        bed_slope=8.0,  # degrees
        surface_slope=12.0,  # degrees
        ice_thickness=150.0,  # m
        latitude=35.5,
        longitude=76.2,
        last_surge_year=1995,
        surge_frequency=25,  # years
        surge_history=[1920, 1945, 1970, 1995]
    )
    
    # Create predictor
    predictor = create_surge_predictor(
        surge_velocity_threshold=100.0,
        normal_velocity_threshold=50.0
    )
    
    # Generate example data
    years = np.arange(2000, 2024)
    n_years = len(years)
    
    # Simulate velocity data with surge-like behavior
    base_velocity = 30 + 10 * np.sin(0.3 * years) + np.random.normal(0, 5, n_years)
    # Add surge event around 2020
    surge_mask = (years >= 2019) & (years <= 2022)
    base_velocity[surge_mask] += 150 * np.exp(-0.5 * ((years[surge_mask] - 2020.5) / 1.5) ** 2)
    
    velocity_data = {
        'velocity': base_velocity
    }
    
    # Simulate geometric data
    terminus_position = 1000 + np.cumsum(base_velocity / 10 + np.random.normal(0, 2, n_years))
    surface_elevation = 4500 + 5 * np.sin(0.2 * years) + np.random.normal(0, 1, n_years)
    
    geometric_data = {
        'terminus_position': terminus_position,
        'surface_elevation': surface_elevation,
        'length': glacier.length + 0.1 * (terminus_position - terminus_position[0]) / 1000,
        'area': glacier.area + 0.5 * (terminus_position - terminus_position[0]) / 1000
    }
    
    # Simulate environmental data
    temperature = -5 + 2 * np.sin(0.1 * years) + np.random.normal(0, 0.5, n_years)
    precipitation = 800 + 100 * np.sin(0.15 * years) + np.random.normal(0, 20, n_years)
    discharge = 50 + 20 * np.sin(0.2 * years) + 0.1 * base_velocity + np.random.normal(0, 3, n_years)
    
    environmental_data = {
        'temperature': temperature,
        'precipitation': precipitation,
        'discharge': discharge
    }
    
    # Perform surge prediction
    print("Performing glacier surge prediction...")
    prediction = predictor.predict_surge_probability(
        glacier, velocity_data, geometric_data, environmental_data, years
    )
    
    print("\nGlacier Surge Prediction Results:")
    print("=" * 40)
    print(f"Glacier: {glacier.glacier_name} ({glacier.glacier_id})")
    
    surge_prob = prediction['surge_probability']
    print(f"\nSurge Probability: {surge_prob['overall_probability']:.3f}")
    print(f"Surge Type: {prediction['surge_type'].value.replace('_', ' ').title()}")
    
    risk = prediction['risk_assessment']
    print(f"Risk Level: {risk['risk_level'].upper()}")
    print(f"Monitoring Priority: {risk['monitoring_priority'].upper()}")
    
    print("\nProbability Components:")
    print(f"  Velocity: {surge_prob['velocity_component']:.3f}")
    print(f"  Geometric: {surge_prob['geometric_component']:.3f}")
    print(f"  Environmental: {surge_prob['environmental_component']:.3f}")
    print(f"  Historical: {surge_prob['historical_component']:.3f}")
    
    # Current state analysis
    current_state = prediction['velocity_analysis']['current_state']
    print(f"\nCurrent State:")
    print(f"  Phase: {current_state['phase'].replace('_', ' ').title()}")
    print(f"  Confidence: {current_state['confidence']:.3f}")
    print(f"  Recent Velocity: {current_state['recent_velocity']:.1f} m/year")
    print(f"  Velocity Trend: {current_state['velocity_trend'].title()}")
    
    # Historical context
    historical = prediction['historical_analysis']
    if historical['has_surge_history']:
        print(f"\nHistorical Context:")
        print(f"  Last Surge: {historical['last_surge_year']}")
        print(f"  Years Since Last: {historical['years_since_last_surge']}")
        if 'expected_next_surge_year' in historical:
            print(f"  Expected Next Surge: {historical['expected_next_surge_year']}")
            print(f"  Cycle Position: {historical['surge_cycle_position']:.2f}")
    
    # Detected events
    velocity_analysis = prediction['velocity_analysis']
    if velocity_analysis.get('anomalies_detected', False):
        surge_events = velocity_analysis.get('surge_events', [])
        print(f"\nDetected Surge Events: {len(surge_events)}")
        for i, event in enumerate(surge_events):
            print(f"  Event {i+1}: {event['start_time']:.1f}-{event['end_time']:.1f} ")
            print(f"    Duration: {event['duration_years']:.1f} years")
            print(f"    Peak Velocity: {event['peak_velocity']:.1f} m/year")
    
    # Potential impacts
    print(f"\nPotential Impacts:")
    print(f"  Terminus Advance: {risk['potential_terminus_advance_km']:.2f} km")
    print(f"  Peak Velocity: {risk['potential_peak_velocity_m_per_year']:.0f} m/year")
    
    print(f"\nGlacier surge analysis completed for {glacier.glacier_name}")