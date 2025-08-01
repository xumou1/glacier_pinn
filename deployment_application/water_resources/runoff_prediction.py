"""Runoff Estimation Module.

This module provides functionality for estimating glacier melt runoff
and water resource contributions from glacier dynamics predictions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class RunoffConfig:
    """Configuration for runoff estimation."""
    # Physical constants
    ice_density: float = 917.0  # kg/m³
    water_density: float = 1000.0  # kg/m³
    latent_heat_fusion: float = 334000.0  # J/kg
    
    # Temperature parameters
    degree_day_factor: float = 4.0  # mm/°C/day
    temperature_threshold: float = 0.0  # °C
    
    # Spatial parameters
    grid_resolution: float = 100.0  # meters
    elevation_lapse_rate: float = 0.0065  # °C/m
    
    # Temporal parameters
    time_step: float = 1.0  # days
    melt_season_start: int = 120  # day of year (May 1)
    melt_season_end: int = 273  # day of year (Sep 30)


class GlacierMeltModel:
    """Model for glacier melt processes."""
    
    def __init__(self, config: RunoffConfig = None):
        """
        Initialize glacier melt model.
        
        Args:
            config: Runoff configuration
        """
        self.config = config or RunoffConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_melt_rate(
        self,
        temperature: jnp.ndarray,
        ice_thickness: jnp.ndarray,
        elevation: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Calculate glacier melt rate using degree-day method.
        
        Args:
            temperature: Air temperature (°C)
            ice_thickness: Ice thickness (m)
            elevation: Elevation (m, optional)
            
        Returns:
            Melt rate (mm/day)
        """
        # Adjust temperature for elevation if provided
        if elevation is not None:
            # Assume reference elevation of 4000m
            temp_adjusted = temperature - (elevation - 4000) * self.config.elevation_lapse_rate
        else:
            temp_adjusted = temperature
        
        # Calculate positive degree days
        positive_temp = jnp.maximum(temp_adjusted - self.config.temperature_threshold, 0.0)
        
        # Calculate melt rate
        melt_rate = self.config.degree_day_factor * positive_temp
        
        # Limit melt rate by available ice
        max_melt = ice_thickness * 1000  # Convert m to mm
        melt_rate = jnp.minimum(melt_rate, max_melt)
        
        return melt_rate
    
    def calculate_energy_balance_melt(
        self,
        net_radiation: jnp.ndarray,
        sensible_heat: jnp.ndarray,
        latent_heat: jnp.ndarray,
        ground_heat: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Calculate melt rate using energy balance method.
        
        Args:
            net_radiation: Net radiation (W/m²)
            sensible_heat: Sensible heat flux (W/m²)
            latent_heat: Latent heat flux (W/m²)
            ground_heat: Ground heat flux (W/m², optional)
            
        Returns:
            Melt rate (mm/day)
        """
        if ground_heat is None:
            ground_heat = jnp.zeros_like(net_radiation)
        
        # Total energy available for melting (W/m²)
        melt_energy = net_radiation + sensible_heat + latent_heat + ground_heat
        
        # Convert to melt rate (mm/day)
        # Energy (W/m²) * seconds_per_day / (latent_heat_fusion * water_density / 1000)
        melt_rate = (melt_energy * 86400) / (self.config.latent_heat_fusion * self.config.water_density / 1000)
        
        # Only positive melt rates
        melt_rate = jnp.maximum(melt_rate, 0.0)
        
        return melt_rate
    
    def calculate_ice_volume_change(
        self,
        velocity_x: jnp.ndarray,
        velocity_y: jnp.ndarray,
        thickness: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        """
        Calculate ice volume change due to flow dynamics.
        
        Args:
            velocity_x: X velocity component (m/year)
            velocity_y: Y velocity component (m/year)
            thickness: Ice thickness (m)
            dx: Grid spacing in x direction (m)
            dy: Grid spacing in y direction (m)
            
        Returns:
            Volume change rate (m/year)
        """
        # Calculate flux divergence
        flux_x = velocity_x * thickness
        flux_y = velocity_y * thickness
        
        # Compute gradients
        dflux_dx = jnp.gradient(flux_x, dx, axis=1)
        dflux_dy = jnp.gradient(flux_y, dy, axis=0)
        
        # Volume change (negative divergence)
        volume_change = -(dflux_dx + dflux_dy)
        
        return volume_change


class SnowMeltModel:
    """Model for snow melt processes."""
    
    def __init__(self, config: RunoffConfig = None):
        """
        Initialize snow melt model.
        
        Args:
            config: Runoff configuration
        """
        self.config = config or RunoffConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_snow_melt(
        self,
        temperature: jnp.ndarray,
        snow_depth: jnp.ndarray,
        solar_radiation: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Calculate snow melt rate.
        
        Args:
            temperature: Air temperature (°C)
            snow_depth: Snow depth (m)
            solar_radiation: Solar radiation (W/m², optional)
            
        Returns:
            Snow melt rate (mm/day)
        """
        # Basic degree-day method
        positive_temp = jnp.maximum(temperature - self.config.temperature_threshold, 0.0)
        base_melt = self.config.degree_day_factor * positive_temp * 0.5  # Snow melts faster than ice
        
        # Radiation enhancement if provided
        if solar_radiation is not None:
            radiation_factor = 1.0 + (solar_radiation / 300.0) * 0.2  # Simple enhancement
            base_melt = base_melt * radiation_factor
        
        # Limit by available snow
        max_snow_melt = snow_depth * 1000  # Convert m to mm
        snow_melt = jnp.minimum(base_melt, max_snow_melt)
        
        return snow_melt


class RunoffEstimator:
    """Main class for runoff estimation from glacier dynamics."""
    
    def __init__(
        self,
        glacier_melt_model: GlacierMeltModel = None,
        snow_melt_model: SnowMeltModel = None,
        config: RunoffConfig = None
    ):
        """
        Initialize runoff estimator.
        
        Args:
            glacier_melt_model: Glacier melt model
            snow_melt_model: Snow melt model
            config: Runoff configuration
        """
        self.config = config or RunoffConfig()
        self.glacier_melt_model = glacier_melt_model or GlacierMeltModel(self.config)
        self.snow_melt_model = snow_melt_model or SnowMeltModel(self.config)
        self.logger = logging.getLogger(__name__)
    
    def estimate_daily_runoff(
        self,
        glacier_predictions: Dict[str, jnp.ndarray],
        meteorological_data: Dict[str, jnp.ndarray],
        catchment_area: float,
        day_of_year: int
    ) -> Dict[str, float]:
        """
        Estimate daily runoff from glacier and meteorological data.
        
        Args:
            glacier_predictions: Glacier dynamics predictions
            meteorological_data: Meteorological forcing data
            catchment_area: Catchment area (km²)
            day_of_year: Day of year (1-365)
            
        Returns:
            Dictionary containing runoff estimates
        """
        # Extract glacier data
        thickness = glacier_predictions['thickness']
        velocity_x = glacier_predictions.get('velocity_x', jnp.zeros_like(thickness))
        velocity_y = glacier_predictions.get('velocity_y', jnp.zeros_like(thickness))
        
        # Extract meteorological data
        temperature = meteorological_data['temperature']
        precipitation = meteorological_data.get('precipitation', jnp.zeros_like(temperature))
        snow_depth = meteorological_data.get('snow_depth', jnp.zeros_like(temperature))
        
        # Calculate glacier melt
        glacier_melt = self.glacier_melt_model.calculate_melt_rate(
            temperature, thickness
        )
        
        # Calculate snow melt
        snow_melt = self.snow_melt_model.calculate_snow_melt(
            temperature, snow_depth
        )
        
        # Calculate total melt water
        total_melt = glacier_melt + snow_melt
        
        # Add precipitation contribution
        liquid_precipitation = jnp.where(
            temperature > 0,
            precipitation,
            0.0  # Snow doesn't contribute immediately
        )
        
        # Total water input (mm/day)
        total_water = total_melt + liquid_precipitation
        
        # Convert to volumetric flow (m³/s)
        # mm/day * km² * 1000 m²/km² * 1 m/1000 mm * 1 day/86400 s
        runoff_volume = jnp.sum(total_water) * catchment_area * 1000 / 86400
        
        # Calculate components
        glacier_contribution = jnp.sum(glacier_melt) * catchment_area * 1000 / 86400
        snow_contribution = jnp.sum(snow_melt) * catchment_area * 1000 / 86400
        rain_contribution = jnp.sum(liquid_precipitation) * catchment_area * 1000 / 86400
        
        return {
            'total_runoff': float(runoff_volume),  # m³/s
            'glacier_runoff': float(glacier_contribution),  # m³/s
            'snow_runoff': float(snow_contribution),  # m³/s
            'rain_runoff': float(rain_contribution),  # m³/s
            'glacier_fraction': float(glacier_contribution / runoff_volume) if runoff_volume > 0 else 0.0,
            'day_of_year': day_of_year,
            'is_melt_season': self._is_melt_season(day_of_year)
        }
    
    def estimate_seasonal_runoff(
        self,
        glacier_predictions_series: List[Dict[str, jnp.ndarray]],
        meteorological_series: List[Dict[str, jnp.ndarray]],
        catchment_area: float,
        start_day: int = 1
    ) -> Dict[str, List[float]]:
        """
        Estimate seasonal runoff patterns.
        
        Args:
            glacier_predictions_series: Time series of glacier predictions
            meteorological_series: Time series of meteorological data
            catchment_area: Catchment area (km²)
            start_day: Starting day of year
            
        Returns:
            Dictionary containing seasonal runoff time series
        """
        daily_runoff = []
        
        for i, (glacier_data, met_data) in enumerate(zip(glacier_predictions_series, meteorological_series)):
            day_of_year = (start_day + i - 1) % 365 + 1
            
            runoff = self.estimate_daily_runoff(
                glacier_data, met_data, catchment_area, day_of_year
            )
            daily_runoff.append(runoff)
        
        # Aggregate results
        result = {
            'total_runoff': [r['total_runoff'] for r in daily_runoff],
            'glacier_runoff': [r['glacier_runoff'] for r in daily_runoff],
            'snow_runoff': [r['snow_runoff'] for r in daily_runoff],
            'rain_runoff': [r['rain_runoff'] for r in daily_runoff],
            'glacier_fraction': [r['glacier_fraction'] for r in daily_runoff],
            'days_of_year': [r['day_of_year'] for r in daily_runoff]
        }
        
        # Calculate seasonal statistics
        result['seasonal_stats'] = self._calculate_seasonal_stats(result)
        
        return result
    
    def estimate_annual_water_yield(
        self,
        seasonal_runoff: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Estimate annual water yield from seasonal runoff.
        
        Args:
            seasonal_runoff: Seasonal runoff data
            
        Returns:
            Annual water yield statistics
        """
        total_runoff = seasonal_runoff['total_runoff']
        glacier_runoff = seasonal_runoff['glacier_runoff']
        
        # Convert m³/s to annual volume (m³)
        seconds_per_year = 365.25 * 24 * 3600
        
        annual_volume = np.sum(total_runoff) * seconds_per_year / len(total_runoff)
        glacier_volume = np.sum(glacier_runoff) * seconds_per_year / len(glacier_runoff)
        
        return {
            'annual_volume': annual_volume,  # m³/year
            'glacier_volume': glacier_volume,  # m³/year
            'glacier_contribution_fraction': glacier_volume / annual_volume if annual_volume > 0 else 0.0,
            'peak_flow': np.max(total_runoff),  # m³/s
            'low_flow': np.min(total_runoff),  # m³/s
            'mean_flow': np.mean(total_runoff),  # m³/s
            'flow_variability': np.std(total_runoff) / np.mean(total_runoff) if np.mean(total_runoff) > 0 else 0.0
        }
    
    def _is_melt_season(self, day_of_year: int) -> bool:
        """Check if day is in melt season."""
        return self.config.melt_season_start <= day_of_year <= self.config.melt_season_end
    
    def _calculate_seasonal_stats(self, runoff_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate seasonal statistics."""
        total_runoff = np.array(runoff_data['total_runoff'])
        glacier_runoff = np.array(runoff_data['glacier_runoff'])
        days = np.array(runoff_data['days_of_year'])
        
        # Define seasons
        seasons = {
            'spring': (60, 151),   # Mar-May
            'summer': (152, 243),  # Jun-Aug
            'autumn': (244, 334),  # Sep-Nov
            'winter': [(335, 365), (1, 59)]  # Dec-Feb
        }
        
        seasonal_stats = {}
        
        for season, period in seasons.items():
            if isinstance(period, list):  # Winter spans year boundary
                mask = np.logical_or(
                    (days >= period[0][0]) & (days <= period[0][1]),
                    (days >= period[1][0]) & (days <= period[1][1])
                )
            else:
                mask = (days >= period[0]) & (days <= period[1])
            
            if np.any(mask):
                seasonal_stats[season] = {
                    'mean_total_runoff': float(np.mean(total_runoff[mask])),
                    'mean_glacier_runoff': float(np.mean(glacier_runoff[mask])),
                    'max_runoff': float(np.max(total_runoff[mask])),
                    'min_runoff': float(np.min(total_runoff[mask]))
                }
            else:
                seasonal_stats[season] = {
                    'mean_total_runoff': 0.0,
                    'mean_glacier_runoff': 0.0,
                    'max_runoff': 0.0,
                    'min_runoff': 0.0
                }
        
        return seasonal_stats


def create_runoff_estimator(
    config: Optional[RunoffConfig] = None,
    **kwargs
) -> RunoffEstimator:
    """
    Create a runoff estimator with specified configuration.
    
    Args:
        config: Runoff configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured runoff estimator
    """
    if config is None:
        config = RunoffConfig(**kwargs)
    
    glacier_model = GlacierMeltModel(config)
    snow_model = SnowMeltModel(config)
    
    return RunoffEstimator(glacier_model, snow_model, config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create runoff estimator
    estimator = create_runoff_estimator()
    
    # Mock glacier predictions
    x, y = np.meshgrid(np.linspace(0, 5000, 50), np.linspace(0, 3000, 30))
    mock_glacier = {
        'thickness': jnp.array(100 + 50 * np.sin(x/1000) * np.cos(y/1000)),
        'velocity_x': jnp.array(10 * np.ones_like(x)),
        'velocity_y': jnp.array(5 * np.ones_like(x))
    }
    
    # Mock meteorological data
    mock_met = {
        'temperature': jnp.array(5.0 * np.ones_like(x)),  # 5°C
        'precipitation': jnp.array(2.0 * np.ones_like(x)),  # 2 mm/day
        'snow_depth': jnp.array(0.5 * np.ones_like(x))  # 0.5 m
    }
    
    # Estimate daily runoff
    catchment_area = 100.0  # km²
    runoff = estimator.estimate_daily_runoff(
        mock_glacier, mock_met, catchment_area, 150  # Day 150 (May 30)
    )
    
    print("Daily runoff estimation:")
    for key, value in runoff.items():
        print(f"  {key}: {value}")
    
    # Estimate seasonal runoff (simplified example)
    seasonal_data = []
    for day in range(120, 274):  # Melt season
        # Vary temperature seasonally
        temp_variation = 5.0 + 10.0 * np.sin((day - 120) * np.pi / 154)
        mock_met_daily = {
            'temperature': jnp.array(temp_variation * np.ones_like(x)),
            'precipitation': jnp.array(2.0 * np.ones_like(x)),
            'snow_depth': jnp.array(np.maximum(0.5 - (day - 120) * 0.003, 0.0) * np.ones_like(x))
        }
        
        daily_runoff = estimator.estimate_daily_runoff(
            mock_glacier, mock_met_daily, catchment_area, day
        )
        seasonal_data.append(daily_runoff)
    
    # Extract time series
    days = [r['day_of_year'] for r in seasonal_data]
    total_runoff = [r['total_runoff'] for r in seasonal_data]
    glacier_runoff = [r['glacier_runoff'] for r in seasonal_data]
    
    print(f"\nSeasonal runoff summary:")
    print(f"  Peak total runoff: {max(total_runoff):.2f} m³/s")
    print(f"  Mean glacier contribution: {np.mean([r['glacier_fraction'] for r in seasonal_data]):.2%}")