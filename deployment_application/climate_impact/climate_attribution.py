"""Climate Scenario Analysis Module.

This module provides functionality for generating and analyzing
climate scenarios for glacier impact assessment.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import stats, interpolate
import json


class EmissionScenario(Enum):
    """IPCC emission scenarios."""
    SSP1_1_9 = "ssp1-1.9"  # Very low emissions
    SSP1_2_6 = "ssp1-2.6"  # Low emissions
    SSP2_4_5 = "ssp2-4.5"  # Intermediate emissions
    SSP3_7_0 = "ssp3-7.0"  # High emissions
    SSP5_8_5 = "ssp5-8.5"  # Very high emissions


class TimeHorizon(Enum):
    """Time horizons for projections."""
    NEAR_TERM = "2021-2040"  # Near-term (20 years)
    MID_TERM = "2041-2060"   # Mid-term (40 years)
    LONG_TERM = "2081-2100"  # Long-term (80 years)


class ConfidenceLevel(Enum):
    """Confidence levels for projections."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ClimateVariable:
    """Climate variable definition."""
    name: str
    units: str
    description: str
    baseline_value: float
    uncertainty_range: Tuple[float, float]  # (min, max)
    seasonal_pattern: Optional[List[float]] = None  # 12 monthly values
    spatial_pattern: Optional[Dict[str, float]] = None  # elevation/aspect adjustments


@dataclass
class ClimateProjection:
    """Climate projection for a specific variable and scenario."""
    variable: ClimateVariable
    emission_scenario: EmissionScenario
    time_horizon: TimeHorizon
    
    # Projected changes
    mean_change: float  # Mean change from baseline
    uncertainty_range: Tuple[float, float]  # (min, max) change
    confidence_level: ConfidenceLevel
    
    # Temporal patterns
    annual_values: List[float]  # Annual values for projection period
    seasonal_changes: List[float]  # 12 monthly change factors
    
    # Spatial patterns
    elevation_gradient: float  # Change per 100m elevation
    aspect_adjustments: Dict[str, float]  # N, S, E, W adjustments
    
    # Metadata
    data_source: str
    model_ensemble: List[str]
    downscaling_method: str
    creation_date: datetime


@dataclass
class ClimateScenario:
    """Complete climate scenario with multiple variables."""
    scenario_id: str
    name: str
    description: str
    emission_scenario: EmissionScenario
    time_horizon: TimeHorizon
    
    # Climate projections
    temperature_projection: ClimateProjection
    precipitation_projection: ClimateProjection
    humidity_projection: Optional[ClimateProjection] = None
    wind_projection: Optional[ClimateProjection] = None
    radiation_projection: Optional[ClimateProjection] = None
    
    # Extreme events
    extreme_temperature_changes: Dict[str, float] = None  # hot/cold extremes
    extreme_precipitation_changes: Dict[str, float] = None  # drought/flood
    
    # Regional characteristics
    region: str = "Tibetan Plateau"
    elevation_range: Tuple[float, float] = (3000, 6000)  # m
    spatial_resolution: float = 1.0  # km
    
    def __post_init__(self):
        """Initialize default extreme event changes."""
        if self.extreme_temperature_changes is None:
            self.extreme_temperature_changes = {
                'hot_days_change': 0.0,  # days/year
                'cold_days_change': 0.0,  # days/year
                'heat_wave_intensity': 0.0,  # °C
                'cold_wave_intensity': 0.0  # °C
            }
        
        if self.extreme_precipitation_changes is None:
            self.extreme_precipitation_changes = {
                'heavy_precipitation_change': 0.0,  # %
                'drought_frequency_change': 0.0,  # events/decade
                'wet_spell_length_change': 0.0,  # days
                'dry_spell_length_change': 0.0  # days
            }


class ClimateDataProcessor:
    """Process and analyze climate data."""
    
    def __init__(self):
        """
        Initialize climate data processor.
        """
        self.logger = logging.getLogger(__name__)
    
    def process_temperature_data(
        self,
        raw_data: np.ndarray,
        baseline_period: Tuple[int, int] = (1981, 2010)
    ) -> Dict[str, Any]:
        """
        Process temperature data and calculate statistics.
        
        Args:
            raw_data: Raw temperature data (time series)
            baseline_period: Baseline period for anomaly calculation
            
        Returns:
            Processed temperature statistics
        """
        # Calculate basic statistics
        mean_temp = float(np.mean(raw_data))
        std_temp = float(np.std(raw_data))
        min_temp = float(np.min(raw_data))
        max_temp = float(np.max(raw_data))
        
        # Calculate trends
        years = np.arange(len(raw_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, raw_data)
        
        # Calculate percentiles
        percentiles = np.percentile(raw_data, [5, 10, 25, 50, 75, 90, 95])
        
        # Seasonal analysis (assuming monthly data)
        if len(raw_data) >= 12:
            seasonal_means = []
            for month in range(12):
                month_data = raw_data[month::12]
                seasonal_means.append(float(np.mean(month_data)))
        else:
            seasonal_means = [mean_temp] * 12
        
        return {
            'mean': mean_temp,
            'std': std_temp,
            'min': min_temp,
            'max': max_temp,
            'trend_slope': float(slope),
            'trend_r_squared': float(r_value**2),
            'trend_p_value': float(p_value),
            'percentiles': {
                'p05': float(percentiles[0]),
                'p10': float(percentiles[1]),
                'p25': float(percentiles[2]),
                'p50': float(percentiles[3]),
                'p75': float(percentiles[4]),
                'p90': float(percentiles[5]),
                'p95': float(percentiles[6])
            },
            'seasonal_means': seasonal_means
        }
    
    def process_precipitation_data(
        self,
        raw_data: np.ndarray,
        baseline_period: Tuple[int, int] = (1981, 2010)
    ) -> Dict[str, Any]:
        """
        Process precipitation data and calculate statistics.
        
        Args:
            raw_data: Raw precipitation data (time series)
            baseline_period: Baseline period for anomaly calculation
            
        Returns:
            Processed precipitation statistics
        """
        # Remove negative values (if any)
        raw_data = np.maximum(raw_data, 0)
        
        # Calculate basic statistics
        total_precip = float(np.sum(raw_data))
        mean_precip = float(np.mean(raw_data))
        std_precip = float(np.std(raw_data))
        
        # Calculate wet day statistics
        wet_days = raw_data > 1.0  # Days with >1mm precipitation
        wet_day_count = int(np.sum(wet_days))
        wet_day_intensity = float(np.mean(raw_data[wet_days])) if wet_day_count > 0 else 0.0
        
        # Calculate extreme precipitation
        p95_threshold = np.percentile(raw_data[raw_data > 0], 95) if np.any(raw_data > 0) else 0
        extreme_precip_days = int(np.sum(raw_data > p95_threshold))
        
        # Seasonal analysis
        if len(raw_data) >= 12:
            seasonal_totals = []
            for month in range(12):
                month_data = raw_data[month::12]
                seasonal_totals.append(float(np.sum(month_data)))
        else:
            seasonal_totals = [total_precip / 12] * 12
        
        # Dry spell analysis
        dry_spells = self._analyze_dry_spells(raw_data)
        
        return {
            'total': total_precip,
            'mean': mean_precip,
            'std': std_precip,
            'wet_day_count': wet_day_count,
            'wet_day_intensity': wet_day_intensity,
            'extreme_precip_days': extreme_precip_days,
            'p95_threshold': float(p95_threshold),
            'seasonal_totals': seasonal_totals,
            'dry_spells': dry_spells
        }
    
    def _analyze_dry_spells(
        self,
        precip_data: np.ndarray,
        threshold: float = 1.0
    ) -> Dict[str, float]:
        """Analyze dry spell characteristics."""
        dry_days = precip_data < threshold
        
        # Find consecutive dry periods
        dry_spells = []
        current_spell = 0
        
        for is_dry in dry_days:
            if is_dry:
                current_spell += 1
            else:
                if current_spell > 0:
                    dry_spells.append(current_spell)
                current_spell = 0
        
        # Add final spell if data ends with dry period
        if current_spell > 0:
            dry_spells.append(current_spell)
        
        if dry_spells:
            return {
                'mean_length': float(np.mean(dry_spells)),
                'max_length': float(np.max(dry_spells)),
                'frequency': float(len(dry_spells)),
                'total_dry_days': float(np.sum(dry_spells))
            }
        else:
            return {
                'mean_length': 0.0,
                'max_length': 0.0,
                'frequency': 0.0,
                'total_dry_days': 0.0
            }


class ScenarioGenerator:
    """Generate climate scenarios based on emission pathways."""
    
    def __init__(self):
        """
        Initialize scenario generator.
        """
        self.logger = logging.getLogger(__name__)
        self.data_processor = ClimateDataProcessor()
        
        # Default climate sensitivity parameters
        self.climate_sensitivity = {
            EmissionScenario.SSP1_1_9: {'temp': 1.5, 'precip': 0.02},
            EmissionScenario.SSP1_2_6: {'temp': 2.0, 'precip': 0.03},
            EmissionScenario.SSP2_4_5: {'temp': 2.7, 'precip': 0.05},
            EmissionScenario.SSP3_7_0: {'temp': 3.6, 'precip': 0.07},
            EmissionScenario.SSP5_8_5: {'temp': 4.4, 'precip': 0.09}
        }
    
    def generate_temperature_projection(
        self,
        emission_scenario: EmissionScenario,
        time_horizon: TimeHorizon,
        baseline_temperature: float = -5.0,  # °C
        elevation: float = 4000.0  # m
    ) -> ClimateProjection:
        """
        Generate temperature projection for given scenario.
        
        Args:
            emission_scenario: Emission scenario
            time_horizon: Time horizon for projection
            baseline_temperature: Baseline temperature (°C)
            elevation: Elevation for adjustment (m)
            
        Returns:
            Temperature projection
        """
        # Get climate sensitivity
        sensitivity = self.climate_sensitivity[emission_scenario]['temp']
        
        # Time horizon adjustments
        time_factors = {
            TimeHorizon.NEAR_TERM: 0.4,
            TimeHorizon.MID_TERM: 0.7,
            TimeHorizon.LONG_TERM: 1.0
        }
        time_factor = time_factors[time_horizon]
        
        # Calculate mean temperature change
        mean_change = sensitivity * time_factor
        
        # Elevation adjustment (enhanced warming at high elevation)
        elevation_factor = 1.0 + (elevation - 3000) / 3000 * 0.3
        mean_change *= elevation_factor
        
        # Uncertainty range (±30% of mean change)
        uncertainty_range = (
            mean_change * 0.7,
            mean_change * 1.3
        )
        
        # Generate annual values with trend and variability
        years = 20  # 20-year periods
        annual_values = []
        for year in range(years):
            # Linear trend with interannual variability
            trend_value = baseline_temperature + mean_change * (year + 1) / years
            variability = np.random.normal(0, 0.5)  # ±0.5°C variability
            annual_values.append(trend_value + variability)
        
        # Seasonal changes (winter warming > summer warming)
        seasonal_changes = [
            1.2, 1.3, 1.1, 0.9, 0.8, 0.7,  # Dec-May (winter/spring)
            0.7, 0.8, 0.9, 1.0, 1.1, 1.2   # Jun-Nov (summer/autumn)
        ]
        seasonal_changes = [c * mean_change for c in seasonal_changes]
        
        # Elevation gradient (°C per 100m)
        elevation_gradient = -0.6 + mean_change * 0.1  # Lapse rate changes
        
        # Aspect adjustments
        aspect_adjustments = {
            'north': 1.1,  # North-facing slopes warm more
            'south': 0.9,  # South-facing slopes warm less
            'east': 1.0,
            'west': 1.0
        }
        
        # Confidence level based on scenario
        confidence_levels = {
            EmissionScenario.SSP1_1_9: ConfidenceLevel.MEDIUM,
            EmissionScenario.SSP1_2_6: ConfidenceLevel.HIGH,
            EmissionScenario.SSP2_4_5: ConfidenceLevel.HIGH,
            EmissionScenario.SSP3_7_0: ConfidenceLevel.MEDIUM,
            EmissionScenario.SSP5_8_5: ConfidenceLevel.MEDIUM
        }
        
        # Create climate variable
        temp_variable = ClimateVariable(
            name="temperature",
            units="°C",
            description="Mean annual temperature",
            baseline_value=baseline_temperature,
            uncertainty_range=(-10.0, 5.0)
        )
        
        return ClimateProjection(
            variable=temp_variable,
            emission_scenario=emission_scenario,
            time_horizon=time_horizon,
            mean_change=mean_change,
            uncertainty_range=uncertainty_range,
            confidence_level=confidence_levels[emission_scenario],
            annual_values=annual_values,
            seasonal_changes=seasonal_changes,
            elevation_gradient=elevation_gradient,
            aspect_adjustments=aspect_adjustments,
            data_source="CMIP6 ensemble",
            model_ensemble=["CESM2", "GFDL-ESM4", "MPI-ESM1-2-HR", "UKESM1-0-LL"],
            downscaling_method="Statistical downscaling",
            creation_date=datetime.now()
        )
    
    def generate_precipitation_projection(
        self,
        emission_scenario: EmissionScenario,
        time_horizon: TimeHorizon,
        baseline_precipitation: float = 400.0,  # mm/year
        elevation: float = 4000.0  # m
    ) -> ClimateProjection:
        """
        Generate precipitation projection for given scenario.
        
        Args:
            emission_scenario: Emission scenario
            time_horizon: Time horizon for projection
            baseline_precipitation: Baseline precipitation (mm/year)
            elevation: Elevation for adjustment (m)
            
        Returns:
            Precipitation projection
        """
        # Get climate sensitivity
        sensitivity = self.climate_sensitivity[emission_scenario]['precip']
        
        # Time horizon adjustments
        time_factors = {
            TimeHorizon.NEAR_TERM: 0.5,
            TimeHorizon.MID_TERM: 0.75,
            TimeHorizon.LONG_TERM: 1.0
        }
        time_factor = time_factors[time_horizon]
        
        # Calculate mean precipitation change (as fraction)
        mean_change_fraction = sensitivity * time_factor
        
        # Elevation adjustment (orographic enhancement)
        elevation_factor = 1.0 + (elevation - 3000) / 1000 * 0.1
        mean_change_fraction *= elevation_factor
        
        # Convert to absolute change
        mean_change = baseline_precipitation * mean_change_fraction
        
        # Uncertainty range (larger for precipitation)
        uncertainty_range = (
            mean_change * 0.5,
            mean_change * 1.5
        )
        
        # Generate annual values with trend and high variability
        years = 20
        annual_values = []
        for year in range(years):
            # Linear trend with high interannual variability
            trend_value = baseline_precipitation + mean_change * (year + 1) / years
            variability = np.random.normal(0, baseline_precipitation * 0.2)
            annual_values.append(max(0, trend_value + variability))
        
        # Seasonal changes (monsoon enhancement)
        seasonal_changes = [
            0.5, 0.6, 0.8, 1.2, 1.5, 2.0,  # Dec-May (dry season to monsoon)
            2.2, 2.0, 1.8, 1.2, 0.8, 0.6   # Jun-Nov (monsoon to dry season)
        ]
        seasonal_changes = [c * mean_change for c in seasonal_changes]
        
        # Elevation gradient (mm per 100m)
        elevation_gradient = 50.0 + mean_change * 0.01  # Orographic effect
        
        # Aspect adjustments
        aspect_adjustments = {
            'south': 1.2,  # Windward slopes receive more
            'north': 0.8,  # Leeward slopes receive less
            'east': 1.1,
            'west': 0.9
        }
        
        # Confidence level (lower for precipitation)
        confidence_levels = {
            EmissionScenario.SSP1_1_9: ConfidenceLevel.LOW,
            EmissionScenario.SSP1_2_6: ConfidenceLevel.MEDIUM,
            EmissionScenario.SSP2_4_5: ConfidenceLevel.MEDIUM,
            EmissionScenario.SSP3_7_0: ConfidenceLevel.LOW,
            EmissionScenario.SSP5_8_5: ConfidenceLevel.LOW
        }
        
        # Create climate variable
        precip_variable = ClimateVariable(
            name="precipitation",
            units="mm/year",
            description="Annual precipitation",
            baseline_value=baseline_precipitation,
            uncertainty_range=(100.0, 800.0)
        )
        
        return ClimateProjection(
            variable=precip_variable,
            emission_scenario=emission_scenario,
            time_horizon=time_horizon,
            mean_change=mean_change,
            uncertainty_range=uncertainty_range,
            confidence_level=confidence_levels[emission_scenario],
            annual_values=annual_values,
            seasonal_changes=seasonal_changes,
            elevation_gradient=elevation_gradient,
            aspect_adjustments=aspect_adjustments,
            data_source="CMIP6 ensemble",
            model_ensemble=["CESM2", "GFDL-ESM4", "MPI-ESM1-2-HR", "UKESM1-0-LL"],
            downscaling_method="Statistical downscaling",
            creation_date=datetime.now()
        )
    
    def generate_complete_scenario(
        self,
        scenario_id: str,
        emission_scenario: EmissionScenario,
        time_horizon: TimeHorizon,
        baseline_temperature: float = -5.0,
        baseline_precipitation: float = 400.0,
        elevation: float = 4000.0
    ) -> ClimateScenario:
        """
        Generate complete climate scenario with all variables.
        
        Args:
            scenario_id: Unique scenario identifier
            emission_scenario: Emission scenario
            time_horizon: Time horizon for projection
            baseline_temperature: Baseline temperature (°C)
            baseline_precipitation: Baseline precipitation (mm/year)
            elevation: Elevation for adjustment (m)
            
        Returns:
            Complete climate scenario
        """
        # Generate temperature projection
        temp_projection = self.generate_temperature_projection(
            emission_scenario, time_horizon, baseline_temperature, elevation
        )
        
        # Generate precipitation projection
        precip_projection = self.generate_precipitation_projection(
            emission_scenario, time_horizon, baseline_precipitation, elevation
        )
        
        # Generate extreme event changes
        temp_change = temp_projection.mean_change
        precip_change_fraction = precip_projection.mean_change / baseline_precipitation
        
        extreme_temperature_changes = {
            'hot_days_change': temp_change * 10,  # More hot days
            'cold_days_change': -temp_change * 15,  # Fewer cold days
            'heat_wave_intensity': temp_change * 1.5,  # Stronger heat waves
            'cold_wave_intensity': temp_change * 0.5  # Weaker cold waves
        }
        
        extreme_precipitation_changes = {
            'heavy_precipitation_change': precip_change_fraction * 150,  # % increase
            'drought_frequency_change': -precip_change_fraction * 2,  # events/decade
            'wet_spell_length_change': precip_change_fraction * 5,  # days
            'dry_spell_length_change': -precip_change_fraction * 10  # days
        }
        
        # Create scenario name and description
        scenario_names = {
            EmissionScenario.SSP1_1_9: "Very Low Emissions",
            EmissionScenario.SSP1_2_6: "Low Emissions",
            EmissionScenario.SSP2_4_5: "Intermediate Emissions",
            EmissionScenario.SSP3_7_0: "High Emissions",
            EmissionScenario.SSP5_8_5: "Very High Emissions"
        }
        
        name = f"{scenario_names[emission_scenario]} - {time_horizon.value}"
        description = f"Climate scenario for {emission_scenario.value} emission pathway, {time_horizon.value} time horizon"
        
        return ClimateScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            emission_scenario=emission_scenario,
            time_horizon=time_horizon,
            temperature_projection=temp_projection,
            precipitation_projection=precip_projection,
            extreme_temperature_changes=extreme_temperature_changes,
            extreme_precipitation_changes=extreme_precipitation_changes
        )
    
    def generate_scenario_ensemble(
        self,
        base_scenario_id: str,
        emission_scenarios: List[EmissionScenario] = None,
        time_horizons: List[TimeHorizon] = None,
        **kwargs
    ) -> List[ClimateScenario]:
        """
        Generate ensemble of climate scenarios.
        
        Args:
            base_scenario_id: Base identifier for scenarios
            emission_scenarios: List of emission scenarios
            time_horizons: List of time horizons
            **kwargs: Additional parameters for scenario generation
            
        Returns:
            List of climate scenarios
        """
        if emission_scenarios is None:
            emission_scenarios = [EmissionScenario.SSP2_4_5, EmissionScenario.SSP5_8_5]
        
        if time_horizons is None:
            time_horizons = [TimeHorizon.MID_TERM, TimeHorizon.LONG_TERM]
        
        scenarios = []
        
        for emission_scenario in emission_scenarios:
            for time_horizon in time_horizons:
                scenario_id = f"{base_scenario_id}_{emission_scenario.value}_{time_horizon.value}"
                
                scenario = self.generate_complete_scenario(
                    scenario_id=scenario_id,
                    emission_scenario=emission_scenario,
                    time_horizon=time_horizon,
                    **kwargs
                )
                
                scenarios.append(scenario)
        
        return scenarios


def create_climate_scenario(
    scenario_id: str,
    emission_scenario: str = "ssp2-4.5",
    time_horizon: str = "2041-2060",
    **kwargs
) -> ClimateScenario:
    """
    Create a climate scenario with specified parameters.
    
    Args:
        scenario_id: Unique scenario identifier
        emission_scenario: Emission scenario (ssp1-1.9, ssp1-2.6, ssp2-4.5, ssp3-7.0, ssp5-8.5)
        time_horizon: Time horizon (2021-2040, 2041-2060, 2081-2100)
        **kwargs: Additional parameters
        
    Returns:
        Climate scenario
    """
    # Convert string inputs to enums
    emission_enum = EmissionScenario(emission_scenario)
    time_enum = TimeHorizon(time_horizon)
    
    # Create scenario generator
    generator = ScenarioGenerator()
    
    return generator.generate_complete_scenario(
        scenario_id=scenario_id,
        emission_scenario=emission_enum,
        time_horizon=time_enum,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create scenario generator
    generator = ScenarioGenerator()
    
    # Generate scenarios for different emission pathways
    scenarios = generator.generate_scenario_ensemble(
        base_scenario_id="tibet_glacier",
        emission_scenarios=[EmissionScenario.SSP2_4_5, EmissionScenario.SSP5_8_5],
        time_horizons=[TimeHorizon.MID_TERM, TimeHorizon.LONG_TERM],
        baseline_temperature=-5.0,
        baseline_precipitation=400.0,
        elevation=4500.0
    )
    
    print("Climate Scenario Analysis Results:")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.name}")
        print(f"ID: {scenario.scenario_id}")
        print(f"Emission Scenario: {scenario.emission_scenario.value}")
        print(f"Time Horizon: {scenario.time_horizon.value}")
        
        # Temperature projection
        temp_proj = scenario.temperature_projection
        print(f"\nTemperature Changes:")
        print(f"  Mean Change: {temp_proj.mean_change:.2f}°C")
        print(f"  Uncertainty Range: {temp_proj.uncertainty_range[0]:.2f} to {temp_proj.uncertainty_range[1]:.2f}°C")
        print(f"  Confidence: {temp_proj.confidence_level.value}")
        print(f"  Elevation Gradient: {temp_proj.elevation_gradient:.3f}°C/100m")
        
        # Precipitation projection
        precip_proj = scenario.precipitation_projection
        print(f"\nPrecipitation Changes:")
        print(f"  Mean Change: {precip_proj.mean_change:.1f}mm/year")
        print(f"  Relative Change: {precip_proj.mean_change/precip_proj.variable.baseline_value*100:.1f}%")
        print(f"  Uncertainty Range: {precip_proj.uncertainty_range[0]:.1f} to {precip_proj.uncertainty_range[1]:.1f}mm/year")
        print(f"  Confidence: {precip_proj.confidence_level.value}")
        
        # Extreme events
        print(f"\nExtreme Event Changes:")
        temp_extremes = scenario.extreme_temperature_changes
        print(f"  Hot Days Change: {temp_extremes['hot_days_change']:.1f} days/year")
        print(f"  Cold Days Change: {temp_extremes['cold_days_change']:.1f} days/year")
        
        precip_extremes = scenario.extreme_precipitation_changes
        print(f"  Heavy Precipitation Change: {precip_extremes['heavy_precipitation_change']:.1f}%")
        print(f"  Drought Frequency Change: {precip_extremes['drought_frequency_change']:.1f} events/decade")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature changes
    emission_labels = []
    temp_changes = []
    temp_uncertainties = []
    
    for scenario in scenarios:
        if scenario.time_horizon == TimeHorizon.LONG_TERM:
            emission_labels.append(scenario.emission_scenario.value)
            temp_changes.append(scenario.temperature_projection.mean_change)
            temp_uncertainties.append([
                scenario.temperature_projection.mean_change - scenario.temperature_projection.uncertainty_range[0],
                scenario.temperature_projection.uncertainty_range[1] - scenario.temperature_projection.mean_change
            ])
    
    temp_uncertainties = np.array(temp_uncertainties).T
    ax1.errorbar(emission_labels, temp_changes, yerr=temp_uncertainties, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_title('Temperature Change by 2081-2100', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temperature Change (°C)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Precipitation changes
    precip_changes = []
    precip_uncertainties = []
    
    for scenario in scenarios:
        if scenario.time_horizon == TimeHorizon.LONG_TERM:
            precip_changes.append(scenario.precipitation_projection.mean_change)
            precip_uncertainties.append([
                scenario.precipitation_projection.mean_change - scenario.precipitation_projection.uncertainty_range[0],
                scenario.precipitation_projection.uncertainty_range[1] - scenario.precipitation_projection.mean_change
            ])
    
    precip_uncertainties = np.array(precip_uncertainties).T
    ax2.errorbar(emission_labels, precip_changes, yerr=precip_uncertainties,
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8, color='blue')
    ax2.set_title('Precipitation Change by 2081-2100', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Precipitation Change (mm/year)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Seasonal temperature patterns
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, scenario in enumerate(scenarios):
        if scenario.time_horizon == TimeHorizon.LONG_TERM:
            seasonal_temp = scenario.temperature_projection.seasonal_changes
            ax3.plot(months, seasonal_temp, 'o-', linewidth=2, markersize=6,
                    label=f'{scenario.emission_scenario.value}')
    
    ax3.set_title('Seasonal Temperature Changes', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Temperature Change (°C)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Seasonal precipitation patterns
    for i, scenario in enumerate(scenarios):
        if scenario.time_horizon == TimeHorizon.LONG_TERM:
            seasonal_precip = scenario.precipitation_projection.seasonal_changes
            ax4.plot(months, seasonal_precip, 's-', linewidth=2, markersize=6,
                    label=f'{scenario.emission_scenario.value}')
    
    ax4.set_title('Seasonal Precipitation Changes', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Precipitation Change (mm)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nGenerated {len(scenarios)} climate scenarios for glacier impact assessment")