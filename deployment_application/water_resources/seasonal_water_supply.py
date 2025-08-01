"""Water Availability Analysis Module.

This module provides functionality for analyzing water resource availability
based on glacier dynamics and runoff predictions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum


class WaterStressLevel(Enum):
    """Water stress levels based on availability."""
    ABUNDANT = "abundant"  # > 1700 m³/person/year
    ADEQUATE = "adequate"  # 1000-1700 m³/person/year
    STRESSED = "stressed"  # 500-1000 m³/person/year
    SCARCE = "scarce"     # 200-500 m³/person/year
    CRITICAL = "critical"  # < 200 m³/person/year


class SeasonalPattern(Enum):
    """Seasonal water availability patterns."""
    GLACIER_DOMINATED = "glacier_dominated"  # Peak in summer
    PRECIPITATION_DOMINATED = "precipitation_dominated"  # Peak in monsoon
    MIXED = "mixed"  # Multiple peaks
    UNIFORM = "uniform"  # Relatively constant


@dataclass
class WaterDemand:
    """Water demand specification."""
    domestic: float = 0.0  # m³/day
    agricultural: float = 0.0  # m³/day
    industrial: float = 0.0  # m³/day
    environmental: float = 0.0  # m³/day
    
    @property
    def total(self) -> float:
        """Total water demand."""
        return self.domestic + self.agricultural + self.industrial + self.environmental


@dataclass
class WaterAvailabilityConfig:
    """Configuration for water availability analysis."""
    # Reliability thresholds
    reliability_threshold: float = 0.95  # 95% reliability
    drought_threshold: float = 0.7  # 70% of mean flow
    flood_threshold: float = 2.0  # 200% of mean flow
    
    # Storage parameters
    reservoir_capacity: float = 0.0  # m³
    groundwater_capacity: float = 0.0  # m³
    
    # Environmental flow requirements
    environmental_flow_fraction: float = 0.1  # 10% of mean flow
    
    # Population and economic parameters
    population: int = 0
    gdp_per_capita: float = 0.0  # USD/year
    water_price: float = 0.001  # USD/m³


class WaterBalanceCalculator:
    """Calculator for water balance analysis."""
    
    def __init__(self, config: WaterAvailabilityConfig = None):
        """
        Initialize water balance calculator.
        
        Args:
            config: Water availability configuration
        """
        self.config = config or WaterAvailabilityConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_daily_balance(
        self,
        supply: float,
        demand: WaterDemand,
        storage: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate daily water balance.
        
        Args:
            supply: Daily water supply (m³/day)
            demand: Water demand specification
            storage: Available storage (m³)
            
        Returns:
            Water balance components
        """
        total_demand = demand.total
        environmental_flow = max(
            supply * self.config.environmental_flow_fraction,
            demand.environmental
        )
        
        # Available water after environmental requirements
        available_water = max(0, supply - environmental_flow)
        
        # Calculate deficit or surplus
        deficit = max(0, total_demand - available_water)
        surplus = max(0, available_water - total_demand)
        
        # Storage utilization
        storage_used = min(deficit, storage)
        remaining_deficit = deficit - storage_used
        
        # Updated storage
        new_storage = storage - storage_used + min(surplus, self.config.reservoir_capacity - storage)
        
        return {
            'supply': supply,
            'demand': total_demand,
            'environmental_flow': environmental_flow,
            'available_water': available_water,
            'deficit': deficit,
            'surplus': surplus,
            'storage_used': storage_used,
            'remaining_deficit': remaining_deficit,
            'storage': new_storage,
            'reliability': 1.0 if remaining_deficit == 0 else 0.0
        }
    
    def calculate_seasonal_balance(
        self,
        supply_series: List[float],
        demand_series: List[WaterDemand],
        initial_storage: float = 0.0
    ) -> Dict[str, List[float]]:
        """
        Calculate seasonal water balance.
        
        Args:
            supply_series: Time series of water supply
            demand_series: Time series of water demand
            initial_storage: Initial storage volume
            
        Returns:
            Time series of water balance components
        """
        results = {
            'supply': [],
            'demand': [],
            'deficit': [],
            'surplus': [],
            'storage': [],
            'reliability': []
        }
        
        current_storage = initial_storage
        
        for supply, demand in zip(supply_series, demand_series):
            balance = self.calculate_daily_balance(supply, demand, current_storage)
            
            for key in results.keys():
                results[key].append(balance[key])
            
            current_storage = balance['storage']
        
        return results


class ReliabilityAnalyzer:
    """Analyzer for water supply reliability."""
    
    def __init__(self, config: WaterAvailabilityConfig = None):
        """
        Initialize reliability analyzer.
        
        Args:
            config: Water availability configuration
        """
        self.config = config or WaterAvailabilityConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_reliability_metrics(
        self,
        supply_series: List[float],
        demand_series: List[float]
    ) -> Dict[str, float]:
        """
        Calculate water supply reliability metrics.
        
        Args:
            supply_series: Time series of water supply
            demand_series: Time series of water demand
            
        Returns:
            Reliability metrics
        """
        supply = np.array(supply_series)
        demand = np.array(demand_series)
        
        # Basic reliability (fraction of time demand is met)
        reliability = np.mean(supply >= demand)
        
        # Vulnerability (average deficit when it occurs)
        deficits = np.maximum(0, demand - supply)
        vulnerability = np.mean(deficits[deficits > 0]) if np.any(deficits > 0) else 0.0
        
        # Resilience (how quickly system recovers from failure)
        failures = supply < demand
        if np.any(failures):
            # Find failure periods
            failure_starts = np.where(np.diff(np.concatenate(([False], failures))))[0]
            failure_ends = np.where(np.diff(np.concatenate((failures, [False]))))[0]
            
            if len(failure_starts) > 0 and len(failure_ends) > 0:
                failure_durations = failure_ends - failure_starts + 1
                resilience = 1.0 / np.mean(failure_durations) if len(failure_durations) > 0 else 1.0
            else:
                resilience = 1.0
        else:
            resilience = 1.0
        
        # Sustainability index
        sustainability = reliability * (1 - vulnerability / np.mean(demand)) * resilience
        
        return {
            'reliability': reliability,
            'vulnerability': vulnerability,
            'resilience': resilience,
            'sustainability': sustainability,
            'mean_supply': float(np.mean(supply)),
            'mean_demand': float(np.mean(demand)),
            'supply_variability': float(np.std(supply) / np.mean(supply)) if np.mean(supply) > 0 else 0.0
        }
    
    def analyze_drought_risk(
        self,
        supply_series: List[float],
        return_periods: List[int] = None
    ) -> Dict[str, float]:
        """
        Analyze drought risk and return periods.
        
        Args:
            supply_series: Time series of water supply
            return_periods: Return periods to analyze (years)
            
        Returns:
            Drought risk analysis
        """
        if return_periods is None:
            return_periods = [2, 5, 10, 20, 50, 100]
        
        supply = np.array(supply_series)
        mean_supply = np.mean(supply)
        
        # Define drought as flow below threshold
        drought_threshold = mean_supply * self.config.drought_threshold
        drought_events = supply < drought_threshold
        
        # Calculate drought characteristics
        drought_frequency = np.mean(drought_events)
        drought_severity = np.mean(supply[drought_events]) / mean_supply if np.any(drought_events) else 1.0
        
        # Estimate return period flows using Gumbel distribution
        sorted_flows = np.sort(supply)
        n = len(sorted_flows)
        
        return_flows = {}
        for rp in return_periods:
            # Gumbel distribution parameters
            prob = 1 - 1/rp
            rank = int(prob * n)
            return_flows[f'{rp}_year'] = float(sorted_flows[min(rank, n-1)])
        
        return {
            'drought_frequency': drought_frequency,
            'drought_severity': drought_severity,
            'drought_threshold': drought_threshold,
            **return_flows
        }


class WaterStressAssessment:
    """Assessment of water stress levels."""
    
    def __init__(self, config: WaterAvailabilityConfig = None):
        """
        Initialize water stress assessment.
        
        Args:
            config: Water availability configuration
        """
        self.config = config or WaterAvailabilityConfig()
        self.logger = logging.getLogger(__name__)
    
    def assess_water_stress(
        self,
        annual_supply: float,
        population: int = None
    ) -> Dict[str, Union[str, float]]:
        """
        Assess water stress level based on per capita availability.
        
        Args:
            annual_supply: Annual water supply (m³/year)
            population: Population count
            
        Returns:
            Water stress assessment
        """
        pop = population or self.config.population
        
        if pop <= 0:
            return {
                'stress_level': 'unknown',
                'per_capita_availability': 0.0,
                'stress_index': 0.0
            }
        
        per_capita = annual_supply / pop
        
        # Determine stress level
        if per_capita > 1700:
            stress_level = WaterStressLevel.ABUNDANT
            stress_index = 0.0
        elif per_capita > 1000:
            stress_level = WaterStressLevel.ADEQUATE
            stress_index = 0.2
        elif per_capita > 500:
            stress_level = WaterStressLevel.STRESSED
            stress_index = 0.5
        elif per_capita > 200:
            stress_level = WaterStressLevel.SCARCE
            stress_index = 0.8
        else:
            stress_level = WaterStressLevel.CRITICAL
            stress_index = 1.0
        
        return {
            'stress_level': stress_level.value,
            'per_capita_availability': per_capita,
            'stress_index': stress_index,
            'population': pop
        }
    
    def calculate_water_security_index(
        self,
        reliability_metrics: Dict[str, float],
        stress_assessment: Dict[str, Union[str, float]],
        infrastructure_score: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive water security index.
        
        Args:
            reliability_metrics: Water supply reliability metrics
            stress_assessment: Water stress assessment
            infrastructure_score: Infrastructure adequacy score (0-1)
            
        Returns:
            Water security index components
        """
        # Availability component (0-1)
        availability = 1.0 - stress_assessment['stress_index']
        
        # Reliability component (0-1)
        reliability = reliability_metrics['reliability']
        
        # Quality component (assumed good for glacier water)
        quality = 0.9
        
        # Infrastructure component
        infrastructure = infrastructure_score
        
        # Overall water security index
        wsi = (availability + reliability + quality + infrastructure) / 4.0
        
        return {
            'water_security_index': wsi,
            'availability_score': availability,
            'reliability_score': reliability,
            'quality_score': quality,
            'infrastructure_score': infrastructure
        }


class SeasonalPatternAnalyzer:
    """Analyzer for seasonal water availability patterns."""
    
    def __init__(self):
        """
        Initialize seasonal pattern analyzer.
        """
        self.logger = logging.getLogger(__name__)
    
    def identify_seasonal_pattern(
        self,
        monthly_supply: List[float],
        monthly_glacier_fraction: List[float]
    ) -> Dict[str, Union[str, float, List[float]]]:
        """
        Identify seasonal water availability pattern.
        
        Args:
            monthly_supply: Monthly water supply values
            monthly_glacier_fraction: Monthly glacier contribution fractions
            
        Returns:
            Seasonal pattern analysis
        """
        supply = np.array(monthly_supply)
        glacier_frac = np.array(monthly_glacier_fraction)
        
        # Normalize supply to identify peaks
        normalized_supply = supply / np.mean(supply)
        
        # Find peak months
        peak_threshold = 1.2  # 20% above mean
        peak_months = np.where(normalized_supply > peak_threshold)[0]
        
        # Analyze glacier contribution
        mean_glacier_frac = np.mean(glacier_frac)
        
        # Determine pattern type
        if len(peak_months) == 0:
            pattern = SeasonalPattern.UNIFORM
        elif mean_glacier_frac > 0.6 and np.any(peak_months >= 5) and np.any(peak_months <= 8):
            # High glacier contribution with summer peaks
            pattern = SeasonalPattern.GLACIER_DOMINATED
        elif mean_glacier_frac < 0.3 and np.any(peak_months >= 6) and np.any(peak_months <= 9):
            # Low glacier contribution with monsoon peaks
            pattern = SeasonalPattern.PRECIPITATION_DOMINATED
        else:
            pattern = SeasonalPattern.MIXED
        
        # Calculate seasonality index
        seasonality_index = np.std(normalized_supply)
        
        # Peak timing
        peak_month = int(np.argmax(supply)) + 1  # 1-indexed
        
        return {
            'pattern_type': pattern.value,
            'seasonality_index': seasonality_index,
            'peak_month': peak_month,
            'peak_months': peak_months.tolist(),
            'mean_glacier_contribution': mean_glacier_frac,
            'monthly_variability': float(np.std(supply) / np.mean(supply))
        }
    
    def calculate_seasonal_reliability(
        self,
        monthly_supply: List[float],
        monthly_demand: List[float]
    ) -> Dict[str, float]:
        """
        Calculate seasonal reliability metrics.
        
        Args:
            monthly_supply: Monthly water supply values
            monthly_demand: Monthly water demand values
            
        Returns:
            Seasonal reliability metrics
        """
        supply = np.array(monthly_supply)
        demand = np.array(monthly_demand)
        
        # Define seasons
        seasons = {
            'spring': [2, 3, 4],    # Mar-May
            'summer': [5, 6, 7],    # Jun-Aug
            'autumn': [8, 9, 10],   # Sep-Nov
            'winter': [11, 0, 1]    # Dec-Feb
        }
        
        seasonal_reliability = {}
        
        for season, months in seasons.items():
            season_supply = supply[months]
            season_demand = demand[months]
            
            reliability = np.mean(season_supply >= season_demand)
            seasonal_reliability[f'{season}_reliability'] = reliability
        
        return seasonal_reliability


class WaterAvailabilityAnalyzer:
    """Main class for comprehensive water availability analysis."""
    
    def __init__(self, config: WaterAvailabilityConfig = None):
        """
        Initialize water availability analyzer.
        
        Args:
            config: Water availability configuration
        """
        self.config = config or WaterAvailabilityConfig()
        self.balance_calculator = WaterBalanceCalculator(self.config)
        self.reliability_analyzer = ReliabilityAnalyzer(self.config)
        self.stress_assessment = WaterStressAssessment(self.config)
        self.pattern_analyzer = SeasonalPatternAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_water_availability(
        self,
        runoff_data: Dict[str, List[float]],
        demand_data: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive water availability analysis.
        
        Args:
            runoff_data: Runoff time series data
            demand_data: Water demand time series data
            
        Returns:
            Comprehensive water availability analysis
        """
        supply_series = runoff_data['total_runoff']
        
        # Default demand if not provided
        if demand_data is None:
            mean_supply = np.mean(supply_series)
            demand_series = [mean_supply * 0.3] * len(supply_series)  # 30% of supply
        else:
            demand_series = demand_data.get('total_demand', [0] * len(supply_series))
        
        # Calculate annual totals
        annual_supply = np.sum(supply_series) * 365.25 / len(supply_series)  # Convert to annual
        annual_demand = np.sum(demand_series) * 365.25 / len(demand_series)
        
        # Reliability analysis
        reliability_metrics = self.reliability_analyzer.calculate_reliability_metrics(
            supply_series, demand_series
        )
        
        # Drought risk analysis
        drought_analysis = self.reliability_analyzer.analyze_drought_risk(supply_series)
        
        # Water stress assessment
        stress_assessment = self.stress_assessment.assess_water_stress(annual_supply)
        
        # Water security index
        security_index = self.stress_assessment.calculate_water_security_index(
            reliability_metrics, stress_assessment
        )
        
        # Seasonal pattern analysis (if monthly data available)
        seasonal_analysis = {}
        if len(supply_series) >= 12:
            # Assume monthly data for first year
            monthly_supply = supply_series[:12]
            monthly_glacier_frac = runoff_data.get('glacier_fraction', [0.5] * 12)[:12]
            
            seasonal_analysis = self.pattern_analyzer.identify_seasonal_pattern(
                monthly_supply, monthly_glacier_frac
            )
            
            if demand_data:
                monthly_demand = demand_series[:12]
                seasonal_reliability = self.pattern_analyzer.calculate_seasonal_reliability(
                    monthly_supply, monthly_demand
                )
                seasonal_analysis.update(seasonal_reliability)
        
        return {
            'summary': {
                'annual_supply': annual_supply,
                'annual_demand': annual_demand,
                'supply_demand_ratio': annual_supply / annual_demand if annual_demand > 0 else float('inf'),
                'water_security_index': security_index['water_security_index']
            },
            'reliability': reliability_metrics,
            'drought_risk': drought_analysis,
            'water_stress': stress_assessment,
            'security_index': security_index,
            'seasonal_patterns': seasonal_analysis
        }
    
    def generate_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate water management recommendations based on analysis.
        
        Args:
            analysis_results: Water availability analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Reliability recommendations
        reliability = analysis_results['reliability']['reliability']
        if reliability < 0.8:
            recommendations.append(
                f"Water supply reliability is {reliability:.1%}. Consider developing additional water sources or storage."
            )
        
        # Water stress recommendations
        stress_level = analysis_results['water_stress']['stress_level']
        if stress_level in ['stressed', 'scarce', 'critical']:
            recommendations.append(
                f"Water stress level is {stress_level}. Implement water conservation measures and demand management."
            )
        
        # Seasonal pattern recommendations
        if 'seasonal_patterns' in analysis_results:
            pattern = analysis_results['seasonal_patterns'].get('pattern_type')
            if pattern == 'glacier_dominated':
                recommendations.append(
                    "Water supply is glacier-dominated. Monitor glacier changes and plan for long-term supply variations."
                )
            elif pattern == 'precipitation_dominated':
                recommendations.append(
                    "Water supply is precipitation-dominated. Develop rainwater harvesting and storage systems."
                )
        
        # Drought risk recommendations
        drought_freq = analysis_results['drought_risk']['drought_frequency']
        if drought_freq > 0.2:
            recommendations.append(
                f"Drought frequency is {drought_freq:.1%}. Develop drought contingency plans and emergency reserves."
            )
        
        # Infrastructure recommendations
        wsi = analysis_results['security_index']['water_security_index']
        if wsi < 0.6:
            recommendations.append(
                f"Water security index is {wsi:.2f}. Invest in water infrastructure and management systems."
            )
        
        return recommendations


def create_water_availability_analyzer(
    config: Optional[WaterAvailabilityConfig] = None,
    **kwargs
) -> WaterAvailabilityAnalyzer:
    """
    Create a water availability analyzer with specified configuration.
    
    Args:
        config: Water availability configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured water availability analyzer
    """
    if config is None:
        config = WaterAvailabilityConfig(**kwargs)
    
    return WaterAvailabilityAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create analyzer
    config = WaterAvailabilityConfig(
        population=10000,
        reservoir_capacity=1000000,  # 1 million m³
        reliability_threshold=0.95
    )
    analyzer = create_water_availability_analyzer(config)
    
    # Mock runoff data (daily for one year)
    days = np.arange(365)
    base_flow = 50  # m³/s
    seasonal_variation = 30 * np.sin(2 * np.pi * (days - 120) / 365)  # Peak in summer
    noise = np.random.normal(0, 5, 365)
    daily_runoff = np.maximum(base_flow + seasonal_variation + noise, 5)
    
    # Mock glacier contribution
    glacier_fraction = 0.6 + 0.3 * np.sin(2 * np.pi * (days - 120) / 365)
    glacier_fraction = np.clip(glacier_fraction, 0.2, 0.9)
    
    runoff_data = {
        'total_runoff': daily_runoff.tolist(),
        'glacier_fraction': glacier_fraction.tolist()
    }
    
    # Analyze water availability
    results = analyzer.analyze_water_availability(runoff_data)
    
    print("Water Availability Analysis Results:")
    print("=" * 40)
    
    # Summary
    summary = results['summary']
    print(f"Annual Supply: {summary['annual_supply']:,.0f} m³/year")
    print(f"Supply-Demand Ratio: {summary['supply_demand_ratio']:.2f}")
    print(f"Water Security Index: {summary['water_security_index']:.2f}")
    
    # Water stress
    stress = results['water_stress']
    print(f"\nWater Stress Level: {stress['stress_level']}")
    print(f"Per Capita Availability: {stress['per_capita_availability']:.0f} m³/person/year")
    
    # Reliability
    reliability = results['reliability']
    print(f"\nSupply Reliability: {reliability['reliability']:.1%}")
    print(f"Sustainability Index: {reliability['sustainability']:.2f}")
    
    # Seasonal patterns
    if 'seasonal_patterns' in results:
        patterns = results['seasonal_patterns']
        print(f"\nSeasonal Pattern: {patterns['pattern_type']}")
        print(f"Peak Month: {patterns['peak_month']}")
        print(f"Mean Glacier Contribution: {patterns['mean_glacier_contribution']:.1%}")
    
    # Recommendations
    recommendations = analyzer.generate_recommendations(results)
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")