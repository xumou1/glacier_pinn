"""Seasonal Analysis Module.

This module provides functionality for analyzing seasonal patterns
in glacier dynamics and water resource availability.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import signal, stats
from scipy.fft import fft, fftfreq


class SeasonDefinition(Enum):
    """Different ways to define seasons."""
    METEOROLOGICAL = "meteorological"  # Fixed calendar dates
    ASTRONOMICAL = "astronomical"      # Based on solstices/equinoxes
    HYDROLOGICAL = "hydrological"      # Based on water year
    THERMAL = "thermal"                # Based on temperature patterns
    MONSOON = "monsoon"                # Based on precipitation patterns


class TrendMethod(Enum):
    """Methods for trend analysis."""
    LINEAR = "linear"
    MANN_KENDALL = "mann_kendall"
    THEIL_SEN = "theil_sen"
    POLYNOMIAL = "polynomial"


@dataclass
class SeasonalConfig:
    """Configuration for seasonal analysis."""
    season_definition: SeasonDefinition = SeasonDefinition.METEOROLOGICAL
    water_year_start: int = 274  # October 1 (day of year)
    monsoon_start: int = 152     # June 1
    monsoon_end: int = 243       # August 31
    
    # Analysis parameters
    trend_significance_level: float = 0.05
    harmonic_components: int = 3
    smoothing_window: int = 30  # days
    
    # Threshold definitions
    dry_season_threshold: float = 0.5  # fraction of mean
    wet_season_threshold: float = 1.5  # fraction of mean
    melt_season_temp_threshold: float = 0.0  # °C


class SeasonalDecomposition:
    """Decompose time series into seasonal components."""
    
    def __init__(self, config: SeasonalConfig = None):
        """
        Initialize seasonal decomposition.
        
        Args:
            config: Seasonal analysis configuration
        """
        self.config = config or SeasonalConfig()
        self.logger = logging.getLogger(__name__)
    
    def decompose_time_series(
        self,
        data: np.ndarray,
        time_axis: np.ndarray = None,
        period: float = 365.25
    ) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data: Time series data
            time_axis: Time axis (days from start)
            period: Period for seasonal decomposition (days)
            
        Returns:
            Decomposed components
        """
        if time_axis is None:
            time_axis = np.arange(len(data))
        
        # Remove NaN values
        valid_mask = ~np.isnan(data)
        clean_data = data[valid_mask]
        clean_time = time_axis[valid_mask]
        
        if len(clean_data) < period:
            self.logger.warning("Data length is less than one period")
            return {
                'original': data,
                'trend': np.full_like(data, np.nanmean(data)),
                'seasonal': np.zeros_like(data),
                'residual': data - np.nanmean(data)
            }
        
        # Calculate trend using moving average
        trend = self._calculate_trend(clean_data, clean_time)
        
        # Remove trend
        detrended = clean_data - trend
        
        # Calculate seasonal component using harmonic analysis
        seasonal = self._calculate_seasonal_component(detrended, clean_time, period)
        
        # Calculate residual
        residual = detrended - seasonal
        
        # Interpolate back to original time axis
        trend_full = np.interp(time_axis, clean_time, trend)
        seasonal_full = np.interp(time_axis, clean_time, seasonal)
        residual_full = data - trend_full - seasonal_full
        
        return {
            'original': data,
            'trend': trend_full,
            'seasonal': seasonal_full,
            'residual': residual_full,
            'detrended': data - trend_full
        }
    
    def _calculate_trend(
        self,
        data: np.ndarray,
        time_axis: np.ndarray
    ) -> np.ndarray:
        """Calculate trend component using robust methods."""
        # Use centered moving average for trend
        window_size = min(int(365.25), len(data) // 3)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        # Apply moving average
        trend = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
        # Handle edges using linear extrapolation
        half_window = window_size // 2
        
        # Left edge
        if half_window > 0:
            left_slope = (trend[half_window + 1] - trend[half_window]) / \
                        (time_axis[half_window + 1] - time_axis[half_window])
            for i in range(half_window):
                trend[i] = trend[half_window] + left_slope * (time_axis[i] - time_axis[half_window])
        
        # Right edge
        if half_window > 0:
            right_slope = (trend[-half_window-1] - trend[-half_window-2]) / \
                         (time_axis[-half_window-1] - time_axis[-half_window-2])
            for i in range(len(trend) - half_window, len(trend)):
                trend[i] = trend[-half_window-1] + right_slope * (time_axis[i] - time_axis[-half_window-1])
        
        return trend
    
    def _calculate_seasonal_component(
        self,
        detrended_data: np.ndarray,
        time_axis: np.ndarray,
        period: float
    ) -> np.ndarray:
        """Calculate seasonal component using harmonic analysis."""
        # Fit harmonic components
        seasonal = np.zeros_like(detrended_data)
        
        for harmonic in range(1, self.config.harmonic_components + 1):
            # Angular frequency
            omega = 2 * np.pi * harmonic / period
            
            # Design matrix for harmonic regression
            A = np.column_stack([
                np.cos(omega * time_axis),
                np.sin(omega * time_axis)
            ])
            
            # Fit coefficients using least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, detrended_data, rcond=None)
                seasonal += A @ coeffs
            except np.linalg.LinAlgError:
                self.logger.warning(f"Failed to fit harmonic {harmonic}")
                continue
        
        return seasonal


class SeasonalStatistics:
    """Calculate seasonal statistics and patterns."""
    
    def __init__(self, config: SeasonalConfig = None):
        """
        Initialize seasonal statistics calculator.
        
        Args:
            config: Seasonal analysis configuration
        """
        self.config = config or SeasonalConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_seasonal_means(
        self,
        data: np.ndarray,
        day_of_year: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate seasonal mean values.
        
        Args:
            data: Time series data
            day_of_year: Day of year for each data point (1-365)
            
        Returns:
            Seasonal mean statistics
        """
        seasons = self._get_season_definitions()
        seasonal_means = {}
        
        for season_name, (start_day, end_day) in seasons.items():
            if isinstance(end_day, tuple):  # Handle winter crossing year boundary
                mask = (day_of_year >= start_day) | (day_of_year <= end_day[1])
            else:
                mask = (day_of_year >= start_day) & (day_of_year <= end_day)
            
            if np.any(mask):
                seasonal_means[f'{season_name}_mean'] = float(np.nanmean(data[mask]))
                seasonal_means[f'{season_name}_std'] = float(np.nanstd(data[mask]))
                seasonal_means[f'{season_name}_count'] = int(np.sum(mask))
            else:
                seasonal_means[f'{season_name}_mean'] = np.nan
                seasonal_means[f'{season_name}_std'] = np.nan
                seasonal_means[f'{season_name}_count'] = 0
        
        return seasonal_means
    
    def calculate_monthly_climatology(
        self,
        data: np.ndarray,
        dates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate monthly climatology.
        
        Args:
            data: Time series data
            dates: Array of day of year values
            
        Returns:
            Monthly climatology statistics
        """
        months = np.arange(1, 13)
        monthly_means = np.zeros(12)
        monthly_stds = np.zeros(12)
        monthly_counts = np.zeros(12, dtype=int)
        
        # Convert day of year to month
        month_boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        
        for i, month in enumerate(months):
            start_day = month_boundaries[i] + 1
            end_day = month_boundaries[i + 1]
            
            mask = (dates >= start_day) & (dates <= end_day)
            
            if np.any(mask):
                monthly_means[i] = np.nanmean(data[mask])
                monthly_stds[i] = np.nanstd(data[mask])
                monthly_counts[i] = np.sum(mask)
            else:
                monthly_means[i] = np.nan
                monthly_stds[i] = np.nan
                monthly_counts[i] = 0
        
        return {
            'months': months,
            'monthly_means': monthly_means,
            'monthly_stds': monthly_stds,
            'monthly_counts': monthly_counts,
            'peak_month': int(months[np.nanargmax(monthly_means)]),
            'minimum_month': int(months[np.nanargmin(monthly_means)]),
            'seasonal_amplitude': float(np.nanmax(monthly_means) - np.nanmin(monthly_means))
        }
    
    def identify_onset_timing(
        self,
        data: np.ndarray,
        day_of_year: np.ndarray,
        threshold_type: str = 'melt_season'
    ) -> Dict[str, float]:
        """
        Identify timing of seasonal onset (e.g., melt season start).
        
        Args:
            data: Time series data (e.g., temperature, runoff)
            day_of_year: Day of year for each data point
            threshold_type: Type of threshold to apply
            
        Returns:
            Onset timing statistics
        """
        if threshold_type == 'melt_season':
            threshold = self.config.melt_season_temp_threshold
            condition = data > threshold
        elif threshold_type == 'wet_season':
            threshold = np.nanmean(data) * self.config.wet_season_threshold
            condition = data > threshold
        elif threshold_type == 'dry_season':
            threshold = np.nanmean(data) * self.config.dry_season_threshold
            condition = data < threshold
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
        
        # Find onset and end dates
        onset_days = []
        end_days = []
        
        # Group by year (assuming multiple years of data)
        years = np.unique(day_of_year // 365)
        
        for year in years:
            year_mask = (day_of_year // 365) == year
            year_doy = day_of_year[year_mask] % 365
            year_condition = condition[year_mask]
            
            if len(year_condition) == 0:
                continue
            
            # Find first sustained period above/below threshold
            sustained_length = 5  # days
            
            for i in range(len(year_condition) - sustained_length):
                if np.all(year_condition[i:i+sustained_length]):
                    onset_days.append(year_doy[i])
                    break
            
            # Find end of season
            for i in range(len(year_condition) - sustained_length - 1, -1, -1):
                if np.all(year_condition[i:i+sustained_length]):
                    end_days.append(year_doy[i + sustained_length - 1])
                    break
        
        if onset_days:
            mean_onset = np.mean(onset_days)
            std_onset = np.std(onset_days)
        else:
            mean_onset = np.nan
            std_onset = np.nan
        
        if end_days:
            mean_end = np.mean(end_days)
            std_end = np.std(end_days)
            season_length = np.mean([end - onset for onset, end in zip(onset_days, end_days)])
        else:
            mean_end = np.nan
            std_end = np.nan
            season_length = np.nan
        
        return {
            f'{threshold_type}_onset_mean': float(mean_onset),
            f'{threshold_type}_onset_std': float(std_onset),
            f'{threshold_type}_end_mean': float(mean_end),
            f'{threshold_type}_end_std': float(std_end),
            f'{threshold_type}_length_mean': float(season_length),
            'threshold_value': float(threshold),
            'years_analyzed': len(onset_days)
        }
    
    def _get_season_definitions(self) -> Dict[str, Tuple]:
        """Get season definitions based on configuration."""
        if self.config.season_definition == SeasonDefinition.METEOROLOGICAL:
            return {
                'spring': (60, 151),   # Mar 1 - May 31
                'summer': (152, 243),  # Jun 1 - Aug 31
                'autumn': (244, 334),  # Sep 1 - Nov 30
                'winter': (335, (59,)) # Dec 1 - Feb 28/29
            }
        elif self.config.season_definition == SeasonDefinition.HYDROLOGICAL:
            start = self.config.water_year_start
            return {
                'early_water_year': (start, start + 91),
                'mid_water_year': (start + 92, start + 183),
                'late_water_year': (start + 184, start + 274),
                'end_water_year': (start + 275, (start - 1,))
            }
        elif self.config.season_definition == SeasonDefinition.MONSOON:
            monsoon_start = self.config.monsoon_start
            monsoon_end = self.config.monsoon_end
            return {
                'pre_monsoon': (60, monsoon_start - 1),
                'monsoon': (monsoon_start, monsoon_end),
                'post_monsoon': (monsoon_end + 1, 334),
                'winter': (335, (59,))
            }
        else:
            # Default to meteorological
            return self._get_season_definitions()


class TrendAnalysis:
    """Analyze long-term trends in seasonal patterns."""
    
    def __init__(self, config: SeasonalConfig = None):
        """
        Initialize trend analysis.
        
        Args:
            config: Seasonal analysis configuration
        """
        self.config = config or SeasonalConfig()
        self.logger = logging.getLogger(__name__)
    
    def analyze_trends(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        method: TrendMethod = TrendMethod.MANN_KENDALL
    ) -> Dict[str, float]:
        """
        Analyze trends in time series data.
        
        Args:
            data: Time series data
            time_axis: Time axis (years or days)
            method: Trend analysis method
            
        Returns:
            Trend analysis results
        """
        # Remove NaN values
        valid_mask = ~np.isnan(data)
        clean_data = data[valid_mask]
        clean_time = time_axis[valid_mask]
        
        if len(clean_data) < 10:
            self.logger.warning("Insufficient data for trend analysis")
            return {
                'trend_slope': np.nan,
                'trend_significance': np.nan,
                'trend_direction': 'insufficient_data'
            }
        
        if method == TrendMethod.LINEAR:
            return self._linear_trend(clean_data, clean_time)
        elif method == TrendMethod.MANN_KENDALL:
            return self._mann_kendall_trend(clean_data, clean_time)
        elif method == TrendMethod.THEIL_SEN:
            return self._theil_sen_trend(clean_data, clean_time)
        else:
            return self._polynomial_trend(clean_data, clean_time)
    
    def _linear_trend(
        self,
        data: np.ndarray,
        time_axis: np.ndarray
    ) -> Dict[str, float]:
        """Calculate linear trend using least squares."""
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_axis, data)
        
        # Determine trend direction
        if p_value < self.config.trend_significance_level:
            if slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        else:
            direction = 'no_significant_trend'
        
        return {
            'trend_slope': float(slope),
            'trend_intercept': float(intercept),
            'trend_r_squared': float(r_value ** 2),
            'trend_p_value': float(p_value),
            'trend_std_error': float(std_err),
            'trend_significance': float(p_value < self.config.trend_significance_level),
            'trend_direction': direction
        }
    
    def _mann_kendall_trend(
        self,
        data: np.ndarray,
        time_axis: np.ndarray
    ) -> Dict[str, float]:
        """Mann-Kendall trend test (non-parametric)."""
        n = len(data)
        
        # Calculate Mann-Kendall statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend direction
        if p_value < self.config.trend_significance_level:
            if s > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        else:
            direction = 'no_significant_trend'
        
        return {
            'trend_slope': float(s / (n * (n - 1) / 2)),  # Normalized slope
            'mann_kendall_s': float(s),
            'mann_kendall_z': float(z),
            'trend_p_value': float(p_value),
            'trend_significance': float(p_value < self.config.trend_significance_level),
            'trend_direction': direction
        }
    
    def _theil_sen_trend(
        self,
        data: np.ndarray,
        time_axis: np.ndarray
    ) -> Dict[str, float]:
        """Theil-Sen trend estimator (robust)."""
        n = len(data)
        slopes = []
        
        # Calculate all pairwise slopes
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time_axis[j] != time_axis[i]:
                    slope = (data[j] - data[i]) / (time_axis[j] - time_axis[i])
                    slopes.append(slope)
        
        if not slopes:
            return {
                'trend_slope': np.nan,
                'trend_significance': np.nan,
                'trend_direction': 'insufficient_data'
            }
        
        # Median slope
        median_slope = np.median(slopes)
        
        # Estimate intercept
        intercept = np.median(data - median_slope * time_axis)
        
        # Simple significance test (could be improved)
        fitted = median_slope * time_axis + intercept
        residuals = data - fitted
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Rough p-value estimation
        t_stat = abs(median_slope) / (rmse / np.sqrt(np.var(time_axis)))
        p_value = 2 * (1 - stats.t.cdf(t_stat, n - 2))
        
        # Determine trend direction
        if p_value < self.config.trend_significance_level:
            if median_slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        else:
            direction = 'no_significant_trend'
        
        return {
            'trend_slope': float(median_slope),
            'trend_intercept': float(intercept),
            'trend_p_value': float(p_value),
            'trend_significance': float(p_value < self.config.trend_significance_level),
            'trend_direction': direction,
            'trend_rmse': float(rmse)
        }
    
    def _polynomial_trend(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        degree: int = 2
    ) -> Dict[str, float]:
        """Polynomial trend analysis."""
        # Fit polynomial
        coeffs = np.polyfit(time_axis, data, degree)
        fitted = np.polyval(coeffs, time_axis)
        
        # Calculate R-squared
        ss_res = np.sum((data - fitted) ** 2)
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Linear component (first derivative at mean time)
        mean_time = np.mean(time_axis)
        if degree >= 1:
            linear_slope = coeffs[-2]  # Linear coefficient
        else:
            linear_slope = 0
        
        return {
            'trend_slope': float(linear_slope),
            'polynomial_coefficients': coeffs.tolist(),
            'trend_r_squared': float(r_squared),
            'polynomial_degree': degree,
            'trend_direction': 'increasing' if linear_slope > 0 else 'decreasing'
        }


class SeasonalAnalyzer:
    """Main class for comprehensive seasonal analysis."""
    
    def __init__(self, config: SeasonalConfig = None):
        """
        Initialize seasonal analyzer.
        
        Args:
            config: Seasonal analysis configuration
        """
        self.config = config or SeasonalConfig()
        self.decomposer = SeasonalDecomposition(self.config)
        self.statistics = SeasonalStatistics(self.config)
        self.trend_analyzer = TrendAnalysis(self.config)
        self.logger = logging.getLogger(__name__)
    
    def analyze_seasonal_patterns(
        self,
        data: Dict[str, np.ndarray],
        time_info: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive seasonal analysis.
        
        Args:
            data: Dictionary containing time series data
            time_info: Dictionary containing time information (day_of_year, year, etc.)
            
        Returns:
            Comprehensive seasonal analysis results
        """
        results = {}
        
        for variable_name, variable_data in data.items():
            self.logger.info(f"Analyzing seasonal patterns for {variable_name}")
            
            # Time series decomposition
            decomposition = self.decomposer.decompose_time_series(
                variable_data,
                time_info.get('time_axis', np.arange(len(variable_data)))
            )
            
            # Seasonal statistics
            seasonal_stats = self.statistics.calculate_seasonal_means(
                variable_data,
                time_info.get('day_of_year', np.arange(len(variable_data)) % 365 + 1)
            )
            
            # Monthly climatology
            monthly_climate = self.statistics.calculate_monthly_climatology(
                variable_data,
                time_info.get('day_of_year', np.arange(len(variable_data)) % 365 + 1)
            )
            
            # Onset timing analysis
            onset_analysis = {}
            if 'temperature' in variable_name.lower():
                onset_analysis = self.statistics.identify_onset_timing(
                    variable_data,
                    time_info.get('day_of_year', np.arange(len(variable_data)) % 365 + 1),
                    'melt_season'
                )
            elif 'runoff' in variable_name.lower() or 'flow' in variable_name.lower():
                onset_analysis = self.statistics.identify_onset_timing(
                    variable_data,
                    time_info.get('day_of_year', np.arange(len(variable_data)) % 365 + 1),
                    'wet_season'
                )
            
            # Trend analysis
            trend_analysis = self.trend_analyzer.analyze_trends(
                variable_data,
                time_info.get('year', np.arange(len(variable_data)) / 365.25)
            )
            
            # Seasonal trend analysis
            seasonal_trends = self._analyze_seasonal_trends(
                variable_data,
                time_info.get('day_of_year', np.arange(len(variable_data)) % 365 + 1),
                time_info.get('year', np.arange(len(variable_data)) / 365.25)
            )
            
            results[variable_name] = {
                'decomposition': decomposition,
                'seasonal_statistics': seasonal_stats,
                'monthly_climatology': monthly_climate,
                'onset_timing': onset_analysis,
                'trend_analysis': trend_analysis,
                'seasonal_trends': seasonal_trends
            }
        
        # Cross-variable analysis
        if len(data) > 1:
            results['cross_variable_analysis'] = self._analyze_cross_variable_patterns(data, time_info)
        
        return results
    
    def _analyze_seasonal_trends(
        self,
        data: np.ndarray,
        day_of_year: np.ndarray,
        year: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze trends within each season."""
        seasons = self.statistics._get_season_definitions()
        seasonal_trends = {}
        
        for season_name, (start_day, end_day) in seasons.items():
            if isinstance(end_day, tuple):  # Handle winter crossing year boundary
                mask = (day_of_year >= start_day) | (day_of_year <= end_day[0])
            else:
                mask = (day_of_year >= start_day) & (day_of_year <= end_day)
            
            if np.sum(mask) > 10:  # Minimum data points for trend analysis
                seasonal_data = data[mask]
                seasonal_years = year[mask]
                
                trend_result = self.trend_analyzer.analyze_trends(
                    seasonal_data, seasonal_years
                )
                seasonal_trends[season_name] = trend_result
            else:
                seasonal_trends[season_name] = {
                    'trend_slope': np.nan,
                    'trend_significance': np.nan,
                    'trend_direction': 'insufficient_data'
                }
        
        return seasonal_trends
    
    def _analyze_cross_variable_patterns(
        self,
        data: Dict[str, np.ndarray],
        time_info: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze relationships between different variables."""
        variables = list(data.keys())
        correlations = {}
        
        # Calculate seasonal correlations
        day_of_year = time_info.get('day_of_year', np.arange(len(list(data.values())[0])) % 365 + 1)
        seasons = self.statistics._get_season_definitions()
        
        for season_name, (start_day, end_day) in seasons.items():
            if isinstance(end_day, tuple):
                mask = (day_of_year >= start_day) | (day_of_year <= end_day[0])
            else:
                mask = (day_of_year >= start_day) & (day_of_year <= end_day)
            
            season_correlations = {}
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables[i+1:], i+1):
                    data1 = data[var1][mask]
                    data2 = data[var2][mask]
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(data1) | np.isnan(data2))
                    if np.sum(valid_mask) > 5:
                        corr_coeff, p_value = stats.pearsonr(data1[valid_mask], data2[valid_mask])
                        season_correlations[f'{var1}_vs_{var2}'] = {
                            'correlation': float(corr_coeff),
                            'p_value': float(p_value),
                            'significant': p_value < self.config.trend_significance_level
                        }
            
            correlations[season_name] = season_correlations
        
        return {
            'seasonal_correlations': correlations,
            'variables_analyzed': variables
        }
    
    def generate_seasonal_summary(
        self,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate human-readable summary of seasonal analysis.
        
        Args:
            analysis_results: Results from analyze_seasonal_patterns
            
        Returns:
            Summary descriptions
        """
        summaries = {}
        
        for variable, results in analysis_results.items():
            if variable == 'cross_variable_analysis':
                continue
            
            summary_parts = []
            
            # Seasonal pattern
            monthly_climate = results.get('monthly_climatology', {})
            if 'peak_month' in monthly_climate:
                peak_month = monthly_climate['peak_month']
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                summary_parts.append(f"Peak occurs in {month_names[peak_month-1]}")
            
            # Trend
            trend = results.get('trend_analysis', {})
            if 'trend_direction' in trend:
                direction = trend['trend_direction']
                if direction != 'no_significant_trend':
                    slope = trend.get('trend_slope', 0)
                    summary_parts.append(f"Shows {direction} trend (slope: {slope:.3f})")
                else:
                    summary_parts.append("No significant long-term trend")
            
            # Seasonal amplitude
            if 'seasonal_amplitude' in monthly_climate:
                amplitude = monthly_climate['seasonal_amplitude']
                summary_parts.append(f"Seasonal amplitude: {amplitude:.2f}")
            
            summaries[variable] = "; ".join(summary_parts)
        
        return summaries


def create_seasonal_analyzer(
    season_definition: SeasonDefinition = SeasonDefinition.METEOROLOGICAL,
    **kwargs
) -> SeasonalAnalyzer:
    """
    Create a seasonal analyzer with specified configuration.
    
    Args:
        season_definition: How to define seasons
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured seasonal analyzer
    """
    config = SeasonalConfig(season_definition=season_definition, **kwargs)
    return SeasonalAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate synthetic data
    days = np.arange(365 * 5)  # 5 years
    years = days / 365.25
    day_of_year = (days % 365) + 1
    
    # Synthetic temperature with trend and seasonal cycle
    base_temp = 5 + 0.1 * years  # Warming trend
    seasonal_temp = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = np.random.normal(0, 2, len(days))
    temperature = base_temp + seasonal_temp + noise
    
    # Synthetic runoff (glacier-influenced)
    base_runoff = 20 + 0.05 * years
    seasonal_runoff = 30 * np.sin(2 * np.pi * (day_of_year - 120) / 365) ** 2
    runoff_noise = np.random.normal(0, 5, len(days))
    runoff = np.maximum(base_runoff + seasonal_runoff + runoff_noise, 1)
    
    # Create analyzer
    analyzer = create_seasonal_analyzer(
        season_definition=SeasonDefinition.METEOROLOGICAL,
        trend_significance_level=0.05
    )
    
    # Prepare data
    data = {
        'temperature': temperature,
        'runoff': runoff
    }
    
    time_info = {
        'day_of_year': day_of_year,
        'year': years,
        'time_axis': days
    }
    
    # Perform analysis
    print("Performing seasonal analysis...")
    results = analyzer.analyze_seasonal_patterns(data, time_info)
    
    # Generate summaries
    summaries = analyzer.generate_seasonal_summary(results)
    
    print("\nSeasonal Analysis Results:")
    print("=" * 40)
    
    for variable, summary in summaries.items():
        print(f"\n{variable.upper()}:")
        print(f"  {summary}")
        
        # Detailed results
        var_results = results[variable]
        
        # Monthly climatology
        monthly = var_results['monthly_climatology']
        print(f"  Peak month: {monthly['peak_month']}")
        print(f"  Seasonal amplitude: {monthly['seasonal_amplitude']:.2f}")
        
        # Trend analysis
        trend = var_results['trend_analysis']
        print(f"  Trend: {trend['trend_direction']} (p={trend.get('trend_p_value', 'N/A'):.3f})")
        
        # Seasonal statistics
        seasonal_stats = var_results['seasonal_statistics']
        for season in ['spring', 'summer', 'autumn', 'winter']:
            mean_key = f'{season}_mean'
            if mean_key in seasonal_stats:
                print(f"  {season.capitalize()}: {seasonal_stats[mean_key]:.2f}")
    
    # Plot results for temperature
    if 'temperature' in results:
        temp_results = results['temperature']
        decomp = temp_results['decomposition']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original and trend
        axes[0, 0].plot(days, decomp['original'], alpha=0.7, label='Original')
        axes[0, 0].plot(days, decomp['trend'], 'r-', linewidth=2, label='Trend')
        axes[0, 0].set_title('Temperature: Original and Trend')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        
        # Seasonal component
        axes[0, 1].plot(days[:365], decomp['seasonal'][:365])
        axes[0, 1].set_title('Seasonal Component (First Year)')
        axes[0, 1].set_ylabel('Temperature (°C)')
        
        # Monthly climatology
        monthly = temp_results['monthly_climatology']
        axes[1, 0].bar(monthly['months'], monthly['monthly_means'])
        axes[1, 0].set_title('Monthly Climatology')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Temperature (°C)')
        
        # Residuals
        axes[1, 1].plot(days, decomp['residual'], alpha=0.7)
        axes[1, 1].set_title('Residuals')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Temperature (°C)')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nAnalysis completed for {len(days)} days of data")
    print(f"Variables analyzed: {list(data.keys())}")