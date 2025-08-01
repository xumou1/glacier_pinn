"""Hydrological Modeling Module.

This module provides comprehensive hydrological modeling capabilities
for glacier-fed river systems and watershed analysis.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from abc import ABC, abstractmethod


class RoutingMethod(Enum):
    """River routing methods."""
    MUSKINGUM = "muskingum"
    KINEMATIC_WAVE = "kinematic_wave"
    DIFFUSION_WAVE = "diffusion_wave"
    UNIT_HYDROGRAPH = "unit_hydrograph"


class EvapotranspirationMethod(Enum):
    """Evapotranspiration calculation methods."""
    PENMAN_MONTEITH = "penman_monteith"
    PRIESTLEY_TAYLOR = "priestley_taylor"
    HARGREAVES = "hargreaves"
    BLANEY_CRIDDLE = "blaney_criddle"


@dataclass
class WatershedProperties:
    """Watershed physical properties."""
    area: float  # km²
    mean_elevation: float  # m
    mean_slope: float  # degrees
    drainage_density: float  # km/km²
    channel_length: float  # km
    channel_slope: float  # m/m
    land_use_fractions: Dict[str, float] = None  # fraction of each land use type
    soil_properties: Dict[str, float] = None  # soil hydraulic properties
    
    def __post_init__(self):
        if self.land_use_fractions is None:
            self.land_use_fractions = {
                'glacier': 0.3,
                'bare_rock': 0.4,
                'vegetation': 0.2,
                'water': 0.1
            }
        
        if self.soil_properties is None:
            self.soil_properties = {
                'porosity': 0.4,
                'field_capacity': 0.25,
                'wilting_point': 0.1,
                'saturated_conductivity': 10.0  # mm/day
            }


@dataclass
class HydrologicalParameters:
    """Hydrological model parameters."""
    # Runoff generation
    curve_number: float = 70.0  # SCS curve number
    initial_abstraction_ratio: float = 0.2  # Ia/S ratio
    
    # Evapotranspiration
    crop_coefficient: float = 0.8
    albedo: float = 0.3
    
    # Baseflow
    baseflow_recession_constant: float = 0.95
    initial_baseflow: float = 1.0  # m³/s
    
    # Snow/ice processes
    snow_temperature_threshold: float = 0.0  # °C
    ice_temperature_threshold: float = -2.0  # °C
    degree_day_factor_snow: float = 3.0  # mm/°C/day
    degree_day_factor_ice: float = 6.0  # mm/°C/day
    
    # Routing parameters
    muskingum_k: float = 1.0  # hours
    muskingum_x: float = 0.2  # dimensionless
    
    # Soil parameters
    soil_depth: float = 1000.0  # mm
    infiltration_capacity: float = 50.0  # mm/day


class BaseHydrologicalProcess(ABC):
    """Base class for hydrological processes."""
    
    def __init__(self, parameters: HydrologicalParameters = None):
        self.parameters = parameters or HydrologicalParameters()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate process outputs from inputs."""
        pass


class EvapotranspirationCalculator(BaseHydrologicalProcess):
    """Calculator for evapotranspiration processes."""
    
    def __init__(
        self,
        method: EvapotranspirationMethod = EvapotranspirationMethod.PENMAN_MONTEITH,
        parameters: HydrologicalParameters = None
    ):
        super().__init__(parameters)
        self.method = method
    
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate evapotranspiration.
        
        Args:
            inputs: Dictionary containing meteorological data
            
        Returns:
            Evapotranspiration components
        """
        if self.method == EvapotranspirationMethod.PENMAN_MONTEITH:
            return self._penman_monteith(inputs)
        elif self.method == EvapotranspirationMethod.PRIESTLEY_TAYLOR:
            return self._priestley_taylor(inputs)
        elif self.method == EvapotranspirationMethod.HARGREAVES:
            return self._hargreaves(inputs)
        else:
            return self._blaney_criddle(inputs)
    
    def _penman_monteith(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Penman-Monteith evapotranspiration calculation."""
        # Extract inputs
        temp = inputs['temperature']  # °C
        humidity = inputs.get('relative_humidity', 70.0)  # %
        wind_speed = inputs.get('wind_speed', 2.0)  # m/s
        solar_radiation = inputs.get('solar_radiation', 200.0)  # W/m²
        pressure = inputs.get('atmospheric_pressure', 101.3)  # kPa
        
        # Constants
        gamma = 0.665 * pressure  # Psychrometric constant
        delta = 4098 * (0.6108 * np.exp(17.27 * temp / (temp + 237.3))) / ((temp + 237.3) ** 2)
        
        # Saturation vapor pressure
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        ea = es * humidity / 100.0
        
        # Net radiation (simplified)
        rn = solar_radiation * 0.0864  # Convert W/m² to MJ/m²/day
        
        # Reference evapotranspiration (mm/day)
        et0 = (0.408 * delta * rn + gamma * 900 / (temp + 273) * wind_speed * (es - ea)) / \
              (delta + gamma * (1 + 0.34 * wind_speed))
        
        # Actual evapotranspiration
        et_actual = et0 * self.parameters.crop_coefficient
        
        return {
            'reference_et': float(et0),
            'actual_et': float(et_actual),
            'crop_coefficient': self.parameters.crop_coefficient
        }
    
    def _priestley_taylor(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Priestley-Taylor evapotranspiration calculation."""
        temp = inputs['temperature']
        solar_radiation = inputs.get('solar_radiation', 200.0)
        
        # Slope of saturation vapor pressure curve
        delta = 4098 * (0.6108 * np.exp(17.27 * temp / (temp + 237.3))) / ((temp + 237.3) ** 2)
        
        # Psychrometric constant (simplified)
        gamma = 0.665 * 101.3
        
        # Net radiation
        rn = solar_radiation * 0.0864
        
        # Priestley-Taylor coefficient
        alpha = 1.26
        
        # Evapotranspiration
        et = alpha * delta / (delta + gamma) * rn / 2.45
        
        return {
            'reference_et': float(et),
            'actual_et': float(et * self.parameters.crop_coefficient),
            'crop_coefficient': self.parameters.crop_coefficient
        }
    
    def _hargreaves(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Hargreaves evapotranspiration calculation."""
        temp_mean = inputs['temperature']
        temp_max = inputs.get('temperature_max', temp_mean + 5)
        temp_min = inputs.get('temperature_min', temp_mean - 5)
        solar_radiation = inputs.get('solar_radiation', 200.0)
        
        # Extraterrestrial radiation (simplified)
        ra = solar_radiation * 0.0864 / 0.5  # Approximate
        
        # Hargreaves equation
        et0 = 0.0023 * (temp_mean + 17.8) * np.sqrt(temp_max - temp_min) * ra
        
        return {
            'reference_et': float(et0),
            'actual_et': float(et0 * self.parameters.crop_coefficient),
            'crop_coefficient': self.parameters.crop_coefficient
        }
    
    def _blaney_criddle(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Blaney-Criddle evapotranspiration calculation."""
        temp = inputs['temperature']
        daylight_hours = inputs.get('daylight_hours', 12.0)
        
        # Simplified Blaney-Criddle
        et0 = (0.46 * temp + 8.13) * daylight_hours / 24
        
        return {
            'reference_et': float(et0),
            'actual_et': float(et0 * self.parameters.crop_coefficient),
            'crop_coefficient': self.parameters.crop_coefficient
        }


class RunoffGenerator(BaseHydrologicalProcess):
    """Generator for surface runoff using SCS curve number method."""
    
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate surface runoff using SCS curve number method.
        
        Args:
            inputs: Dictionary containing precipitation and soil moisture
            
        Returns:
            Runoff components
        """
        precipitation = inputs['precipitation']  # mm
        antecedent_moisture = inputs.get('antecedent_moisture', 0.5)  # fraction
        
        # Adjust curve number for antecedent moisture
        cn = self.parameters.curve_number
        if antecedent_moisture < 0.3:  # Dry conditions
            cn_adjusted = cn / (2.281 - 0.01281 * cn)
        elif antecedent_moisture > 0.7:  # Wet conditions
            cn_adjusted = cn / (0.427 + 0.00573 * cn)
        else:  # Normal conditions
            cn_adjusted = cn
        
        # Maximum potential retention
        s = 25400 / cn_adjusted - 254  # mm
        
        # Initial abstraction
        ia = self.parameters.initial_abstraction_ratio * s
        
        # Surface runoff
        if precipitation > ia:
            runoff = (precipitation - ia) ** 2 / (precipitation - ia + s)
        else:
            runoff = 0.0
        
        # Infiltration
        infiltration = min(precipitation - runoff, self.parameters.infiltration_capacity)
        
        return {
            'surface_runoff': float(runoff),
            'infiltration': float(infiltration),
            'curve_number_adjusted': float(cn_adjusted),
            'potential_retention': float(s)
        }


class BaseflowCalculator(BaseHydrologicalProcess):
    """Calculator for baseflow using recession analysis."""
    
    def __init__(self, parameters: HydrologicalParameters = None):
        super().__init__(parameters)
        self.previous_baseflow = self.parameters.initial_baseflow
    
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate baseflow using recession equation.
        
        Args:
            inputs: Dictionary containing groundwater recharge
            
        Returns:
            Baseflow components
        """
        recharge = inputs.get('groundwater_recharge', 0.0)  # mm/day
        watershed_area = inputs.get('watershed_area', 100.0)  # km²
        
        # Convert recharge to flow units
        recharge_flow = recharge * watershed_area * 1000 / 86400  # m³/s
        
        # Baseflow recession
        baseflow = self.previous_baseflow * self.parameters.baseflow_recession_constant + recharge_flow
        
        # Update for next time step
        self.previous_baseflow = baseflow
        
        return {
            'baseflow': float(baseflow),
            'groundwater_recharge': float(recharge),
            'recession_constant': self.parameters.baseflow_recession_constant
        }


class SnowIceModel(BaseHydrologicalProcess):
    """Model for snow and ice melt processes."""
    
    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate snow and ice melt.
        
        Args:
            inputs: Dictionary containing temperature and snow/ice data
            
        Returns:
            Melt components
        """
        temperature = inputs['temperature']  # °C
        snow_depth = inputs.get('snow_depth', 0.0)  # mm water equivalent
        ice_thickness = inputs.get('ice_thickness', 0.0)  # mm water equivalent
        
        # Snow melt
        if temperature > self.parameters.snow_temperature_threshold and snow_depth > 0:
            snow_melt = min(
                self.parameters.degree_day_factor_snow * 
                (temperature - self.parameters.snow_temperature_threshold),
                snow_depth
            )
        else:
            snow_melt = 0.0
        
        # Ice melt
        if temperature > self.parameters.ice_temperature_threshold and ice_thickness > 0:
            ice_melt = min(
                self.parameters.degree_day_factor_ice * 
                (temperature - self.parameters.ice_temperature_threshold),
                ice_thickness
            )
        else:
            ice_melt = 0.0
        
        # Update snow and ice storage
        new_snow_depth = max(0, snow_depth - snow_melt)
        new_ice_thickness = max(0, ice_thickness - ice_melt)
        
        return {
            'snow_melt': float(snow_melt),
            'ice_melt': float(ice_melt),
            'total_melt': float(snow_melt + ice_melt),
            'remaining_snow': float(new_snow_depth),
            'remaining_ice': float(new_ice_thickness)
        }


class RiverRouting:
    """River routing using various methods."""
    
    def __init__(
        self,
        method: RoutingMethod = RoutingMethod.MUSKINGUM,
        parameters: HydrologicalParameters = None
    ):
        self.method = method
        self.parameters = parameters or HydrologicalParameters()
        self.logger = logging.getLogger(__name__)
        
        # Storage for routing calculations
        self.previous_inflow = 0.0
        self.previous_outflow = 0.0
        self.storage = 0.0
    
    def route_flow(
        self,
        inflow: float,
        time_step: float = 1.0
    ) -> Dict[str, float]:
        """
        Route flow through river reach.
        
        Args:
            inflow: Inflow to reach (m³/s)
            time_step: Time step (hours)
            
        Returns:
            Routing results
        """
        if self.method == RoutingMethod.MUSKINGUM:
            return self._muskingum_routing(inflow, time_step)
        elif self.method == RoutingMethod.KINEMATIC_WAVE:
            return self._kinematic_wave_routing(inflow, time_step)
        else:
            # Simple lag routing as fallback
            return self._simple_lag_routing(inflow, time_step)
    
    def _muskingum_routing(
        self,
        inflow: float,
        time_step: float
    ) -> Dict[str, float]:
        """Muskingum routing method."""
        k = self.parameters.muskingum_k  # hours
        x = self.parameters.muskingum_x
        
        # Muskingum coefficients
        c1 = (time_step - 2 * k * x) / (2 * k * (1 - x) + time_step)
        c2 = (time_step + 2 * k * x) / (2 * k * (1 - x) + time_step)
        c3 = (2 * k * (1 - x) - time_step) / (2 * k * (1 - x) + time_step)
        
        # Calculate outflow
        outflow = c1 * inflow + c2 * self.previous_inflow + c3 * self.previous_outflow
        
        # Update storage
        self.storage = k * (x * inflow + (1 - x) * outflow)
        
        # Update previous values
        self.previous_inflow = inflow
        self.previous_outflow = outflow
        
        return {
            'outflow': float(max(0, outflow)),
            'storage': float(self.storage),
            'travel_time': float(k),
            'attenuation': float(1 - x)
        }
    
    def _kinematic_wave_routing(
        self,
        inflow: float,
        time_step: float
    ) -> Dict[str, float]:
        """Simplified kinematic wave routing."""
        # Simplified approach - assume constant velocity
        velocity = 1.0  # m/s (should be calculated from channel properties)
        channel_length = 1000.0  # m (should be from watershed properties)
        
        travel_time = channel_length / velocity / 3600  # hours
        
        # Simple translation with no attenuation
        if travel_time <= time_step:
            outflow = inflow
        else:
            # Linear interpolation for sub-time step routing
            fraction = time_step / travel_time
            outflow = fraction * inflow + (1 - fraction) * self.previous_outflow
        
        self.previous_outflow = outflow
        
        return {
            'outflow': float(max(0, outflow)),
            'storage': float(inflow * travel_time / 24),  # Approximate storage
            'travel_time': float(travel_time),
            'velocity': float(velocity)
        }
    
    def _simple_lag_routing(
        self,
        inflow: float,
        time_step: float
    ) -> Dict[str, float]:
        """Simple lag routing."""
        lag_time = self.parameters.muskingum_k
        
        if lag_time <= time_step:
            outflow = inflow
        else:
            # Exponential lag
            alpha = time_step / lag_time
            outflow = alpha * inflow + (1 - alpha) * self.previous_outflow
        
        self.previous_outflow = outflow
        
        return {
            'outflow': float(max(0, outflow)),
            'storage': float(inflow * lag_time / 24),
            'travel_time': float(lag_time),
            'attenuation': 0.0
        }


class WatershedModel:
    """Comprehensive watershed hydrological model."""
    
    def __init__(
        self,
        watershed_properties: WatershedProperties,
        parameters: HydrologicalParameters = None,
        et_method: EvapotranspirationMethod = EvapotranspirationMethod.PENMAN_MONTEITH,
        routing_method: RoutingMethod = RoutingMethod.MUSKINGUM
    ):
        """
        Initialize watershed model.
        
        Args:
            watershed_properties: Watershed physical properties
            parameters: Hydrological parameters
            et_method: Evapotranspiration calculation method
            routing_method: River routing method
        """
        self.watershed = watershed_properties
        self.parameters = parameters or HydrologicalParameters()
        
        # Initialize process models
        self.et_calculator = EvapotranspirationCalculator(et_method, self.parameters)
        self.runoff_generator = RunoffGenerator(self.parameters)
        self.baseflow_calculator = BaseflowCalculator(self.parameters)
        self.snow_ice_model = SnowIceModel(self.parameters)
        self.river_routing = RiverRouting(routing_method, self.parameters)
        
        # State variables
        self.soil_moisture = self.parameters.soil_depth * 0.5  # mm
        self.snow_storage = 0.0  # mm
        self.ice_storage = 0.0  # mm
        
        self.logger = logging.getLogger(__name__)
    
    def simulate_timestep(
        self,
        meteorological_inputs: Dict[str, float],
        glacier_inputs: Dict[str, float] = None,
        time_step: float = 24.0  # hours
    ) -> Dict[str, float]:
        """
        Simulate one time step of watershed hydrology.
        
        Args:
            meteorological_inputs: Meteorological forcing data
            glacier_inputs: Glacier-specific inputs
            time_step: Time step in hours
            
        Returns:
            Hydrological outputs for the time step
        """
        if glacier_inputs is None:
            glacier_inputs = {}
        
        # Extract inputs
        precipitation = meteorological_inputs.get('precipitation', 0.0)  # mm
        temperature = meteorological_inputs.get('temperature', 0.0)  # °C
        
        # Calculate evapotranspiration
        et_results = self.et_calculator.calculate(meteorological_inputs)
        actual_et = et_results['actual_et']
        
        # Snow and ice processes
        snow_ice_inputs = {
            'temperature': temperature,
            'snow_depth': self.snow_storage,
            'ice_thickness': self.ice_storage
        }
        snow_ice_results = self.snow_ice_model.calculate(snow_ice_inputs)
        
        # Update snow and ice storage
        # Add new snow (precipitation when temp < 0)
        if temperature < 0:
            new_snow = precipitation
            liquid_precipitation = 0.0
        else:
            new_snow = 0.0
            liquid_precipitation = precipitation
        
        self.snow_storage = snow_ice_results['remaining_snow'] + new_snow
        self.ice_storage = snow_ice_results['remaining_ice']
        
        # Add glacier melt from external inputs
        glacier_melt = glacier_inputs.get('glacier_melt', 0.0)  # mm
        total_melt = snow_ice_results['total_melt'] + glacier_melt
        
        # Total water input to soil
        water_input = liquid_precipitation + total_melt
        
        # Calculate runoff
        runoff_inputs = {
            'precipitation': water_input,
            'antecedent_moisture': self.soil_moisture / self.parameters.soil_depth
        }
        runoff_results = self.runoff_generator.calculate(runoff_inputs)
        surface_runoff = runoff_results['surface_runoff']
        infiltration = runoff_results['infiltration']
        
        # Update soil moisture
        self.soil_moisture += infiltration - actual_et
        self.soil_moisture = max(0, min(self.soil_moisture, self.parameters.soil_depth))
        
        # Calculate groundwater recharge (excess soil moisture)
        field_capacity = self.parameters.soil_depth * \
                        self.watershed.soil_properties['field_capacity']
        
        if self.soil_moisture > field_capacity:
            groundwater_recharge = self.soil_moisture - field_capacity
            self.soil_moisture = field_capacity
        else:
            groundwater_recharge = 0.0
        
        # Calculate baseflow
        baseflow_inputs = {
            'groundwater_recharge': groundwater_recharge,
            'watershed_area': self.watershed.area
        }
        baseflow_results = self.baseflow_calculator.calculate(baseflow_inputs)
        baseflow = baseflow_results['baseflow']
        
        # Total runoff generation
        total_runoff_mm = surface_runoff  # mm
        total_runoff_flow = (surface_runoff * self.watershed.area * 1000 / 86400) + baseflow  # m³/s
        
        # Route flow through channel
        routing_results = self.river_routing.route_flow(total_runoff_flow, time_step)
        outlet_flow = routing_results['outflow']
        
        return {
            # Water balance components
            'precipitation': precipitation,
            'evapotranspiration': actual_et,
            'surface_runoff': surface_runoff,
            'baseflow': baseflow,
            'total_runoff': total_runoff_mm,
            'outlet_flow': outlet_flow,
            
            # Snow and ice
            'snow_melt': snow_ice_results['snow_melt'],
            'ice_melt': snow_ice_results['ice_melt'],
            'glacier_melt': glacier_melt,
            'snow_storage': self.snow_storage,
            'ice_storage': self.ice_storage,
            
            # Soil and groundwater
            'soil_moisture': self.soil_moisture,
            'infiltration': infiltration,
            'groundwater_recharge': groundwater_recharge,
            
            # Routing
            'travel_time': routing_results['travel_time'],
            'channel_storage': routing_results['storage'],
            
            # State variables
            'soil_moisture_fraction': self.soil_moisture / self.parameters.soil_depth,
            'water_balance_error': precipitation + total_melt - actual_et - total_runoff_mm - 
                                 (self.snow_storage + self.ice_storage - snow_ice_inputs['snow_depth'] - 
                                  snow_ice_inputs['ice_thickness'])
        }
    
    def simulate_period(
        self,
        meteorological_series: List[Dict[str, float]],
        glacier_series: List[Dict[str, float]] = None,
        time_step: float = 24.0
    ) -> Dict[str, List[float]]:
        """
        Simulate watershed hydrology for a time period.
        
        Args:
            meteorological_series: Time series of meteorological data
            glacier_series: Time series of glacier data
            time_step: Time step in hours
            
        Returns:
            Time series of hydrological outputs
        """
        if glacier_series is None:
            glacier_series = [{}] * len(meteorological_series)
        
        # Initialize output storage
        outputs = {}
        
        for i, (met_data, glacier_data) in enumerate(zip(meteorological_series, glacier_series)):
            result = self.simulate_timestep(met_data, glacier_data, time_step)
            
            # Store results
            for key, value in result.items():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(value)
        
        return outputs
    
    def get_water_balance_summary(
        self,
        simulation_results: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate water balance summary statistics.
        
        Args:
            simulation_results: Results from simulate_period
            
        Returns:
            Water balance summary
        """
        # Convert to numpy arrays for easier calculation
        precip = np.array(simulation_results['precipitation'])
        et = np.array(simulation_results['evapotranspiration'])
        runoff = np.array(simulation_results['total_runoff'])
        
        # Calculate totals
        total_precip = np.sum(precip)
        total_et = np.sum(et)
        total_runoff = np.sum(runoff)
        
        # Calculate fractions
        runoff_coefficient = total_runoff / total_precip if total_precip > 0 else 0.0
        et_fraction = total_et / total_precip if total_precip > 0 else 0.0
        
        # Flow statistics
        outlet_flow = np.array(simulation_results['outlet_flow'])
        mean_flow = np.mean(outlet_flow)
        peak_flow = np.max(outlet_flow)
        low_flow = np.min(outlet_flow)
        
        return {
            'total_precipitation': float(total_precip),
            'total_evapotranspiration': float(total_et),
            'total_runoff': float(total_runoff),
            'runoff_coefficient': float(runoff_coefficient),
            'evapotranspiration_fraction': float(et_fraction),
            'mean_flow': float(mean_flow),
            'peak_flow': float(peak_flow),
            'low_flow': float(low_flow),
            'flow_variability': float(np.std(outlet_flow) / mean_flow) if mean_flow > 0 else 0.0,
            'water_balance_error': float(np.mean(simulation_results['water_balance_error']))
        }


def create_watershed_model(
    watershed_area: float,
    mean_elevation: float = 4000.0,
    glacier_fraction: float = 0.3,
    **kwargs
) -> WatershedModel:
    """
    Create a watershed model with default properties.
    
    Args:
        watershed_area: Watershed area (km²)
        mean_elevation: Mean elevation (m)
        glacier_fraction: Fraction of watershed covered by glaciers
        **kwargs: Additional watershed and parameter specifications
        
    Returns:
        Configured watershed model
    """
    # Default watershed properties
    watershed_props = WatershedProperties(
        area=watershed_area,
        mean_elevation=mean_elevation,
        mean_slope=kwargs.get('mean_slope', 15.0),
        drainage_density=kwargs.get('drainage_density', 2.0),
        channel_length=kwargs.get('channel_length', np.sqrt(watershed_area) * 2),
        channel_slope=kwargs.get('channel_slope', 0.02)
    )
    
    # Update land use fractions
    watershed_props.land_use_fractions['glacier'] = glacier_fraction
    watershed_props.land_use_fractions['bare_rock'] = 0.6 - glacier_fraction
    
    # Default parameters
    parameters = HydrologicalParameters(**{k: v for k, v in kwargs.items() 
                                         if k in HydrologicalParameters.__dataclass_fields__})
    
    return WatershedModel(watershed_props, parameters)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create watershed model
    model = create_watershed_model(
        watershed_area=100.0,  # km²
        mean_elevation=4500.0,  # m
        glacier_fraction=0.4,
        curve_number=65.0
    )
    
    # Generate synthetic meteorological data
    days = 365
    met_data = []
    
    for day in range(days):
        # Seasonal temperature variation
        temp = 5 + 15 * np.sin(2 * np.pi * (day - 80) / 365)
        
        # Random precipitation
        precip = np.random.exponential(2.0) if np.random.random() < 0.3 else 0.0
        
        # Other meteorological variables
        met_data.append({
            'temperature': temp,
            'precipitation': precip,
            'relative_humidity': 70.0,
            'wind_speed': 2.0,
            'solar_radiation': 200 + 100 * np.sin(2 * np.pi * day / 365)
        })
    
    # Generate glacier melt data
    glacier_data = []
    for day in range(days):
        # Glacier melt depends on temperature and season
        temp = met_data[day]['temperature']
        if temp > 0 and 120 <= day <= 270:  # Melt season
            glacier_melt = max(0, temp * 2.0)  # Simple degree-day
        else:
            glacier_melt = 0.0
        
        glacier_data.append({'glacier_melt': glacier_melt})
    
    # Run simulation
    print("Running watershed simulation...")
    results = model.simulate_period(met_data, glacier_data)
    
    # Calculate summary
    summary = model.get_water_balance_summary(results)
    
    print("\nWater Balance Summary:")
    print("=" * 30)
    for key, value in summary.items():
        if 'fraction' in key or 'coefficient' in key or 'variability' in key:
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.1f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Precipitation and runoff
    axes[0, 0].plot(results['precipitation'], label='Precipitation', alpha=0.7)
    axes[0, 0].plot(results['total_runoff'], label='Runoff', alpha=0.7)
    axes[0, 0].set_title('Precipitation and Runoff')
    axes[0, 0].set_ylabel('mm/day')
    axes[0, 0].legend()
    
    # Flow at outlet
    axes[0, 1].plot(results['outlet_flow'])
    axes[0, 1].set_title('Flow at Watershed Outlet')
    axes[0, 1].set_ylabel('m³/s')
    
    # Snow and ice storage
    axes[1, 0].plot(results['snow_storage'], label='Snow', alpha=0.7)
    axes[1, 0].plot(results['ice_storage'], label='Ice', alpha=0.7)
    axes[1, 0].set_title('Snow and Ice Storage')
    axes[1, 0].set_ylabel('mm water equivalent')
    axes[1, 0].legend()
    
    # Soil moisture
    axes[1, 1].plot(results['soil_moisture_fraction'])
    axes[1, 1].set_title('Soil Moisture Fraction')
    axes[1, 1].set_ylabel('Fraction')
    axes[1, 1].set_ylim(0, 1)
    
    for ax in axes.flat:
        ax.set_xlabel('Day of Year')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSimulation completed for {days} days")
    print(f"Peak flow: {max(results['outlet_flow']):.1f} m³/s")
    print(f"Mean flow: {np.mean(results['outlet_flow']):.1f} m³/s")
    print(f"Runoff coefficient: {summary['runoff_coefficient']:.3f}")