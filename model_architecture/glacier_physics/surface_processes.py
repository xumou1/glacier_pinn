#!/usr/bin/env python3
"""
表面过程实现

实现冰川表面过程，包括：
- 表面质量平衡
- 消融过程
- 积累过程
- 表面能量平衡

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod

class SurfaceProcess(nn.Module, ABC):
    """
    表面过程基类
    
    定义冰川表面过程的抽象接口
    """
    
    def __init__(self):
        """
        初始化表面过程
        """
        super(SurfaceProcess, self).__init__()
    
    @abstractmethod
    def compute_surface_balance(self, coordinates: torch.Tensor,
                              time: torch.Tensor,
                              climate_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算表面平衡
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据
            
        Returns:
            Tensor: 表面平衡率
        """
        pass

class SurfaceMassBalance(SurfaceProcess):
    """
    表面质量平衡模型
    
    实现冰川表面质量平衡计算
    """
    
    def __init__(self, 
                 model_type: str = 'degree_day',
                 include_radiation: bool = True,
                 include_wind: bool = False):
        """
        初始化表面质量平衡模型
        
        Args:
            model_type: 模型类型 ('degree_day', 'energy_balance')
            include_radiation: 是否包含辐射
            include_wind: 是否包含风效应
        """
        super(SurfaceMassBalance, self).__init__()
        
        self.model_type = model_type
        self.include_radiation = include_radiation
        self.include_wind = include_wind
        
        # 模型参数
        self.freezing_temp = 273.15  # K
        self.ice_density = 917.0  # kg/m³
        self.water_density = 1000.0  # kg/m³
    
    def compute_accumulation(self, coordinates: torch.Tensor,
                           time: torch.Tensor,
                           precipitation: torch.Tensor,
                           temperature: torch.Tensor,
                           humidity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算积累率
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            precipitation: 降水量 (m/year)
            temperature: 温度 (K)
            humidity: 相对湿度
            
        Returns:
            Tensor: 积累率 (m ice equivalent/year)
        """
        # 降水相态判断
        snow_fraction = self.compute_snow_fraction(temperature, humidity)
        
        # 雪密度修正
        snow_density = self.compute_snow_density(temperature)
        
        # 积累率 = 降雪量 * 密度修正
        accumulation = precipitation * snow_fraction * (snow_density / self.ice_density)
        
        # 海拔效应
        elevation = coordinates[..., -1] if coordinates.shape[-1] > 2 else torch.zeros_like(precipitation)
        elevation_factor = self.compute_elevation_effect(elevation)
        
        accumulation = accumulation * elevation_factor
        
        return accumulation
    
    def compute_snow_fraction(self, temperature: torch.Tensor,
                            humidity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算降雪比例
        
        Args:
            temperature: 温度 (K)
            humidity: 相对湿度
            
        Returns:
            Tensor: 降雪比例 (0-1)
        """
        # 基于温度的简单模型
        temp_celsius = temperature - self.freezing_temp
        
        # Sigmoid函数平滑过渡
        snow_fraction = torch.sigmoid(-2.0 * temp_celsius)
        
        # 湿度修正
        if humidity is not None:
            # 低湿度时减少降雪
            humidity_factor = torch.clamp(humidity / 0.8, 0.5, 1.0)
            snow_fraction = snow_fraction * humidity_factor
        
        return snow_fraction
    
    def compute_snow_density(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        计算雪密度
        
        Args:
            temperature: 温度 (K)
            
        Returns:
            Tensor: 雪密度 (kg/m³)
        """
        temp_celsius = temperature - self.freezing_temp
        
        # 温度依赖的雪密度模型
        # 较冷时雪密度较低，较暖时雪密度较高
        base_density = 300.0  # kg/m³
        temp_effect = 50.0 * torch.clamp(temp_celsius, -20, 0)  # 温度效应
        
        snow_density = base_density + temp_effect
        
        return torch.clamp(snow_density, 100.0, 600.0)
    
    def compute_elevation_effect(self, elevation: torch.Tensor) -> torch.Tensor:
        """
        计算海拔效应
        
        Args:
            elevation: 海拔 (m)
            
        Returns:
            Tensor: 海拔修正因子
        """
        # 海拔梯度效应
        # 每100m海拔增加约5%的降水
        gradient = 0.0005  # 1/m
        reference_elevation = 4000.0  # m
        
        elevation_factor = 1.0 + gradient * (elevation - reference_elevation)
        
        return torch.clamp(elevation_factor, 0.5, 2.0)
    
    def compute_ablation_degree_day(self, coordinates: torch.Tensor,
                                  time: torch.Tensor,
                                  temperature: torch.Tensor,
                                  solar_radiation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算度日模型消融
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            temperature: 温度 (K)
            solar_radiation: 太阳辐射 (W/m²)
            
        Returns:
            Tensor: 消融率 (m ice equivalent/year)
        """
        temp_celsius = temperature - self.freezing_temp
        
        # 正积温
        positive_temp = torch.clamp(temp_celsius, min=0)
        
        # 度日因子
        if self.include_radiation and solar_radiation is not None:
            # 辐射增强的度日因子
            base_ddf = 0.004  # m/day/K
            radiation_factor = 1.0 + 0.0001 * solar_radiation  # 辐射增强
            ddf = base_ddf * radiation_factor
        else:
            # 标准度日因子
            ddf = 0.005  # m/day/K
        
        # 消融率
        ablation = ddf * positive_temp * 365.25  # 转换为年率
        
        # 海拔修正（高海拔消融减少）
        elevation = coordinates[..., -1] if coordinates.shape[-1] > 2 else torch.zeros_like(temperature)
        elevation_factor = torch.exp(-0.0001 * torch.clamp(elevation - 3000, min=0))
        
        ablation = ablation * elevation_factor
        
        return ablation
    
    def compute_ablation_energy_balance(self, coordinates: torch.Tensor,
                                      time: torch.Tensor,
                                      climate_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算能量平衡消融
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据字典
            
        Returns:
            Tensor: 消融率 (m ice equivalent/year)
        """
        # 提取气候变量
        temperature = climate_data['temperature']
        solar_radiation = climate_data.get('solar_radiation', torch.zeros_like(temperature))
        wind_speed = climate_data.get('wind_speed', torch.ones_like(temperature) * 2.0)
        humidity = climate_data.get('humidity', torch.ones_like(temperature) * 0.7)
        
        # 净辐射
        net_radiation = self.compute_net_radiation(
            solar_radiation, temperature, coordinates
        )
        
        # 感热通量
        sensible_heat = self.compute_sensible_heat(
            temperature, wind_speed
        )
        
        # 潜热通量
        latent_heat = self.compute_latent_heat(
            temperature, humidity, wind_speed
        )
        
        # 总能量通量
        total_energy = net_radiation + sensible_heat + latent_heat
        
        # 转换为消融率
        latent_heat_fusion = 334000.0  # J/kg
        ablation = torch.clamp(total_energy, min=0) / (
            self.ice_density * latent_heat_fusion
        ) * 365.25 * 24 * 3600  # 转换为m/year
        
        return ablation
    
    def compute_net_radiation(self, solar_radiation: torch.Tensor,
                            temperature: torch.Tensor,
                            coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算净辐射
        
        Args:
            solar_radiation: 太阳辐射 (W/m²)
            temperature: 温度 (K)
            coordinates: 空间坐标
            
        Returns:
            Tensor: 净辐射 (W/m²)
        """
        # 反照率
        albedo = self.compute_albedo(temperature, coordinates)
        
        # 短波辐射
        shortwave = solar_radiation * (1 - albedo)
        
        # 长波辐射
        stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        emissivity = 0.97  # 冰雪发射率
        
        # 向下长波辐射（简化）
        longwave_down = 0.8 * stefan_boltzmann * (temperature ** 4)
        
        # 向上长波辐射
        longwave_up = emissivity * stefan_boltzmann * (temperature ** 4)
        
        # 净辐射
        net_radiation = shortwave + longwave_down - longwave_up
        
        return net_radiation
    
    def compute_albedo(self, temperature: torch.Tensor,
                     coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算反照率
        
        Args:
            temperature: 温度 (K)
            coordinates: 空间坐标
            
        Returns:
            Tensor: 反照率 (0-1)
        """
        temp_celsius = temperature - self.freezing_temp
        
        # 基础反照率
        fresh_snow_albedo = 0.85
        old_snow_albedo = 0.65
        ice_albedo = 0.35
        
        # 温度依赖的反照率
        if temp_celsius.max() > 0:
            # 有融化时，反照率降低
            melt_factor = torch.sigmoid(2.0 * temp_celsius)
            albedo = fresh_snow_albedo * (1 - melt_factor) + ice_albedo * melt_factor
        else:
            # 无融化时，保持雪的反照率
            albedo = torch.full_like(temperature, old_snow_albedo)
        
        return torch.clamp(albedo, ice_albedo, fresh_snow_albedo)
    
    def compute_sensible_heat(self, temperature: torch.Tensor,
                            wind_speed: torch.Tensor) -> torch.Tensor:
        """
        计算感热通量
        
        Args:
            temperature: 温度 (K)
            wind_speed: 风速 (m/s)
            
        Returns:
            Tensor: 感热通量 (W/m²)
        """
        # 空气密度和比热
        air_density = 1.225  # kg/m³
        specific_heat_air = 1005.0  # J/(kg·K)
        
        # 传热系数
        heat_transfer_coeff = 0.001 * wind_speed + 0.002
        
        # 温度差（假设表面温度为冰点）
        temp_diff = temperature - self.freezing_temp
        
        # 感热通量
        sensible_heat = air_density * specific_heat_air * heat_transfer_coeff * temp_diff
        
        return sensible_heat
    
    def compute_latent_heat(self, temperature: torch.Tensor,
                          humidity: torch.Tensor,
                          wind_speed: torch.Tensor) -> torch.Tensor:
        """
        计算潜热通量
        
        Args:
            temperature: 温度 (K)
            humidity: 相对湿度 (0-1)
            wind_speed: 风速 (m/s)
            
        Returns:
            Tensor: 潜热通量 (W/m²)
        """
        # 饱和水汽压（简化公式）
        temp_celsius = temperature - self.freezing_temp
        sat_vapor_pressure = 611.2 * torch.exp(17.67 * temp_celsius / (temp_celsius + 243.5))
        
        # 实际水汽压
        actual_vapor_pressure = humidity * sat_vapor_pressure
        
        # 表面水汽压（假设为冰点饱和）
        surface_vapor_pressure = 611.2  # Pa
        
        # 水汽压差
        vapor_pressure_diff = actual_vapor_pressure - surface_vapor_pressure
        
        # 传质系数
        mass_transfer_coeff = 0.001 * wind_speed + 0.001
        
        # 潜热通量
        latent_heat_vaporization = 2.45e6  # J/kg
        latent_heat = mass_transfer_coeff * latent_heat_vaporization * vapor_pressure_diff / 1000.0
        
        return latent_heat
    
    def compute_surface_balance(self, coordinates: torch.Tensor,
                              time: torch.Tensor,
                              climate_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算表面平衡
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据
            
        Returns:
            Tensor: 表面平衡率 (m ice equivalent/year)
        """
        # 计算积累
        accumulation = self.compute_accumulation(
            coordinates, time,
            climate_data['precipitation'],
            climate_data['temperature'],
            climate_data.get('humidity', None)
        )
        
        # 计算消融
        if self.model_type == 'degree_day':
            ablation = self.compute_ablation_degree_day(
                coordinates, time,
                climate_data['temperature'],
                climate_data.get('solar_radiation', None)
            )
        elif self.model_type == 'energy_balance':
            ablation = self.compute_ablation_energy_balance(
                coordinates, time, climate_data
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 净表面平衡
        surface_balance = accumulation - ablation
        
        return surface_balance

class SurfaceEnergyBalance(SurfaceProcess):
    """
    表面能量平衡模型
    
    实现详细的表面能量平衡计算
    """
    
    def __init__(self):
        """
        初始化表面能量平衡模型
        """
        super(SurfaceEnergyBalance, self).__init__()
        
        # 物理常数
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.latent_heat_fusion = 334000.0  # J/kg
        self.latent_heat_vaporization = 2.45e6  # J/kg
        self.ice_density = 917.0  # kg/m³
    
    def compute_radiation_balance(self, solar_radiation: torch.Tensor,
                                temperature: torch.Tensor,
                                albedo: torch.Tensor,
                                cloud_cover: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算辐射平衡
        
        Args:
            solar_radiation: 太阳辐射 (W/m²)
            temperature: 温度 (K)
            albedo: 反照率
            cloud_cover: 云量 (0-1)
            
        Returns:
            Tensor: 净辐射 (W/m²)
        """
        # 短波辐射
        shortwave_net = solar_radiation * (1 - albedo)
        
        # 长波辐射
        emissivity = 0.97
        
        # 大气长波辐射
        if cloud_cover is not None:
            # 云的影响
            clear_sky_emissivity = 0.7
            cloud_emissivity = 0.95
            effective_emissivity = clear_sky_emissivity * (1 - cloud_cover) + cloud_emissivity * cloud_cover
        else:
            effective_emissivity = 0.8
        
        longwave_down = effective_emissivity * self.stefan_boltzmann * (temperature ** 4)
        longwave_up = emissivity * self.stefan_boltzmann * (temperature ** 4)
        
        longwave_net = longwave_down - longwave_up
        
        # 净辐射
        net_radiation = shortwave_net + longwave_net
        
        return net_radiation
    
    def compute_turbulent_fluxes(self, temperature: torch.Tensor,
                               air_temperature: torch.Tensor,
                               humidity: torch.Tensor,
                               wind_speed: torch.Tensor,
                               surface_roughness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算湍流通量
        
        Args:
            temperature: 表面温度 (K)
            air_temperature: 空气温度 (K)
            humidity: 相对湿度 (0-1)
            wind_speed: 风速 (m/s)
            surface_roughness: 表面粗糙度 (m)
            
        Returns:
            Tuple: (感热通量, 潜热通量) (W/m²)
        """
        # 空气物理参数
        air_density = 1.225  # kg/m³
        specific_heat_air = 1005.0  # J/(kg·K)
        
        # 传输系数
        von_karman = 0.4
        measurement_height = 2.0  # m
        
        # 粗糙度长度
        z0 = surface_roughness
        
        # 传热和传质系数
        transfer_coeff = (von_karman ** 2) / (
            torch.log(measurement_height / z0) ** 2
        ) * wind_speed
        
        # 感热通量
        sensible_heat = air_density * specific_heat_air * transfer_coeff * (
            air_temperature - temperature
        )
        
        # 潜热通量
        # 饱和水汽压
        def saturation_vapor_pressure(T):
            return 611.2 * torch.exp(17.67 * (T - 273.15) / (T - 29.65))
        
        sat_vp_air = saturation_vapor_pressure(air_temperature)
        sat_vp_surface = saturation_vapor_pressure(temperature)
        
        actual_vp = humidity * sat_vp_air
        
        # 水汽压差
        vapor_pressure_diff = actual_vp - sat_vp_surface
        
        # 潜热通量
        latent_heat = 0.622 * self.latent_heat_vaporization / 1013.25 * transfer_coeff * vapor_pressure_diff
        
        return sensible_heat, latent_heat
    
    def compute_surface_balance(self, coordinates: torch.Tensor,
                              time: torch.Tensor,
                              climate_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算表面能量平衡
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据
            
        Returns:
            Tensor: 表面平衡率 (m ice equivalent/year)
        """
        # 提取气候变量
        solar_radiation = climate_data['solar_radiation']
        temperature = climate_data['temperature']
        air_temperature = climate_data.get('air_temperature', temperature)
        humidity = climate_data['humidity']
        wind_speed = climate_data['wind_speed']
        albedo = climate_data.get('albedo', torch.full_like(temperature, 0.7))
        surface_roughness = climate_data.get('surface_roughness', torch.full_like(temperature, 0.001))
        
        # 辐射平衡
        net_radiation = self.compute_radiation_balance(
            solar_radiation, temperature, albedo,
            climate_data.get('cloud_cover', None)
        )
        
        # 湍流通量
        sensible_heat, latent_heat = self.compute_turbulent_fluxes(
            temperature, air_temperature, humidity, wind_speed, surface_roughness
        )
        
        # 总能量通量
        total_energy = net_radiation + sensible_heat + latent_heat
        
        # 转换为质量平衡
        # 正值表示能量输入（融化），负值表示能量输出（冻结）
        mass_balance = total_energy / (self.ice_density * self.latent_heat_fusion) * 365.25 * 24 * 3600
        
        return -mass_balance  # 负号因为能量输入导致质量损失

class GlacierSurfaceProcesses(nn.Module):
    """
    冰川表面过程综合模型
    
    集成各种表面过程
    """
    
    def __init__(self, 
                 mass_balance_model: str = 'degree_day',
                 include_energy_balance: bool = False):
        """
        初始化冰川表面过程模型
        
        Args:
            mass_balance_model: 质量平衡模型类型
            include_energy_balance: 是否包含能量平衡
        """
        super(GlacierSurfaceProcesses, self).__init__()
        
        self.mass_balance = SurfaceMassBalance(model_type=mass_balance_model)
        
        if include_energy_balance:
            self.energy_balance = SurfaceEnergyBalance()
        else:
            self.energy_balance = None
    
    def compute_comprehensive_surface_balance(self, 
                                            coordinates: torch.Tensor,
                                            time: torch.Tensor,
                                            climate_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算综合表面平衡
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据
            
        Returns:
            Dict: 各种表面平衡结果
        """
        results = {}
        
        # 质量平衡
        mass_balance = self.mass_balance.compute_surface_balance(
            coordinates, time, climate_data
        )
        results['mass_balance'] = mass_balance
        
        # 能量平衡
        if self.energy_balance is not None:
            energy_balance = self.energy_balance.compute_surface_balance(
                coordinates, time, climate_data
            )
            results['energy_balance'] = energy_balance
        
        # 选择最终结果
        if self.energy_balance is not None:
            final_balance = energy_balance
        else:
            final_balance = mass_balance
        
        results['final_surface_balance'] = final_balance
        
        return results
    
    def validate_surface_processes(self, 
                                 coordinates: torch.Tensor,
                                 time: torch.Tensor,
                                 climate_data: Dict[str, torch.Tensor],
                                 observed_balance: torch.Tensor) -> Dict[str, float]:
        """
        验证表面过程
        
        Args:
            coordinates: 空间坐标
            time: 时间坐标
            climate_data: 气候数据
            observed_balance: 观测的表面平衡
            
        Returns:
            Dict: 验证指标
        """
        # 计算预测的表面平衡
        results = self.compute_comprehensive_surface_balance(
            coordinates, time, climate_data
        )
        predicted_balance = results['final_surface_balance']
        
        # 计算误差
        error = torch.abs(predicted_balance - observed_balance)
        relative_error = error / (torch.abs(observed_balance) + 1e-6)
        
        validation_metrics = {
            'mean_absolute_error': torch.mean(error).item(),
            'max_absolute_error': torch.max(error).item(),
            'mean_relative_error': torch.mean(relative_error).item(),
            'max_relative_error': torch.max(relative_error).item(),
            'correlation': torch.corrcoef(torch.stack([
                predicted_balance.flatten(), observed_balance.flatten()
            ]))[0, 1].item(),
            'surface_processes_valid': torch.mean(relative_error).item() < 0.2
        }
        
        return validation_metrics

def create_surface_process_model(model_type: str = 'mass_balance',
                               **kwargs) -> nn.Module:
    """
    创建表面过程模型
    
    Args:
        model_type: 模型类型
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 表面过程模型实例
    """
    if model_type == 'mass_balance':
        return SurfaceMassBalance(**kwargs)
    elif model_type == 'energy_balance':
        return SurfaceEnergyBalance(**kwargs)
    elif model_type == 'comprehensive':
        return GlacierSurfaceProcesses(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试表面过程模型
    surface_processes = GlacierSurfaceProcesses(
        mass_balance_model='energy_balance',
        include_energy_balance=True
    )
    
    # 测试数据
    batch_size = 100
    coordinates = torch.randn(batch_size, 3) * 1000  # x, y, z (m)
    coordinates[:, 2] = torch.abs(coordinates[:, 2]) + 4000  # 海拔 > 4000m
    time = torch.randn(batch_size, 1) * 365  # days
    
    # 气候数据
    climate_data = {
        'temperature': torch.randn(batch_size, 1) * 10 + 268,  # K
        'precipitation': torch.randn(batch_size, 1) * 1 + 2,  # m/year
        'solar_radiation': torch.randn(batch_size, 1) * 100 + 200,  # W/m²
        'humidity': torch.rand(batch_size, 1) * 0.5 + 0.3,  # 0-1
        'wind_speed': torch.randn(batch_size, 1) * 2 + 3,  # m/s
        'air_temperature': torch.randn(batch_size, 1) * 10 + 270,  # K
        'surface_roughness': torch.full((batch_size, 1), 0.001)  # m
    }
    
    # 计算表面平衡
    results = surface_processes.compute_comprehensive_surface_balance(
        coordinates, time, climate_data
    )
    
    print("表面过程计算结果:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状={value.shape}, 范围=[{value.min().item():.3f}, {value.max().item():.3f}] m/year")
    
    # 验证表面过程
    observed_balance = torch.randn(batch_size, 1) * 0.5  # m/year
    validation = surface_processes.validate_surface_processes(
        coordinates, time, climate_data, observed_balance
    )
    
    print("\n表面过程验证:")
    for key, value in validation.items():
        print(f"{key}: {value}")