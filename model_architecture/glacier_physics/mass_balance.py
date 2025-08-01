#!/usr/bin/env python3
"""
冰川质量平衡模型

实现冰川质量平衡的物理过程，包括：
- 积累过程（降雪、雪崩等）
- 消融过程（融化、升华等）
- 质量守恒方程
- 表面质量平衡
- 内部质量变化

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math

class AccumulationModel(nn.Module):
    """
    积累模型
    
    模拟冰川的积累过程
    """
    
    def __init__(self, accumulation_type: str = 'temperature_precipitation',
                 learnable_parameters: bool = True):
        """
        初始化积累模型
        
        Args:
            accumulation_type: 积累模型类型
            learnable_parameters: 是否学习参数
        """
        super(AccumulationModel, self).__init__()
        
        self.accumulation_type = accumulation_type
        
        if learnable_parameters:
            # 温度阈值参数
            self.snow_temperature_threshold = nn.Parameter(torch.tensor(2.0))  # °C
            self.rain_temperature_threshold = nn.Parameter(torch.tensor(0.0))  # °C
            
            # 积累效率参数
            self.accumulation_efficiency = nn.Parameter(torch.tensor(0.8))
            
            # 高程梯度参数
            self.precipitation_gradient = nn.Parameter(torch.tensor(0.0005))  # m⁻¹
            
            # 风吹雪参数
            self.wind_redistribution_factor = nn.Parameter(torch.tensor(0.1))
            
        else:
            self.register_buffer('snow_temperature_threshold', torch.tensor(2.0))
            self.register_buffer('rain_temperature_threshold', torch.tensor(0.0))
            self.register_buffer('accumulation_efficiency', torch.tensor(0.8))
            self.register_buffer('precipitation_gradient', torch.tensor(0.0005))
            self.register_buffer('wind_redistribution_factor', torch.tensor(0.1))
    
    def temperature_precipitation_model(self, temperature: torch.Tensor,
                                     precipitation: torch.Tensor,
                                     elevation: torch.Tensor) -> torch.Tensor:
        """
        基于温度和降水的积累模型
        
        Args:
            temperature: 气温 [°C]
            precipitation: 降水量 [mm/day]
            elevation: 海拔高度 [m]
            
        Returns:
            torch.Tensor: 积累率 [mm w.e./day]
        """
        # 降雪比例
        snow_fraction = torch.sigmoid(
            (self.snow_temperature_threshold - temperature) * 2.0
        )
        
        # 高程效应
        elevation_factor = 1.0 + self.precipitation_gradient * elevation
        
        # 有效降雪
        snowfall = precipitation * snow_fraction * elevation_factor
        
        # 积累效率
        accumulation = snowfall * self.accumulation_efficiency
        
        return accumulation
    
    def wind_redistribution(self, accumulation: torch.Tensor,
                          wind_speed: torch.Tensor,
                          slope: torch.Tensor,
                          aspect: torch.Tensor) -> torch.Tensor:
        """
        风吹雪再分布
        
        Args:
            accumulation: 原始积累
            wind_speed: 风速
            slope: 坡度
            aspect: 坡向
            
        Returns:
            torch.Tensor: 再分布后的积累
        """
        # 风吹雪强度
        wind_effect = self.wind_redistribution_factor * wind_speed * torch.sin(slope)
        
        # 坡向效应（简化）
        aspect_effect = torch.cos(aspect)  # 北坡积累更多
        
        # 再分布
        redistributed_accumulation = accumulation * (1.0 + wind_effect * aspect_effect)
        
        return torch.clamp(redistributed_accumulation, min=0.0)
    
    def avalanche_redistribution(self, accumulation: torch.Tensor,
                               slope: torch.Tensor,
                               slope_threshold: float = 30.0) -> torch.Tensor:
        """
        雪崩再分布
        
        Args:
            accumulation: 原始积累
            slope: 坡度 [度]
            slope_threshold: 雪崩坡度阈值
            
        Returns:
            torch.Tensor: 雪崩调整后的积累
        """
        # 雪崩概率
        avalanche_probability = torch.sigmoid((slope - slope_threshold) * 0.2)
        
        # 雪崩损失
        avalanche_loss = accumulation * avalanche_probability * 0.5
        
        # 调整后的积累
        adjusted_accumulation = accumulation - avalanche_loss
        
        return torch.clamp(adjusted_accumulation, min=0.0)
    
    def forward(self, meteorological_data: Dict[str, torch.Tensor],
               topographical_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            meteorological_data: 气象数据字典
            topographical_data: 地形数据字典
            
        Returns:
            Dict: 积累结果字典
        """
        results = {}
        
        # 基础积累
        if self.accumulation_type == 'temperature_precipitation':
            base_accumulation = self.temperature_precipitation_model(
                meteorological_data['temperature'],
                meteorological_data['precipitation'],
                topographical_data['elevation']
            )
        else:
            raise ValueError(f"未知的积累模型类型: {self.accumulation_type}")
        
        results['base_accumulation'] = base_accumulation
        
        # 风吹雪再分布
        if 'wind_speed' in meteorological_data:
            wind_redistributed = self.wind_redistribution(
                base_accumulation,
                meteorological_data['wind_speed'],
                topographical_data['slope'],
                topographical_data['aspect']
            )
            results['wind_redistributed'] = wind_redistributed
        else:
            wind_redistributed = base_accumulation
        
        # 雪崩调整
        if 'slope' in topographical_data:
            final_accumulation = self.avalanche_redistribution(
                wind_redistributed,
                topographical_data['slope']
            )
        else:
            final_accumulation = wind_redistributed
        
        results['final_accumulation'] = final_accumulation
        
        return results

class AblationModel(nn.Module):
    """
    消融模型
    
    模拟冰川的消融过程
    """
    
    def __init__(self, ablation_type: str = 'degree_day',
                 learnable_parameters: bool = True):
        """
        初始化消融模型
        
        Args:
            ablation_type: 消融模型类型
            learnable_parameters: 是否学习参数
        """
        super(AblationModel, self).__init__()
        
        self.ablation_type = ablation_type
        
        if learnable_parameters:
            # 度日因子
            self.degree_day_factor_snow = nn.Parameter(torch.tensor(3.0))  # mm/°C/day
            self.degree_day_factor_ice = nn.Parameter(torch.tensor(6.0))   # mm/°C/day
            
            # 辐射参数
            self.radiation_factor = nn.Parameter(torch.tensor(0.01))  # mm/(W/m²)/day
            
            # 升华参数
            self.sublimation_factor = nn.Parameter(torch.tensor(0.5))  # mm/day
            
            # 温度阈值
            self.melting_threshold = nn.Parameter(torch.tensor(0.0))  # °C
            
        else:
            self.register_buffer('degree_day_factor_snow', torch.tensor(3.0))
            self.register_buffer('degree_day_factor_ice', torch.tensor(6.0))
            self.register_buffer('radiation_factor', torch.tensor(0.01))
            self.register_buffer('sublimation_factor', torch.tensor(0.5))
            self.register_buffer('melting_threshold', torch.tensor(0.0))
    
    def degree_day_model(self, temperature: torch.Tensor,
                        surface_type: torch.Tensor) -> torch.Tensor:
        """
        度日模型
        
        Args:
            temperature: 气温 [°C]
            surface_type: 表面类型 (0: 雪, 1: 冰)
            
        Returns:
            torch.Tensor: 融化率 [mm w.e./day]
        """
        # 正积温
        positive_temperature = torch.clamp(temperature - self.melting_threshold, min=0.0)
        
        # 根据表面类型选择度日因子
        degree_day_factor = (
            self.degree_day_factor_snow * (1 - surface_type) +
            self.degree_day_factor_ice * surface_type
        )
        
        # 融化率
        melt_rate = degree_day_factor * positive_temperature
        
        return melt_rate
    
    def energy_balance_model(self, temperature: torch.Tensor,
                           solar_radiation: torch.Tensor,
                           longwave_radiation: torch.Tensor,
                           sensible_heat: torch.Tensor,
                           latent_heat: torch.Tensor,
                           albedo: torch.Tensor) -> torch.Tensor:
        """
        能量平衡模型
        
        Args:
            temperature: 气温
            solar_radiation: 太阳辐射
            longwave_radiation: 长波辐射
            sensible_heat: 感热通量
            latent_heat: 潜热通量
            albedo: 反照率
            
        Returns:
            torch.Tensor: 融化率
        """
        # 净短波辐射
        net_shortwave = solar_radiation * (1 - albedo)
        
        # 净能量
        net_energy = net_shortwave + longwave_radiation + sensible_heat + latent_heat
        
        # 融化潜热 (334 kJ/kg)
        latent_heat_fusion = 334000.0  # J/kg
        
        # 融化率 (只有正能量才能融化)
        melt_rate = torch.clamp(net_energy, min=0.0) / latent_heat_fusion * 86400  # mm/day
        
        return melt_rate
    
    def sublimation_model(self, temperature: torch.Tensor,
                         humidity: torch.Tensor,
                         wind_speed: torch.Tensor,
                         pressure: torch.Tensor) -> torch.Tensor:
        """
        升华模型
        
        Args:
            temperature: 气温
            humidity: 相对湿度
            wind_speed: 风速
            pressure: 气压
            
        Returns:
            torch.Tensor: 升华率
        """
        # 饱和水汽压 (简化公式)
        saturation_vapor_pressure = 611.2 * torch.exp(17.67 * temperature / (temperature + 243.5))
        
        # 实际水汽压
        actual_vapor_pressure = humidity * saturation_vapor_pressure
        
        # 水汽压差
        vapor_pressure_deficit = saturation_vapor_pressure - actual_vapor_pressure
        
        # 升华率 (简化)
        sublimation_rate = self.sublimation_factor * vapor_pressure_deficit * wind_speed / pressure
        
        return torch.clamp(sublimation_rate, min=0.0)
    
    def forward(self, meteorological_data: Dict[str, torch.Tensor],
               surface_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            meteorological_data: 气象数据字典
            surface_data: 表面数据字典
            
        Returns:
            Dict: 消融结果字典
        """
        results = {}
        
        temperature = meteorological_data['temperature']
        
        # 融化
        if self.ablation_type == 'degree_day':
            melt_rate = self.degree_day_model(
                temperature,
                surface_data.get('surface_type', torch.zeros_like(temperature))
            )
        elif self.ablation_type == 'energy_balance':
            melt_rate = self.energy_balance_model(
                temperature,
                meteorological_data['solar_radiation'],
                meteorological_data['longwave_radiation'],
                meteorological_data['sensible_heat'],
                meteorological_data['latent_heat'],
                surface_data['albedo']
            )
        else:
            raise ValueError(f"未知的消融模型类型: {self.ablation_type}")
        
        results['melt_rate'] = melt_rate
        
        # 升华
        if all(key in meteorological_data for key in ['humidity', 'wind_speed', 'pressure']):
            sublimation_rate = self.sublimation_model(
                temperature,
                meteorological_data['humidity'],
                meteorological_data['wind_speed'],
                meteorological_data['pressure']
            )
            results['sublimation_rate'] = sublimation_rate
        else:
            sublimation_rate = torch.zeros_like(temperature)
        
        # 总消融
        total_ablation = melt_rate + sublimation_rate
        results['total_ablation'] = total_ablation
        
        return results

class MassConservation(nn.Module):
    """
    质量守恒方程
    
    实现冰川的质量守恒
    """
    
    def __init__(self, spatial_dim: int = 2):
        """
        初始化质量守恒
        
        Args:
            spatial_dim: 空间维度
        """
        super(MassConservation, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.ice_density = 917.0  # kg/m³
        self.water_density = 1000.0  # kg/m³
    
    def thickness_evolution(self, thickness: torch.Tensor,
                          thickness_gradients: torch.Tensor,
                          velocity: torch.Tensor,
                          velocity_divergence: torch.Tensor,
                          accumulation_rate: torch.Tensor,
                          ablation_rate: torch.Tensor) -> torch.Tensor:
        """
        厚度演化方程
        
        Args:
            thickness: 冰川厚度
            thickness_gradients: 厚度梯度
            velocity: 速度场
            velocity_divergence: 速度散度
            accumulation_rate: 积累率
            ablation_rate: 消融率
            
        Returns:
            torch.Tensor: 厚度变化率
        """
        # 对流项: v · ∇h
        advection_term = torch.zeros_like(thickness)
        for i in range(min(self.spatial_dim, velocity.shape[-1])):
            advection_term += velocity[..., i] * thickness_gradients[..., i]
        
        # 散度项: h * ∇ · v
        divergence_term = thickness * velocity_divergence
        
        # 质量平衡项 (转换为冰当量)
        net_mass_balance = (accumulation_rate - ablation_rate) * (self.water_density / self.ice_density)
        
        # 厚度演化: ∂h/∂t + v·∇h + h∇·v = SMB
        thickness_change = -advection_term - divergence_term + net_mass_balance
        
        return thickness_change
    
    def volume_conservation(self, volume: torch.Tensor,
                          volume_flux: torch.Tensor,
                          accumulation_rate: torch.Tensor,
                          ablation_rate: torch.Tensor,
                          area: torch.Tensor) -> torch.Tensor:
        """
        体积守恒方程
        
        Args:
            volume: 冰川体积
            volume_flux: 体积通量
            accumulation_rate: 积累率
            ablation_rate: 消融率
            area: 面积
            
        Returns:
            torch.Tensor: 体积变化率
        """
        # 净质量平衡
        net_mass_balance = accumulation_rate - ablation_rate
        
        # 体积变化
        volume_change = -volume_flux + net_mass_balance * area * (self.water_density / self.ice_density)
        
        return volume_change
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 质量守恒残差字典
        """
        results = {}
        
        # 厚度演化
        if all(key in fields for key in ['thickness', 'thickness_gradients', 'velocity', 
                                       'velocity_divergence', 'accumulation_rate', 'ablation_rate']):
            thickness_change = self.thickness_evolution(
                fields['thickness'],
                fields['thickness_gradients'],
                fields['velocity'],
                fields['velocity_divergence'],
                fields['accumulation_rate'],
                fields['ablation_rate']
            )
            results['thickness_evolution'] = thickness_change
        
        # 体积守恒
        if all(key in fields for key in ['volume', 'volume_flux', 'accumulation_rate', 
                                       'ablation_rate', 'area']):
            volume_change = self.volume_conservation(
                fields['volume'],
                fields['volume_flux'],
                fields['accumulation_rate'],
                fields['ablation_rate'],
                fields['area']
            )
            results['volume_conservation'] = volume_change
        
        return results

class SurfaceMassBalance(nn.Module):
    """
    表面质量平衡模型
    
    集成积累和消融过程
    """
    
    def __init__(self, accumulation_type: str = 'temperature_precipitation',
                 ablation_type: str = 'degree_day'):
        """
        初始化表面质量平衡模型
        
        Args:
            accumulation_type: 积累模型类型
            ablation_type: 消融模型类型
        """
        super(SurfaceMassBalance, self).__init__()
        
        self.accumulation_model = AccumulationModel(accumulation_type)
        self.ablation_model = AblationModel(ablation_type)
    
    def compute_equilibrium_line_altitude(self, elevation: torch.Tensor,
                                        mass_balance: torch.Tensor) -> torch.Tensor:
        """
        计算平衡线高度
        
        Args:
            elevation: 海拔高度
            mass_balance: 质量平衡
            
        Returns:
            torch.Tensor: 平衡线高度
        """
        # 找到质量平衡为零的高度
        # 简化实现：线性插值
        zero_crossings = torch.where(torch.diff(torch.sign(mass_balance), dim=-1) != 0)
        
        if len(zero_crossings[0]) > 0:
            # 取第一个零点
            idx = zero_crossings[0][0]
            ela = elevation[..., idx]
        else:
            # 如果没有零点，返回中位数高度
            ela = torch.median(elevation, dim=-1)[0]
        
        return ela
    
    def forward(self, meteorological_data: Dict[str, torch.Tensor],
               topographical_data: Dict[str, torch.Tensor],
               surface_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            meteorological_data: 气象数据
            topographical_data: 地形数据
            surface_data: 表面数据
            
        Returns:
            Dict: 表面质量平衡结果
        """
        # 积累
        accumulation_results = self.accumulation_model(meteorological_data, topographical_data)
        
        # 消融
        ablation_results = self.ablation_model(meteorological_data, surface_data)
        
        # 净质量平衡
        net_mass_balance = accumulation_results['final_accumulation'] - ablation_results['total_ablation']
        
        # 平衡线高度
        if 'elevation' in topographical_data:
            ela = self.compute_equilibrium_line_altitude(
                topographical_data['elevation'],
                net_mass_balance
            )
        else:
            ela = torch.zeros_like(net_mass_balance[..., 0])
        
        results = {
            'accumulation': accumulation_results['final_accumulation'],
            'ablation': ablation_results['total_ablation'],
            'net_mass_balance': net_mass_balance,
            'equilibrium_line_altitude': ela
        }
        
        # 添加详细结果
        results.update({f'accumulation_{k}': v for k, v in accumulation_results.items()})
        results.update({f'ablation_{k}': v for k, v in ablation_results.items()})
        
        return results

class MassBalanceModel(nn.Module):
    """
    完整的质量平衡模型
    
    集成所有质量平衡组件
    """
    
    def __init__(self, spatial_dim: int = 2,
                 accumulation_type: str = 'temperature_precipitation',
                 ablation_type: str = 'degree_day'):
        """
        初始化质量平衡模型
        
        Args:
            spatial_dim: 空间维度
            accumulation_type: 积累模型类型
            ablation_type: 消融模型类型
        """
        super(MassBalanceModel, self).__init__()
        
        self.spatial_dim = spatial_dim
        
        # 核心组件
        self.surface_mass_balance = SurfaceMassBalance(accumulation_type, ablation_type)
        self.mass_conservation = MassConservation(spatial_dim)
    
    def forward(self, meteorological_data: Dict[str, torch.Tensor],
               topographical_data: Dict[str, torch.Tensor],
               surface_data: Dict[str, torch.Tensor],
               dynamic_fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            meteorological_data: 气象数据
            topographical_data: 地形数据
            surface_data: 表面数据
            dynamic_fields: 动力学场
            
        Returns:
            Dict: 完整的质量平衡结果
        """
        # 表面质量平衡
        smb_results = self.surface_mass_balance(
            meteorological_data, topographical_data, surface_data
        )
        
        # 准备质量守恒计算的场
        conservation_fields = dynamic_fields.copy()
        conservation_fields['accumulation_rate'] = smb_results['accumulation']
        conservation_fields['ablation_rate'] = smb_results['ablation']
        
        # 质量守恒
        conservation_results = self.mass_conservation(conservation_fields)
        
        # 合并结果
        results = {}
        results.update(smb_results)
        results.update(conservation_results)
        
        return results

if __name__ == "__main__":
    # 测试质量平衡组件
    batch_size = 32
    spatial_points = 100
    
    # 准备测试数据
    meteorological_data = {
        'temperature': torch.randn(batch_size, spatial_points) * 10,  # -10 to 10°C
        'precipitation': torch.abs(torch.randn(batch_size, spatial_points)) * 5,  # 0-5 mm/day
        'wind_speed': torch.abs(torch.randn(batch_size, spatial_points)) * 10,  # 0-10 m/s
        'humidity': torch.rand(batch_size, spatial_points),  # 0-1
        'pressure': torch.ones(batch_size, spatial_points) * 101325,  # Pa
        'solar_radiation': torch.abs(torch.randn(batch_size, spatial_points)) * 300,  # W/m²
        'longwave_radiation': torch.randn(batch_size, spatial_points) * 50,  # W/m²
        'sensible_heat': torch.randn(batch_size, spatial_points) * 20,  # W/m²
        'latent_heat': torch.randn(batch_size, spatial_points) * 20,  # W/m²
    }
    
    topographical_data = {
        'elevation': torch.abs(torch.randn(batch_size, spatial_points)) * 3000 + 3000,  # 3000-6000m
        'slope': torch.abs(torch.randn(batch_size, spatial_points)) * 45,  # 0-45°
        'aspect': torch.rand(batch_size, spatial_points) * 2 * math.pi,  # 0-2π
    }
    
    surface_data = {
        'surface_type': torch.rand(batch_size, spatial_points),  # 0: snow, 1: ice
        'albedo': torch.rand(batch_size, spatial_points) * 0.5 + 0.3,  # 0.3-0.8
    }
    
    dynamic_fields = {
        'thickness': torch.abs(torch.randn(batch_size, spatial_points)) * 100,  # m
        'thickness_gradients': torch.randn(batch_size, spatial_points, 2),
        'velocity': torch.randn(batch_size, spatial_points, 2),
        'velocity_divergence': torch.randn(batch_size, spatial_points),
        'volume': torch.abs(torch.randn(batch_size, spatial_points)) * 1e6,  # m³
        'volume_flux': torch.randn(batch_size, spatial_points),
        'area': torch.abs(torch.randn(batch_size, spatial_points)) * 1e4,  # m²
    }
    
    # 测试表面质量平衡
    smb_model = SurfaceMassBalance()
    smb_results = smb_model(meteorological_data, topographical_data, surface_data)
    print(f"表面质量平衡 - 净质量平衡形状: {smb_results['net_mass_balance'].shape}")
    print(f"表面质量平衡 - 平衡线高度形状: {smb_results['equilibrium_line_altitude'].shape}")
    
    # 测试完整质量平衡模型
    mass_balance_model = MassBalanceModel()
    mb_results = mass_balance_model(
        meteorological_data, topographical_data, surface_data, dynamic_fields
    )
    print(f"质量平衡模型输出键: {list(mb_results.keys())}")
    
    if 'thickness_evolution' in mb_results:
        print(f"厚度演化形状: {mb_results['thickness_evolution'].shape}")