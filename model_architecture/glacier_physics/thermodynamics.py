#!/usr/bin/env python3
"""
冰川热力学模型

实现冰川的热力学过程，包括：
- 温度分布
- 热传导
- 热对流
- 相变过程
- 地热通量
- 摩擦加热

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math

class HeatConduction(nn.Module):
    """
    热传导模型
    
    实现冰川内部的热传导过程
    """
    
    def __init__(self, spatial_dim: int = 3, learnable_parameters: bool = True):
        """
        初始化热传导模型
        
        Args:
            spatial_dim: 空间维度
            learnable_parameters: 是否学习参数
        """
        super(HeatConduction, self).__init__()
        
        self.spatial_dim = spatial_dim
        
        if learnable_parameters:
            # 热导率 [W/(m·K)]
            self.thermal_conductivity = nn.Parameter(torch.tensor(2.1))  # 冰的热导率
            
            # 比热容 [J/(kg·K)]
            self.specific_heat = nn.Parameter(torch.tensor(2050.0))  # 冰的比热容
            
            # 密度 [kg/m³]
            self.density = nn.Parameter(torch.tensor(917.0))  # 冰的密度
            
        else:
            self.register_buffer('thermal_conductivity', torch.tensor(2.1))
            self.register_buffer('specific_heat', torch.tensor(2050.0))
            self.register_buffer('density', torch.tensor(917.0))
        
        # 热扩散率
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat)
    
    def temperature_dependent_properties(self, temperature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        温度相关的热物理性质
        
        Args:
            temperature: 温度 [K]
            
        Returns:
            Dict: 热物理性质字典
        """
        # 温度相关的热导率 (Yen 1981)
        k_temp = self.thermal_conductivity * (1.0 + 0.0057 * (temperature - 273.15))
        
        # 温度相关的比热容
        c_temp = self.specific_heat * (1.0 + 0.0008 * (temperature - 273.15))
        
        # 温度相关的密度
        rho_temp = self.density * (1.0 - 0.0001 * (temperature - 273.15))
        
        return {
            'thermal_conductivity': k_temp,
            'specific_heat': c_temp,
            'density': rho_temp,
            'thermal_diffusivity': k_temp / (rho_temp * c_temp)
        }
    
    def heat_diffusion_equation(self, temperature: torch.Tensor,
                              temperature_laplacian: torch.Tensor,
                              heat_source: torch.Tensor = None) -> torch.Tensor:
        """
        热扩散方程
        
        Args:
            temperature: 温度场
            temperature_laplacian: 温度拉普拉斯算子
            heat_source: 热源项
            
        Returns:
            torch.Tensor: 温度变化率
        """
        # 获取温度相关性质
        properties = self.temperature_dependent_properties(temperature)
        
        # 热扩散项
        diffusion_term = properties['thermal_diffusivity'] * temperature_laplacian
        
        # 热源项
        if heat_source is not None:
            source_term = heat_source / (properties['density'] * properties['specific_heat'])
        else:
            source_term = torch.zeros_like(temperature)
        
        # 温度变化率: ∂T/∂t = α∇²T + Q/(ρc)
        temperature_rate = diffusion_term + source_term
        
        return temperature_rate
    
    def anisotropic_conduction(self, temperature: torch.Tensor,
                             temperature_gradients: torch.Tensor,
                             fabric_tensor: torch.Tensor) -> torch.Tensor:
        """
        各向异性热传导
        
        Args:
            temperature: 温度场
            temperature_gradients: 温度梯度
            fabric_tensor: 组构张量
            
        Returns:
            torch.Tensor: 各向异性热通量
        """
        # 各向异性热导率张量
        k_tensor = self.thermal_conductivity * fabric_tensor
        
        # 热通量: q = -k·∇T
        heat_flux = torch.zeros_like(temperature_gradients)
        for i in range(self.spatial_dim):
            for j in range(self.spatial_dim):
                heat_flux[..., i] -= k_tensor[..., i, j] * temperature_gradients[..., j]
        
        return heat_flux
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 热传导结果字典
        """
        results = {}
        
        temperature = fields['temperature']
        
        # 温度相关性质
        properties = self.temperature_dependent_properties(temperature)
        results.update(properties)
        
        # 热扩散方程
        if 'temperature_laplacian' in fields:
            temperature_rate = self.heat_diffusion_equation(
                temperature,
                fields['temperature_laplacian'],
                fields.get('heat_source')
            )
            results['temperature_rate'] = temperature_rate
        
        # 各向异性传导
        if 'temperature_gradients' in fields and 'fabric_tensor' in fields:
            heat_flux = self.anisotropic_conduction(
                temperature,
                fields['temperature_gradients'],
                fields['fabric_tensor']
            )
            results['heat_flux'] = heat_flux
        
        return results

class HeatAdvection(nn.Module):
    """
    热对流模型
    
    实现冰川流动引起的热对流
    """
    
    def __init__(self, spatial_dim: int = 3):
        """
        初始化热对流模型
        
        Args:
            spatial_dim: 空间维度
        """
        super(HeatAdvection, self).__init__()
        
        self.spatial_dim = spatial_dim
    
    def advection_term(self, temperature: torch.Tensor,
                      temperature_gradients: torch.Tensor,
                      velocity: torch.Tensor) -> torch.Tensor:
        """
        对流项计算
        
        Args:
            temperature: 温度场
            temperature_gradients: 温度梯度
            velocity: 速度场
            
        Returns:
            torch.Tensor: 对流项
        """
        # 对流项: v·∇T
        advection = torch.zeros_like(temperature)
        for i in range(min(self.spatial_dim, velocity.shape[-1])):
            advection += velocity[..., i] * temperature_gradients[..., i]
        
        return advection
    
    def peclet_number(self, velocity: torch.Tensor,
                     characteristic_length: torch.Tensor,
                     thermal_diffusivity: torch.Tensor) -> torch.Tensor:
        """
        计算Peclet数
        
        Args:
            velocity: 速度大小
            characteristic_length: 特征长度
            thermal_diffusivity: 热扩散率
            
        Returns:
            torch.Tensor: Peclet数
        """
        velocity_magnitude = torch.norm(velocity, dim=-1)
        pe = velocity_magnitude * characteristic_length / thermal_diffusivity
        
        return pe
    
    def upwind_scheme(self, temperature: torch.Tensor,
                     temperature_gradients: torch.Tensor,
                     velocity: torch.Tensor,
                     grid_spacing: float) -> torch.Tensor:
        """
        迎风格式
        
        Args:
            temperature: 温度场
            temperature_gradients: 温度梯度
            velocity: 速度场
            grid_spacing: 网格间距
            
        Returns:
            torch.Tensor: 迎风对流项
        """
        # 简化的迎风格式实现
        velocity_magnitude = torch.norm(velocity, dim=-1, keepdim=True)
        
        # 迎风方向的梯度
        upwind_gradient = torch.zeros_like(temperature)
        for i in range(min(self.spatial_dim, velocity.shape[-1])):
            # 根据速度方向选择梯度
            upwind_gradient += torch.where(
                velocity[..., i:i+1] > 0,
                temperature_gradients[..., i:i+1],
                -temperature_gradients[..., i:i+1]
            ).squeeze(-1)
        
        upwind_term = velocity_magnitude.squeeze(-1) * upwind_gradient
        
        return upwind_term
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 热对流结果字典
        """
        results = {}
        
        if all(key in fields for key in ['temperature', 'temperature_gradients', 'velocity']):
            # 标准对流项
            advection = self.advection_term(
                fields['temperature'],
                fields['temperature_gradients'],
                fields['velocity']
            )
            results['advection_term'] = advection
            
            # Peclet数
            if 'thermal_diffusivity' in fields and 'characteristic_length' in fields:
                pe = self.peclet_number(
                    fields['velocity'],
                    fields['characteristic_length'],
                    fields['thermal_diffusivity']
                )
                results['peclet_number'] = pe
            
            # 迎风格式
            if 'grid_spacing' in fields:
                upwind = self.upwind_scheme(
                    fields['temperature'],
                    fields['temperature_gradients'],
                    fields['velocity'],
                    fields['grid_spacing']
                )
                results['upwind_advection'] = upwind
        
        return results

class PhaseChange(nn.Module):
    """
    相变模型
    
    实现冰-水相变过程
    """
    
    def __init__(self, learnable_parameters: bool = True):
        """
        初始化相变模型
        
        Args:
            learnable_parameters: 是否学习参数
        """
        super(PhaseChange, self).__init__()
        
        if learnable_parameters:
            # 融化潜热 [J/kg]
            self.latent_heat_fusion = nn.Parameter(torch.tensor(334000.0))
            
            # 融化点 [K]
            self.melting_point = nn.Parameter(torch.tensor(273.15))
            
            # 压力融化系数 [K/Pa]
            self.pressure_melting_coefficient = nn.Parameter(torch.tensor(7.4e-8))
            
        else:
            self.register_buffer('latent_heat_fusion', torch.tensor(334000.0))
            self.register_buffer('melting_point', torch.tensor(273.15))
            self.register_buffer('pressure_melting_coefficient', torch.tensor(7.4e-8))
    
    def pressure_melting_point(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        压力相关的融化点
        
        Args:
            pressure: 压力 [Pa]
            
        Returns:
            torch.Tensor: 融化点 [K]
        """
        # Clausius-Clapeyron关系
        melting_point = self.melting_point - self.pressure_melting_coefficient * pressure
        
        return melting_point
    
    def enthalpy_method(self, temperature: torch.Tensor,
                       pressure: torch.Tensor,
                       specific_heat_ice: torch.Tensor,
                       specific_heat_water: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        焓方法处理相变
        
        Args:
            temperature: 温度
            pressure: 压力
            specific_heat_ice: 冰的比热容
            specific_heat_water: 水的比热容
            
        Returns:
            Dict: 相变结果
        """
        # 压力相关融化点
        T_m = self.pressure_melting_point(pressure)
        
        # 液体分数
        liquid_fraction = torch.zeros_like(temperature)
        
        # 完全固态
        solid_mask = temperature < T_m
        
        # 完全液态
        liquid_mask = temperature > T_m
        liquid_fraction[liquid_mask] = 1.0
        
        # 相变区间（简化为瞬时相变）
        phase_change_mask = torch.abs(temperature - T_m) < 0.01
        liquid_fraction[phase_change_mask] = 0.5
        
        # 有效比热容（包含潜热效应）
        effective_specific_heat = (
            specific_heat_ice * (1 - liquid_fraction) +
            specific_heat_water * liquid_fraction
        )
        
        # 在相变点附近添加潜热贡献
        latent_heat_contribution = torch.zeros_like(temperature)
        latent_heat_contribution[phase_change_mask] = self.latent_heat_fusion / 0.02  # 假设2K的相变区间
        
        effective_specific_heat += latent_heat_contribution
        
        return {
            'liquid_fraction': liquid_fraction,
            'effective_specific_heat': effective_specific_heat,
            'melting_point': T_m
        }
    
    def stefan_problem(self, temperature: torch.Tensor,
                      temperature_gradients: torch.Tensor,
                      interface_velocity: torch.Tensor,
                      thermal_conductivity_ice: torch.Tensor,
                      thermal_conductivity_water: torch.Tensor) -> torch.Tensor:
        """
        Stefan问题（移动边界相变）
        
        Args:
            temperature: 温度
            temperature_gradients: 温度梯度
            interface_velocity: 界面速度
            thermal_conductivity_ice: 冰的热导率
            thermal_conductivity_water: 水的热导率
            
        Returns:
            torch.Tensor: Stefan条件残差
        """
        # Stefan条件: ρL·v_interface = k_ice·∇T_ice - k_water·∇T_water
        # 简化实现
        heat_flux_ice = thermal_conductivity_ice * torch.norm(temperature_gradients, dim=-1)
        heat_flux_water = thermal_conductivity_water * torch.norm(temperature_gradients, dim=-1)
        
        stefan_residual = (
            917.0 * self.latent_heat_fusion * interface_velocity -
            (heat_flux_ice - heat_flux_water)
        )
        
        return stefan_residual
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 相变结果字典
        """
        results = {}
        
        temperature = fields['temperature']
        pressure = fields.get('pressure', torch.zeros_like(temperature))
        
        # 焓方法
        if 'specific_heat_ice' in fields and 'specific_heat_water' in fields:
            enthalpy_results = self.enthalpy_method(
                temperature,
                pressure,
                fields['specific_heat_ice'],
                fields['specific_heat_water']
            )
            results.update(enthalpy_results)
        
        # Stefan问题
        if all(key in fields for key in ['temperature_gradients', 'interface_velocity',
                                       'thermal_conductivity_ice', 'thermal_conductivity_water']):
            stefan_residual = self.stefan_problem(
                temperature,
                fields['temperature_gradients'],
                fields['interface_velocity'],
                fields['thermal_conductivity_ice'],
                fields['thermal_conductivity_water']
            )
            results['stefan_residual'] = stefan_residual
        
        return results

class HeatSources(nn.Module):
    """
    热源模型
    
    实现各种热源
    """
    
    def __init__(self, learnable_parameters: bool = True):
        """
        初始化热源模型
        
        Args:
            learnable_parameters: 是否学习参数
        """
        super(HeatSources, self).__init__()
        
        if learnable_parameters:
            # 地热通量 [W/m²]
            self.geothermal_flux = nn.Parameter(torch.tensor(0.05))
            
            # 摩擦系数
            self.friction_coefficient = nn.Parameter(torch.tensor(0.1))
            
        else:
            self.register_buffer('geothermal_flux', torch.tensor(0.05))
            self.register_buffer('friction_coefficient', torch.tensor(0.1))
    
    def geothermal_heating(self, depth: torch.Tensor,
                          geothermal_gradient: torch.Tensor = None) -> torch.Tensor:
        """
        地热加热
        
        Args:
            depth: 深度
            geothermal_gradient: 地热梯度
            
        Returns:
            torch.Tensor: 地热加热率
        """
        if geothermal_gradient is not None:
            # 变化的地热梯度
            heating_rate = geothermal_gradient * depth
        else:
            # 常数地热通量
            heating_rate = self.geothermal_flux * torch.ones_like(depth)
        
        return heating_rate
    
    def friction_heating(self, velocity: torch.Tensor,
                        stress: torch.Tensor,
                        strain_rate: torch.Tensor) -> torch.Tensor:
        """
        摩擦加热
        
        Args:
            velocity: 速度
            stress: 应力
            strain_rate: 应变率
            
        Returns:
            torch.Tensor: 摩擦加热率
        """
        # 粘性耗散: τ:ε̇
        if stress.dim() > strain_rate.dim():
            # 应力张量与应变率张量的双点积
            viscous_dissipation = torch.sum(stress * strain_rate.unsqueeze(-1), dim=(-2, -1))
        else:
            # 简化：标量应力和应变率
            viscous_dissipation = stress * strain_rate
        
        # 基底摩擦
        velocity_magnitude = torch.norm(velocity, dim=-1)
        basal_friction = self.friction_coefficient * velocity_magnitude**2
        
        total_friction_heating = viscous_dissipation + basal_friction
        
        return total_friction_heating
    
    def strain_heating(self, strain_rate_tensor: torch.Tensor,
                      stress_tensor: torch.Tensor) -> torch.Tensor:
        """
        应变加热
        
        Args:
            strain_rate_tensor: 应变率张量
            stress_tensor: 应力张量
            
        Returns:
            torch.Tensor: 应变加热率
        """
        # 应变加热: σ:ε̇
        strain_heating = torch.sum(
            stress_tensor * strain_rate_tensor,
            dim=(-2, -1)
        )
        
        return strain_heating
    
    def radiative_heating(self, temperature: torch.Tensor,
                         surface_temperature: torch.Tensor,
                         emissivity: torch.Tensor = None) -> torch.Tensor:
        """
        辐射加热
        
        Args:
            temperature: 温度
            surface_temperature: 表面温度
            emissivity: 发射率
            
        Returns:
            torch.Tensor: 辐射加热率
        """
        if emissivity is None:
            emissivity = torch.ones_like(temperature) * 0.97  # 冰的发射率
        
        stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        
        # Stefan-Boltzmann定律
        radiative_flux = (
            emissivity * stefan_boltzmann *
            (surface_temperature**4 - temperature**4)
        )
        
        return radiative_flux
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 热源结果字典
        """
        results = {}
        
        # 地热加热
        if 'depth' in fields:
            geothermal = self.geothermal_heating(
                fields['depth'],
                fields.get('geothermal_gradient')
            )
            results['geothermal_heating'] = geothermal
        
        # 摩擦加热
        if all(key in fields for key in ['velocity', 'stress', 'strain_rate']):
            friction = self.friction_heating(
                fields['velocity'],
                fields['stress'],
                fields['strain_rate']
            )
            results['friction_heating'] = friction
        
        # 应变加热
        if 'strain_rate_tensor' in fields and 'stress_tensor' in fields:
            strain = self.strain_heating(
                fields['strain_rate_tensor'],
                fields['stress_tensor']
            )
            results['strain_heating'] = strain
        
        # 辐射加热
        if 'temperature' in fields and 'surface_temperature' in fields:
            radiative = self.radiative_heating(
                fields['temperature'],
                fields['surface_temperature'],
                fields.get('emissivity')
            )
            results['radiative_heating'] = radiative
        
        # 总热源
        total_heat_source = torch.zeros_like(fields['temperature'])
        for key, value in results.items():
            if 'heating' in key:
                total_heat_source += value
        
        results['total_heat_source'] = total_heat_source
        
        return results

class ThermodynamicsModel(nn.Module):
    """
    完整的热力学模型
    
    集成所有热力学组件
    """
    
    def __init__(self, spatial_dim: int = 3):
        """
        初始化热力学模型
        
        Args:
            spatial_dim: 空间维度
        """
        super(ThermodynamicsModel, self).__init__()
        
        self.spatial_dim = spatial_dim
        
        # 核心组件
        self.heat_conduction = HeatConduction(spatial_dim)
        self.heat_advection = HeatAdvection(spatial_dim)
        self.phase_change = PhaseChange()
        self.heat_sources = HeatSources()
    
    def energy_equation(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        能量方程
        
        Args:
            fields: 场变量字典
            
        Returns:
            torch.Tensor: 能量方程残差
        """
        # 热传导结果
        conduction_results = self.heat_conduction(fields)
        
        # 热对流结果
        advection_results = self.heat_advection(fields)
        
        # 相变结果
        phase_results = self.phase_change(fields)
        
        # 热源结果
        source_results = self.heat_sources(fields)
        
        # 能量方程: ρc(∂T/∂t + v·∇T) = ∇·(k∇T) + Q
        temperature_rate = conduction_results.get('temperature_rate', torch.zeros_like(fields['temperature']))
        advection_term = advection_results.get('advection_term', torch.zeros_like(fields['temperature']))
        heat_source = source_results.get('total_heat_source', torch.zeros_like(fields['temperature']))
        
        # 有效比热容（考虑相变）
        effective_specific_heat = phase_results.get('effective_specific_heat', 
                                                  conduction_results.get('specific_heat', 2050.0))
        density = conduction_results.get('density', 917.0)
        
        # 能量方程残差
        energy_residual = (
            density * effective_specific_heat * advection_term -
            temperature_rate * density * effective_specific_heat +
            heat_source
        )
        
        return energy_residual
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 场变量字典
            
        Returns:
            Dict: 完整的热力学结果
        """
        results = {}
        
        # 各组件结果
        conduction_results = self.heat_conduction(fields)
        advection_results = self.heat_advection(fields)
        phase_results = self.phase_change(fields)
        source_results = self.heat_sources(fields)
        
        # 合并结果
        results.update({f'conduction_{k}': v for k, v in conduction_results.items()})
        results.update({f'advection_{k}': v for k, v in advection_results.items()})
        results.update({f'phase_{k}': v for k, v in phase_results.items()})
        results.update({f'source_{k}': v for k, v in source_results.items()})
        
        # 能量方程
        energy_residual = self.energy_equation(fields)
        results['energy_equation_residual'] = energy_residual
        
        return results

if __name__ == "__main__":
    # 测试热力学组件
    batch_size = 16
    spatial_points = 64
    spatial_dim = 3
    
    # 准备测试数据
    fields = {
        'temperature': torch.randn(batch_size, spatial_points) * 10 + 263,  # 253-273K
        'temperature_gradients': torch.randn(batch_size, spatial_points, spatial_dim),
        'temperature_laplacian': torch.randn(batch_size, spatial_points),
        'velocity': torch.randn(batch_size, spatial_points, spatial_dim) * 0.1,  # m/year
        'pressure': torch.abs(torch.randn(batch_size, spatial_points)) * 1e6,  # Pa
        'depth': torch.abs(torch.randn(batch_size, spatial_points)) * 100,  # m
        'stress': torch.abs(torch.randn(batch_size, spatial_points)) * 1e5,  # Pa
        'strain_rate': torch.abs(torch.randn(batch_size, spatial_points)) * 1e-8,  # s⁻¹
        'stress_tensor': torch.randn(batch_size, spatial_points, 3, 3) * 1e5,
        'strain_rate_tensor': torch.randn(batch_size, spatial_points, 3, 3) * 1e-8,
        'surface_temperature': torch.randn(batch_size, spatial_points) * 5 + 268,  # K
        'specific_heat_ice': torch.ones(batch_size, spatial_points) * 2050,
        'specific_heat_water': torch.ones(batch_size, spatial_points) * 4186,
        'thermal_diffusivity': torch.ones(batch_size, spatial_points) * 1e-6,
        'characteristic_length': torch.ones(batch_size, spatial_points) * 100,
        'grid_spacing': 10.0,
        'fabric_tensor': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, spatial_points, 1, 1),
        'interface_velocity': torch.randn(batch_size, spatial_points) * 0.01,
        'thermal_conductivity_ice': torch.ones(batch_size, spatial_points) * 2.1,
        'thermal_conductivity_water': torch.ones(batch_size, spatial_points) * 0.6,
    }
    
    # 测试热传导
    heat_conduction = HeatConduction(spatial_dim)
    conduction_results = heat_conduction(fields)
    print(f"热传导 - 温度变化率形状: {conduction_results['temperature_rate'].shape}")
    
    # 测试热对流
    heat_advection = HeatAdvection(spatial_dim)
    advection_results = heat_advection(fields)
    print(f"热对流 - 对流项形状: {advection_results['advection_term'].shape}")
    
    # 测试相变
    phase_change = PhaseChange()
    phase_results = phase_change(fields)
    print(f"相变 - 液体分数形状: {phase_results['liquid_fraction'].shape}")
    
    # 测试热源
    heat_sources = HeatSources()
    source_results = heat_sources(fields)
    print(f"热源 - 总热源形状: {source_results['total_heat_source'].shape}")
    
    # 测试完整热力学模型
    thermodynamics_model = ThermodynamicsModel(spatial_dim)
    thermo_results = thermodynamics_model(fields)
    print(f"热力学模型输出键: {list(thermo_results.keys())}")
    print(f"能量方程残差形状: {thermo_results['energy_equation_residual'].shape}")