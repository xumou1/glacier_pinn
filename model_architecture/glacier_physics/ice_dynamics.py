#!/usr/bin/env python3
"""
冰川动力学模型

实现冰川动力学的物理方程，包括：
- Glen流动定律
- 应力-应变关系
- 冰川流动方程
- 滑动定律
- 温度依赖性

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math

class GlenFlowLaw(nn.Module):
    """
    Glen流动定律实现
    
    描述冰川冰的流变行为
    """
    
    def __init__(self, glen_exponent: float = 3.0, 
                 rate_factor_method: str = 'temperature_dependent',
                 learnable_parameters: bool = True):
        """
        初始化Glen流动定律
        
        Args:
            glen_exponent: Glen指数 (通常为3)
            rate_factor_method: 速率因子计算方法
            learnable_parameters: 是否学习参数
        """
        super(GlenFlowLaw, self).__init__()
        
        self.glen_exponent = glen_exponent
        self.rate_factor_method = rate_factor_method
        
        # 可学习参数
        if learnable_parameters:
            self.log_rate_factor = nn.Parameter(torch.tensor(-16.0))  # log(A)
            self.activation_energy = nn.Parameter(torch.tensor(60000.0))  # Q (J/mol)
        else:
            self.register_buffer('log_rate_factor', torch.tensor(-16.0))
            self.register_buffer('activation_energy', torch.tensor(60000.0))
        
        # 物理常数
        self.gas_constant = 8.314  # J/(mol·K)
        self.reference_temperature = 273.15  # K
    
    def compute_rate_factor(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        计算速率因子A(T)
        
        Args:
            temperature: 温度 [K]
            
        Returns:
            torch.Tensor: 速率因子
        """
        if self.rate_factor_method == 'constant':
            return torch.exp(self.log_rate_factor).expand_as(temperature)
        
        elif self.rate_factor_method == 'temperature_dependent':
            # Arrhenius关系
            inv_temp = 1.0 / temperature
            inv_ref_temp = 1.0 / self.reference_temperature
            
            exponent = -self.activation_energy / self.gas_constant * (inv_temp - inv_ref_temp)
            rate_factor = torch.exp(self.log_rate_factor + exponent)
            
            return rate_factor
        
        else:
            raise ValueError(f"未知的速率因子方法: {self.rate_factor_method}")
    
    def compute_effective_stress(self, stress_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算有效应力
        
        Args:
            stress_tensor: 应力张量 [..., 2, 2] 或 [..., 3, 3]
            
        Returns:
            torch.Tensor: 有效应力
        """
        # 计算应力偏量
        if stress_tensor.shape[-1] == 2:
            # 2D情况
            trace = stress_tensor[..., 0, 0] + stress_tensor[..., 1, 1]
            mean_stress = trace / 2.0
            
            deviatoric = stress_tensor.clone()
            deviatoric[..., 0, 0] -= mean_stress
            deviatoric[..., 1, 1] -= mean_stress
            
            # 第二不变量
            tau_xx = deviatoric[..., 0, 0]
            tau_yy = deviatoric[..., 1, 1]
            tau_xy = deviatoric[..., 0, 1]
            
            second_invariant = 0.5 * (tau_xx**2 + tau_yy**2) + tau_xy**2
            
        else:
            # 3D情况
            trace = torch.trace(stress_tensor)
            mean_stress = trace / 3.0
            
            deviatoric = stress_tensor.clone()
            for i in range(3):
                deviatoric[..., i, i] -= mean_stress
            
            # 第二不变量
            second_invariant = 0.5 * torch.sum(deviatoric**2, dim=(-2, -1))
        
        effective_stress = torch.sqrt(second_invariant)
        return effective_stress
    
    def compute_strain_rate(self, stress_tensor: torch.Tensor, 
                          temperature: torch.Tensor) -> torch.Tensor:
        """
        计算应变率张量
        
        Args:
            stress_tensor: 应力张量
            temperature: 温度
            
        Returns:
            torch.Tensor: 应变率张量
        """
        # 计算有效应力
        effective_stress = self.compute_effective_stress(stress_tensor)
        
        # 计算速率因子
        rate_factor = self.compute_rate_factor(temperature)
        
        # Glen流动定律
        effective_strain_rate = rate_factor * (effective_stress ** (self.glen_exponent - 1))
        
        # 计算应力偏量
        if stress_tensor.shape[-1] == 2:
            trace = stress_tensor[..., 0, 0] + stress_tensor[..., 1, 1]
            mean_stress = trace / 2.0
            
            deviatoric = stress_tensor.clone()
            deviatoric[..., 0, 0] -= mean_stress
            deviatoric[..., 1, 1] -= mean_stress
        else:
            trace = torch.trace(stress_tensor)
            mean_stress = trace / 3.0
            
            deviatoric = stress_tensor.clone()
            for i in range(3):
                deviatoric[..., i, i] -= mean_stress
        
        # 应变率张量
        strain_rate = effective_strain_rate.unsqueeze(-1).unsqueeze(-1) * deviatoric
        
        return strain_rate
    
    def forward(self, stress_tensor: torch.Tensor, 
               temperature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            stress_tensor: 应力张量
            temperature: 温度
            
        Returns:
            Dict: 包含应变率等信息的字典
        """
        strain_rate = self.compute_strain_rate(stress_tensor, temperature)
        effective_stress = self.compute_effective_stress(stress_tensor)
        rate_factor = self.compute_rate_factor(temperature)
        
        return {
            'strain_rate': strain_rate,
            'effective_stress': effective_stress,
            'rate_factor': rate_factor
        }

class IceFlowEquations(nn.Module):
    """
    冰川流动方程
    
    实现冰川流动的控制方程
    """
    
    def __init__(self, spatial_dim: int = 2, include_vertical: bool = False):
        """
        初始化冰川流动方程
        
        Args:
            spatial_dim: 空间维度
            include_vertical: 是否包含垂直维度
        """
        super(IceFlowEquations, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.include_vertical = include_vertical
        
        # 物理常数
        self.ice_density = 917.0  # kg/m³
        self.gravity = 9.81  # m/s²
        
        # Glen流动定律
        self.glen_flow = GlenFlowLaw()
    
    def compute_stress_tensor(self, velocity_gradients: torch.Tensor, 
                            pressure: torch.Tensor) -> torch.Tensor:
        """
        计算应力张量
        
        Args:
            velocity_gradients: 速度梯度张量
            pressure: 压力
            
        Returns:
            torch.Tensor: 应力张量
        """
        # 应变率张量
        strain_rate = 0.5 * (velocity_gradients + velocity_gradients.transpose(-2, -1))
        
        # 动态粘度（从Glen流动定律反推）
        # TODO: 实现动态粘度计算
        dynamic_viscosity = torch.ones_like(pressure) * 1e13  # Pa·s
        
        # 应力张量 = 2 * μ * ε̇ - p * I
        stress_tensor = 2 * dynamic_viscosity.unsqueeze(-1).unsqueeze(-1) * strain_rate
        
        # 减去压力项
        for i in range(self.spatial_dim):
            stress_tensor[..., i, i] -= pressure
        
        return stress_tensor
    
    def momentum_conservation(self, velocity: torch.Tensor, 
                            velocity_gradients: torch.Tensor,
                            pressure: torch.Tensor,
                            pressure_gradients: torch.Tensor,
                            surface_elevation: torch.Tensor,
                            surface_gradients: torch.Tensor) -> torch.Tensor:
        """
        动量守恒方程
        
        Args:
            velocity: 速度场
            velocity_gradients: 速度梯度
            pressure: 压力
            pressure_gradients: 压力梯度
            surface_elevation: 表面高程
            surface_gradients: 表面梯度
            
        Returns:
            torch.Tensor: 动量守恒残差
        """
        # 计算应力张量
        stress_tensor = self.compute_stress_tensor(velocity_gradients, pressure)
        
        # 应力散度
        # TODO: 实现应力散度计算
        stress_divergence = torch.zeros_like(velocity)
        
        # 重力项
        gravity_force = torch.zeros_like(velocity)
        gravity_force[..., -1] = -self.ice_density * self.gravity  # 垂直方向
        
        # 表面坡度引起的重力分量
        if surface_gradients is not None:
            for i in range(min(self.spatial_dim, surface_gradients.shape[-1])):
                gravity_force[..., i] += self.ice_density * self.gravity * surface_gradients[..., i]
        
        # 动量守恒: ∇·σ + ρg = 0
        momentum_residual = stress_divergence + gravity_force
        
        return momentum_residual
    
    def mass_conservation(self, velocity: torch.Tensor, 
                        velocity_divergence: torch.Tensor) -> torch.Tensor:
        """
        质量守恒方程（不可压缩）
        
        Args:
            velocity: 速度场
            velocity_divergence: 速度散度
            
        Returns:
            torch.Tensor: 质量守恒残差
        """
        # 不可压缩条件: ∇·v = 0
        return velocity_divergence
    
    def surface_evolution(self, surface_elevation: torch.Tensor,
                        surface_velocity: torch.Tensor,
                        accumulation_rate: torch.Tensor,
                        ablation_rate: torch.Tensor) -> torch.Tensor:
        """
        表面演化方程
        
        Args:
            surface_elevation: 表面高程
            surface_velocity: 表面速度
            accumulation_rate: 积累率
            ablation_rate: 消融率
            
        Returns:
            torch.Tensor: 表面演化残差
        """
        # ∂h/∂t + ∇·(h*v) = accumulation - ablation
        # TODO: 实现完整的表面演化方程
        
        # 简化版本
        net_mass_balance = accumulation_rate - ablation_rate
        
        return net_mass_balance
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 包含各种场的字典
            
        Returns:
            Dict: 物理残差字典
        """
        residuals = {}
        
        # 动量守恒
        if all(key in fields for key in ['velocity', 'velocity_gradients', 'pressure', 'pressure_gradients']):
            momentum_residual = self.momentum_conservation(
                fields['velocity'],
                fields['velocity_gradients'],
                fields['pressure'],
                fields['pressure_gradients'],
                fields.get('surface_elevation'),
                fields.get('surface_gradients')
            )
            residuals['momentum'] = momentum_residual
        
        # 质量守恒
        if 'velocity_divergence' in fields:
            mass_residual = self.mass_conservation(
                fields['velocity'],
                fields['velocity_divergence']
            )
            residuals['mass'] = mass_residual
        
        # 表面演化
        if all(key in fields for key in ['surface_elevation', 'surface_velocity', 'accumulation_rate', 'ablation_rate']):
            surface_residual = self.surface_evolution(
                fields['surface_elevation'],
                fields['surface_velocity'],
                fields['accumulation_rate'],
                fields['ablation_rate']
            )
            residuals['surface'] = surface_residual
        
        return residuals

class SlidingLaw(nn.Module):
    """
    冰川滑动定律
    
    描述冰川底部滑动行为
    """
    
    def __init__(self, sliding_law_type: str = 'weertman',
                 learnable_parameters: bool = True):
        """
        初始化滑动定律
        
        Args:
            sliding_law_type: 滑动定律类型
            learnable_parameters: 是否学习参数
        """
        super(SlidingLaw, self).__init__()
        
        self.sliding_law_type = sliding_law_type
        
        if learnable_parameters:
            # Weertman滑动参数
            self.log_sliding_coefficient = nn.Parameter(torch.tensor(-10.0))
            self.sliding_exponent = nn.Parameter(torch.tensor(3.0))
            
            # Coulomb摩擦参数
            self.friction_coefficient = nn.Parameter(torch.tensor(0.5))
            
        else:
            self.register_buffer('log_sliding_coefficient', torch.tensor(-10.0))
            self.register_buffer('sliding_exponent', torch.tensor(3.0))
            self.register_buffer('friction_coefficient', torch.tensor(0.5))
    
    def weertman_sliding(self, basal_stress: torch.Tensor, 
                        effective_pressure: torch.Tensor) -> torch.Tensor:
        """
        Weertman滑动定律
        
        Args:
            basal_stress: 底部剪切应力
            effective_pressure: 有效压力
            
        Returns:
            torch.Tensor: 滑动速度
        """
        sliding_coefficient = torch.exp(self.log_sliding_coefficient)
        
        # v_b = C * |τ_b|^m * N^(-q)
        # 简化版本: v_b = C * |τ_b|^m
        sliding_velocity = sliding_coefficient * (torch.abs(basal_stress) ** self.sliding_exponent)
        
        # 保持应力方向
        sliding_velocity = sliding_velocity * torch.sign(basal_stress)
        
        return sliding_velocity
    
    def coulomb_friction(self, basal_stress: torch.Tensor,
                        normal_stress: torch.Tensor) -> torch.Tensor:
        """
        Coulomb摩擦定律
        
        Args:
            basal_stress: 底部剪切应力
            normal_stress: 法向应力
            
        Returns:
            torch.Tensor: 摩擦应力
        """
        max_friction = self.friction_coefficient * torch.abs(normal_stress)
        friction_stress = torch.clamp(torch.abs(basal_stress), max=max_friction)
        
        return friction_stress * torch.sign(basal_stress)
    
    def forward(self, basal_stress: torch.Tensor,
               effective_pressure: Optional[torch.Tensor] = None,
               normal_stress: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            basal_stress: 底部剪切应力
            effective_pressure: 有效压力
            normal_stress: 法向应力
            
        Returns:
            Dict: 滑动结果字典
        """
        results = {}
        
        if self.sliding_law_type == 'weertman':
            sliding_velocity = self.weertman_sliding(basal_stress, effective_pressure)
            results['sliding_velocity'] = sliding_velocity
            
        elif self.sliding_law_type == 'coulomb':
            friction_stress = self.coulomb_friction(basal_stress, normal_stress)
            results['friction_stress'] = friction_stress
            
        else:
            raise ValueError(f"未知的滑动定律类型: {self.sliding_law_type}")
        
        return results

class TemperatureEvolution(nn.Module):
    """
    温度演化方程
    
    描述冰川内部温度分布的演化
    """
    
    def __init__(self, include_advection: bool = True,
                 include_strain_heating: bool = True):
        """
        初始化温度演化
        
        Args:
            include_advection: 是否包含对流项
            include_strain_heating: 是否包含应变加热
        """
        super(TemperatureEvolution, self).__init__()
        
        self.include_advection = include_advection
        self.include_strain_heating = include_strain_heating
        
        # 热力学参数
        self.thermal_conductivity = 2.1  # W/(m·K)
        self.specific_heat = 2009.0  # J/(kg·K)
        self.ice_density = 917.0  # kg/m³
        
        # 可学习参数
        self.log_thermal_diffusivity = nn.Parameter(
            torch.tensor(math.log(self.thermal_conductivity / (self.ice_density * self.specific_heat)))
        )
    
    def compute_thermal_diffusivity(self) -> torch.Tensor:
        """
        计算热扩散率
        
        Returns:
            torch.Tensor: 热扩散率
        """
        return torch.exp(self.log_thermal_diffusivity)
    
    def strain_heating_rate(self, stress_tensor: torch.Tensor,
                          strain_rate_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算应变加热率
        
        Args:
            stress_tensor: 应力张量
            strain_rate_tensor: 应变率张量
            
        Returns:
            torch.Tensor: 应变加热率
        """
        # H = σ : ε̇ (应力和应变率的双点积)
        heating_rate = torch.sum(stress_tensor * strain_rate_tensor, dim=(-2, -1))
        
        return heating_rate
    
    def forward(self, temperature: torch.Tensor,
               temperature_gradients: torch.Tensor,
               temperature_laplacian: torch.Tensor,
               velocity: Optional[torch.Tensor] = None,
               stress_tensor: Optional[torch.Tensor] = None,
               strain_rate_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        温度演化方程
        
        Args:
            temperature: 温度场
            temperature_gradients: 温度梯度
            temperature_laplacian: 温度拉普拉斯算子
            velocity: 速度场
            stress_tensor: 应力张量
            strain_rate_tensor: 应变率张量
            
        Returns:
            torch.Tensor: 温度演化残差
        """
        thermal_diffusivity = self.compute_thermal_diffusivity()
        
        # 扩散项
        diffusion_term = thermal_diffusivity * temperature_laplacian
        
        # 对流项
        advection_term = torch.zeros_like(temperature)
        if self.include_advection and velocity is not None:
            # v · ∇T
            for i in range(min(velocity.shape[-1], temperature_gradients.shape[-1])):
                advection_term += velocity[..., i] * temperature_gradients[..., i]
        
        # 应变加热项
        strain_heating_term = torch.zeros_like(temperature)
        if self.include_strain_heating and stress_tensor is not None and strain_rate_tensor is not None:
            heating_rate = self.strain_heating_rate(stress_tensor, strain_rate_tensor)
            strain_heating_term = heating_rate / (self.ice_density * self.specific_heat)
        
        # 温度演化方程: ∂T/∂t + v·∇T = κ∇²T + H/(ρc)
        temperature_residual = diffusion_term - advection_term + strain_heating_term
        
        return temperature_residual

class IceDynamicsModel(nn.Module):
    """
    完整的冰川动力学模型
    
    集成所有冰川动力学组件
    """
    
    def __init__(self, spatial_dim: int = 2, 
                 include_temperature: bool = True,
                 include_sliding: bool = True):
        """
        初始化冰川动力学模型
        
        Args:
            spatial_dim: 空间维度
            include_temperature: 是否包含温度演化
            include_sliding: 是否包含滑动
        """
        super(IceDynamicsModel, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.include_temperature = include_temperature
        self.include_sliding = include_sliding
        
        # 核心组件
        self.glen_flow = GlenFlowLaw()
        self.flow_equations = IceFlowEquations(spatial_dim)
        
        if include_sliding:
            self.sliding_law = SlidingLaw()
        
        if include_temperature:
            self.temperature_evolution = TemperatureEvolution()
    
    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fields: 包含所有场变量的字典
            
        Returns:
            Dict: 物理残差和诊断量字典
        """
        results = {}
        
        # 流动方程
        flow_residuals = self.flow_equations(fields)
        results.update(flow_residuals)
        
        # Glen流动定律
        if 'stress_tensor' in fields and 'temperature' in fields:
            glen_results = self.glen_flow(fields['stress_tensor'], fields['temperature'])
            results.update(glen_results)
        
        # 滑动定律
        if self.include_sliding and 'basal_stress' in fields:
            sliding_results = self.sliding_law(
                fields['basal_stress'],
                fields.get('effective_pressure'),
                fields.get('normal_stress')
            )
            results.update(sliding_results)
        
        # 温度演化
        if self.include_temperature and all(key in fields for key in ['temperature', 'temperature_gradients', 'temperature_laplacian']):
            temperature_residual = self.temperature_evolution(
                fields['temperature'],
                fields['temperature_gradients'],
                fields['temperature_laplacian'],
                fields.get('velocity'),
                fields.get('stress_tensor'),
                fields.get('strain_rate')
            )
            results['temperature_evolution'] = temperature_residual
        
        return results

if __name__ == "__main__":
    # 测试冰川动力学组件
    batch_size = 32
    spatial_points = 100
    
    # 测试Glen流动定律
    glen_flow = GlenFlowLaw()
    stress_tensor = torch.randn(batch_size, spatial_points, 2, 2)
    temperature = torch.ones(batch_size, spatial_points) * 273.15  # 0°C
    
    glen_results = glen_flow(stress_tensor, temperature)
    print(f"Glen流动定律 - 应变率形状: {glen_results['strain_rate'].shape}")
    print(f"Glen流动定律 - 有效应力形状: {glen_results['effective_stress'].shape}")
    
    # 测试滑动定律
    sliding_law = SlidingLaw()
    basal_stress = torch.randn(batch_size, spatial_points, 2)
    effective_pressure = torch.abs(torch.randn(batch_size, spatial_points))
    
    sliding_results = sliding_law(basal_stress, effective_pressure)
    print(f"滑动定律 - 滑动速度形状: {sliding_results['sliding_velocity'].shape}")
    
    # 测试完整动力学模型
    ice_dynamics = IceDynamicsModel(spatial_dim=2)
    
    fields = {
        'velocity': torch.randn(batch_size, spatial_points, 2),
        'velocity_gradients': torch.randn(batch_size, spatial_points, 2, 2),
        'velocity_divergence': torch.randn(batch_size, spatial_points),
        'pressure': torch.randn(batch_size, spatial_points),
        'pressure_gradients': torch.randn(batch_size, spatial_points, 2),
        'stress_tensor': stress_tensor,
        'temperature': temperature,
        'temperature_gradients': torch.randn(batch_size, spatial_points, 2),
        'temperature_laplacian': torch.randn(batch_size, spatial_points),
        'basal_stress': basal_stress,
        'effective_pressure': effective_pressure
    }
    
    dynamics_results = ice_dynamics(fields)
    print(f"冰川动力学模型输出键: {list(dynamics_results.keys())}")