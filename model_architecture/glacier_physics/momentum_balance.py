#!/usr/bin/env python3
"""
动量平衡方程实现

实现冰川动量平衡的物理定律，包括：
- Stokes方程
- 应力-应变关系
- 边界条件
- 粘性流动

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod

class MomentumBalanceLaw(nn.Module, ABC):
    """
    动量平衡定律基类
    
    定义冰川动量平衡的抽象接口
    """
    
    def __init__(self):
        """
        初始化动量平衡定律
        """
        super(MomentumBalanceLaw, self).__init__()
    
    @abstractmethod
    def compute_momentum_balance(self, velocity: torch.Tensor,
                               pressure: torch.Tensor,
                               coordinates: torch.Tensor,
                               time: torch.Tensor) -> torch.Tensor:
        """
        计算动量平衡
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            time: 时间坐标
            
        Returns:
            Tensor: 动量平衡残差
        """
        pass
    
    @abstractmethod
    def compute_stress_tensor(self, velocity: torch.Tensor,
                            coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算应力张量
        
        Args:
            velocity: 速度场
            coordinates: 空间坐标
            
        Returns:
            Tensor: 应力张量
        """
        pass

class StokesEquation(MomentumBalanceLaw):
    """
    Stokes方程实现
    
    实现冰川的Stokes方程：∇·σ + ρg = 0
    """
    
    def __init__(self, ice_density: float = 917.0, gravity: float = 9.81):
        """
        初始化Stokes方程
        
        Args:
            ice_density: 冰的密度 (kg/m³)
            gravity: 重力加速度 (m/s²)
        """
        super(StokesEquation, self).__init__()
        self.ice_density = ice_density
        self.gravity = gravity
    
    def compute_strain_rate_tensor(self, velocity: torch.Tensor,
                                 coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算应变率张量
        
        Args:
            velocity: 速度场
            coordinates: 空间坐标
            
        Returns:
            Tensor: 应变率张量
        """
        batch_size = velocity.shape[0]
        dim = velocity.shape[-1]
        
        # 计算速度梯度
        velocity_gradients = []
        for i in range(dim):
            grad_vi = torch.autograd.grad(
                outputs=velocity[..., i],
                inputs=coordinates,
                grad_outputs=torch.ones_like(velocity[..., i]),
                create_graph=True,
                retain_graph=True
            )[0]
            velocity_gradients.append(grad_vi)
        
        # 构建应变率张量：ε_ij = 1/2 * (∂v_i/∂x_j + ∂v_j/∂x_i)
        strain_rate = torch.zeros(batch_size, dim, dim)
        
        for i in range(dim):
            for j in range(dim):
                strain_rate[:, i, j] = 0.5 * (
                    velocity_gradients[i][..., j] + velocity_gradients[j][..., i]
                )
        
        return strain_rate
    
    def compute_effective_strain_rate(self, strain_rate: torch.Tensor) -> torch.Tensor:
        """
        计算有效应变率
        
        Args:
            strain_rate: 应变率张量
            
        Returns:
            Tensor: 有效应变率
        """
        # 计算第二不变量：ε_eff = sqrt(1/2 * ε_ij * ε_ij)
        strain_rate_squared = torch.sum(strain_rate * strain_rate, dim=(-2, -1))
        effective_strain_rate = torch.sqrt(0.5 * strain_rate_squared + 1e-12)
        
        return effective_strain_rate
    
    def compute_viscosity(self, effective_strain_rate: torch.Tensor,
                        temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算粘性系数
        
        Args:
            effective_strain_rate: 有效应变率
            temperature: 温度场
            
        Returns:
            Tensor: 粘性系数
        """
        # Glen流动定律参数
        A = 2.4e-24  # Pa^-3 s^-1 (at 0°C)
        n = 3.0  # Glen指数
        
        # 温度依赖性（Arrhenius关系）
        if temperature is not None:
            Q = 60000.0  # J/mol (activation energy)
            R = 8.314  # J/(mol·K)
            T_ref = 273.15  # K
            
            # 温度修正因子
            temp_factor = torch.exp(-Q/R * (1/temperature - 1/T_ref))
            A_temp = A * temp_factor
        else:
            A_temp = A
        
        # Glen流动定律：η = 1/2 * A^(-1/n) * ε_eff^((1-n)/n)
        viscosity = 0.5 * (A_temp ** (-1/n)) * (effective_strain_rate ** ((1-n)/n))
        
        # 限制粘性系数范围
        viscosity = torch.clamp(viscosity, min=1e10, max=1e16)
        
        return viscosity
    
    def compute_stress_tensor(self, velocity: torch.Tensor,
                            coordinates: torch.Tensor,
                            temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算应力张量
        
        Args:
            velocity: 速度场
            coordinates: 空间坐标
            temperature: 温度场
            
        Returns:
            Tensor: 应力张量
        """
        # 计算应变率张量
        strain_rate = self.compute_strain_rate_tensor(velocity, coordinates)
        
        # 计算有效应变率
        effective_strain_rate = self.compute_effective_strain_rate(strain_rate)
        
        # 计算粘性系数
        viscosity = self.compute_viscosity(effective_strain_rate, temperature)
        
        # 计算偏应力张量：τ_ij = 2η * ε_ij
        stress_tensor = 2 * viscosity.unsqueeze(-1).unsqueeze(-1) * strain_rate
        
        return stress_tensor
    
    def compute_stress_divergence(self, stress_tensor: torch.Tensor,
                                coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算应力散度
        
        Args:
            stress_tensor: 应力张量
            coordinates: 空间坐标
            
        Returns:
            Tensor: 应力散度
        """
        batch_size = stress_tensor.shape[0]
        dim = stress_tensor.shape[-1]
        
        stress_divergence = torch.zeros(batch_size, dim)
        
        # 计算 ∇·σ
        for i in range(dim):
            for j in range(dim):
                grad_stress_ij = torch.autograd.grad(
                    outputs=stress_tensor[:, i, j],
                    inputs=coordinates,
                    grad_outputs=torch.ones_like(stress_tensor[:, i, j]),
                    create_graph=True,
                    retain_graph=True
                )[0]
                stress_divergence[:, i] += grad_stress_ij[:, j]
        
        return stress_divergence
    
    def compute_momentum_balance(self, velocity: torch.Tensor,
                               pressure: torch.Tensor,
                               coordinates: torch.Tensor,
                               time: torch.Tensor,
                               temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算动量平衡残差
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            time: 时间坐标
            temperature: 温度场
            
        Returns:
            Tensor: 动量平衡残差
        """
        # 计算应力张量
        stress_tensor = self.compute_stress_tensor(velocity, coordinates, temperature)
        
        # 计算应力散度
        stress_divergence = self.compute_stress_divergence(stress_tensor, coordinates)
        
        # 计算压力梯度
        pressure_gradient = torch.autograd.grad(
            outputs=pressure,
            inputs=coordinates,
            grad_outputs=torch.ones_like(pressure),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 重力项
        dim = coordinates.shape[-1]
        gravity_force = torch.zeros_like(stress_divergence)
        if dim >= 2:  # 假设z是第二个坐标（垂直方向）
            gravity_force[:, -1] = -self.ice_density * self.gravity
        
        # Stokes方程：∇·σ - ∇p + ρg = 0
        momentum_residual = stress_divergence - pressure_gradient + gravity_force
        
        return momentum_residual

class ShallowIceApproximation(MomentumBalanceLaw):
    """
    浅冰近似实现
    
    实现冰川的浅冰近似方程
    """
    
    def __init__(self, ice_density: float = 917.0, gravity: float = 9.81):
        """
        初始化浅冰近似
        
        Args:
            ice_density: 冰的密度 (kg/m³)
            gravity: 重力加速度 (m/s²)
        """
        super(ShallowIceApproximation, self).__init__()
        self.ice_density = ice_density
        self.gravity = gravity
    
    def compute_driving_stress(self, thickness: torch.Tensor,
                             surface_gradient: torch.Tensor) -> torch.Tensor:
        """
        计算驱动应力
        
        Args:
            thickness: 冰川厚度
            surface_gradient: 表面梯度
            
        Returns:
            Tensor: 驱动应力
        """
        # 驱动应力：τ_d = ρgh∇s
        driving_stress = self.ice_density * self.gravity * thickness.unsqueeze(-1) * surface_gradient
        
        return driving_stress
    
    def compute_basal_stress(self, velocity: torch.Tensor,
                           sliding_coefficient: torch.Tensor) -> torch.Tensor:
        """
        计算基底应力
        
        Args:
            velocity: 基底速度
            sliding_coefficient: 滑动系数
            
        Returns:
            Tensor: 基底应力
        """
        # 线性滑动定律：τ_b = β * v_b
        basal_stress = sliding_coefficient.unsqueeze(-1) * velocity
        
        return basal_stress
    
    def compute_stress_tensor(self, velocity: torch.Tensor,
                            coordinates: torch.Tensor) -> torch.Tensor:
        """
        计算应力张量（浅冰近似）
        
        Args:
            velocity: 速度场
            coordinates: 空间坐标
            
        Returns:
            Tensor: 应力张量
        """
        # 在浅冰近似中，主要考虑水平剪切应力
        batch_size = velocity.shape[0]
        dim = coordinates.shape[-1]
        
        # 计算水平速度梯度
        if dim >= 2:
            du_dx = torch.autograd.grad(
                outputs=velocity[..., 0],
                inputs=coordinates,
                grad_outputs=torch.ones_like(velocity[..., 0]),
                create_graph=True,
                retain_graph=True
            )[0][..., 0]
            
            dv_dy = torch.autograd.grad(
                outputs=velocity[..., 1] if velocity.shape[-1] > 1 else torch.zeros_like(velocity[..., 0]),
                inputs=coordinates,
                grad_outputs=torch.ones_like(velocity[..., 0]),
                create_graph=True,
                retain_graph=True
            )[0][..., 1] if dim > 1 else torch.zeros_like(du_dx)
        
        # 构建简化的应力张量
        stress_tensor = torch.zeros(batch_size, dim, dim)
        if dim >= 2:
            stress_tensor[:, 0, 0] = du_dx
            stress_tensor[:, 1, 1] = dv_dy
            stress_tensor[:, 0, 1] = stress_tensor[:, 1, 0] = 0.5 * (du_dx + dv_dy)
        
        return stress_tensor
    
    def compute_momentum_balance(self, velocity: torch.Tensor,
                               pressure: torch.Tensor,
                               coordinates: torch.Tensor,
                               time: torch.Tensor,
                               thickness: torch.Tensor,
                               surface_gradient: torch.Tensor,
                               sliding_coefficient: torch.Tensor) -> torch.Tensor:
        """
        计算动量平衡残差（浅冰近似）
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            time: 时间坐标
            thickness: 冰川厚度
            surface_gradient: 表面梯度
            sliding_coefficient: 滑动系数
            
        Returns:
            Tensor: 动量平衡残差
        """
        # 计算驱动应力
        driving_stress = self.compute_driving_stress(thickness, surface_gradient)
        
        # 计算基底应力
        basal_stress = self.compute_basal_stress(velocity, sliding_coefficient)
        
        # 浅冰近似动量平衡：τ_d = τ_b
        momentum_residual = driving_stress - basal_stress
        
        return momentum_residual

class ViscousFlowModel(nn.Module):
    """
    粘性流动模型
    
    实现冰川的粘性流动
    """
    
    def __init__(self, flow_law_type: str = 'glen'):
        """
        初始化粘性流动模型
        
        Args:
            flow_law_type: 流动定律类型
        """
        super(ViscousFlowModel, self).__init__()
        self.flow_law_type = flow_law_type
    
    def glen_flow_law(self, stress: torch.Tensor,
                     temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Glen流动定律
        
        Args:
            stress: 应力
            temperature: 温度
            
        Returns:
            Tensor: 应变率
        """
        # Glen流动定律参数
        A = 2.4e-24  # Pa^-3 s^-1
        n = 3.0
        
        # 温度依赖性
        if temperature is not None:
            Q = 60000.0  # J/mol
            R = 8.314  # J/(mol·K)
            T_ref = 273.15  # K
            
            temp_factor = torch.exp(-Q/R * (1/temperature - 1/T_ref))
            A_temp = A * temp_factor
        else:
            A_temp = A
        
        # Glen流动定律：ε = A * τ^n
        strain_rate = A_temp * (torch.abs(stress) ** n) * torch.sign(stress)
        
        return strain_rate
    
    def compute_flow_velocity(self, stress: torch.Tensor,
                            thickness: torch.Tensor,
                            temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算流动速度
        
        Args:
            stress: 应力
            thickness: 厚度
            temperature: 温度
            
        Returns:
            Tensor: 流动速度
        """
        if self.flow_law_type == 'glen':
            strain_rate = self.glen_flow_law(stress, temperature)
        else:
            raise ValueError(f"不支持的流动定律类型: {self.flow_law_type}")
        
        # 积分应变率得到速度（简化）
        velocity = strain_rate * thickness
        
        return velocity

class GlacierMomentumBalance(nn.Module):
    """
    冰川动量平衡综合模型
    
    集成各种动量平衡组件
    """
    
    def __init__(self, approximation: str = 'stokes',
                 ice_density: float = 917.0,
                 gravity: float = 9.81):
        """
        初始化冰川动量平衡模型
        
        Args:
            approximation: 近似类型 ('stokes' 或 'shallow_ice')
            ice_density: 冰的密度
            gravity: 重力加速度
        """
        super(GlacierMomentumBalance, self).__init__()
        
        if approximation == 'stokes':
            self.momentum_eq = StokesEquation(ice_density, gravity)
        elif approximation == 'shallow_ice':
            self.momentum_eq = ShallowIceApproximation(ice_density, gravity)
        else:
            raise ValueError(f"不支持的近似类型: {approximation}")
        
        self.viscous_flow = ViscousFlowModel()
        self.approximation = approximation
    
    def compute_momentum_residual(self, velocity: torch.Tensor,
                                pressure: torch.Tensor,
                                coordinates: torch.Tensor,
                                time: torch.Tensor,
                                **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算动量平衡残差
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            time: 时间坐标
            **kwargs: 其他参数
            
        Returns:
            Dict: 各种残差
        """
        results = {}
        
        if self.approximation == 'stokes':
            momentum_residual = self.momentum_eq.compute_momentum_balance(
                velocity, pressure, coordinates, time,
                kwargs.get('temperature', None)
            )
            
            stress_tensor = self.momentum_eq.compute_stress_tensor(
                velocity, coordinates, kwargs.get('temperature', None)
            )
            
        elif self.approximation == 'shallow_ice':
            momentum_residual = self.momentum_eq.compute_momentum_balance(
                velocity, pressure, coordinates, time,
                kwargs['thickness'], kwargs['surface_gradient'],
                kwargs['sliding_coefficient']
            )
            
            stress_tensor = self.momentum_eq.compute_stress_tensor(
                velocity, coordinates
            )
        
        results.update({
            'momentum_residual': momentum_residual,
            'stress_tensor': stress_tensor
        })
        
        return results
    
    def validate_momentum_balance(self, velocity: torch.Tensor,
                                pressure: torch.Tensor,
                                coordinates: torch.Tensor,
                                time: torch.Tensor,
                                **kwargs) -> Dict[str, float]:
        """
        验证动量平衡
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            time: 时间坐标
            **kwargs: 其他参数
            
        Returns:
            Dict: 验证指标
        """
        results = self.compute_momentum_residual(
            velocity, pressure, coordinates, time, **kwargs
        )
        
        momentum_residual = results['momentum_residual']
        
        # 计算统计指标
        mean_residual = torch.mean(torch.norm(momentum_residual, dim=-1)).item()
        max_residual = torch.max(torch.norm(momentum_residual, dim=-1)).item()
        std_residual = torch.std(torch.norm(momentum_residual, dim=-1)).item()
        
        validation_metrics = {
            'mean_momentum_residual': mean_residual,
            'max_momentum_residual': max_residual,
            'momentum_residual_std': std_residual,
            'momentum_balance_satisfied': mean_residual < 1e-3
        }
        
        return validation_metrics

def create_momentum_balance_model(approximation: str = 'stokes',
                                **kwargs) -> nn.Module:
    """
    创建动量平衡模型
    
    Args:
        approximation: 近似类型
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 动量平衡模型实例
    """
    return GlacierMomentumBalance(approximation=approximation, **kwargs)

if __name__ == "__main__":
    # 测试动量平衡模型
    momentum_balance = GlacierMomentumBalance(approximation='stokes')
    
    # 测试数据
    batch_size = 100
    velocity = torch.randn(batch_size, 2, requires_grad=True) * 10  # m/year
    pressure = torch.randn(batch_size, 1, requires_grad=True) * 1e5  # Pa
    coordinates = torch.randn(batch_size, 2, requires_grad=True) * 1000  # m
    time = torch.randn(batch_size, 1, requires_grad=True) * 365  # days
    temperature = torch.randn(batch_size, 1) * 10 + 268  # K
    
    # 计算动量平衡残差
    results = momentum_balance.compute_momentum_residual(
        velocity, pressure, coordinates, time, temperature=temperature
    )
    
    print("动量平衡计算结果:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状={value.shape}, 范围=[{value.min().item():.2e}, {value.max().item():.2e}]")
    
    # 验证动量平衡
    validation = momentum_balance.validate_momentum_balance(
        velocity, pressure, coordinates, time, temperature=temperature
    )
    
    print("\n动量平衡验证:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    # 测试浅冰近似
    print("\n测试浅冰近似:")
    shallow_ice = GlacierMomentumBalance(approximation='shallow_ice')
    
    thickness = torch.randn(batch_size, 1) * 100 + 200  # m
    surface_gradient = torch.randn(batch_size, 2) * 0.1  # 无量纲
    sliding_coefficient = torch.randn(batch_size, 1) * 1e6 + 1e6  # Pa·s/m
    
    shallow_results = shallow_ice.compute_momentum_residual(
        velocity, pressure, coordinates, time,
        thickness=thickness,
        surface_gradient=surface_gradient,
        sliding_coefficient=sliding_coefficient
    )
    
    print("浅冰近似结果:")
    for key, value in shallow_results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状={value.shape}, 范围=[{value.min().item():.2e}, {value.max().item():.2e}]")