#!/usr/bin/env python3
"""
自动微分实现

提供PINNs所需的自动微分功能，包括：
- 一阶和高阶导数计算
- 梯度和散度计算
- 拉普拉斯算子
- 雅可比矩阵和海塞矩阵

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import warnings

class AutoDiff:
    """
    自动微分工具类
    
    提供各种微分计算功能
    """
    
    @staticmethod
    def gradient(outputs: torch.Tensor, inputs: torch.Tensor, 
                grad_outputs: Optional[torch.Tensor] = None,
                retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算梯度
        
        Args:
            outputs: 输出张量
            inputs: 输入张量
            grad_outputs: 梯度输出
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 梯度张量
        """
        if grad_outputs is None:
            grad_outputs = torch.ones_like(outputs)
            
        gradients = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=True
        )[0]
        
        return gradients
    
    @staticmethod
    def divergence(vector_field: torch.Tensor, coordinates: torch.Tensor,
                  retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算向量场的散度
        
        Args:
            vector_field: 向量场 [batch_size, vector_dim]
            coordinates: 坐标 [batch_size, coord_dim]
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 散度
        """
        batch_size, vector_dim = vector_field.shape
        _, coord_dim = coordinates.shape
        
        if vector_dim != coord_dim:
            raise ValueError(f"向量场维度({vector_dim})必须等于坐标维度({coord_dim})")
        
        div = torch.zeros(batch_size, 1, device=vector_field.device)
        
        for i in range(vector_dim):
            # 计算第i个分量对第i个坐标的偏导数
            component = vector_field[:, i:i+1]
            coord = coordinates[:, i:i+1]
            
            partial_derivative = AutoDiff.gradient(
                outputs=component,
                inputs=coord,
                retain_graph=retain_graph,
                create_graph=create_graph
            )
            
            div += partial_derivative
        
        return div
    
    @staticmethod
    def laplacian(scalar_field: torch.Tensor, coordinates: torch.Tensor,
                 retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算标量场的拉普拉斯算子
        
        Args:
            scalar_field: 标量场 [batch_size, 1]
            coordinates: 坐标 [batch_size, coord_dim]
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 拉普拉斯算子结果
        """
        batch_size, coord_dim = coordinates.shape
        
        laplacian = torch.zeros_like(scalar_field)
        
        for i in range(coord_dim):
            coord = coordinates[:, i:i+1]
            
            # 一阶导数
            first_derivative = AutoDiff.gradient(
                outputs=scalar_field,
                inputs=coord,
                retain_graph=True,
                create_graph=True
            )
            
            # 二阶导数
            second_derivative = AutoDiff.gradient(
                outputs=first_derivative,
                inputs=coord,
                retain_graph=retain_graph,
                create_graph=create_graph
            )
            
            laplacian += second_derivative
        
        return laplacian
    
    @staticmethod
    def jacobian(outputs: torch.Tensor, inputs: torch.Tensor,
                retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算雅可比矩阵
        
        Args:
            outputs: 输出张量 [batch_size, output_dim]
            inputs: 输入张量 [batch_size, input_dim]
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 雅可比矩阵 [batch_size, output_dim, input_dim]
        """
        batch_size, output_dim = outputs.shape
        _, input_dim = inputs.shape
        
        jacobian_matrix = torch.zeros(batch_size, output_dim, input_dim, 
                                    device=outputs.device, dtype=outputs.dtype)
        
        for i in range(output_dim):
            output_component = outputs[:, i:i+1]
            
            gradients = AutoDiff.gradient(
                outputs=output_component,
                inputs=inputs,
                retain_graph=retain_graph if i < output_dim - 1 else retain_graph,
                create_graph=create_graph
            )
            
            jacobian_matrix[:, i, :] = gradients
        
        return jacobian_matrix
    
    @staticmethod
    def hessian(scalar_field: torch.Tensor, coordinates: torch.Tensor,
               retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算海塞矩阵
        
        Args:
            scalar_field: 标量场 [batch_size, 1]
            coordinates: 坐标 [batch_size, coord_dim]
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 海塞矩阵 [batch_size, coord_dim, coord_dim]
        """
        batch_size, coord_dim = coordinates.shape
        
        hessian_matrix = torch.zeros(batch_size, coord_dim, coord_dim,
                                   device=scalar_field.device, dtype=scalar_field.dtype)
        
        # 计算一阶导数
        first_derivatives = []
        for i in range(coord_dim):
            coord = coordinates[:, i:i+1]
            first_deriv = AutoDiff.gradient(
                outputs=scalar_field,
                inputs=coord,
                retain_graph=True,
                create_graph=True
            )
            first_derivatives.append(first_deriv)
        
        # 计算二阶导数
        for i in range(coord_dim):
            for j in range(coord_dim):
                coord_j = coordinates[:, j:j+1]
                
                second_deriv = AutoDiff.gradient(
                    outputs=first_derivatives[i],
                    inputs=coord_j,
                    retain_graph=retain_graph if (i < coord_dim - 1 or j < coord_dim - 1) else retain_graph,
                    create_graph=create_graph
                )
                
                hessian_matrix[:, i, j] = second_deriv.squeeze()
        
        return hessian_matrix
    
    @staticmethod
    def curl_2d(vector_field: torch.Tensor, coordinates: torch.Tensor,
               retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算2D向量场的旋度
        
        Args:
            vector_field: 2D向量场 [batch_size, 2] (vx, vy)
            coordinates: 2D坐标 [batch_size, 2] (x, y)
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 旋度 [batch_size, 1]
        """
        if vector_field.shape[1] != 2 or coordinates.shape[1] != 2:
            raise ValueError("curl_2d只适用于2D向量场和坐标")
        
        vx = vector_field[:, 0:1]
        vy = vector_field[:, 1:2]
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        
        # ∂vy/∂x
        dvydx = AutoDiff.gradient(
            outputs=vy,
            inputs=x,
            retain_graph=True,
            create_graph=create_graph
        )
        
        # ∂vx/∂y
        dvxdy = AutoDiff.gradient(
            outputs=vx,
            inputs=y,
            retain_graph=retain_graph,
            create_graph=create_graph
        )
        
        curl = dvydx - dvxdy
        return curl
    
    @staticmethod
    def strain_rate_tensor_2d(velocity_field: torch.Tensor, coordinates: torch.Tensor,
                             retain_graph: bool = True, create_graph: bool = True) -> torch.Tensor:
        """
        计算2D应变率张量
        
        Args:
            velocity_field: 速度场 [batch_size, 2] (vx, vy)
            coordinates: 坐标 [batch_size, 2] (x, y)
            retain_graph: 是否保留计算图
            create_graph: 是否创建计算图
            
        Returns:
            Tensor: 应变率张量 [batch_size, 2, 2]
        """
        if velocity_field.shape[1] != 2 or coordinates.shape[1] != 2:
            raise ValueError("strain_rate_tensor_2d只适用于2D速度场和坐标")
        
        batch_size = velocity_field.shape[0]
        strain_tensor = torch.zeros(batch_size, 2, 2, device=velocity_field.device)
        
        vx = velocity_field[:, 0:1]
        vy = velocity_field[:, 1:2]
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        
        # ∂vx/∂x
        dvxdx = AutoDiff.gradient(vx, x, retain_graph=True, create_graph=create_graph)
        
        # ∂vy/∂y
        dvydy = AutoDiff.gradient(vy, y, retain_graph=True, create_graph=create_graph)
        
        # ∂vx/∂y
        dvxdy = AutoDiff.gradient(vx, y, retain_graph=True, create_graph=create_graph)
        
        # ∂vy/∂x
        dvydx = AutoDiff.gradient(vy, x, retain_graph=retain_graph, create_graph=create_graph)
        
        # 应变率张量分量
        strain_tensor[:, 0, 0] = dvxdx.squeeze()
        strain_tensor[:, 1, 1] = dvydy.squeeze()
        strain_tensor[:, 0, 1] = 0.5 * (dvxdy + dvydx).squeeze()
        strain_tensor[:, 1, 0] = strain_tensor[:, 0, 1]
        
        return strain_tensor

class GlacierDifferentialOperators:
    """
    冰川专用微分算子
    
    针对冰川物理过程的特化微分计算
    """
    
    @staticmethod
    def mass_balance_operator(thickness: torch.Tensor, velocity: torch.Tensor,
                            coordinates: torch.Tensor, accumulation: torch.Tensor,
                            ablation: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        质量平衡算子
        
        ∂h/∂t + ∇·(h*v) = a - b
        
        Args:
            thickness: 冰川厚度 [batch_size, 1]
            velocity: 速度场 [batch_size, 2]
            coordinates: 空间坐标 [batch_size, 2]
            accumulation: 积累率 [batch_size, 1]
            ablation: 消融率 [batch_size, 1]
            time: 时间坐标 [batch_size, 1]
            
        Returns:
            Tensor: 质量平衡残差
        """
        # ∂h/∂t
        dhdt = AutoDiff.gradient(
            outputs=thickness,
            inputs=time,
            retain_graph=True,
            create_graph=True
        )
        
        # h*v
        flux = thickness.unsqueeze(-1) * velocity  # [batch_size, 2]
        
        # ∇·(h*v)
        flux_divergence = AutoDiff.divergence(
            vector_field=flux,
            coordinates=coordinates,
            retain_graph=True,
            create_graph=True
        )
        
        # 质量平衡方程残差
        residual = dhdt + flux_divergence - (accumulation - ablation)
        
        return residual
    
    @staticmethod
    def momentum_balance_operator(velocity: torch.Tensor, pressure: torch.Tensor,
                                coordinates: torch.Tensor, viscosity: torch.Tensor,
                                gravity: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        动量平衡算子
        
        ρ(∂v/∂t + v·∇v) = -∇p + ∇·τ + ρg
        
        Args:
            velocity: 速度场 [batch_size, 2]
            pressure: 压力场 [batch_size, 1]
            coordinates: 空间坐标 [batch_size, 2]
            viscosity: 粘度 [batch_size, 1]
            gravity: 重力加速度 [batch_size, 2]
            density: 密度 [batch_size, 1]
            
        Returns:
            Tensor: 动量平衡残差 [batch_size, 2]
        """
        # TODO: 实现动量平衡算子
        pass
    
    @staticmethod
    def glen_flow_law(strain_rate: torch.Tensor, stress: torch.Tensor,
                     temperature: torch.Tensor, A: torch.Tensor,
                     n: float = 3.0) -> torch.Tensor:
        """
        Glen流动律
        
        ε̇ = A(T) * τ^n
        
        Args:
            strain_rate: 应变率张量 [batch_size, 2, 2]
            stress: 应力张量 [batch_size, 2, 2]
            temperature: 温度 [batch_size, 1]
            A: 流动律参数 [batch_size, 1]
            n: Glen指数
            
        Returns:
            Tensor: Glen流动律残差
        """
        # TODO: 实现Glen流动律
        pass
    
    @staticmethod
    def effective_stress(stress_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算有效应力
        
        Args:
            stress_tensor: 应力张量 [batch_size, 2, 2]
            
        Returns:
            Tensor: 有效应力 [batch_size, 1]
        """
        # 计算应力张量的第二不变量
        # τ_eff = sqrt(0.5 * τ_ij * τ_ij)
        
        stress_squared = torch.sum(stress_tensor * stress_tensor, dim=(-2, -1), keepdim=True)
        effective_stress = torch.sqrt(0.5 * stress_squared)
        
        return effective_stress
    
    @staticmethod
    def effective_strain_rate(strain_rate_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算有效应变率
        
        Args:
            strain_rate_tensor: 应变率张量 [batch_size, 2, 2]
            
        Returns:
            Tensor: 有效应变率 [batch_size, 1]
        """
        # 计算应变率张量的第二不变量
        # ε̇_eff = sqrt(0.5 * ε̇_ij * ε̇_ij)
        
        strain_rate_squared = torch.sum(strain_rate_tensor * strain_rate_tensor, 
                                      dim=(-2, -1), keepdim=True)
        effective_strain_rate = torch.sqrt(0.5 * strain_rate_squared)
        
        return effective_strain_rate

def test_autodiff():
    """
    测试自动微分功能
    """
    # 创建测试数据
    x = torch.randn(100, 2, requires_grad=True)
    
    # 测试函数: f(x,y) = x^2 + y^2
    f = torch.sum(x**2, dim=1, keepdim=True)
    
    # 计算梯度
    grad_f = AutoDiff.gradient(f, x)
    print(f"梯度形状: {grad_f.shape}")
    
    # 计算拉普拉斯算子
    laplacian_f = AutoDiff.laplacian(f, x)
    print(f"拉普拉斯算子形状: {laplacian_f.shape}")
    
    # 计算海塞矩阵
    hessian_f = AutoDiff.hessian(f, x)
    print(f"海塞矩阵形状: {hessian_f.shape}")

if __name__ == "__main__":
    test_autodiff()