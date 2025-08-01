#!/usr/bin/env python3
"""
损失函数实现

提供PINNs训练所需的各种损失函数，包括：
- 数据损失
- 物理损失
- 边界条件损失
- 正则化损失
- 自适应权重损失

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """
    基础损失函数类
    
    所有损失函数的基类
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        """
        初始化基础损失
        
        Args:
            weight: 损失权重
            reduction: 损失聚合方式
        """
        super(BaseLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播
        """
        pass
        
    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        损失聚合
        
        Args:
            loss: 损失张量
            
        Returns:
            Tensor: 聚合后的损失
        """
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"不支持的reduction方式: {self.reduction}")

class DataLoss(BaseLoss):
    """
    数据损失函数
    
    计算预测值与观测数据之间的损失
    """
    
    def __init__(self, loss_type: str = 'mse', weight: float = 1.0, 
                 reduction: str = 'mean', robust: bool = False):
        """
        初始化数据损失
        
        Args:
            loss_type: 损失类型 ('mse', 'mae', 'huber', 'quantile')
            weight: 损失权重
            reduction: 损失聚合方式
            robust: 是否使用鲁棒损失
        """
        super(DataLoss, self).__init__(weight, reduction)
        self.loss_type = loss_type
        self.robust = robust
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               weights: Optional[torch.Tensor] = None,
               uncertainties: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算数据损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            weights: 数据权重
            uncertainties: 数据不确定性
            
        Returns:
            Tensor: 数据损失
        """
        # 基础损失计算
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'mae':
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predictions, targets, reduction='none')
        elif self.loss_type == 'quantile':
            loss = self._quantile_loss(predictions, targets)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        # 考虑不确定性
        if uncertainties is not None:
            # 基于不确定性的加权
            loss = loss / (uncertainties**2 + 1e-8) + torch.log(uncertainties + 1e-8)
        
        # 应用数据权重
        if weights is not None:
            loss = loss * weights
        
        # 鲁棒损失处理
        if self.robust:
            loss = self._robust_loss(loss)
        
        return self.weight * self._reduce_loss(loss)
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                      quantile: float = 0.5) -> torch.Tensor:
        """
        分位数损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            quantile: 分位数
            
        Returns:
            Tensor: 分位数损失
        """
        errors = targets - predictions
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss
    
    def _robust_loss(self, loss: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """
        鲁棒损失处理
        
        Args:
            loss: 原始损失
            threshold: 阈值
            
        Returns:
            Tensor: 鲁棒损失
        """
        # 使用Huber损失的思想
        mask = loss <= threshold
        robust_loss = torch.where(mask, loss, threshold * (2 * torch.sqrt(loss) - 1))
        return robust_loss

class PhysicsLoss(BaseLoss):
    """
    物理损失函数
    
    计算物理方程残差损失
    """
    
    def __init__(self, equation_type: str, weight: float = 1.0, 
                 reduction: str = 'mean', adaptive: bool = False):
        """
        初始化物理损失
        
        Args:
            equation_type: 方程类型
            weight: 损失权重
            reduction: 损失聚合方式
            adaptive: 是否使用自适应权重
        """
        super(PhysicsLoss, self).__init__(weight, reduction)
        self.equation_type = equation_type
        self.adaptive = adaptive
        
        if adaptive:
            self.adaptive_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, residuals: torch.Tensor, 
               equation_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算物理损失
        
        Args:
            residuals: 物理方程残差
            equation_weights: 方程权重
            
        Returns:
            Tensor: 物理损失
        """
        # 计算残差的平方
        loss = torch.square(residuals)
        
        # 应用方程权重
        if equation_weights is not None:
            loss = loss * equation_weights
        
        # 自适应权重
        if self.adaptive:
            loss = loss * torch.exp(-self.adaptive_weight)
        
        return self.weight * self._reduce_loss(loss)

class BoundaryLoss(BaseLoss):
    """
    边界条件损失函数
    
    计算边界条件违反的损失
    """
    
    def __init__(self, boundary_type: str, weight: float = 1.0, 
                 reduction: str = 'mean', penalty_factor: float = 1.0):
        """
        初始化边界损失
        
        Args:
            boundary_type: 边界类型 ('dirichlet', 'neumann', 'robin')
            weight: 损失权重
            reduction: 损失聚合方式
            penalty_factor: 惩罚因子
        """
        super(BoundaryLoss, self).__init__(weight, reduction)
        self.boundary_type = boundary_type
        self.penalty_factor = penalty_factor
        
    def forward(self, boundary_predictions: torch.Tensor,
               boundary_conditions: torch.Tensor,
               boundary_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算边界损失
        
        Args:
            boundary_predictions: 边界预测值
            boundary_conditions: 边界条件值
            boundary_weights: 边界权重
            
        Returns:
            Tensor: 边界损失
        """
        # 计算边界条件违反
        if self.boundary_type == 'dirichlet':
            # Dirichlet边界条件: u = g
            loss = torch.square(boundary_predictions - boundary_conditions)
        elif self.boundary_type == 'neumann':
            # Neumann边界条件: ∂u/∂n = g
            loss = torch.square(boundary_predictions - boundary_conditions)
        elif self.boundary_type == 'robin':
            # Robin边界条件: αu + β∂u/∂n = g
            # TODO: 实现Robin边界条件
            loss = torch.square(boundary_predictions - boundary_conditions)
        else:
            raise ValueError(f"不支持的边界类型: {self.boundary_type}")
        
        # 应用边界权重
        if boundary_weights is not None:
            loss = loss * boundary_weights
        
        # 应用惩罚因子
        loss = loss * self.penalty_factor
        
        return self.weight * self._reduce_loss(loss)

class RegularizationLoss(BaseLoss):
    """
    正则化损失函数
    
    防止过拟合的正则化项
    """
    
    def __init__(self, reg_type: str = 'l2', weight: float = 1e-4, 
                 reduction: str = 'mean'):
        """
        初始化正则化损失
        
        Args:
            reg_type: 正则化类型 ('l1', 'l2', 'elastic')
            weight: 正则化权重
            reduction: 损失聚合方式
        """
        super(RegularizationLoss, self).__init__(weight, reduction)
        self.reg_type = reg_type
        
    def forward(self, model: nn.Module, l1_ratio: float = 0.5) -> torch.Tensor:
        """
        计算正则化损失
        
        Args:
            model: 神经网络模型
            l1_ratio: L1正则化比例（用于elastic net）
            
        Returns:
            Tensor: 正则化损失
        """
        reg_loss = 0.0
        
        for param in model.parameters():
            if param.requires_grad:
                if self.reg_type == 'l1':
                    reg_loss += torch.sum(torch.abs(param))
                elif self.reg_type == 'l2':
                    reg_loss += torch.sum(torch.square(param))
                elif self.reg_type == 'elastic':
                    reg_loss += l1_ratio * torch.sum(torch.abs(param)) + \
                               (1 - l1_ratio) * torch.sum(torch.square(param))
                else:
                    raise ValueError(f"不支持的正则化类型: {self.reg_type}")
        
        return self.weight * reg_loss

class GradientPenalty(BaseLoss):
    """
    梯度惩罚损失
    
    约束梯度的大小或平滑性
    """
    
    def __init__(self, penalty_type: str = 'gradient_norm', weight: float = 1.0,
                 reduction: str = 'mean', target_norm: float = 1.0):
        """
        初始化梯度惩罚
        
        Args:
            penalty_type: 惩罚类型 ('gradient_norm', 'gradient_smoothness')
            weight: 惩罚权重
            reduction: 损失聚合方式
            target_norm: 目标梯度范数
        """
        super(GradientPenalty, self).__init__(weight, reduction)
        self.penalty_type = penalty_type
        self.target_norm = target_norm
        
    def forward(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        计算梯度惩罚
        
        Args:
            outputs: 网络输出
            inputs: 网络输入
            
        Returns:
            Tensor: 梯度惩罚损失
        """
        from .automatic_differentiation import AutoDiff
        
        # 计算梯度
        gradients = AutoDiff.gradient(outputs, inputs)
        
        if self.penalty_type == 'gradient_norm':
            # 梯度范数惩罚
            grad_norm = torch.norm(gradients, dim=-1, keepdim=True)
            penalty = torch.square(grad_norm - self.target_norm)
        elif self.penalty_type == 'gradient_smoothness':
            # 梯度平滑性惩罚
            # TODO: 实现梯度平滑性惩罚
            penalty = torch.square(gradients)
        else:
            raise ValueError(f"不支持的梯度惩罚类型: {self.penalty_type}")
        
        return self.weight * self._reduce_loss(penalty)

class AdaptiveWeightLoss(nn.Module):
    """
    自适应权重损失
    
    动态调整不同损失项的权重
    """
    
    def __init__(self, loss_names: List[str], initial_weights: Optional[List[float]] = None,
                 adaptation_method: str = 'gradnorm'):
        """
        初始化自适应权重损失
        
        Args:
            loss_names: 损失名称列表
            initial_weights: 初始权重
            adaptation_method: 自适应方法 ('gradnorm', 'uncertainty')
        """
        super(AdaptiveWeightLoss, self).__init__()
        
        self.loss_names = loss_names
        self.adaptation_method = adaptation_method
        
        # 初始化权重
        if initial_weights is None:
            initial_weights = [1.0] * len(loss_names)
        
        if adaptation_method == 'gradnorm':
            self.log_weights = nn.Parameter(torch.log(torch.tensor(initial_weights, dtype=torch.float32)))
        elif adaptation_method == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(len(loss_names)))
        else:
            raise ValueError(f"不支持的自适应方法: {adaptation_method}")
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算自适应加权损失
        
        Args:
            losses: 各项损失字典
            
        Returns:
            Tuple: (总损失, 权重字典)
        """
        if self.adaptation_method == 'gradnorm':
            weights = torch.exp(self.log_weights)
            total_loss = sum(w * losses[name] for w, name in zip(weights, self.loss_names))
            weight_dict = {name: w.item() for name, w in zip(self.loss_names, weights)}
        
        elif self.adaptation_method == 'uncertainty':
            # 基于不确定性的权重
            total_loss = 0
            weight_dict = {}
            
            for i, name in enumerate(self.loss_names):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * losses[name] + self.log_vars[i]
                total_loss += weighted_loss
                weight_dict[name] = precision.item()
        
        return total_loss, weight_dict

class GlacierPhysicsLoss(PhysicsLoss):
    """
    冰川物理损失
    
    专门针对冰川物理过程的损失函数
    """
    
    def __init__(self, physics_type: str, weight: float = 1.0, 
                 reduction: str = 'mean', **kwargs):
        """
        初始化冰川物理损失
        
        Args:
            physics_type: 物理类型 ('mass_balance', 'momentum', 'glen_flow')
            weight: 损失权重
            reduction: 损失聚合方式
        """
        super(GlacierPhysicsLoss, self).__init__(physics_type, weight, reduction)
        self.physics_params = kwargs
        
    def mass_balance_loss(self, thickness: torch.Tensor, velocity: torch.Tensor,
                         coordinates: torch.Tensor, time: torch.Tensor,
                         accumulation: torch.Tensor, ablation: torch.Tensor) -> torch.Tensor:
        """
        质量平衡损失
        
        Args:
            thickness: 冰川厚度
            velocity: 速度场
            coordinates: 空间坐标
            time: 时间坐标
            accumulation: 积累率
            ablation: 消融率
            
        Returns:
            Tensor: 质量平衡损失
        """
        from .automatic_differentiation import GlacierDifferentialOperators
        
        # 计算质量平衡残差
        residual = GlacierDifferentialOperators.mass_balance_operator(
            thickness, velocity, coordinates, accumulation, ablation, time
        )
        
        return self.forward(residual)
    
    def momentum_balance_loss(self, velocity: torch.Tensor, pressure: torch.Tensor,
                            coordinates: torch.Tensor, **physics_params) -> torch.Tensor:
        """
        动量平衡损失
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coordinates: 空间坐标
            **physics_params: 物理参数
            
        Returns:
            Tensor: 动量平衡损失
        """
        # TODO: 实现动量平衡损失
        pass
    
    def glen_flow_loss(self, strain_rate: torch.Tensor, stress: torch.Tensor,
                      temperature: torch.Tensor, **glen_params) -> torch.Tensor:
        """
        Glen流动律损失
        
        Args:
            strain_rate: 应变率张量
            stress: 应力张量
            temperature: 温度
            **glen_params: Glen流动律参数
            
        Returns:
            Tensor: Glen流动律损失
        """
        # TODO: 实现Glen流动律损失
        pass

class CompositeLoss(nn.Module):
    """
    复合损失函数
    
    组合多个损失函数
    """
    
    def __init__(self, loss_functions: Dict[str, BaseLoss], 
                 loss_weights: Optional[Dict[str, float]] = None,
                 adaptive_weights: bool = False):
        """
        初始化复合损失
        
        Args:
            loss_functions: 损失函数字典
            loss_weights: 损失权重字典
            adaptive_weights: 是否使用自适应权重
        """
        super(CompositeLoss, self).__init__()
        
        self.loss_functions = nn.ModuleDict(loss_functions)
        
        if loss_weights is None:
            loss_weights = {name: 1.0 for name in loss_functions.keys()}
        self.loss_weights = loss_weights
        
        if adaptive_weights:
            self.adaptive_loss = AdaptiveWeightLoss(list(loss_functions.keys()))
        else:
            self.adaptive_loss = None
    
    def forward(self, **loss_inputs) -> Dict[str, torch.Tensor]:
        """
        计算复合损失
        
        Args:
            **loss_inputs: 各损失函数的输入
            
        Returns:
            Dict: 损失结果字典
        """
        losses = {}
        
        # 计算各项损失
        for name, loss_fn in self.loss_functions.items():
            if name in loss_inputs:
                losses[name] = loss_fn(**loss_inputs[name])
            else:
                losses[name] = torch.tensor(0.0, device=next(loss_fn.parameters()).device)
        
        # 计算总损失
        if self.adaptive_loss is not None:
            total_loss, adaptive_weights = self.adaptive_loss(losses)
            losses['adaptive_weights'] = adaptive_weights
        else:
            total_loss = sum(self.loss_weights.get(name, 1.0) * loss 
                           for name, loss in losses.items())
        
        losses['total'] = total_loss
        
        return losses

if __name__ == "__main__":
    # 测试损失函数
    predictions = torch.randn(100, 3)
    targets = torch.randn(100, 3)
    
    # 数据损失
    data_loss = DataLoss(loss_type='mse')
    loss_value = data_loss(predictions, targets)
    print(f"数据损失: {loss_value.item():.4f}")
    
    # 物理损失
    residuals = torch.randn(100, 1)
    physics_loss = PhysicsLoss('mass_balance')
    physics_loss_value = physics_loss(residuals)
    print(f"物理损失: {physics_loss_value.item():.4f}")