#!/usr/bin/env python3
"""
时间约束管理

实现时间相关的约束，包括：
- 时间连续性约束
- 因果性约束
- 时间序列一致性
- 季节性约束

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
import datetime

class TemporalConstraintType(Enum):
    """时间约束类型枚举"""
    CONTINUITY = "continuity"  # 连续性约束
    CAUSALITY = "causality"  # 因果性约束
    MONOTONICITY = "monotonicity"  # 单调性约束
    PERIODICITY = "periodicity"  # 周期性约束
    SMOOTHNESS = "smoothness"  # 平滑性约束

@dataclass
class TemporalWindow:
    """时间窗口定义"""
    start_time: float
    end_time: float
    time_step: float
    window_type: str = "sliding"  # sliding, fixed, adaptive

class TemporalConstraintBase(ABC):
    """
    时间约束基类
    
    定义时间约束的抽象接口
    """
    
    def __init__(self, constraint_type: TemporalConstraintType, 
                 temporal_window: TemporalWindow, weight: float = 1.0):
        """
        初始化时间约束
        
        Args:
            constraint_type: 约束类型
            temporal_window: 时间窗口
            weight: 约束权重
        """
        self.constraint_type = constraint_type
        self.temporal_window = temporal_window
        self.weight = weight
        self.is_active = True
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              time_coords: torch.Tensor) -> torch.Tensor:
        """
        计算时间约束损失
        
        Args:
            predictions: 时间序列预测 [batch_size, time_steps, features]
            time_coords: 时间坐标 [time_steps]
            
        Returns:
            Tensor: 约束损失
        """
        pass
    
    @abstractmethod
    def validate_constraint(self, predictions: torch.Tensor, 
                          time_coords: torch.Tensor) -> bool:
        """
        验证时间约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            bool: 约束是否满足
        """
        pass

class TemporalContinuityConstraint(TemporalConstraintBase):
    """
    时间连续性约束
    
    确保时间序列的连续性和平滑性
    """
    
    def __init__(self, temporal_window: TemporalWindow, 
                 continuity_order: int = 1, tolerance: float = 0.1, weight: float = 1.0):
        """
        初始化时间连续性约束
        
        Args:
            temporal_window: 时间窗口
            continuity_order: 连续性阶数 (1=一阶导数, 2=二阶导数)
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(TemporalConstraintType.CONTINUITY, temporal_window, weight)
        self.continuity_order = continuity_order
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              time_coords: torch.Tensor) -> torch.Tensor:
        """
        计算连续性约束损失
        
        Args:
            predictions: 时间序列预测 [batch_size, time_steps, features]
            time_coords: 时间坐标 [time_steps]
            
        Returns:
            Tensor: 连续性损失
        """
        device = predictions.device
        batch_size, time_steps, features = predictions.shape
        
        if time_steps < self.continuity_order + 1:
            return torch.tensor(0.0, device=device)
        
        # 计算时间步长
        dt = time_coords[1:] - time_coords[:-1]
        
        total_loss = torch.tensor(0.0, device=device)
        
        if self.continuity_order == 1:
            # 一阶导数连续性（速度连续性）
            # 计算一阶差分
            first_diff = (predictions[:, 1:, :] - predictions[:, :-1, :]) / dt.unsqueeze(0).unsqueeze(-1)
            
            # 计算一阶差分的变化率
            if first_diff.shape[1] > 1:
                second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
                continuity_loss = torch.mean(second_diff ** 2)
                total_loss += continuity_loss
        
        elif self.continuity_order == 2:
            # 二阶导数连续性（加速度连续性）
            # 计算二阶差分
            first_diff = (predictions[:, 1:, :] - predictions[:, :-1, :]) / dt.unsqueeze(0).unsqueeze(-1)
            
            if first_diff.shape[1] > 1:
                dt_second = dt[1:]
                second_diff = (first_diff[:, 1:, :] - first_diff[:, :-1, :]) / dt_second.unsqueeze(0).unsqueeze(-1)
                
                # 计算二阶差分的变化率
                if second_diff.shape[1] > 1:
                    third_diff = second_diff[:, 1:, :] - second_diff[:, :-1, :]
                    continuity_loss = torch.mean(third_diff ** 2)
                    total_loss += continuity_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          time_coords: torch.Tensor) -> bool:
        """
        验证连续性约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            bool: 约束是否满足
        """
        if predictions.shape[1] < self.continuity_order + 1:
            return True
        
        # 计算时间步长
        dt = time_coords[1:] - time_coords[:-1]
        
        # 计算一阶差分
        first_diff = (predictions[:, 1:, :] - predictions[:, :-1, :]) / dt.unsqueeze(0).unsqueeze(-1)
        
        # 检查一阶差分的变化是否在容忍范围内
        if first_diff.shape[1] > 1:
            second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
            max_change = torch.max(torch.abs(second_diff))
            
            if max_change > self.tolerance:
                return False
        
        return True

class CausalityConstraint(TemporalConstraintBase):
    """
    因果性约束
    
    确保因果关系的时间顺序
    """
    
    def __init__(self, temporal_window: TemporalWindow, 
                 causal_relationships: List[Tuple[int, int, float]], weight: float = 1.0):
        """
        初始化因果性约束
        
        Args:
            temporal_window: 时间窗口
            causal_relationships: 因果关系列表 [(cause_feature, effect_feature, delay)]
            weight: 约束权重
        """
        super().__init__(TemporalConstraintType.CAUSALITY, temporal_window, weight)
        self.causal_relationships = causal_relationships
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              time_coords: torch.Tensor) -> torch.Tensor:
        """
        计算因果性约束损失
        
        Args:
            predictions: 时间序列预测 [batch_size, time_steps, features]
            time_coords: 时间坐标 [time_steps]
            
        Returns:
            Tensor: 因果性损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        dt = torch.mean(time_coords[1:] - time_coords[:-1])
        
        for cause_idx, effect_idx, delay in self.causal_relationships:
            if cause_idx >= predictions.shape[2] or effect_idx >= predictions.shape[2]:
                continue
            
            # 计算延迟步数
            delay_steps = int(delay / dt)
            
            if delay_steps >= predictions.shape[1]:
                continue
            
            # 提取因果变量
            cause = predictions[:, :-delay_steps, cause_idx] if delay_steps > 0 else predictions[:, :, cause_idx]
            effect = predictions[:, delay_steps:, effect_idx] if delay_steps > 0 else predictions[:, :, effect_idx]
            
            # 计算因果关系强度（使用相关性）
            if cause.shape[1] == effect.shape[1] and cause.shape[1] > 1:
                # 标准化
                cause_norm = (cause - torch.mean(cause, dim=1, keepdim=True)) / (torch.std(cause, dim=1, keepdim=True) + 1e-8)
                effect_norm = (effect - torch.mean(effect, dim=1, keepdim=True)) / (torch.std(effect, dim=1, keepdim=True) + 1e-8)
                
                # 计算相关性
                correlation = torch.mean(cause_norm * effect_norm, dim=1)
                
                # 因果性损失：期望正相关
                causality_loss = torch.mean(torch.relu(-correlation))  # 惩罚负相关
                total_loss += causality_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          time_coords: torch.Tensor) -> bool:
        """
        验证因果性约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            bool: 约束是否满足
        """
        dt = torch.mean(time_coords[1:] - time_coords[:-1])
        
        for cause_idx, effect_idx, delay in self.causal_relationships:
            if cause_idx >= predictions.shape[2] or effect_idx >= predictions.shape[2]:
                continue
            
            delay_steps = int(delay / dt)
            
            if delay_steps >= predictions.shape[1]:
                continue
            
            # 提取因果变量
            cause = predictions[:, :-delay_steps, cause_idx] if delay_steps > 0 else predictions[:, :, cause_idx]
            effect = predictions[:, delay_steps:, effect_idx] if delay_steps > 0 else predictions[:, :, effect_idx]
            
            if cause.shape[1] == effect.shape[1] and cause.shape[1] > 1:
                # 计算相关性
                cause_flat = cause.flatten()
                effect_flat = effect.flatten()
                
                if len(cause_flat) > 1 and len(effect_flat) > 1:
                    correlation = torch.corrcoef(torch.stack([cause_flat, effect_flat]))[0, 1]
                    
                    # 因果关系应该是正相关
                    if correlation < 0.1:  # 最小相关性阈值
                        return False
        
        return True

class MonotonicityConstraint(TemporalConstraintBase):
    """
    单调性约束
    
    确保某些变量在时间上的单调性
    """
    
    def __init__(self, temporal_window: TemporalWindow, 
                 monotonic_features: Dict[int, str], tolerance: float = 0.01, weight: float = 1.0):
        """
        初始化单调性约束
        
        Args:
            temporal_window: 时间窗口
            monotonic_features: 单调特征字典 {feature_idx: 'increasing'/'decreasing'}
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(TemporalConstraintType.MONOTONICITY, temporal_window, weight)
        self.monotonic_features = monotonic_features
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              time_coords: torch.Tensor) -> torch.Tensor:
        """
        计算单调性约束损失
        
        Args:
            predictions: 时间序列预测 [batch_size, time_steps, features]
            time_coords: 时间坐标 [time_steps]
            
        Returns:
            Tensor: 单调性损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        for feature_idx, monotonic_type in self.monotonic_features.items():
            if feature_idx >= predictions.shape[2]:
                continue
            
            feature_values = predictions[:, :, feature_idx]  # [batch_size, time_steps]
            
            # 计算时间差分
            time_diff = feature_values[:, 1:] - feature_values[:, :-1]
            
            if monotonic_type == 'increasing':
                # 惩罚递减的部分
                violation = torch.relu(-time_diff - self.tolerance)
                monotonic_loss = torch.mean(violation ** 2)
            
            elif monotonic_type == 'decreasing':
                # 惩罚递增的部分
                violation = torch.relu(time_diff - self.tolerance)
                monotonic_loss = torch.mean(violation ** 2)
            
            else:
                monotonic_loss = torch.tensor(0.0, device=device)
            
            total_loss += monotonic_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          time_coords: torch.Tensor) -> bool:
        """
        验证单调性约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            bool: 约束是否满足
        """
        for feature_idx, monotonic_type in self.monotonic_features.items():
            if feature_idx >= predictions.shape[2]:
                continue
            
            feature_values = predictions[:, :, feature_idx]
            time_diff = feature_values[:, 1:] - feature_values[:, :-1]
            
            if monotonic_type == 'increasing':
                # 检查是否有显著的递减
                if torch.any(time_diff < -self.tolerance):
                    return False
            
            elif monotonic_type == 'decreasing':
                # 检查是否有显著的递增
                if torch.any(time_diff > self.tolerance):
                    return False
        
        return True

class PeriodicityConstraint(TemporalConstraintBase):
    """
    周期性约束
    
    确保时间序列的周期性模式
    """
    
    def __init__(self, temporal_window: TemporalWindow, 
                 periodic_features: Dict[int, float], tolerance: float = 0.1, weight: float = 1.0):
        """
        初始化周期性约束
        
        Args:
            temporal_window: 时间窗口
            periodic_features: 周期性特征字典 {feature_idx: period}
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(TemporalConstraintType.PERIODICITY, temporal_window, weight)
        self.periodic_features = periodic_features
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              time_coords: torch.Tensor) -> torch.Tensor:
        """
        计算周期性约束损失
        
        Args:
            predictions: 时间序列预测 [batch_size, time_steps, features]
            time_coords: 时间坐标 [time_steps]
            
        Returns:
            Tensor: 周期性损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        dt = torch.mean(time_coords[1:] - time_coords[:-1])
        
        for feature_idx, period in self.periodic_features.items():
            if feature_idx >= predictions.shape[2]:
                continue
            
            # 计算周期对应的时间步数
            period_steps = int(period / dt)
            
            if period_steps >= predictions.shape[1]:
                continue
            
            feature_values = predictions[:, :, feature_idx]  # [batch_size, time_steps]
            
            # 计算周期性损失
            if feature_values.shape[1] > period_steps:
                # 比较相隔一个周期的值
                periodic_diff = feature_values[:, period_steps:] - feature_values[:, :-period_steps]
                periodic_loss = torch.mean(periodic_diff ** 2)
                total_loss += periodic_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          time_coords: torch.Tensor) -> bool:
        """
        验证周期性约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            bool: 约束是否满足
        """
        dt = torch.mean(time_coords[1:] - time_coords[:-1])
        
        for feature_idx, period in self.periodic_features.items():
            if feature_idx >= predictions.shape[2]:
                continue
            
            period_steps = int(period / dt)
            
            if period_steps >= predictions.shape[1]:
                continue
            
            feature_values = predictions[:, :, feature_idx]
            
            if feature_values.shape[1] > period_steps:
                # 检查周期性
                periodic_diff = feature_values[:, period_steps:] - feature_values[:, :-period_steps]
                max_diff = torch.max(torch.abs(periodic_diff))
                
                if max_diff > self.tolerance:
                    return False
        
        return True

class TemporalConstraintManager:
    """
    时间约束管理器
    
    管理和协调多个时间约束
    """
    
    def __init__(self, global_temporal_window: TemporalWindow):
        """
        初始化时间约束管理器
        
        Args:
            global_temporal_window: 全局时间窗口
        """
        self.global_temporal_window = global_temporal_window
        self.constraints: List[TemporalConstraintBase] = []
        self.constraint_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def add_constraint(self, constraint: TemporalConstraintBase, 
                     constraint_id: str = None) -> str:
        """
        添加时间约束
        
        Args:
            constraint: 约束对象
            constraint_id: 约束ID
            
        Returns:
            str: 约束ID
        """
        if constraint_id is None:
            constraint_id = f"temporal_constraint_{len(self.constraints)}"
        
        self.constraints.append(constraint)
        self.constraint_weights[constraint_id] = constraint.weight
        
        self.logger.info(f"添加时间约束: {constraint_id}, 类型: {constraint.constraint_type}")
        
        return constraint_id
    
    def compute_total_temporal_loss(self, predictions: torch.Tensor, 
                                  time_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总时间约束损失
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            Dict: 时间约束损失字典
        """
        constraint_losses = {}
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    loss = constraint.compute_constraint_loss(predictions, time_coords)
                    constraint_losses[constraint_id] = loss
                    total_loss += loss
                    
                except Exception as e:
                    self.logger.warning(f"时间约束 {constraint_id} 计算失败: {e}")
                    constraint_losses[constraint_id] = torch.tensor(0.0)
        
        constraint_losses['total_temporal_loss'] = total_loss
        
        return constraint_losses
    
    def validate_all_temporal_constraints(self, predictions: torch.Tensor, 
                                        time_coords: torch.Tensor) -> Dict[str, bool]:
        """
        验证所有时间约束
        
        Args:
            predictions: 时间序列预测
            time_coords: 时间坐标
            
        Returns:
            Dict: 约束验证结果字典
        """
        validation_results = {}
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    is_valid = constraint.validate_constraint(predictions, time_coords)
                    validation_results[constraint_id] = is_valid
                    
                except Exception as e:
                    self.logger.warning(f"时间约束 {constraint_id} 验证失败: {e}")
                    validation_results[constraint_id] = False
        
        return validation_results

def create_temporal_constraint_manager(
    temporal_window: TemporalWindow,
    constraint_config: Dict[str, Any] = None
) -> TemporalConstraintManager:
    """
    创建时间约束管理器
    
    Args:
        temporal_window: 时间窗口
        constraint_config: 约束配置
        
    Returns:
        TemporalConstraintManager: 时间约束管理器实例
    """
    manager = TemporalConstraintManager(temporal_window)
    
    if constraint_config:
        # 添加连续性约束
        if 'continuity' in constraint_config:
            continuity_config = constraint_config['continuity']
            continuity_constraint = TemporalContinuityConstraint(
                temporal_window=temporal_window,
                continuity_order=continuity_config.get('order', 1),
                tolerance=continuity_config.get('tolerance', 0.1),
                weight=continuity_config.get('weight', 1.0)
            )
            manager.add_constraint(continuity_constraint, 'temporal_continuity')
        
        # 添加因果性约束
        if 'causality' in constraint_config:
            causality_config = constraint_config['causality']
            causality_constraint = CausalityConstraint(
                temporal_window=temporal_window,
                causal_relationships=causality_config.get('relationships', []),
                weight=causality_config.get('weight', 1.0)
            )
            manager.add_constraint(causality_constraint, 'causality')
        
        # 添加单调性约束
        if 'monotonicity' in constraint_config:
            monotonicity_config = constraint_config['monotonicity']
            monotonicity_constraint = MonotonicityConstraint(
                temporal_window=temporal_window,
                monotonic_features=monotonicity_config.get('features', {}),
                tolerance=monotonicity_config.get('tolerance', 0.01),
                weight=monotonicity_config.get('weight', 1.0)
            )
            manager.add_constraint(monotonicity_constraint, 'monotonicity')
        
        # 添加周期性约束
        if 'periodicity' in constraint_config:
            periodicity_config = constraint_config['periodicity']
            periodicity_constraint = PeriodicityConstraint(
                temporal_window=temporal_window,
                periodic_features=periodicity_config.get('features', {}),
                tolerance=periodicity_config.get('tolerance', 0.1),
                weight=periodicity_config.get('weight', 1.0)
            )
            manager.add_constraint(periodicity_constraint, 'periodicity')
    
    return manager

if __name__ == "__main__":
    # 测试时间约束管理
    
    # 创建时间窗口
    temporal_window = TemporalWindow(
        start_time=0.0,
        end_time=10.0,
        time_step=0.1,
        window_type="sliding"
    )
    
    # 创建约束配置
    constraint_config = {
        'continuity': {
            'order': 1,
            'tolerance': 0.1,
            'weight': 1.0
        },
        'causality': {
            'relationships': [(0, 1, 0.5), (1, 2, 1.0)],  # (cause, effect, delay)
            'weight': 0.5
        },
        'monotonicity': {
            'features': {0: 'increasing', 2: 'decreasing'},
            'tolerance': 0.01,
            'weight': 0.8
        },
        'periodicity': {
            'features': {1: 2.0},  # feature 1 has period 2.0
            'tolerance': 0.1,
            'weight': 0.3
        }
    }
    
    # 创建时间约束管理器
    manager = create_temporal_constraint_manager(temporal_window, constraint_config)
    
    # 创建测试数据
    batch_size = 10
    time_steps = 100
    features = 3
    
    # 生成时间坐标
    time_coords = torch.linspace(0, 10, time_steps)
    
    # 生成测试预测数据
    predictions = torch.randn(batch_size, time_steps, features)
    
    # 添加一些模式使其更符合约束
    # 单调递增的特征0
    predictions[:, :, 0] = torch.cumsum(torch.abs(predictions[:, :, 0]), dim=1)
    
    # 周期性的特征1
    t_expanded = time_coords.unsqueeze(0).expand(batch_size, -1)
    predictions[:, :, 1] = torch.sin(2 * np.pi * t_expanded / 2.0) + 0.1 * predictions[:, :, 1]
    
    # 单调递减的特征2
    predictions[:, :, 2] = -torch.cumsum(torch.abs(predictions[:, :, 2]), dim=1)
    
    # 计算时间约束损失
    temporal_losses = manager.compute_total_temporal_loss(predictions, time_coords)
    print("=== 时间约束损失 ===")
    for constraint_id, loss in temporal_losses.items():
        print(f"{constraint_id}: {loss.item():.6f}")
    
    # 验证时间约束
    validation_results = manager.validate_all_temporal_constraints(predictions, time_coords)
    print("\n=== 时间约束验证结果 ===")
    for constraint_id, is_valid in validation_results.items():
        print(f"{constraint_id}: {'通过' if is_valid else '失败'}")
    
    print("\n时间约束管理测试完成！")