#!/usr/bin/env python3
"""
空间约束管理

实现空间相关的约束，包括：
- 边界条件约束
- 空间连续性约束
- 几何约束
- 拓扑约束

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

class SpatialConstraintType(Enum):
    """空间约束类型枚举"""
    BOUNDARY = "boundary"  # 边界约束
    CONTINUITY = "continuity"  # 连续性约束
    SYMMETRY = "symmetry"  # 对称性约束
    GEOMETRY = "geometry"  # 几何约束
    TOPOLOGY = "topology"  # 拓扑约束

@dataclass
class SpatialDomain:
    """空间域定义"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    dimension: int = 2

@dataclass
class BoundaryCondition:
    """边界条件定义"""
    boundary_type: str  # 'dirichlet', 'neumann', 'robin'
    boundary_location: str  # 'left', 'right', 'top', 'bottom', 'front', 'back'
    value: Union[float, Callable]  # 边界值或函数
    feature_idx: int  # 特征索引

class SpatialConstraintBase(ABC):
    """
    空间约束基类
    
    定义空间约束的抽象接口
    """
    
    def __init__(self, constraint_type: SpatialConstraintType, 
                 spatial_domain: SpatialDomain, weight: float = 1.0):
        """
        初始化空间约束
        
        Args:
            constraint_type: 约束类型
            spatial_domain: 空间域
            weight: 约束权重
        """
        self.constraint_type = constraint_type
        self.spatial_domain = spatial_domain
        self.weight = weight
        self.is_active = True
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算空间约束损失
        
        Args:
            predictions: 空间预测 [batch_size, spatial_points, features]
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            
        Returns:
            Tensor: 约束损失
        """
        pass
    
    @abstractmethod
    def validate_constraint(self, predictions: torch.Tensor, 
                          spatial_coords: torch.Tensor) -> bool:
        """
        验证空间约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            bool: 约束是否满足
        """
        pass

class BoundaryConstraint(SpatialConstraintBase):
    """
    边界约束
    
    实现各种边界条件
    """
    
    def __init__(self, spatial_domain: SpatialDomain, 
                 boundary_conditions: List[BoundaryCondition], 
                 tolerance: float = 0.01, weight: float = 1.0):
        """
        初始化边界约束
        
        Args:
            spatial_domain: 空间域
            boundary_conditions: 边界条件列表
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(SpatialConstraintType.BOUNDARY, spatial_domain, weight)
        self.boundary_conditions = boundary_conditions
        self.tolerance = tolerance
    
    def _identify_boundary_points(self, spatial_coords: torch.Tensor, 
                                boundary_location: str) -> torch.Tensor:
        """
        识别边界点
        
        Args:
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            boundary_location: 边界位置
            
        Returns:
            Tensor: 边界点掩码
        """
        x, y = spatial_coords[:, 0], spatial_coords[:, 1]
        
        if boundary_location == 'left':
            mask = torch.abs(x - self.spatial_domain.x_min) < self.tolerance
        elif boundary_location == 'right':
            mask = torch.abs(x - self.spatial_domain.x_max) < self.tolerance
        elif boundary_location == 'bottom':
            mask = torch.abs(y - self.spatial_domain.y_min) < self.tolerance
        elif boundary_location == 'top':
            mask = torch.abs(y - self.spatial_domain.y_max) < self.tolerance
        elif boundary_location == 'front' and spatial_coords.shape[1] > 2:
            z = spatial_coords[:, 2]
            mask = torch.abs(z - self.spatial_domain.z_min) < self.tolerance
        elif boundary_location == 'back' and spatial_coords.shape[1] > 2:
            z = spatial_coords[:, 2]
            mask = torch.abs(z - self.spatial_domain.z_max) < self.tolerance
        else:
            mask = torch.zeros(spatial_coords.shape[0], dtype=torch.bool, device=spatial_coords.device)
        
        return mask
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算边界约束损失
        
        Args:
            predictions: 空间预测 [batch_size, spatial_points, features]
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            
        Returns:
            Tensor: 边界约束损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        for bc in self.boundary_conditions:
            # 识别边界点
            boundary_mask = self._identify_boundary_points(spatial_coords, bc.boundary_location)
            
            if not torch.any(boundary_mask):
                continue
            
            # 提取边界点的预测值
            boundary_predictions = predictions[:, boundary_mask, bc.feature_idx]  # [batch_size, boundary_points]
            boundary_coords = spatial_coords[boundary_mask]  # [boundary_points, spatial_dim]
            
            if bc.boundary_type == 'dirichlet':
                # Dirichlet边界条件：u = g
                if callable(bc.value):
                    target_values = bc.value(boundary_coords)  # [boundary_points]
                    if isinstance(target_values, np.ndarray):
                        target_values = torch.from_numpy(target_values).to(device)
                    target_values = target_values.unsqueeze(0).expand(boundary_predictions.shape[0], -1)
                else:
                    target_values = torch.full_like(boundary_predictions, bc.value)
                
                boundary_loss = torch.mean((boundary_predictions - target_values) ** 2)
                total_loss += boundary_loss
            
            elif bc.boundary_type == 'neumann':
                # Neumann边界条件：∂u/∂n = g
                # 需要计算法向导数
                if boundary_predictions.shape[1] > 1:
                    # 简化实现：使用有限差分近似法向导数
                    if bc.boundary_location in ['left', 'right']:
                        # x方向导数
                        if boundary_predictions.shape[1] > 1:
                            normal_derivative = boundary_predictions[:, 1:] - boundary_predictions[:, :-1]
                        else:
                            normal_derivative = torch.zeros_like(boundary_predictions[:, :1])
                    elif bc.boundary_location in ['bottom', 'top']:
                        # y方向导数
                        if boundary_predictions.shape[1] > 1:
                            normal_derivative = boundary_predictions[:, 1:] - boundary_predictions[:, :-1]
                        else:
                            normal_derivative = torch.zeros_like(boundary_predictions[:, :1])
                    else:
                        normal_derivative = torch.zeros_like(boundary_predictions[:, :1])
                    
                    if callable(bc.value):
                        target_derivative = bc.value(boundary_coords[:-1] if normal_derivative.shape[1] > 0 else boundary_coords[:1])
                        if isinstance(target_derivative, np.ndarray):
                            target_derivative = torch.from_numpy(target_derivative).to(device)
                        target_derivative = target_derivative.unsqueeze(0).expand(normal_derivative.shape[0], -1)
                    else:
                        target_derivative = torch.full_like(normal_derivative, bc.value)
                    
                    if normal_derivative.shape[1] > 0:
                        boundary_loss = torch.mean((normal_derivative - target_derivative) ** 2)
                        total_loss += boundary_loss
            
            elif bc.boundary_type == 'robin':
                # Robin边界条件：αu + β∂u/∂n = g
                # 简化实现
                alpha, beta, g = 1.0, 1.0, bc.value  # 可以扩展为更复杂的参数
                
                if callable(g):
                    target_values = g(boundary_coords)
                    if isinstance(target_values, np.ndarray):
                        target_values = torch.from_numpy(target_values).to(device)
                    target_values = target_values.unsqueeze(0).expand(boundary_predictions.shape[0], -1)
                else:
                    target_values = torch.full_like(boundary_predictions, g)
                
                # 简化的Robin条件
                robin_values = alpha * boundary_predictions
                boundary_loss = torch.mean((robin_values - target_values) ** 2)
                total_loss += boundary_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          spatial_coords: torch.Tensor) -> bool:
        """
        验证边界约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            bool: 约束是否满足
        """
        for bc in self.boundary_conditions:
            boundary_mask = self._identify_boundary_points(spatial_coords, bc.boundary_location)
            
            if not torch.any(boundary_mask):
                continue
            
            boundary_predictions = predictions[:, boundary_mask, bc.feature_idx]
            boundary_coords = spatial_coords[boundary_mask]
            
            if bc.boundary_type == 'dirichlet':
                if callable(bc.value):
                    target_values = bc.value(boundary_coords)
                    if isinstance(target_values, np.ndarray):
                        target_values = torch.from_numpy(target_values).to(predictions.device)
                    target_values = target_values.unsqueeze(0).expand(boundary_predictions.shape[0], -1)
                else:
                    target_values = torch.full_like(boundary_predictions, bc.value)
                
                max_error = torch.max(torch.abs(boundary_predictions - target_values))
                if max_error > self.tolerance:
                    return False
        
        return True

class SpatialContinuityConstraint(SpatialConstraintBase):
    """
    空间连续性约束
    
    确保空间预测的连续性和平滑性
    """
    
    def __init__(self, spatial_domain: SpatialDomain, 
                 smoothness_order: int = 1, tolerance: float = 0.1, weight: float = 1.0):
        """
        初始化空间连续性约束
        
        Args:
            spatial_domain: 空间域
            smoothness_order: 平滑性阶数
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(SpatialConstraintType.CONTINUITY, spatial_domain, weight)
        self.smoothness_order = smoothness_order
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算空间连续性约束损失
        
        Args:
            predictions: 空间预测 [batch_size, spatial_points, features]
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            
        Returns:
            Tensor: 连续性损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        batch_size, spatial_points, features = predictions.shape
        
        # 计算空间梯度的平滑性
        for feature_idx in range(features):
            feature_values = predictions[:, :, feature_idx]  # [batch_size, spatial_points]
            
            # 计算相邻点之间的差异
            if spatial_points > 1:
                # 简化实现：计算相邻点的差异
                spatial_diff = feature_values[:, 1:] - feature_values[:, :-1]  # [batch_size, spatial_points-1]
                
                if self.smoothness_order == 1:
                    # 一阶平滑性：梯度变化
                    if spatial_diff.shape[1] > 1:
                        gradient_diff = spatial_diff[:, 1:] - spatial_diff[:, :-1]
                        smoothness_loss = torch.mean(gradient_diff ** 2)
                        total_loss += smoothness_loss
                
                elif self.smoothness_order == 2:
                    # 二阶平滑性：曲率变化
                    if spatial_diff.shape[1] > 1:
                        gradient_diff = spatial_diff[:, 1:] - spatial_diff[:, :-1]
                        if gradient_diff.shape[1] > 1:
                            curvature_diff = gradient_diff[:, 1:] - gradient_diff[:, :-1]
                            smoothness_loss = torch.mean(curvature_diff ** 2)
                            total_loss += smoothness_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          spatial_coords: torch.Tensor) -> bool:
        """
        验证空间连续性约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            bool: 约束是否满足
        """
        batch_size, spatial_points, features = predictions.shape
        
        for feature_idx in range(features):
            feature_values = predictions[:, :, feature_idx]
            
            if spatial_points > 1:
                spatial_diff = feature_values[:, 1:] - feature_values[:, :-1]
                
                if spatial_diff.shape[1] > 1:
                    gradient_diff = spatial_diff[:, 1:] - spatial_diff[:, :-1]
                    max_gradient_change = torch.max(torch.abs(gradient_diff))
                    
                    if max_gradient_change > self.tolerance:
                        return False
        
        return True

class SymmetryConstraint(SpatialConstraintBase):
    """
    对称性约束
    
    确保空间预测的对称性
    """
    
    def __init__(self, spatial_domain: SpatialDomain, 
                 symmetry_axes: List[str], tolerance: float = 0.01, weight: float = 1.0):
        """
        初始化对称性约束
        
        Args:
            spatial_domain: 空间域
            symmetry_axes: 对称轴列表 ['x', 'y', 'z']
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(SpatialConstraintType.SYMMETRY, spatial_domain, weight)
        self.symmetry_axes = symmetry_axes
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算对称性约束损失
        
        Args:
            predictions: 空间预测 [batch_size, spatial_points, features]
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            
        Returns:
            Tensor: 对称性损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        for axis in self.symmetry_axes:
            if axis == 'x':
                # x轴对称
                x_center = (self.spatial_domain.x_min + self.spatial_domain.x_max) / 2
                x_coords = spatial_coords[:, 0]
                
                # 找到对称点对
                for i, x in enumerate(x_coords):
                    symmetric_x = 2 * x_center - x
                    
                    # 找到最接近的对称点
                    distances = torch.abs(x_coords - symmetric_x)
                    closest_idx = torch.argmin(distances)
                    
                    if distances[closest_idx] < self.tolerance:
                        # 计算对称性损失
                        symmetry_diff = predictions[:, i, :] - predictions[:, closest_idx, :]
                        symmetry_loss = torch.mean(symmetry_diff ** 2)
                        total_loss += symmetry_loss
            
            elif axis == 'y':
                # y轴对称
                y_center = (self.spatial_domain.y_min + self.spatial_domain.y_max) / 2
                y_coords = spatial_coords[:, 1]
                
                for i, y in enumerate(y_coords):
                    symmetric_y = 2 * y_center - y
                    
                    distances = torch.abs(y_coords - symmetric_y)
                    closest_idx = torch.argmin(distances)
                    
                    if distances[closest_idx] < self.tolerance:
                        symmetry_diff = predictions[:, i, :] - predictions[:, closest_idx, :]
                        symmetry_loss = torch.mean(symmetry_diff ** 2)
                        total_loss += symmetry_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          spatial_coords: torch.Tensor) -> bool:
        """
        验证对称性约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            bool: 约束是否满足
        """
        for axis in self.symmetry_axes:
            if axis == 'x':
                x_center = (self.spatial_domain.x_min + self.spatial_domain.x_max) / 2
                x_coords = spatial_coords[:, 0]
                
                for i, x in enumerate(x_coords):
                    symmetric_x = 2 * x_center - x
                    distances = torch.abs(x_coords - symmetric_x)
                    closest_idx = torch.argmin(distances)
                    
                    if distances[closest_idx] < self.tolerance:
                        symmetry_diff = predictions[:, i, :] - predictions[:, closest_idx, :]
                        max_diff = torch.max(torch.abs(symmetry_diff))
                        
                        if max_diff > self.tolerance:
                            return False
        
        return True

class GeometryConstraint(SpatialConstraintBase):
    """
    几何约束
    
    确保预测满足几何形状约束
    """
    
    def __init__(self, spatial_domain: SpatialDomain, 
                 geometry_functions: List[Callable], tolerance: float = 0.01, weight: float = 1.0):
        """
        初始化几何约束
        
        Args:
            spatial_domain: 空间域
            geometry_functions: 几何函数列表
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(SpatialConstraintType.GEOMETRY, spatial_domain, weight)
        self.geometry_functions = geometry_functions
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: torch.Tensor, 
                              spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        计算几何约束损失
        
        Args:
            predictions: 空间预测 [batch_size, spatial_points, features]
            spatial_coords: 空间坐标 [spatial_points, spatial_dim]
            
        Returns:
            Tensor: 几何约束损失
        """
        device = predictions.device
        total_loss = torch.tensor(0.0, device=device)
        
        for geometry_func in self.geometry_functions:
            try:
                # 应用几何函数
                geometry_values = geometry_func(spatial_coords, predictions)
                
                if isinstance(geometry_values, torch.Tensor):
                    # 几何约束损失：期望几何函数值为0
                    geometry_loss = torch.mean(geometry_values ** 2)
                    total_loss += geometry_loss
                    
            except Exception as e:
                self.logger.warning(f"几何函数计算失败: {e}")
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: torch.Tensor, 
                          spatial_coords: torch.Tensor) -> bool:
        """
        验证几何约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            bool: 约束是否满足
        """
        for geometry_func in self.geometry_functions:
            try:
                geometry_values = geometry_func(spatial_coords, predictions)
                
                if isinstance(geometry_values, torch.Tensor):
                    max_violation = torch.max(torch.abs(geometry_values))
                    
                    if max_violation > self.tolerance:
                        return False
                        
            except Exception as e:
                self.logger.warning(f"几何函数验证失败: {e}")
                return False
        
        return True

class SpatialConstraintManager:
    """
    空间约束管理器
    
    管理和协调多个空间约束
    """
    
    def __init__(self, global_spatial_domain: SpatialDomain):
        """
        初始化空间约束管理器
        
        Args:
            global_spatial_domain: 全局空间域
        """
        self.global_spatial_domain = global_spatial_domain
        self.constraints: List[SpatialConstraintBase] = []
        self.constraint_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def add_constraint(self, constraint: SpatialConstraintBase, 
                     constraint_id: str = None) -> str:
        """
        添加空间约束
        
        Args:
            constraint: 约束对象
            constraint_id: 约束ID
            
        Returns:
            str: 约束ID
        """
        if constraint_id is None:
            constraint_id = f"spatial_constraint_{len(self.constraints)}"
        
        self.constraints.append(constraint)
        self.constraint_weights[constraint_id] = constraint.weight
        
        self.logger.info(f"添加空间约束: {constraint_id}, 类型: {constraint.constraint_type}")
        
        return constraint_id
    
    def compute_total_spatial_loss(self, predictions: torch.Tensor, 
                                 spatial_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总空间约束损失
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            Dict: 空间约束损失字典
        """
        constraint_losses = {}
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    loss = constraint.compute_constraint_loss(predictions, spatial_coords)
                    constraint_losses[constraint_id] = loss
                    total_loss += loss
                    
                except Exception as e:
                    self.logger.warning(f"空间约束 {constraint_id} 计算失败: {e}")
                    constraint_losses[constraint_id] = torch.tensor(0.0)
        
        constraint_losses['total_spatial_loss'] = total_loss
        
        return constraint_losses
    
    def validate_all_spatial_constraints(self, predictions: torch.Tensor, 
                                       spatial_coords: torch.Tensor) -> Dict[str, bool]:
        """
        验证所有空间约束
        
        Args:
            predictions: 空间预测
            spatial_coords: 空间坐标
            
        Returns:
            Dict: 约束验证结果字典
        """
        validation_results = {}
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    is_valid = constraint.validate_constraint(predictions, spatial_coords)
                    validation_results[constraint_id] = is_valid
                    
                except Exception as e:
                    self.logger.warning(f"空间约束 {constraint_id} 验证失败: {e}")
                    validation_results[constraint_id] = False
        
        return validation_results

def create_spatial_constraint_manager(
    spatial_domain: SpatialDomain,
    constraint_config: Dict[str, Any] = None
) -> SpatialConstraintManager:
    """
    创建空间约束管理器
    
    Args:
        spatial_domain: 空间域
        constraint_config: 约束配置
        
    Returns:
        SpatialConstraintManager: 空间约束管理器实例
    """
    manager = SpatialConstraintManager(spatial_domain)
    
    if constraint_config:
        # 添加边界约束
        if 'boundary' in constraint_config:
            boundary_config = constraint_config['boundary']
            boundary_conditions = []
            
            for bc_config in boundary_config.get('conditions', []):
                bc = BoundaryCondition(
                    boundary_type=bc_config['type'],
                    boundary_location=bc_config['location'],
                    value=bc_config['value'],
                    feature_idx=bc_config['feature_idx']
                )
                boundary_conditions.append(bc)
            
            if boundary_conditions:
                boundary_constraint = BoundaryConstraint(
                    spatial_domain=spatial_domain,
                    boundary_conditions=boundary_conditions,
                    tolerance=boundary_config.get('tolerance', 0.01),
                    weight=boundary_config.get('weight', 1.0)
                )
                manager.add_constraint(boundary_constraint, 'boundary')
        
        # 添加连续性约束
        if 'continuity' in constraint_config:
            continuity_config = constraint_config['continuity']
            continuity_constraint = SpatialContinuityConstraint(
                spatial_domain=spatial_domain,
                smoothness_order=continuity_config.get('order', 1),
                tolerance=continuity_config.get('tolerance', 0.1),
                weight=continuity_config.get('weight', 1.0)
            )
            manager.add_constraint(continuity_constraint, 'spatial_continuity')
        
        # 添加对称性约束
        if 'symmetry' in constraint_config:
            symmetry_config = constraint_config['symmetry']
            symmetry_constraint = SymmetryConstraint(
                spatial_domain=spatial_domain,
                symmetry_axes=symmetry_config.get('axes', []),
                tolerance=symmetry_config.get('tolerance', 0.01),
                weight=symmetry_config.get('weight', 1.0)
            )
            manager.add_constraint(symmetry_constraint, 'symmetry')
        
        # 添加几何约束
        if 'geometry' in constraint_config:
            geometry_config = constraint_config['geometry']
            geometry_constraint = GeometryConstraint(
                spatial_domain=spatial_domain,
                geometry_functions=geometry_config.get('functions', []),
                tolerance=geometry_config.get('tolerance', 0.01),
                weight=geometry_config.get('weight', 1.0)
            )
            manager.add_constraint(geometry_constraint, 'geometry')
    
    return manager

if __name__ == "__main__":
    # 测试空间约束管理
    
    # 创建空间域
    spatial_domain = SpatialDomain(
        x_min=0.0, x_max=10.0,
        y_min=0.0, y_max=10.0,
        dimension=2
    )
    
    # 创建边界条件
    def boundary_func(coords):
        return torch.zeros(coords.shape[0])
    
    # 创建约束配置
    constraint_config = {
        'boundary': {
            'conditions': [
                {
                    'type': 'dirichlet',
                    'location': 'left',
                    'value': 0.0,
                    'feature_idx': 0
                },
                {
                    'type': 'dirichlet',
                    'location': 'right',
                    'value': 1.0,
                    'feature_idx': 0
                }
            ],
            'tolerance': 0.01,
            'weight': 1.0
        },
        'continuity': {
            'order': 1,
            'tolerance': 0.1,
            'weight': 0.5
        },
        'symmetry': {
            'axes': ['x'],
            'tolerance': 0.01,
            'weight': 0.3
        }
    }
    
    # 创建空间约束管理器
    manager = create_spatial_constraint_manager(spatial_domain, constraint_config)
    
    # 创建测试数据
    batch_size = 5
    spatial_points = 100
    features = 2
    
    # 生成空间坐标
    x = torch.linspace(0, 10, int(np.sqrt(spatial_points)))
    y = torch.linspace(0, 10, int(np.sqrt(spatial_points)))
    X, Y = torch.meshgrid(x, y, indexing='ij')
    spatial_coords = torch.stack([X.flatten(), Y.flatten()], dim=1)[:spatial_points]
    
    # 生成测试预测数据
    predictions = torch.randn(batch_size, spatial_points, features)
    
    # 添加一些模式使其更符合约束
    # 在左边界设置为0，右边界设置为1
    left_mask = torch.abs(spatial_coords[:, 0] - 0.0) < 0.1
    right_mask = torch.abs(spatial_coords[:, 0] - 10.0) < 0.1
    
    predictions[:, left_mask, 0] = 0.0
    predictions[:, right_mask, 0] = 1.0
    
    # 计算空间约束损失
    spatial_losses = manager.compute_total_spatial_loss(predictions, spatial_coords)
    print("=== 空间约束损失 ===")
    for constraint_id, loss in spatial_losses.items():
        print(f"{constraint_id}: {loss.item():.6f}")
    
    # 验证空间约束
    validation_results = manager.validate_all_spatial_constraints(predictions, spatial_coords)
    print("\n=== 空间约束验证结果 ===")
    for constraint_id, is_valid in validation_results.items():
        print(f"{constraint_id}: {'通过' if is_valid else '失败'}")
    
    print("\n空间约束管理测试完成！")