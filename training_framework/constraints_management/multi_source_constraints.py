#!/usr/bin/env python3
"""
多源约束管理

实现多源数据约束的集成和管理，包括：
- 多源数据一致性约束
- 数据源权重分配
- 约束冲突解决
- 动态约束调整

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

class DataSourceType(Enum):
    """数据源类型枚举"""
    RGI = "rgi"  # RGI冰川轮廓
    FARINOTTI = "farinotti"  # Farinotti厚度数据
    MILLAN = "millan"  # Millan速度数据
    HUGONNET = "hugonnet"  # Hugonnet高程变化
    DUSSAILLANT = "dussaillant"  # Dussaillant质量变化
    AUXILIARY = "auxiliary"  # 辅助数据

@dataclass
class DataSourceInfo:
    """数据源信息"""
    source_type: DataSourceType
    reliability: float  # 可靠性评分 [0, 1]
    temporal_coverage: Tuple[str, str]  # 时间覆盖范围
    spatial_resolution: float  # 空间分辨率 (米)
    uncertainty: float  # 不确定性水平
    weight: float = 1.0  # 权重

class ConstraintType(Enum):
    """约束类型枚举"""
    CONSISTENCY = "consistency"  # 一致性约束
    BOUNDARY = "boundary"  # 边界约束
    PHYSICAL = "physical"  # 物理约束
    TEMPORAL = "temporal"  # 时间约束
    SPATIAL = "spatial"  # 空间约束

class MultiSourceConstraintBase(ABC):
    """
    多源约束基类
    
    定义多源约束的抽象接口
    """
    
    def __init__(self, constraint_type: ConstraintType, weight: float = 1.0):
        """
        初始化多源约束
        
        Args:
            constraint_type: 约束类型
            weight: 约束权重
        """
        self.constraint_type = constraint_type
        self.weight = weight
        self.is_active = True
    
    @abstractmethod
    def compute_constraint_loss(self, predictions: Dict[str, torch.Tensor], 
                              data_sources: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算约束损失
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Tensor: 约束损失
        """
        pass
    
    @abstractmethod
    def validate_constraint(self, predictions: Dict[str, torch.Tensor], 
                          data_sources: Dict[str, torch.Tensor]) -> bool:
        """
        验证约束是否满足
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            bool: 约束是否满足
        """
        pass

class DataConsistencyConstraint(MultiSourceConstraintBase):
    """
    数据一致性约束
    
    确保不同数据源之间的一致性
    """
    
    def __init__(self, source_pairs: List[Tuple[str, str]], 
                 tolerance: float = 0.1, weight: float = 1.0):
        """
        初始化数据一致性约束
        
        Args:
            source_pairs: 需要保持一致性的数据源对
            tolerance: 容忍度
            weight: 约束权重
        """
        super().__init__(ConstraintType.CONSISTENCY, weight)
        self.source_pairs = source_pairs
        self.tolerance = tolerance
    
    def compute_constraint_loss(self, predictions: Dict[str, torch.Tensor], 
                              data_sources: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算一致性约束损失
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Tensor: 一致性损失
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for source1, source2 in self.source_pairs:
            if source1 in data_sources and source2 in data_sources:
                # 计算数据源之间的差异
                diff = torch.abs(data_sources[source1] - data_sources[source2])
                
                # 使用Huber损失来处理异常值
                consistency_loss = torch.where(
                    diff <= self.tolerance,
                    0.5 * diff ** 2,
                    self.tolerance * (diff - 0.5 * self.tolerance)
                )
                
                total_loss += torch.mean(consistency_loss)
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: Dict[str, torch.Tensor], 
                          data_sources: Dict[str, torch.Tensor]) -> bool:
        """
        验证一致性约束
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            bool: 约束是否满足
        """
        for source1, source2 in self.source_pairs:
            if source1 in data_sources and source2 in data_sources:
                diff = torch.abs(data_sources[source1] - data_sources[source2])
                if torch.max(diff) > self.tolerance:
                    return False
        return True

class CrossValidationConstraint(MultiSourceConstraintBase):
    """
    交叉验证约束
    
    使用一个数据源验证另一个数据源的预测
    """
    
    def __init__(self, validation_pairs: Dict[str, str], 
                 confidence_threshold: float = 0.8, weight: float = 1.0):
        """
        初始化交叉验证约束
        
        Args:
            validation_pairs: 验证对 {prediction_source: validation_source}
            confidence_threshold: 置信度阈值
            weight: 约束权重
        """
        super().__init__(ConstraintType.CONSISTENCY, weight)
        self.validation_pairs = validation_pairs
        self.confidence_threshold = confidence_threshold
    
    def compute_constraint_loss(self, predictions: Dict[str, torch.Tensor], 
                              data_sources: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算交叉验证约束损失
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Tensor: 交叉验证损失
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for pred_source, val_source in self.validation_pairs.items():
            if pred_source in predictions and val_source in data_sources:
                # 计算预测与验证数据的差异
                pred_data = predictions[pred_source]
                val_data = data_sources[val_source]
                
                # 使用加权MSE损失
                mse_loss = torch.mean((pred_data - val_data) ** 2)
                
                # 根据验证数据的置信度调整损失
                confidence_weight = self.confidence_threshold
                validation_loss = confidence_weight * mse_loss
                
                total_loss += validation_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: Dict[str, torch.Tensor], 
                          data_sources: Dict[str, torch.Tensor]) -> bool:
        """
        验证交叉验证约束
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            bool: 约束是否满足
        """
        for pred_source, val_source in self.validation_pairs.items():
            if pred_source in predictions and val_source in data_sources:
                pred_data = predictions[pred_source]
                val_data = data_sources[val_source]
                
                # 计算相关系数
                correlation = torch.corrcoef(torch.stack([
                    pred_data.flatten(), val_data.flatten()
                ]))[0, 1]
                
                if correlation < self.confidence_threshold:
                    return False
        return True

class UncertaintyWeightedConstraint(MultiSourceConstraintBase):
    """
    不确定性加权约束
    
    根据数据源的不确定性调整约束权重
    """
    
    def __init__(self, source_uncertainties: Dict[str, float], 
                 uncertainty_model: str = 'inverse', weight: float = 1.0):
        """
        初始化不确定性加权约束
        
        Args:
            source_uncertainties: 数据源不确定性字典
            uncertainty_model: 不确定性模型 ('inverse', 'exponential')
            weight: 约束权重
        """
        super().__init__(ConstraintType.CONSISTENCY, weight)
        self.source_uncertainties = source_uncertainties
        self.uncertainty_model = uncertainty_model
    
    def _compute_uncertainty_weights(self) -> Dict[str, float]:
        """
        计算基于不确定性的权重
        
        Returns:
            Dict: 权重字典
        """
        weights = {}
        
        for source, uncertainty in self.source_uncertainties.items():
            if self.uncertainty_model == 'inverse':
                # 反比权重：不确定性越高，权重越低
                weights[source] = 1.0 / (1.0 + uncertainty)
            elif self.uncertainty_model == 'exponential':
                # 指数权重：不确定性越高，权重指数衰减
                weights[source] = np.exp(-uncertainty)
            else:
                weights[source] = 1.0
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def compute_constraint_loss(self, predictions: Dict[str, torch.Tensor], 
                              data_sources: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算不确定性加权约束损失
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Tensor: 加权约束损失
        """
        uncertainty_weights = self._compute_uncertainty_weights()
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # 计算加权一致性损失
        sources = list(data_sources.keys())
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                if source1 in uncertainty_weights and source2 in uncertainty_weights:
                    # 计算两个数据源的差异
                    diff = torch.abs(data_sources[source1] - data_sources[source2])
                    
                    # 使用不确定性权重
                    combined_weight = (uncertainty_weights[source1] + uncertainty_weights[source2]) / 2
                    weighted_loss = combined_weight * torch.mean(diff ** 2)
                    
                    total_loss += weighted_loss
        
        return self.weight * total_loss
    
    def validate_constraint(self, predictions: Dict[str, torch.Tensor], 
                          data_sources: Dict[str, torch.Tensor]) -> bool:
        """
        验证不确定性加权约束
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            bool: 约束是否满足
        """
        uncertainty_weights = self._compute_uncertainty_weights()
        
        # 检查高权重数据源的一致性
        high_weight_sources = [
            source for source, weight in uncertainty_weights.items() 
            if weight > 0.5
        ]
        
        for i, source1 in enumerate(high_weight_sources):
            for source2 in high_weight_sources[i+1:]:
                if source1 in data_sources and source2 in data_sources:
                    diff = torch.abs(data_sources[source1] - data_sources[source2])
                    # 高权重数据源应该有更严格的一致性要求
                    if torch.max(diff) > 0.05:  # 5%的容忍度
                        return False
        
        return True

class MultiSourceConstraintManager:
    """
    多源约束管理器
    
    管理和协调多个数据源约束
    """
    
    def __init__(self, data_source_info: Dict[str, DataSourceInfo]):
        """
        初始化多源约束管理器
        
        Args:
            data_source_info: 数据源信息字典
        """
        self.data_source_info = data_source_info
        self.constraints: List[MultiSourceConstraintBase] = []
        self.constraint_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def add_constraint(self, constraint: MultiSourceConstraintBase, 
                     constraint_id: str = None) -> str:
        """
        添加约束
        
        Args:
            constraint: 约束对象
            constraint_id: 约束ID
            
        Returns:
            str: 约束ID
        """
        if constraint_id is None:
            constraint_id = f"constraint_{len(self.constraints)}"
        
        self.constraints.append(constraint)
        self.constraint_weights[constraint_id] = constraint.weight
        
        self.logger.info(f"添加约束: {constraint_id}, 类型: {constraint.constraint_type}")
        
        return constraint_id
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """
        移除约束
        
        Args:
            constraint_id: 约束ID
            
        Returns:
            bool: 是否成功移除
        """
        if constraint_id in self.constraint_weights:
            # 找到对应的约束索引
            constraint_index = list(self.constraint_weights.keys()).index(constraint_id)
            
            # 移除约束
            del self.constraints[constraint_index]
            del self.constraint_weights[constraint_id]
            
            self.logger.info(f"移除约束: {constraint_id}")
            return True
        
        return False
    
    def compute_total_constraint_loss(self, predictions: Dict[str, torch.Tensor], 
                                    data_sources: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算总约束损失
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Dict: 约束损失字典
        """
        constraint_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    loss = constraint.compute_constraint_loss(predictions, data_sources)
                    constraint_losses[constraint_id] = loss
                    total_loss += loss
                    
                except Exception as e:
                    self.logger.warning(f"约束 {constraint_id} 计算失败: {e}")
                    constraint_losses[constraint_id] = torch.tensor(0.0)
        
        constraint_losses['total_constraint_loss'] = total_loss
        
        return constraint_losses
    
    def validate_all_constraints(self, predictions: Dict[str, torch.Tensor], 
                               data_sources: Dict[str, torch.Tensor]) -> Dict[str, bool]:
        """
        验证所有约束
        
        Args:
            predictions: 模型预测字典
            data_sources: 数据源字典
            
        Returns:
            Dict: 约束验证结果字典
        """
        validation_results = {}
        
        for i, constraint in enumerate(self.constraints):
            if constraint.is_active:
                constraint_id = list(self.constraint_weights.keys())[i]
                
                try:
                    is_valid = constraint.validate_constraint(predictions, data_sources)
                    validation_results[constraint_id] = is_valid
                    
                except Exception as e:
                    self.logger.warning(f"约束 {constraint_id} 验证失败: {e}")
                    validation_results[constraint_id] = False
        
        return validation_results
    
    def update_constraint_weights(self, weight_updates: Dict[str, float]):
        """
        更新约束权重
        
        Args:
            weight_updates: 权重更新字典
        """
        for constraint_id, new_weight in weight_updates.items():
            if constraint_id in self.constraint_weights:
                self.constraint_weights[constraint_id] = new_weight
                
                # 更新对应约束对象的权重
                constraint_index = list(self.constraint_weights.keys()).index(constraint_id)
                self.constraints[constraint_index].weight = new_weight
                
                self.logger.info(f"更新约束权重: {constraint_id} -> {new_weight}")
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """
        获取约束统计信息
        
        Returns:
            Dict: 约束统计信息
        """
        stats = {
            'total_constraints': len(self.constraints),
            'active_constraints': sum(1 for c in self.constraints if c.is_active),
            'constraint_types': {},
            'total_weight': sum(self.constraint_weights.values())
        }
        
        # 统计约束类型
        for constraint in self.constraints:
            constraint_type = constraint.constraint_type.value
            if constraint_type not in stats['constraint_types']:
                stats['constraint_types'][constraint_type] = 0
            stats['constraint_types'][constraint_type] += 1
        
        return stats

def create_multi_source_constraint_manager(
    data_source_info: Dict[str, DataSourceInfo],
    constraint_config: Dict[str, Any] = None
) -> MultiSourceConstraintManager:
    """
    创建多源约束管理器
    
    Args:
        data_source_info: 数据源信息字典
        constraint_config: 约束配置
        
    Returns:
        MultiSourceConstraintManager: 约束管理器实例
    """
    manager = MultiSourceConstraintManager(data_source_info)
    
    if constraint_config:
        # 添加数据一致性约束
        if 'consistency_pairs' in constraint_config:
            consistency_constraint = DataConsistencyConstraint(
                source_pairs=constraint_config['consistency_pairs'],
                tolerance=constraint_config.get('consistency_tolerance', 0.1),
                weight=constraint_config.get('consistency_weight', 1.0)
            )
            manager.add_constraint(consistency_constraint, 'data_consistency')
        
        # 添加交叉验证约束
        if 'validation_pairs' in constraint_config:
            cross_validation_constraint = CrossValidationConstraint(
                validation_pairs=constraint_config['validation_pairs'],
                confidence_threshold=constraint_config.get('confidence_threshold', 0.8),
                weight=constraint_config.get('validation_weight', 1.0)
            )
            manager.add_constraint(cross_validation_constraint, 'cross_validation')
        
        # 添加不确定性加权约束
        if 'source_uncertainties' in constraint_config:
            uncertainty_constraint = UncertaintyWeightedConstraint(
                source_uncertainties=constraint_config['source_uncertainties'],
                uncertainty_model=constraint_config.get('uncertainty_model', 'inverse'),
                weight=constraint_config.get('uncertainty_weight', 1.0)
            )
            manager.add_constraint(uncertainty_constraint, 'uncertainty_weighted')
    
    return manager

if __name__ == "__main__":
    # 测试多源约束管理
    
    # 创建数据源信息
    data_sources = {
        'rgi': DataSourceInfo(
            source_type=DataSourceType.RGI,
            reliability=0.9,
            temporal_coverage=('2000', '2020'),
            spatial_resolution=30.0,
            uncertainty=0.05
        ),
        'farinotti': DataSourceInfo(
            source_type=DataSourceType.FARINOTTI,
            reliability=0.8,
            temporal_coverage=('2000', '2020'),
            spatial_resolution=100.0,
            uncertainty=0.15
        ),
        'millan': DataSourceInfo(
            source_type=DataSourceType.MILLAN,
            reliability=0.85,
            temporal_coverage=('2017', '2018'),
            spatial_resolution=50.0,
            uncertainty=0.10
        )
    }
    
    # 创建约束配置
    constraint_config = {
        'consistency_pairs': [('rgi', 'farinotti'), ('farinotti', 'millan')],
        'consistency_tolerance': 0.1,
        'consistency_weight': 1.0,
        'validation_pairs': {'rgi': 'farinotti'},
        'confidence_threshold': 0.8,
        'validation_weight': 0.5,
        'source_uncertainties': {'rgi': 0.05, 'farinotti': 0.15, 'millan': 0.10},
        'uncertainty_model': 'inverse',
        'uncertainty_weight': 0.3
    }
    
    # 创建约束管理器
    manager = create_multi_source_constraint_manager(data_sources, constraint_config)
    
    # 创建测试数据
    device = torch.device('cpu')
    predictions = {
        'rgi': torch.randn(100, 50, device=device),
        'farinotti': torch.randn(100, 50, device=device),
        'millan': torch.randn(100, 50, device=device)
    }
    
    data_sources_tensor = {
        'rgi': torch.randn(100, 50, device=device),
        'farinotti': torch.randn(100, 50, device=device),
        'millan': torch.randn(100, 50, device=device)
    }
    
    # 计算约束损失
    constraint_losses = manager.compute_total_constraint_loss(predictions, data_sources_tensor)
    print("=== 约束损失 ===")
    for constraint_id, loss in constraint_losses.items():
        print(f"{constraint_id}: {loss.item():.6f}")
    
    # 验证约束
    validation_results = manager.validate_all_constraints(predictions, data_sources_tensor)
    print("\n=== 约束验证结果 ===")
    for constraint_id, is_valid in validation_results.items():
        print(f"{constraint_id}: {'通过' if is_valid else '失败'}")
    
    # 获取统计信息
    stats = manager.get_constraint_statistics()
    print("\n=== 约束统计信息 ===")
    print(f"总约束数: {stats['total_constraints']}")
    print(f"活跃约束数: {stats['active_constraints']}")
    print(f"约束类型分布: {stats['constraint_types']}")
    print(f"总权重: {stats['total_weight']:.2f}")
    
    print("\n多源约束管理测试完成！")