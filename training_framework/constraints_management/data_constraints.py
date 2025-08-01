#!/usr/bin/env python3
"""
数据约束模块

实现PINNs模型的数据约束，包括观测数据拟合、
数据质量权重、多源数据融合等功能。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
from abc import ABC, abstractmethod

class DataConstraint(ABC):
    """
    数据约束基类
    
    定义数据约束的通用接口，包括：
    - 数据拟合损失计算
    - 数据质量评估
    - 权重调整
    - 不确定性量化
    """
    
    def __init__(self, 
                 constraint_config: Dict[str, Any]):
        """
        初始化数据约束
        
        Args:
            constraint_config: 约束配置参数
        """
        self.constraint_config = constraint_config
        self.constraint_weight = constraint_config.get('weight', 1.0)
        self.data_type = constraint_config.get('data_type', 'unknown')
        self.uncertainty_estimation = constraint_config.get('uncertainty_estimation', False)
        
    @abstractmethod
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算数据拟合损失
        
        Args:
            model_outputs: 模型输出
            observed_data: 观测数据
            data_weights: 数据权重
            
        Returns:
            数据拟合损失
        """
        pass
        
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        计算数据质量权重
        
        Args:
            observed_data: 观测数据
            
        Returns:
            数据质量权重
        """
        # 默认实现：均匀权重
        data_size = len(list(observed_data.values())[0])
        return jnp.ones(data_size)
        
    def estimate_uncertainty(self, 
                           model_outputs: Dict[str, jnp.ndarray],
                           observed_data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        估计数据不确定性
        
        Args:
            model_outputs: 模型输出
            observed_data: 观测数据
            
        Returns:
            不确定性估计
        """
        uncertainties = {}
        
        for key in observed_data.keys():
            if key in model_outputs:
                residual = model_outputs[key] - observed_data[key]
                uncertainty = jnp.std(residual)
                uncertainties[key] = uncertainty
                
        return uncertainties

class VelocityDataConstraint(DataConstraint):
    """
    速度数据约束
    
    处理冰川表面速度观测数据的拟合，
    包括InSAR、GPS等观测数据。
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        super().__init__(constraint_config)
        self.velocity_components = constraint_config.get(
            'velocity_components', ['velocity_x', 'velocity_y']
        )
        self.measurement_error = constraint_config.get('measurement_error', 0.1)
        self.outlier_threshold = constraint_config.get('outlier_threshold', 3.0)
        
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算速度数据拟合损失
        """
        total_loss = 0.0
        
        for component in self.velocity_components:
            if component in model_outputs and component in observed_data:
                predicted = model_outputs[component]
                observed = observed_data[component]
                
                # 计算残差
                residual = predicted - observed
                
                # 应用数据权重
                if data_weights is not None:
                    weighted_residual = residual * data_weights
                else:
                    weighted_residual = residual
                    
                # 考虑测量误差
                normalized_residual = weighted_residual / self.measurement_error
                
                # 计算损失（L2范数）
                component_loss = jnp.mean(normalized_residual**2)
                total_loss += component_loss
                
        return self.constraint_weight * total_loss
        
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        基于速度数据质量计算权重
        """
        # 计算速度幅值
        velocity_x = observed_data.get('velocity_x', jnp.zeros(1))
        velocity_y = observed_data.get('velocity_y', jnp.zeros(1))
        velocity_magnitude = jnp.sqrt(velocity_x**2 + velocity_y**2)
        
        # 基于速度幅值的权重
        # 高速区域权重较高，低速区域权重较低
        velocity_weights = jnp.where(
            velocity_magnitude > 10.0,  # m/year
            1.0,
            0.5 + 0.5 * velocity_magnitude / 10.0
        )
        
        # 异常值检测和权重调整
        velocity_median = jnp.median(velocity_magnitude)
        velocity_mad = jnp.median(jnp.abs(velocity_magnitude - velocity_median))
        
        # 使用修正的Z分数检测异常值
        modified_z_score = 0.6745 * (velocity_magnitude - velocity_median) / velocity_mad
        outlier_mask = jnp.abs(modified_z_score) > self.outlier_threshold
        
        # 降低异常值权重
        quality_weights = jnp.where(outlier_mask, 0.1, velocity_weights)
        
        return quality_weights
        
    def detect_velocity_outliers(self, 
                               observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        检测速度数据中的异常值
        
        Returns:
            异常值掩码（True表示异常值）
        """
        velocity_x = observed_data.get('velocity_x', jnp.zeros(1))
        velocity_y = observed_data.get('velocity_y', jnp.zeros(1))
        velocity_magnitude = jnp.sqrt(velocity_x**2 + velocity_y**2)
        
        # 使用四分位距方法检测异常值
        q1 = jnp.percentile(velocity_magnitude, 25)
        q3 = jnp.percentile(velocity_magnitude, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (velocity_magnitude < lower_bound) | (velocity_magnitude > upper_bound)
        
        return outlier_mask

class ThicknessDataConstraint(DataConstraint):
    """
    厚度数据约束
    
    处理冰川厚度观测数据的拟合，
    包括探地雷达、重力测量等数据。
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        super().__init__(constraint_config)
        self.measurement_error = constraint_config.get('measurement_error', 5.0)  # meters
        self.min_thickness = constraint_config.get('min_thickness', 0.0)
        self.max_thickness = constraint_config.get('max_thickness', 1000.0)
        
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算厚度数据拟合损失
        """
        if 'thickness' not in model_outputs or 'thickness' not in observed_data:
            return 0.0
            
        predicted_thickness = model_outputs['thickness']
        observed_thickness = observed_data['thickness']
        
        # 确保厚度为非负值
        predicted_thickness = jnp.maximum(predicted_thickness, 0.0)
        
        # 计算残差
        residual = predicted_thickness - observed_thickness
        
        # 应用数据权重
        if data_weights is not None:
            weighted_residual = residual * data_weights
        else:
            weighted_residual = residual
            
        # 考虑测量误差
        normalized_residual = weighted_residual / self.measurement_error
        
        # 计算损失
        thickness_loss = jnp.mean(normalized_residual**2)
        
        # 添加厚度范围约束
        range_penalty = self._compute_range_penalty(predicted_thickness)
        
        total_loss = thickness_loss + range_penalty
        
        return self.constraint_weight * total_loss
        
    def _compute_range_penalty(self, thickness: jnp.ndarray) -> float:
        """
        计算厚度范围约束惩罚
        """
        # 厚度不能为负
        negative_penalty = jnp.sum(jnp.maximum(-thickness, 0.0)**2)
        
        # 厚度不能超过最大值
        excess_penalty = jnp.sum(jnp.maximum(thickness - self.max_thickness, 0.0)**2)
        
        return 0.1 * (negative_penalty + excess_penalty)
        
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        基于厚度数据质量计算权重
        """
        thickness = observed_data.get('thickness', jnp.zeros(1))
        
        # 基于厚度值的权重
        # 较厚的冰川区域权重较高
        thickness_weights = jnp.where(
            thickness > 50.0,  # meters
            1.0,
            0.3 + 0.7 * thickness / 50.0
        )
        
        # 检查数据一致性
        thickness_std = jnp.std(thickness)
        if thickness_std > 0:
            # 基于局部变异性调整权重
            local_variation = jnp.abs(thickness - jnp.mean(thickness)) / thickness_std
            variation_weights = jnp.exp(-local_variation / 2.0)
            
            quality_weights = thickness_weights * variation_weights
        else:
            quality_weights = thickness_weights
            
        return quality_weights

class ElevationDataConstraint(DataConstraint):
    """
    高程数据约束
    
    处理冰川表面高程观测数据的拟合，
    包括DEM、激光测高等数据。
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        super().__init__(constraint_config)
        self.measurement_error = constraint_config.get('measurement_error', 1.0)  # meters
        self.temporal_consistency = constraint_config.get('temporal_consistency', True)
        
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算高程数据拟合损失
        """
        if 'surface_elevation' not in model_outputs or 'surface_elevation' not in observed_data:
            return 0.0
            
        predicted_elevation = model_outputs['surface_elevation']
        observed_elevation = observed_data['surface_elevation']
        
        # 计算残差
        residual = predicted_elevation - observed_elevation
        
        # 应用数据权重
        if data_weights is not None:
            weighted_residual = residual * data_weights
        else:
            weighted_residual = residual
            
        # 考虑测量误差
        normalized_residual = weighted_residual / self.measurement_error
        
        # 计算损失
        elevation_loss = jnp.mean(normalized_residual**2)
        
        # 添加时间一致性约束
        if self.temporal_consistency:
            temporal_penalty = self._compute_temporal_consistency_penalty(
                model_outputs, observed_data
            )
            elevation_loss += temporal_penalty
            
        return self.constraint_weight * elevation_loss
        
    def _compute_temporal_consistency_penalty(self, 
                                            model_outputs: Dict[str, jnp.ndarray],
                                            observed_data: Dict[str, jnp.ndarray]) -> float:
        """
        计算时间一致性惩罚
        """
        # 简化实现：检查高程变化的合理性
        if 'time' in observed_data and len(observed_data['time']) > 1:
            # 计算高程变化率
            elevation = model_outputs['surface_elevation']
            time = observed_data['time']
            
            # 简单的时间差分
            dt = jnp.diff(time)
            dh = jnp.diff(elevation)
            
            # 高程变化率（m/year）
            elevation_rate = dh / (dt + 1e-8)
            
            # 限制合理的高程变化率（-10 到 +5 m/year）
            rate_penalty = jnp.sum(
                jnp.maximum(elevation_rate - 5.0, 0.0)**2 + 
                jnp.maximum(-elevation_rate - 10.0, 0.0)**2
            )
            
            return 0.01 * rate_penalty
        else:
            return 0.0
            
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        基于高程数据质量计算权重
        """
        elevation = observed_data.get('surface_elevation', jnp.zeros(1))
        
        # 基于高程的权重
        # 高海拔区域权重较高（冰川核心区域）
        elevation_weights = jnp.where(
            elevation > 4000.0,  # meters
            1.0,
            0.5 + 0.5 * jnp.maximum(elevation - 3000.0, 0.0) / 1000.0
        )
        
        # 检查高程梯度的合理性
        if len(elevation) > 1:
            elevation_gradient = jnp.abs(jnp.diff(elevation))
            max_gradient = jnp.max(elevation_gradient)
            
            if max_gradient > 0:
                # 基于梯度平滑性的权重
                gradient_weights = jnp.ones_like(elevation)
                gradient_weights = gradient_weights.at[1:].set(
                    jnp.exp(-elevation_gradient / (0.1 * max_gradient + 1e-8))
                )
                
                quality_weights = elevation_weights * gradient_weights
            else:
                quality_weights = elevation_weights
        else:
            quality_weights = elevation_weights
            
        return quality_weights

class TemperatureDataConstraint(DataConstraint):
    """
    温度数据约束
    
    处理冰川温度观测数据的拟合，
    包括钻孔温度、遥感温度等数据。
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        super().__init__(constraint_config)
        self.measurement_error = constraint_config.get('measurement_error', 0.5)  # Kelvin
        self.temperature_range = constraint_config.get('temperature_range', (-50.0, 10.0))  # Celsius
        
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算温度数据拟合损失
        """
        if 'temperature' not in model_outputs or 'temperature' not in observed_data:
            return 0.0
            
        predicted_temperature = model_outputs['temperature']
        observed_temperature = observed_data['temperature']
        
        # 计算残差
        residual = predicted_temperature - observed_temperature
        
        # 应用数据权重
        if data_weights is not None:
            weighted_residual = residual * data_weights
        else:
            weighted_residual = residual
            
        # 考虑测量误差
        normalized_residual = weighted_residual / self.measurement_error
        
        # 计算损失
        temperature_loss = jnp.mean(normalized_residual**2)
        
        # 添加温度范围约束
        range_penalty = self._compute_temperature_range_penalty(predicted_temperature)
        
        total_loss = temperature_loss + range_penalty
        
        return self.constraint_weight * total_loss
        
    def _compute_temperature_range_penalty(self, temperature: jnp.ndarray) -> float:
        """
        计算温度范围约束惩罚
        """
        min_temp, max_temp = self.temperature_range
        
        # 温度超出合理范围的惩罚
        low_penalty = jnp.sum(jnp.maximum(min_temp - temperature, 0.0)**2)
        high_penalty = jnp.sum(jnp.maximum(temperature - max_temp, 0.0)**2)
        
        return 0.1 * (low_penalty + high_penalty)
        
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        基于温度数据质量计算权重
        """
        temperature = observed_data.get('temperature', jnp.zeros(1))
        
        # 基于温度值的权重
        # 接近融点的温度数据权重较高
        melting_point = 0.0  # Celsius
        temp_distance = jnp.abs(temperature - melting_point)
        
        temperature_weights = jnp.exp(-temp_distance / 10.0)  # 10度的特征距离
        
        # 检查温度的物理合理性
        min_temp, max_temp = self.temperature_range
        physical_mask = (temperature >= min_temp) & (temperature <= max_temp)
        
        quality_weights = temperature_weights * physical_mask.astype(float)
        
        return quality_weights

class MultiSourceDataConstraint(DataConstraint):
    """
    多源数据约束
    
    处理多种观测数据源的融合和约束，
    包括数据源权重、一致性检查等。
    """
    
    def __init__(self, constraint_config: Dict[str, Any]):
        super().__init__(constraint_config)
        self.data_sources = constraint_config.get('data_sources', [])
        self.source_weights = constraint_config.get('source_weights', {})
        self.consistency_check = constraint_config.get('consistency_check', True)
        
    def compute_data_loss(self, 
                         model_outputs: Dict[str, jnp.ndarray],
                         observed_data: Dict[str, jnp.ndarray],
                         data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算多源数据拟合损失
        """
        total_loss = 0.0
        
        for source in self.data_sources:
            source_weight = self.source_weights.get(source, 1.0)
            
            # 为每个数据源计算损失
            source_loss = self._compute_source_specific_loss(
                model_outputs, observed_data, source, data_weights
            )
            
            total_loss += source_weight * source_loss
            
        # 添加数据一致性约束
        if self.consistency_check:
            consistency_penalty = self._compute_consistency_penalty(
                model_outputs, observed_data
            )
            total_loss += consistency_penalty
            
        return self.constraint_weight * total_loss
        
    def _compute_source_specific_loss(self, 
                                     model_outputs: Dict[str, jnp.ndarray],
                                     observed_data: Dict[str, jnp.ndarray],
                                     source: str,
                                     data_weights: Optional[jnp.ndarray]) -> float:
        """
        计算特定数据源的损失
        """
        source_loss = 0.0
        
        # 根据数据源类型选择相应的变量
        if source == 'velocity':
            for var in ['velocity_x', 'velocity_y']:
                if var in model_outputs and var in observed_data:
                    residual = model_outputs[var] - observed_data[var]
                    if data_weights is not None:
                        residual = residual * data_weights
                    source_loss += jnp.mean(residual**2)
                    
        elif source == 'thickness':
            if 'thickness' in model_outputs and 'thickness' in observed_data:
                residual = model_outputs['thickness'] - observed_data['thickness']
                if data_weights is not None:
                    residual = residual * data_weights
                source_loss += jnp.mean(residual**2)
                
        elif source == 'elevation':
            if 'surface_elevation' in model_outputs and 'surface_elevation' in observed_data:
                residual = model_outputs['surface_elevation'] - observed_data['surface_elevation']
                if data_weights is not None:
                    residual = residual * data_weights
                source_loss += jnp.mean(residual**2)
                
        return source_loss
        
    def _compute_consistency_penalty(self, 
                                   model_outputs: Dict[str, jnp.ndarray],
                                   observed_data: Dict[str, jnp.ndarray]) -> float:
        """
        计算数据一致性惩罚
        """
        consistency_penalty = 0.0
        
        # 检查厚度和表面高程的一致性
        if ('thickness' in model_outputs and 'surface_elevation' in model_outputs and
            'bed_elevation' in observed_data):
            
            predicted_thickness = model_outputs['thickness']
            predicted_surface = model_outputs['surface_elevation']
            observed_bed = observed_data['bed_elevation']
            
            # 一致性检查：surface = bed + thickness
            consistency_residual = predicted_surface - (observed_bed + predicted_thickness)
            consistency_penalty += 0.1 * jnp.mean(consistency_residual**2)
            
        # 检查速度和厚度的物理一致性
        if ('velocity_x' in model_outputs and 'velocity_y' in model_outputs and
            'thickness' in model_outputs):
            
            velocity_magnitude = jnp.sqrt(
                model_outputs['velocity_x']**2 + model_outputs['velocity_y']**2
            )
            thickness = model_outputs['thickness']
            
            # 薄冰区域不应有高速度
            thin_ice_mask = thickness < 10.0  # meters
            high_velocity_mask = velocity_magnitude > 100.0  # m/year
            
            inconsistent_mask = thin_ice_mask & high_velocity_mask
            consistency_penalty += 0.05 * jnp.sum(inconsistent_mask.astype(float))
            
        return consistency_penalty
        
    def compute_data_quality_weights(self, 
                                   observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        基于多源数据质量计算权重
        """
        # 获取数据点数量
        data_size = len(list(observed_data.values())[0])
        quality_weights = jnp.ones(data_size)
        
        # 为每个数据源计算权重
        source_weights_list = []
        
        for source in self.data_sources:
            if source == 'velocity':
                if 'velocity_x' in observed_data and 'velocity_y' in observed_data:
                    velocity_constraint = VelocityDataConstraint(self.constraint_config)
                    source_weight = velocity_constraint.compute_data_quality_weights(observed_data)
                    source_weights_list.append(source_weight)
                    
            elif source == 'thickness':
                if 'thickness' in observed_data:
                    thickness_constraint = ThicknessDataConstraint(self.constraint_config)
                    source_weight = thickness_constraint.compute_data_quality_weights(observed_data)
                    source_weights_list.append(source_weight)
                    
            elif source == 'elevation':
                if 'surface_elevation' in observed_data:
                    elevation_constraint = ElevationDataConstraint(self.constraint_config)
                    source_weight = elevation_constraint.compute_data_quality_weights(observed_data)
                    source_weights_list.append(source_weight)
                    
        # 组合多个数据源的权重
        if source_weights_list:
            # 使用几何平均
            combined_weights = jnp.ones(data_size)
            for weights in source_weights_list:
                combined_weights *= weights
            quality_weights = combined_weights**(1.0 / len(source_weights_list))
            
        return quality_weights

def create_data_constraint(constraint_type: str,
                          constraint_config: Dict[str, Any]) -> DataConstraint:
    """
    工厂函数：创建数据约束
    
    Args:
        constraint_type: 约束类型
        constraint_config: 约束配置
        
    Returns:
        数据约束实例
    """
    if constraint_type == 'velocity':
        return VelocityDataConstraint(constraint_config)
    elif constraint_type == 'thickness':
        return ThicknessDataConstraint(constraint_config)
    elif constraint_type == 'elevation':
        return ElevationDataConstraint(constraint_config)
    elif constraint_type == 'temperature':
        return TemperatureDataConstraint(constraint_config)
    elif constraint_type == 'multi_source':
        return MultiSourceDataConstraint(constraint_config)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

class DataConstraintManager:
    """
    数据约束管理器
    
    管理多个数据约束的组合和计算
    """
    
    def __init__(self, constraints: List[DataConstraint]):
        """
        初始化约束管理器
        
        Args:
            constraints: 数据约束列表
        """
        self.constraints = constraints
        
    def compute_total_data_loss(self, 
                               model_outputs: Dict[str, jnp.ndarray],
                               observed_data: Dict[str, jnp.ndarray],
                               data_weights: Optional[jnp.ndarray] = None) -> float:
        """
        计算总的数据约束损失
        
        Args:
            model_outputs: 模型输出
            observed_data: 观测数据
            data_weights: 数据权重
            
        Returns:
            总的数据损失
        """
        total_loss = 0.0
        
        for constraint in self.constraints:
            constraint_loss = constraint.compute_data_loss(
                model_outputs, observed_data, data_weights
            )
            total_loss += constraint_loss
            
        return total_loss
        
    def compute_individual_data_losses(self, 
                                      model_outputs: Dict[str, jnp.ndarray],
                                      observed_data: Dict[str, jnp.ndarray],
                                      data_weights: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """
        计算各个数据约束的损失
        
        Args:
            model_outputs: 模型输出
            observed_data: 观测数据
            data_weights: 数据权重
            
        Returns:
            各约束的损失字典
        """
        losses = {}
        
        for i, constraint in enumerate(self.constraints):
            constraint_name = f"{type(constraint).__name__}_{i}"
            constraint_loss = constraint.compute_data_loss(
                model_outputs, observed_data, data_weights
            )
            losses[constraint_name] = constraint_loss
            
        return losses
        
    def compute_adaptive_weights(self, 
                                observed_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        计算自适应数据权重
        
        Args:
            observed_data: 观测数据
            
        Returns:
            自适应权重
        """
        # 获取数据点数量
        data_size = len(list(observed_data.values())[0])
        adaptive_weights = jnp.ones(data_size)
        
        # 为每个约束计算权重并组合
        constraint_weights = []
        
        for constraint in self.constraints:
            weights = constraint.compute_data_quality_weights(observed_data)
            constraint_weights.append(weights)
            
        if constraint_weights:
            # 使用加权平均组合权重
            total_weight = sum(constraint.constraint_weight for constraint in self.constraints)
            
            for i, weights in enumerate(constraint_weights):
                weight_factor = self.constraints[i].constraint_weight / total_weight
                adaptive_weights += weight_factor * weights
                
            adaptive_weights /= len(constraint_weights)
            
        return adaptive_weights

if __name__ == "__main__":
    # 测试代码
    print("Data constraints module loaded successfully")
    
    # 创建示例约束
    velocity_constraint = create_data_constraint(
        'velocity',
        {'weight': 1.0, 'measurement_error': 0.1}
    )
    
    thickness_constraint = create_data_constraint(
        'thickness',
        {'weight': 0.8, 'measurement_error': 5.0}
    )
    
    elevation_constraint = create_data_constraint(
        'elevation',
        {'weight': 0.6, 'measurement_error': 1.0}
    )
    
    # 创建约束管理器
    constraint_manager = DataConstraintManager([
        velocity_constraint,
        thickness_constraint,
        elevation_constraint
    ])
    
    print(f"Created constraints:")
    print(f"Velocity constraint: {type(velocity_constraint).__name__}")
    print(f"Thickness constraint: {type(thickness_constraint).__name__}")
    print(f"Elevation constraint: {type(elevation_constraint).__name__}")
    print(f"Constraint manager with {len(constraint_manager.constraints)} constraints")