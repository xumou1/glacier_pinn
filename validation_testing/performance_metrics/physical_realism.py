#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物理现实性评估模块

该模块实现了用于评估模型预测物理现实性的各种指标，包括物理约束检验、
守恒定律验证、因果关系检查、热力学一致性和地球物理合理性等。

主要功能:
- 物理约束验证
- 守恒定律检验
- 因果关系分析
- 热力学一致性
- 地球物理合理性
- 时空连续性

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicalConstraintType(Enum):
    """物理约束类型"""
    # 基本物理约束
    NON_NEGATIVE = "non_negative"          # 非负约束
    BOUNDED = "bounded"                    # 有界约束
    MONOTONIC = "monotonic"                # 单调性约束
    CONTINUITY = "continuity"              # 连续性约束
    
    # 冰川物理约束
    ICE_THICKNESS_POSITIVE = "ice_thickness_positive"  # 冰厚度非负
    VELOCITY_REALISTIC = "velocity_realistic"          # 速度合理性
    TEMPERATURE_RANGE = "temperature_range"            # 温度范围
    MASS_BALANCE_SIGN = "mass_balance_sign"            # 质量平衡符号
    
    # 地形约束
    ELEVATION_CONSISTENCY = "elevation_consistency"    # 高程一致性
    SLOPE_REALISTIC = "slope_realistic"                # 坡度合理性
    ASPECT_VALID = "aspect_valid"                      # 坡向有效性
    
    # 时间约束
    TEMPORAL_SMOOTHNESS = "temporal_smoothness"        # 时间平滑性
    CAUSALITY = "causality"                            # 因果性
    TREND_CONSISTENCY = "trend_consistency"            # 趋势一致性

class ConservationLaw(Enum):
    """守恒定律类型"""
    MASS_CONSERVATION = "mass_conservation"            # 质量守恒
    ENERGY_CONSERVATION = "energy_conservation"        # 能量守恒
    MOMENTUM_CONSERVATION = "momentum_conservation"    # 动量守恒
    VOLUME_CONSERVATION = "volume_conservation"        # 体积守恒

class PhysicalRealismMetric(Enum):
    """物理现实性指标"""
    # 约束违反指标
    CONSTRAINT_VIOLATION_RATE = "constraint_violation_rate"    # 约束违反率
    CONSTRAINT_VIOLATION_MAGNITUDE = "constraint_violation_magnitude"  # 约束违反幅度
    
    # 守恒定律指标
    CONSERVATION_ERROR = "conservation_error"                  # 守恒误差
    CONSERVATION_RELATIVE_ERROR = "conservation_relative_error"  # 相对守恒误差
    
    # 物理一致性指标
    PHYSICAL_CONSISTENCY = "physical_consistency"              # 物理一致性
    THERMODYNAMIC_CONSISTENCY = "thermodynamic_consistency"    # 热力学一致性
    
    # 因果性指标
    CAUSALITY_VIOLATION = "causality_violation"                # 因果性违反
    TEMPORAL_CONSISTENCY = "temporal_consistency"              # 时间一致性
    
    # 空间一致性指标
    SPATIAL_SMOOTHNESS = "spatial_smoothness"                  # 空间平滑性
    GRADIENT_REALISM = "gradient_realism"                      # 梯度现实性
    
    # 综合指标
    OVERALL_REALISM_SCORE = "overall_realism_score"            # 总体现实性得分
    PHYSICAL_PLAUSIBILITY = "physical_plausibility"            # 物理合理性

@dataclass
class PhysicalConstraint:
    """物理约束定义"""
    name: str
    constraint_type: PhysicalConstraintType
    check_function: Callable[[np.ndarray], np.ndarray]
    description: str
    severity: float = 1.0  # 违反严重程度权重
    tolerance: float = 1e-6  # 容忍度

@dataclass
class PhysicalRealismConfig:
    """物理现实性评估配置"""
    # 基本设置
    metrics: List[PhysicalRealismMetric] = field(default_factory=lambda: [
        PhysicalRealismMetric.CONSTRAINT_VIOLATION_RATE,
        PhysicalRealismMetric.CONSERVATION_ERROR,
        PhysicalRealismMetric.PHYSICAL_CONSISTENCY,
        PhysicalRealismMetric.SPATIAL_SMOOTHNESS
    ])
    
    # 约束设置
    constraints: List[PhysicalConstraint] = field(default_factory=list)
    enable_constraint_checking: bool = True
    constraint_tolerance: float = 1e-6
    
    # 守恒定律设置
    conservation_laws: List[ConservationLaw] = field(default_factory=lambda: [
        ConservationLaw.MASS_CONSERVATION
    ])
    conservation_tolerance: float = 1e-3
    
    # 物理参数范围
    ice_thickness_range: Tuple[float, float] = (0.0, 1000.0)  # 米
    velocity_range: Tuple[float, float] = (0.0, 1000.0)       # 米/年
    temperature_range: Tuple[float, float] = (-50.0, 10.0)    # 摄氏度
    elevation_range: Tuple[float, float] = (0.0, 9000.0)      # 米
    slope_range: Tuple[float, float] = (0.0, 90.0)            # 度
    
    # 时空一致性设置
    enable_temporal_consistency: bool = True
    temporal_smoothness_threshold: float = 0.1
    enable_spatial_consistency: bool = True
    spatial_smoothness_threshold: float = 0.1
    
    # 因果性设置
    enable_causality_check: bool = True
    causality_lag_max: int = 10  # 最大滞后期
    
    # 热力学设置
    enable_thermodynamic_check: bool = True
    melting_point: float = 0.0  # 摄氏度
    freezing_point: float = 0.0  # 摄氏度
    
    # 统计设置
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./physical_realism_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

@dataclass
class PhysicalRealismResult:
    """物理现实性评估结果"""
    metric_name: str
    value: float
    violation_count: int = 0
    total_count: int = 0
    violation_locations: Optional[np.ndarray] = None
    severity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConstraintChecker:
    """约束检查器"""
    
    def __init__(self, config: PhysicalRealismConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化默认约束
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """初始化默认约束"""
        default_constraints = [
            PhysicalConstraint(
                name="ice_thickness_non_negative",
                constraint_type=PhysicalConstraintType.ICE_THICKNESS_POSITIVE,
                check_function=lambda x: x >= 0,
                description="冰厚度必须非负",
                tolerance=self.config.constraint_tolerance
            ),
            PhysicalConstraint(
                name="velocity_realistic",
                constraint_type=PhysicalConstraintType.VELOCITY_REALISTIC,
                check_function=lambda x: (x >= self.config.velocity_range[0]) & 
                                       (x <= self.config.velocity_range[1]),
                description="速度必须在合理范围内",
                tolerance=self.config.constraint_tolerance
            ),
            PhysicalConstraint(
                name="temperature_range",
                constraint_type=PhysicalConstraintType.TEMPERATURE_RANGE,
                check_function=lambda x: (x >= self.config.temperature_range[0]) & 
                                       (x <= self.config.temperature_range[1]),
                description="温度必须在合理范围内",
                tolerance=self.config.constraint_tolerance
            ),
            PhysicalConstraint(
                name="elevation_consistency",
                constraint_type=PhysicalConstraintType.ELEVATION_CONSISTENCY,
                check_function=lambda x: (x >= self.config.elevation_range[0]) & 
                                       (x <= self.config.elevation_range[1]),
                description="高程必须在合理范围内",
                tolerance=self.config.constraint_tolerance
            )
        ]
        
        # 添加到配置中
        if not self.config.constraints:
            self.config.constraints = default_constraints
    
    def check_constraints(self, data: Dict[str, np.ndarray]) -> Dict[str, PhysicalRealismResult]:
        """检查物理约束"""
        results = {}
        
        try:
            for constraint in self.config.constraints:
                if constraint.name in data or self._can_derive_variable(constraint.name, data):
                    result = self._check_single_constraint(constraint, data)
                    if result:
                        results[constraint.name] = result
            
            return results
        
        except Exception as e:
            self.logger.error(f"约束检查失败: {e}")
            return {}
    
    def _can_derive_variable(self, variable_name: str, data: Dict[str, np.ndarray]) -> bool:
        """检查是否可以从现有数据推导变量"""
        # 简化实现，实际中可以更复杂
        derivable_vars = {
            'ice_thickness_non_negative': ['ice_thickness', 'thickness', 'h'],
            'velocity_realistic': ['velocity', 'speed', 'v'],
            'temperature_range': ['temperature', 'temp', 'T'],
            'elevation_consistency': ['elevation', 'dem', 'z']
        }
        
        if variable_name in derivable_vars:
            return any(var in data for var in derivable_vars[variable_name])
        
        return False
    
    def _check_single_constraint(self, constraint: PhysicalConstraint, 
                               data: Dict[str, np.ndarray]) -> Optional[PhysicalRealismResult]:
        """检查单个约束"""
        try:
            # 获取相关数据
            variable_data = self._get_variable_data(constraint.name, data)
            
            if variable_data is None:
                return None
            
            # 应用约束检查
            constraint_satisfied = constraint.check_function(variable_data)
            
            # 计算违反统计
            violation_mask = ~constraint_satisfied
            violation_count = np.sum(violation_mask)
            total_count = len(variable_data)
            violation_rate = violation_count / total_count if total_count > 0 else 0
            
            # 计算违反幅度
            if violation_count > 0:
                violation_magnitude = self._compute_violation_magnitude(
                    variable_data, violation_mask, constraint
                )
            else:
                violation_magnitude = 0.0
            
            return PhysicalRealismResult(
                metric_name=constraint.name,
                value=violation_rate,
                violation_count=violation_count,
                total_count=total_count,
                violation_locations=np.where(violation_mask)[0] if violation_count > 0 else None,
                severity_score=violation_magnitude * constraint.severity,
                metadata={
                    'constraint_type': constraint.constraint_type.value,
                    'description': constraint.description,
                    'tolerance': constraint.tolerance
                }
            )
        
        except Exception as e:
            self.logger.warning(f"约束 {constraint.name} 检查失败: {e}")
            return None
    
    def _get_variable_data(self, constraint_name: str, data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """获取约束相关的变量数据"""
        # 变量名映射
        variable_mapping = {
            'ice_thickness_non_negative': ['ice_thickness', 'thickness', 'h'],
            'velocity_realistic': ['velocity', 'speed', 'v'],
            'temperature_range': ['temperature', 'temp', 'T'],
            'elevation_consistency': ['elevation', 'dem', 'z']
        }
        
        if constraint_name in variable_mapping:
            for var_name in variable_mapping[constraint_name]:
                if var_name in data:
                    return data[var_name]
        
        # 直接查找
        if constraint_name in data:
            return data[constraint_name]
        
        return None
    
    def _compute_violation_magnitude(self, data: np.ndarray, 
                                   violation_mask: np.ndarray,
                                   constraint: PhysicalConstraint) -> float:
        """计算违反幅度"""
        try:
            violated_data = data[violation_mask]
            
            if constraint.constraint_type == PhysicalConstraintType.NON_NEGATIVE:
                # 负值的绝对值
                return np.mean(np.abs(violated_data[violated_data < 0]))
            
            elif constraint.constraint_type in [
                PhysicalConstraintType.VELOCITY_REALISTIC,
                PhysicalConstraintType.TEMPERATURE_RANGE,
                PhysicalConstraintType.ELEVATION_CONSISTENCY
            ]:
                # 超出范围的程度
                if constraint.constraint_type == PhysicalConstraintType.VELOCITY_REALISTIC:
                    range_bounds = self.config.velocity_range
                elif constraint.constraint_type == PhysicalConstraintType.TEMPERATURE_RANGE:
                    range_bounds = self.config.temperature_range
                else:
                    range_bounds = self.config.elevation_range
                
                lower_violations = violated_data[violated_data < range_bounds[0]]
                upper_violations = violated_data[violated_data > range_bounds[1]]
                
                magnitude = 0.0
                if len(lower_violations) > 0:
                    magnitude += np.mean(range_bounds[0] - lower_violations)
                if len(upper_violations) > 0:
                    magnitude += np.mean(upper_violations - range_bounds[1])
                
                return magnitude
            
            else:
                # 默认：违反值的标准差
                return np.std(violated_data)
        
        except Exception as e:
            self.logger.warning(f"违反幅度计算失败: {e}")
            return 0.0

class ConservationChecker:
    """守恒定律检查器"""
    
    def __init__(self, config: PhysicalRealismConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_conservation_laws(self, data: Dict[str, np.ndarray],
                              coordinates: Optional[np.ndarray] = None,
                              time_steps: Optional[np.ndarray] = None) -> Dict[str, PhysicalRealismResult]:
        """检查守恒定律"""
        results = {}
        
        try:
            for law in self.config.conservation_laws:
                result = self._check_single_conservation_law(law, data, coordinates, time_steps)
                if result:
                    results[law.value] = result
            
            return results
        
        except Exception as e:
            self.logger.error(f"守恒定律检查失败: {e}")
            return {}
    
    def _check_single_conservation_law(self, law: ConservationLaw,
                                     data: Dict[str, np.ndarray],
                                     coordinates: Optional[np.ndarray] = None,
                                     time_steps: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查单个守恒定律"""
        try:
            if law == ConservationLaw.MASS_CONSERVATION:
                return self._check_mass_conservation(data, coordinates, time_steps)
            elif law == ConservationLaw.ENERGY_CONSERVATION:
                return self._check_energy_conservation(data, coordinates, time_steps)
            elif law == ConservationLaw.MOMENTUM_CONSERVATION:
                return self._check_momentum_conservation(data, coordinates, time_steps)
            elif law == ConservationLaw.VOLUME_CONSERVATION:
                return self._check_volume_conservation(data, coordinates, time_steps)
            
            return None
        
        except Exception as e:
            self.logger.warning(f"守恒定律 {law.value} 检查失败: {e}")
            return None
    
    def _check_mass_conservation(self, data: Dict[str, np.ndarray],
                               coordinates: Optional[np.ndarray] = None,
                               time_steps: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查质量守恒"""
        try:
            # 需要质量平衡、冰厚度变化等数据
            required_vars = ['mass_balance', 'ice_thickness']
            
            if not all(var in data for var in required_vars):
                return None
            
            mass_balance = data['mass_balance']
            ice_thickness = data['ice_thickness']
            
            if time_steps is not None and len(time_steps) > 1:
                # 时间序列分析
                dt = np.diff(time_steps)
                dh_dt = np.diff(ice_thickness, axis=0) / dt[:, None]
                
                # 质量守恒方程: dh/dt = mass_balance / ice_density
                ice_density = 917.0  # kg/m³
                expected_dh_dt = mass_balance[:-1] / ice_density
                
                # 计算守恒误差
                conservation_error = np.abs(dh_dt - expected_dh_dt)
                mean_error = np.mean(conservation_error)
                relative_error = mean_error / (np.mean(np.abs(expected_dh_dt)) + 1e-10)
                
                # 判断违反
                violation_mask = conservation_error > self.config.conservation_tolerance
                violation_rate = np.mean(violation_mask)
                
                return PhysicalRealismResult(
                    metric_name='mass_conservation',
                    value=mean_error,
                    violation_count=np.sum(violation_mask),
                    total_count=violation_mask.size,
                    severity_score=relative_error,
                    metadata={
                        'relative_error': relative_error,
                        'violation_rate': violation_rate,
                        'conservation_type': 'mass'
                    }
                )
            
            else:
                # 静态分析：检查质量平衡的合理性
                mass_balance_stats = {
                    'mean': np.mean(mass_balance),
                    'std': np.std(mass_balance),
                    'min': np.min(mass_balance),
                    'max': np.max(mass_balance)
                }
                
                # 简单的合理性检查
                unrealistic_mask = np.abs(mass_balance) > 10.0  # 10 m/year 阈值
                violation_rate = np.mean(unrealistic_mask)
                
                return PhysicalRealismResult(
                    metric_name='mass_conservation',
                    value=violation_rate,
                    violation_count=np.sum(unrealistic_mask),
                    total_count=len(mass_balance),
                    severity_score=violation_rate,
                    metadata={
                        'mass_balance_stats': mass_balance_stats,
                        'conservation_type': 'mass'
                    }
                )
        
        except Exception as e:
            self.logger.warning(f"质量守恒检查失败: {e}")
            return None
    
    def _check_energy_conservation(self, data: Dict[str, np.ndarray],
                                 coordinates: Optional[np.ndarray] = None,
                                 time_steps: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查能量守恒"""
        # 简化实现
        try:
            if 'temperature' in data and 'heat_flux' in data:
                temperature = data['temperature']
                heat_flux = data['heat_flux']
                
                # 简单的能量平衡检查
                energy_imbalance = np.abs(heat_flux - np.gradient(temperature))
                mean_imbalance = np.mean(energy_imbalance)
                
                violation_mask = energy_imbalance > self.config.conservation_tolerance
                violation_rate = np.mean(violation_mask)
                
                return PhysicalRealismResult(
                    metric_name='energy_conservation',
                    value=mean_imbalance,
                    violation_count=np.sum(violation_mask),
                    total_count=len(energy_imbalance),
                    severity_score=violation_rate,
                    metadata={'conservation_type': 'energy'}
                )
            
            return None
        
        except Exception as e:
            self.logger.warning(f"能量守恒检查失败: {e}")
            return None
    
    def _check_momentum_conservation(self, data: Dict[str, np.ndarray],
                                   coordinates: Optional[np.ndarray] = None,
                                   time_steps: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查动量守恒"""
        # 简化实现
        return None
    
    def _check_volume_conservation(self, data: Dict[str, np.ndarray],
                                 coordinates: Optional[np.ndarray] = None,
                                 time_steps: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查体积守恒"""
        # 简化实现
        return None

class CausalityChecker:
    """因果性检查器"""
    
    def __init__(self, config: PhysicalRealismConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_causality(self, data: Dict[str, np.ndarray],
                       time_steps: Optional[np.ndarray] = None) -> Dict[str, PhysicalRealismResult]:
        """检查因果性"""
        results = {}
        
        try:
            if time_steps is None or len(time_steps) < 3:
                return results
            
            # 检查时间序列的因果关系
            for var_name, var_data in data.items():
                if var_data.ndim >= 2:  # 时间序列数据
                    result = self._check_temporal_causality(var_name, var_data, time_steps)
                    if result:
                        results[f'{var_name}_causality'] = result
            
            return results
        
        except Exception as e:
            self.logger.error(f"因果性检查失败: {e}")
            return {}
    
    def _check_temporal_causality(self, var_name: str, var_data: np.ndarray,
                                time_steps: np.ndarray) -> Optional[PhysicalRealismResult]:
        """检查时间因果性"""
        try:
            # 检查是否存在未来信息影响过去的情况
            n_times = var_data.shape[0]
            
            causality_violations = 0
            total_checks = 0
            
            for t in range(1, n_times):
                for lag in range(1, min(self.config.causality_lag_max, t)):
                    # 检查 t-lag 时刻的值是否异常依赖于 t 时刻的值
                    current_values = var_data[t]
                    past_values = var_data[t - lag]
                    
                    # 简单的因果性检查：过去值不应该与未来值有过强的相关性
                    if np.std(current_values) > 0 and np.std(past_values) > 0:
                        correlation = np.corrcoef(current_values.flatten(), past_values.flatten())[0, 1]
                        
                        # 如果相关性过高，可能存在因果性问题
                        if abs(correlation) > 0.95:  # 阈值
                            causality_violations += 1
                        
                        total_checks += 1
            
            violation_rate = causality_violations / total_checks if total_checks > 0 else 0
            
            return PhysicalRealismResult(
                metric_name=f'{var_name}_causality',
                value=violation_rate,
                violation_count=causality_violations,
                total_count=total_checks,
                severity_score=violation_rate,
                metadata={
                    'variable': var_name,
                    'max_lag': self.config.causality_lag_max
                }
            )
        
        except Exception as e:
            self.logger.warning(f"变量 {var_name} 因果性检查失败: {e}")
            return None

class SpatialConsistencyChecker:
    """空间一致性检查器"""
    
    def __init__(self, config: PhysicalRealismConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_spatial_consistency(self, data: Dict[str, np.ndarray],
                                coordinates: Optional[np.ndarray] = None) -> Dict[str, PhysicalRealismResult]:
        """检查空间一致性"""
        results = {}
        
        try:
            for var_name, var_data in data.items():
                if var_data.ndim >= 2:  # 空间数据
                    result = self._check_spatial_smoothness(var_name, var_data, coordinates)
                    if result:
                        results[f'{var_name}_spatial_smoothness'] = result
            
            return results
        
        except Exception as e:
            self.logger.error(f"空间一致性检查失败: {e}")
            return {}
    
    def _check_spatial_smoothness(self, var_name: str, var_data: np.ndarray,
                                coordinates: Optional[np.ndarray] = None) -> Optional[PhysicalRealismResult]:
        """检查空间平滑性"""
        try:
            # 计算空间梯度
            if var_data.ndim == 2:
                # 2D数据
                grad_y, grad_x = np.gradient(var_data)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            elif var_data.ndim == 3 and var_data.shape[0] > 1:
                # 3D时间序列数据，取最后一个时间步
                last_time_data = var_data[-1]
                grad_y, grad_x = np.gradient(last_time_data)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            else:
                return None
            
            # 检查梯度的合理性
            mean_gradient = np.mean(gradient_magnitude)
            std_gradient = np.std(gradient_magnitude)
            
            # 检测异常大的梯度（可能表示不平滑）
            threshold = mean_gradient + 3 * std_gradient
            large_gradient_mask = gradient_magnitude > threshold
            
            violation_rate = np.mean(large_gradient_mask)
            
            return PhysicalRealismResult(
                metric_name=f'{var_name}_spatial_smoothness',
                value=mean_gradient,
                violation_count=np.sum(large_gradient_mask),
                total_count=gradient_magnitude.size,
                severity_score=violation_rate,
                metadata={
                    'variable': var_name,
                    'mean_gradient': mean_gradient,
                    'std_gradient': std_gradient,
                    'violation_rate': violation_rate
                }
            )
        
        except Exception as e:
            self.logger.warning(f"变量 {var_name} 空间平滑性检查失败: {e}")
            return None

class PhysicalRealismAnalyzer:
    """物理现实性分析器"""
    
    def __init__(self, config: PhysicalRealismConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化检查器
        self.constraint_checker = ConstraintChecker(config)
        self.conservation_checker = ConservationChecker(config)
        self.causality_checker = CausalityChecker(config)
        self.spatial_checker = SpatialConsistencyChecker(config)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
    
    def analyze_physical_realism(self, 
                               data: Dict[str, np.ndarray],
                               coordinates: Optional[np.ndarray] = None,
                               time_steps: Optional[np.ndarray] = None) -> Dict[str, PhysicalRealismResult]:
        """分析物理现实性"""
        try:
            results = {}
            
            # 约束检查
            if self.config.enable_constraint_checking:
                constraint_results = self.constraint_checker.check_constraints(data)
                results.update(constraint_results)
            
            # 守恒定律检查
            conservation_results = self.conservation_checker.check_conservation_laws(
                data, coordinates, time_steps
            )
            results.update(conservation_results)
            
            # 因果性检查
            if self.config.enable_causality_check:
                causality_results = self.causality_checker.check_causality(data, time_steps)
                results.update(causality_results)
            
            # 空间一致性检查
            if self.config.enable_spatial_consistency:
                spatial_results = self.spatial_checker.check_spatial_consistency(data, coordinates)
                results.update(spatial_results)
            
            # 计算综合指标
            overall_results = self._compute_overall_metrics(results)
            results.update(overall_results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"物理现实性分析失败: {e}")
            return {}
    
    def _compute_overall_metrics(self, results: Dict[str, PhysicalRealismResult]) -> Dict[str, PhysicalRealismResult]:
        """计算综合指标"""
        overall_results = {}
        
        try:
            if not results:
                return overall_results
            
            # 总体现实性得分
            if PhysicalRealismMetric.OVERALL_REALISM_SCORE in self.config.metrics:
                violation_rates = []
                severity_scores = []
                
                for result in results.values():
                    if hasattr(result, 'value') and hasattr(result, 'severity_score'):
                        violation_rates.append(result.value)
                        severity_scores.append(result.severity_score)
                
                if violation_rates:
                    # 综合得分：1 - 加权平均违反率
                    mean_violation_rate = np.mean(violation_rates)
                    mean_severity = np.mean(severity_scores)
                    
                    overall_score = 1.0 - (0.7 * mean_violation_rate + 0.3 * mean_severity)
                    overall_score = max(0.0, min(1.0, overall_score))  # 限制在[0,1]
                    
                    overall_results['overall_realism_score'] = PhysicalRealismResult(
                        metric_name='overall_realism_score',
                        value=overall_score,
                        severity_score=mean_severity,
                        metadata={
                            'mean_violation_rate': mean_violation_rate,
                            'mean_severity': mean_severity,
                            'n_checks': len(violation_rates)
                        }
                    )
            
            # 物理合理性
            if PhysicalRealismMetric.PHYSICAL_PLAUSIBILITY in self.config.metrics:
                # 基于约束违反的合理性评分
                constraint_results = [r for r in results.values() 
                                    if 'constraint' in r.metadata.get('constraint_type', '')]
                
                if constraint_results:
                    plausibility_scores = []
                    for result in constraint_results:
                        # 合理性 = 1 - 违反率
                        plausibility = 1.0 - result.value
                        plausibility_scores.append(plausibility)
                    
                    mean_plausibility = np.mean(plausibility_scores)
                    
                    overall_results['physical_plausibility'] = PhysicalRealismResult(
                        metric_name='physical_plausibility',
                        value=mean_plausibility,
                        metadata={
                            'n_constraints': len(constraint_results),
                            'individual_scores': plausibility_scores
                        }
                    )
        
        except Exception as e:
            self.logger.warning(f"综合指标计算失败: {e}")
        
        return overall_results
    
    def plot_results(self, 
                    results: Dict[str, PhysicalRealismResult],
                    data: Dict[str, np.ndarray]):
        """绘制物理现实性分析结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('物理现实性分析结果', fontsize=16, fontweight='bold')
            
            # 约束违反率
            ax = axes[0, 0]
            constraint_results = {k: v for k, v in results.items() 
                                if 'constraint' in v.metadata.get('constraint_type', '')}
            
            if constraint_results:
                names = list(constraint_results.keys())
                violation_rates = [r.value for r in constraint_results.values()]
                
                bars = ax.bar(range(len(names)), violation_rates)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('违反率')
                ax.set_title('物理约束违反率')
                ax.grid(True, alpha=0.3)
                
                # 添加颜色编码
                for i, bar in enumerate(bars):
                    if violation_rates[i] > 0.1:
                        bar.set_color('red')
                    elif violation_rates[i] > 0.05:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
            
            # 守恒误差
            ax = axes[0, 1]
            conservation_results = {k: v for k, v in results.items() 
                                  if 'conservation' in k}
            
            if conservation_results:
                names = list(conservation_results.keys())
                errors = [r.value for r in conservation_results.values()]
                
                ax.bar(range(len(names)), errors)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('守恒误差')
                ax.set_title('守恒定律误差')
                ax.grid(True, alpha=0.3)
            
            # 综合得分
            ax = axes[1, 0]
            if 'overall_realism_score' in results:
                score = results['overall_realism_score'].value
                
                # 创建仪表盘样式的图
                theta = np.linspace(0, np.pi, 100)
                r = np.ones_like(theta)
                
                ax.plot(theta, r, 'k-', linewidth=2)
                
                # 得分指针
                score_angle = np.pi * (1 - score)
                ax.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
                
                ax.set_xlim(0, np.pi)
                ax.set_ylim(0, 1.2)
                ax.set_title(f'总体现实性得分: {score:.3f}')
                ax.set_xticks([0, np.pi/2, np.pi])
                ax.set_xticklabels(['0', '0.5', '1.0'])
                ax.grid(True, alpha=0.3)
            
            # 空间平滑性示例
            ax = axes[1, 1]
            spatial_results = {k: v for k, v in results.items() 
                             if 'spatial_smoothness' in k}
            
            if spatial_results and data:
                # 选择第一个空间数据进行可视化
                for var_name, var_data in data.items():
                    if var_data.ndim >= 2:
                        if var_data.ndim == 3:
                            plot_data = var_data[-1]  # 最后时间步
                        else:
                            plot_data = var_data
                        
                        im = ax.imshow(plot_data, cmap='viridis', aspect='auto')
                        ax.set_title(f'{var_name} 空间分布')
                        plt.colorbar(im, ax=ax)
                        break
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/physical_realism_analysis.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, PhysicalRealismResult],
                       data: Dict[str, np.ndarray]) -> str:
        """生成物理现实性分析报告"""
        from datetime import datetime
        
        report_lines = [
            "="*80,
            "物理现实性分析报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析变量数: {len(data)}",
            f"检查项目数: {len(results)}",
            "",
            "1. 物理约束检查",
            "-"*40
        ]
        
        # 约束检查结果
        constraint_results = {k: v for k, v in results.items() 
                            if 'constraint' in v.metadata.get('constraint_type', '')}
        
        if constraint_results:
            for name, result in constraint_results.items():
                status = "✓ 通过" if result.value < 0.01 else "✗ 违反"
                report_lines.append(f"{name}: {status} (违反率: {result.value:.4f})")
                if result.violation_count > 0:
                    report_lines.append(f"  违反数量: {result.violation_count}/{result.total_count}")
                    report_lines.append(f"  严重程度: {result.severity_score:.4f}")
        else:
            report_lines.append("未进行约束检查")
        
        # 守恒定律检查
        conservation_results = {k: v for k, v in results.items() if 'conservation' in k}
        if conservation_results:
            report_lines.extend([
                "",
                "2. 守恒定律检查",
                "-"*40
            ])
            
            for name, result in conservation_results.items():
                report_lines.append(f"{name}: 误差 = {result.value:.6f}")
                if 'relative_error' in result.metadata:
                    report_lines.append(f"  相对误差: {result.metadata['relative_error']:.6f}")
        
        # 因果性检查
        causality_results = {k: v for k, v in results.items() if 'causality' in k}
        if causality_results:
            report_lines.extend([
                "",
                "3. 因果性检查",
                "-"*40
            ])
            
            for name, result in causality_results.items():
                status = "✓ 正常" if result.value < 0.1 else "⚠ 异常"
                report_lines.append(f"{name}: {status} (违反率: {result.value:.4f})")
        
        # 空间一致性检查
        spatial_results = {k: v for k, v in results.items() if 'spatial' in k}
        if spatial_results:
            report_lines.extend([
                "",
                "4. 空间一致性检查",
                "-"*40
            ])
            
            for name, result in spatial_results.items():
                report_lines.append(f"{name}: 平均梯度 = {result.value:.6f}")
                if 'violation_rate' in result.metadata:
                    report_lines.append(f"  异常梯度率: {result.metadata['violation_rate']:.4f}")
        
        # 综合评估
        if 'overall_realism_score' in results:
            overall_score = results['overall_realism_score']
            report_lines.extend([
                "",
                "5. 综合评估",
                "-"*40,
                f"总体现实性得分: {overall_score.value:.4f} (0-1, 越高越好)"
            ])
            
            if overall_score.value >= 0.8:
                assessment = "优秀 - 模型预测具有很高的物理现实性"
            elif overall_score.value >= 0.6:
                assessment = "良好 - 模型预测基本符合物理规律"
            elif overall_score.value >= 0.4:
                assessment = "一般 - 模型预测存在一些物理问题"
            else:
                assessment = "较差 - 模型预测存在严重的物理问题"
            
            report_lines.append(f"评估结果: {assessment}")
        
        if 'physical_plausibility' in results:
            plausibility = results['physical_plausibility']
            report_lines.append(f"物理合理性: {plausibility.value:.4f}")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_physical_realism_analyzer(config: Optional[PhysicalRealismConfig] = None) -> PhysicalRealismAnalyzer:
    """创建物理现实性分析器"""
    if config is None:
        config = PhysicalRealismConfig()
    
    return PhysicalRealismAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    print("开始物理现实性分析测试...")
    
    # 生成测试数据
    np.random.seed(42)
    n_points = 100
    n_times = 10
    
    # 创建测试数据
    test_data = {
        'ice_thickness': np.abs(np.random.normal(50, 20, (n_times, n_points))),  # 确保非负
        'velocity': np.abs(np.random.normal(100, 30, (n_times, n_points))),     # 确保非负
        'temperature': np.random.normal(-10, 5, (n_times, n_points)),
        'mass_balance': np.random.normal(0, 2, (n_times, n_points)),
        'elevation': np.random.uniform(3000, 6000, n_points)
    }
    
    # 添加一些违反约束的数据
    test_data['ice_thickness'][0, :5] = -10  # 负厚度
    test_data['velocity'][0, :3] = 2000      # 过大速度
    test_data['temperature'][0, :2] = 50     # 过高温度
    
    # 坐标和时间
    coordinates = np.random.uniform(-1, 1, (n_points, 2))
    time_steps = np.arange(n_times)
    
    # 创建配置
    config = PhysicalRealismConfig(
        metrics=[
            PhysicalRealismMetric.CONSTRAINT_VIOLATION_RATE,
            PhysicalRealismMetric.CONSERVATION_ERROR,
            PhysicalRealismMetric.PHYSICAL_CONSISTENCY,
            PhysicalRealismMetric.OVERALL_REALISM_SCORE,
            PhysicalRealismMetric.PHYSICAL_PLAUSIBILITY
        ],
        enable_constraint_checking=True,
        enable_causality_check=True,
        enable_spatial_consistency=True,
        enable_plotting=True,
        verbose=True
    )
    
    # 创建分析器
    analyzer = create_physical_realism_analyzer(config)
    
    # 进行分析
    results = analyzer.analyze_physical_realism(
        data=test_data,
        coordinates=coordinates,
        time_steps=time_steps
    )
    
    # 打印结果
    print("\n物理现实性分析完成！")
    print(f"检查了 {len(results)} 个项目")
    
    # 生成报告
    report = analyzer.generate_report(results, test_data)
    print("\n" + report)
    
    # 绘制结果
    analyzer.plot_results(results, test_data)
    
    print("\n物理现实性分析测试完成！")