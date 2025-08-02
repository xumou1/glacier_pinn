#!/usr/bin/env python3
"""
因果性检验

实现物理信息神经网络中因果性的验证，包括：
- 时间因果性检验
- 空间因果性检验
- 物理因果性检验
- 信息传播速度检验
- 格兰杰因果性检验
- 冰川动力学因果性检验

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
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import networkx as nx

class CausalityType(Enum):
    """因果性类型枚举"""
    TEMPORAL = "temporal"  # 时间因果性
    SPATIAL = "spatial"  # 空间因果性
    PHYSICAL = "physical"  # 物理因果性
    INFORMATION_PROPAGATION = "information_propagation"  # 信息传播
    GRANGER = "granger"  # 格兰杰因果性
    GLACIER_DYNAMICS = "glacier_dynamics"  # 冰川动力学因果性

class CausalDirection(Enum):
    """因果方向枚举"""
    FORWARD = "forward"  # 正向因果
    BACKWARD = "backward"  # 反向因果
    BIDIRECTIONAL = "bidirectional"  # 双向因果
    NO_CAUSALITY = "no_causality"  # 无因果关系

class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"  # 基础验证
    INTERMEDIATE = "intermediate"  # 中级验证
    ADVANCED = "advanced"  # 高级验证
    COMPREHENSIVE = "comprehensive"  # 全面验证

@dataclass
class CausalityConfig:
    """因果性检验配置"""
    # 基础配置
    tolerance: float = 1e-6  # 容差
    significance_level: float = 0.05  # 显著性水平
    
    # 时间配置
    max_time_lag: int = 10  # 最大时间滞后
    time_step: float = 0.1  # 时间步长
    temporal_window: int = 50  # 时间窗口大小
    
    # 空间配置
    max_spatial_distance: float = 10.0  # 最大空间距离
    spatial_resolution: float = 0.1  # 空间分辨率
    neighbor_radius: float = 1.0  # 邻域半径
    
    # 物理参数
    light_speed: float = 3e8  # 光速 [m/s]
    sound_speed: float = 343.0  # 声速 [m/s]
    ice_wave_speed: float = 3000.0  # 冰中波速 [m/s]
    thermal_diffusivity: float = 1.09e-6  # 热扩散率 [m²/s]
    
    # 采样配置
    num_test_points: int = 1000  # 测试点数量
    spatial_domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)  # 空间域
    temporal_domain: Tuple[float, float] = (0.0, 10.0)  # 时间域
    
    # 统计配置
    bootstrap_samples: int = 1000  # 自举样本数
    confidence_level: float = 0.95  # 置信水平
    min_samples_for_test: int = 30  # 测试所需最小样本数
    
    # 格兰杰因果性配置
    granger_max_lags: int = 5  # 格兰杰因果性最大滞后
    var_model_lags: int = 3  # VAR模型滞后阶数
    
    # 可视化配置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_directory: str = "./causality_plots"
    
    # 日志配置
    log_level: str = "INFO"
    detailed_logging: bool = False

class CausalityValidatorBase(ABC):
    """
    因果性验证器基类
    
    定义因果性验证的通用接口
    """
    
    def __init__(self, config: CausalityConfig):
        """
        初始化验证器
        
        Args:
            config: 配置
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # 创建绘图目录
        if config.save_plots:
            Path(config.plot_directory).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """验证因果性"""
        pass
    
    def _generate_test_points(self, num_points: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        生成测试点
        
        Args:
            num_points: 测试点数量
            
        Returns:
            Dict[str, torch.Tensor]: 测试点数据
        """
        if num_points is None:
            num_points = self.config.num_test_points
        
        # 空间坐标
        x_min, x_max, y_min, y_max = self.config.spatial_domain
        x = torch.linspace(x_min, x_max, int(np.sqrt(num_points)))
        y = torch.linspace(y_min, y_max, int(np.sqrt(num_points)))
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 时间坐标
        t_min, t_max = self.config.temporal_domain
        t = torch.linspace(t_min, t_max, num_points)
        
        # 随机采样
        indices = torch.randperm(X.numel())[:num_points]
        x_flat = X.flatten()[indices]
        y_flat = Y.flatten()[indices]
        t_sample = t[:len(x_flat)]
        
        return {
            'x': x_flat.unsqueeze(1),
            'y': y_flat.unsqueeze(1),
            't': t_sample.unsqueeze(1)
        }
    
    def _compute_derivatives(self, model: nn.Module, inputs: torch.Tensor, 
                           output_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算导数
        
        Args:
            model: 模型
            inputs: 输入张量
            output_idx: 输出索引
            
        Returns:
            Dict[str, torch.Tensor]: 导数字典
        """
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        if outputs.dim() > 1:
            u = outputs[:, output_idx:output_idx+1]
        else:
            u = outputs.unsqueeze(1)
        
        # 一阶导数
        grad_u = torch.autograd.grad(
            outputs=u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        derivatives = {
            'u': u,
            'u_x': grad_u[:, 0:1],
            'u_y': grad_u[:, 1:2],
            'u_t': grad_u[:, 2:3]
        }
        
        return derivatives
    
    def _statistical_test(self, data1: np.ndarray, data2: np.ndarray, 
                         test_type: str = "correlation") -> Dict[str, float]:
        """
        统计检验
        
        Args:
            data1: 数据1
            data2: 数据2
            test_type: 检验类型
            
        Returns:
            Dict[str, float]: 检验结果
        """
        if test_type == "correlation":
            corr, p_value = stats.pearsonr(data1.flatten(), data2.flatten())
            return {'statistic': corr, 'p_value': p_value}
        
        elif test_type == "mutual_info":
            # 离散化数据
            bins = 20
            data1_disc = np.digitize(data1.flatten(), np.linspace(data1.min(), data1.max(), bins))
            data2_disc = np.digitize(data2.flatten(), np.linspace(data2.min(), data2.max(), bins))
            mi = mutual_info_score(data1_disc, data2_disc)
            return {'statistic': mi, 'p_value': None}
        
        elif test_type == "ks_test":
            statistic, p_value = stats.ks_2samp(data1.flatten(), data2.flatten())
            return {'statistic': statistic, 'p_value': p_value}
        
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")
    
    def _check_significance(self, p_value: float) -> bool:
        """
        检查显著性
        
        Args:
            p_value: p值
            
        Returns:
            bool: 是否显著
        """
        return p_value < self.config.significance_level if p_value is not None else False

class TemporalCausalityValidator(CausalityValidatorBase):
    """时间因果性验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证时间因果性
        
        检查模型是否违反时间因果性：
        1. 未来不能影响过去
        2. 信息传播有限速度
        3. 时间序列的因果关系
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        
        # 生成时间序列数据
        time_series_results = self._test_temporal_causality(model, inputs)
        
        # 检查未来影响过去
        future_past_results = self._test_future_past_influence(model, inputs)
        
        # 检查信息传播速度
        propagation_results = self._test_information_propagation(model, inputs)
        
        # 综合结果
        overall_passed = (
            time_series_results['passed'] and
            future_past_results['passed'] and
            propagation_results['passed']
        )
        
        result = {
            'passed': overall_passed,
            'time_series_causality': time_series_results,
            'future_past_influence': future_past_results,
            'information_propagation': propagation_results,
            'validation_type': 'temporal_causality'
        }
        
        # 记录日志
        status = "PASSED" if overall_passed else "FAILED"
        self.logger.info(f"时间因果性验证: {status}")
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_temporal_causality(inputs.detach(), result)
        
        return result
    
    def _test_temporal_causality(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试时间序列因果性
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 固定空间位置，变化时间
        x_fixed = inputs[0, 0].item()
        y_fixed = inputs[0, 1].item()
        
        # 生成时间序列
        t_min, t_max = self.config.temporal_domain
        time_points = torch.linspace(t_min, t_max, self.config.temporal_window)
        
        time_series_inputs = torch.stack([
            torch.full_like(time_points, x_fixed),
            torch.full_like(time_points, y_fixed),
            time_points
        ], dim=1)
        
        # 模型预测
        with torch.no_grad():
            predictions = model(time_series_inputs)
        
        # 如果是多输出，选择第一个输出
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]
        
        # 转换为numpy
        time_series = predictions.numpy()
        
        # 检查时间序列的因果性
        causality_violations = 0
        total_tests = 0
        
        # 滑动窗口检查
        window_size = min(10, len(time_series) // 2)
        
        for i in range(len(time_series) - window_size):
            past_window = time_series[i:i+window_size//2]
            future_window = time_series[i+window_size//2:i+window_size]
            
            # 检查未来是否"预测"过去（反因果性）
            if len(past_window) > 1 and len(future_window) > 1:
                # 计算相关性
                corr_result = self._statistical_test(future_window, past_window, "correlation")
                
                # 如果未来与过去高度相关且显著，可能存在反因果性
                if (abs(corr_result['statistic']) > 0.8 and 
                    self._check_significance(corr_result['p_value'])):
                    causality_violations += 1
                
                total_tests += 1
        
        violation_rate = causality_violations / max(total_tests, 1)
        passed = violation_rate < 0.1  # 允许10%的违反率
        
        return {
            'passed': passed,
            'violation_rate': violation_rate,
            'total_tests': total_tests,
            'violations': causality_violations,
            'time_series': time_series
        }
    
    def _test_future_past_influence(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试未来对过去的影响
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 选择两个时间点：t1 < t2
        t_early = 0.3
        t_late = 0.7
        
        # 固定空间位置
        spatial_points = inputs[:100, :2]  # 取前100个空间点
        
        # 早期时间点
        early_inputs = torch.cat([
            spatial_points,
            torch.full((spatial_points.shape[0], 1), t_early)
        ], dim=1)
        
        # 晚期时间点
        late_inputs = torch.cat([
            spatial_points,
            torch.full((spatial_points.shape[0], 1), t_late)
        ], dim=1)
        
        # 计算导数
        early_derivatives = self._compute_derivatives(model, early_inputs)
        late_derivatives = self._compute_derivatives(model, late_inputs)
        
        # 检查早期时间点的值是否依赖于晚期时间点的信息
        # 通过比较时间导数的符号和大小
        early_time_grad = early_derivatives['u_t'].detach().numpy()
        late_time_grad = late_derivatives['u_t'].detach().numpy()
        
        # 计算相关性
        corr_result = self._statistical_test(early_time_grad, late_time_grad, "correlation")
        
        # 如果早期和晚期的时间导数高度相关，可能存在反因果性
        violation_detected = (
            abs(corr_result['statistic']) > 0.9 and
            self._check_significance(corr_result['p_value'])
        )
        
        passed = not violation_detected
        
        return {
            'passed': passed,
            'correlation': corr_result['statistic'],
            'p_value': corr_result['p_value'],
            'violation_detected': violation_detected,
            'early_time_grad': early_time_grad,
            'late_time_grad': late_time_grad
        }
    
    def _test_information_propagation(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试信息传播速度
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 选择两个空间点
        point1 = torch.tensor([[0.0, 0.0, 0.5]])
        point2 = torch.tensor([[1.0, 0.0, 0.5]])
        
        # 计算空间距离
        distance = torch.norm(point2[:, :2] - point1[:, :2]).item()
        
        # 在不同时间计算这两点的值
        time_points = torch.linspace(0.0, 1.0, 50)
        
        values1 = []
        values2 = []
        
        for t in time_points:
            input1 = torch.cat([point1[:, :2], t.unsqueeze(0).unsqueeze(0)], dim=1)
            input2 = torch.cat([point2[:, :2], t.unsqueeze(0).unsqueeze(0)], dim=1)
            
            with torch.no_grad():
                val1 = model(input1)
                val2 = model(input2)
            
            if val1.dim() > 1:
                val1 = val1[:, 0]
            if val2.dim() > 1:
                val2 = val2[:, 0]
            
            values1.append(val1.item())
            values2.append(val2.item())
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # 计算互相关
        cross_corr = signal.correlate(values1, values2, mode='full')
        lags = signal.correlation_lags(len(values1), len(values2), mode='full')
        
        # 找到最大相关性对应的滞后
        max_corr_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[max_corr_idx]
        
        # 计算隐含的传播速度
        time_step = (time_points[1] - time_points[0]).item()
        time_delay = abs(optimal_lag) * time_step
        
        if time_delay > 0:
            implied_speed = distance / time_delay
        else:
            implied_speed = float('inf')
        
        # 检查是否超过物理极限速度
        max_physical_speed = max(self.config.light_speed, self.config.ice_wave_speed)
        speed_violation = implied_speed > max_physical_speed
        
        passed = not speed_violation
        
        return {
            'passed': passed,
            'distance': distance,
            'time_delay': time_delay,
            'implied_speed': implied_speed,
            'max_physical_speed': max_physical_speed,
            'speed_violation': speed_violation,
            'cross_correlation': cross_corr,
            'lags': lags,
            'values1': values1,
            'values2': values2
        }
    
    def _plot_temporal_causality(self, inputs: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制时间因果性验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # 时间序列
            if 'time_series' in result['time_series_causality']:
                time_series = result['time_series_causality']['time_series']
                t_min, t_max = self.config.temporal_domain
                time_points = np.linspace(t_min, t_max, len(time_series))
                
                axes[0].plot(time_points, time_series, 'b-', linewidth=2)
                axes[0].set_title('Time Series at Fixed Location')
                axes[0].set_xlabel('Time')
                axes[0].set_ylabel('Value')
                axes[0].grid(True, alpha=0.3)
            
            # 时间导数比较
            if 'early_time_grad' in result['future_past_influence']:
                early_grad = result['future_past_influence']['early_time_grad']
                late_grad = result['future_past_influence']['late_time_grad']
                
                axes[1].scatter(early_grad, late_grad, alpha=0.6)
                axes[1].plot([early_grad.min(), early_grad.max()], 
                           [early_grad.min(), early_grad.max()], 'r--', alpha=0.5)
                axes[1].set_title('Early vs Late Time Derivatives')
                axes[1].set_xlabel('Early Time Derivative')
                axes[1].set_ylabel('Late Time Derivative')
                axes[1].grid(True, alpha=0.3)
            
            # 信息传播
            if 'values1' in result['information_propagation']:
                values1 = result['information_propagation']['values1']
                values2 = result['information_propagation']['values2']
                time_points = np.linspace(0, 1, len(values1))
                
                axes[2].plot(time_points, values1, 'b-', label='Point 1', linewidth=2)
                axes[2].plot(time_points, values2, 'r-', label='Point 2', linewidth=2)
                axes[2].set_title('Information Propagation Between Points')
                axes[2].set_xlabel('Time')
                axes[2].set_ylabel('Value')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            # 互相关
            if 'cross_correlation' in result['information_propagation']:
                cross_corr = result['information_propagation']['cross_correlation']
                lags = result['information_propagation']['lags']
                
                axes[3].plot(lags, cross_corr, 'g-', linewidth=2)
                axes[3].axvline(0, color='red', linestyle='--', alpha=0.5)
                axes[3].set_title('Cross-Correlation')
                axes[3].set_xlabel('Lag')
                axes[3].set_ylabel('Correlation')
                axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "temporal_causality.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制时间因果性图失败: {e}")

class SpatialCausalityValidator(CausalityValidatorBase):
    """空间因果性验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证空间因果性
        
        检查模型是否违反空间因果性：
        1. 远距离点不应瞬时影响
        2. 空间信息传播有限速度
        3. 局部性原理
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        
        # 测试空间局部性
        locality_results = self._test_spatial_locality(model, inputs)
        
        # 测试空间传播速度
        propagation_results = self._test_spatial_propagation(model, inputs)
        
        # 测试远程相关性
        correlation_results = self._test_long_range_correlation(model, inputs)
        
        # 综合结果
        overall_passed = (
            locality_results['passed'] and
            propagation_results['passed'] and
            correlation_results['passed']
        )
        
        result = {
            'passed': overall_passed,
            'spatial_locality': locality_results,
            'spatial_propagation': propagation_results,
            'long_range_correlation': correlation_results,
            'validation_type': 'spatial_causality'
        }
        
        # 记录日志
        status = "PASSED" if overall_passed else "FAILED"
        self.logger.info(f"空间因果性验证: {status}")
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_spatial_causality(inputs.detach(), result)
        
        return result
    
    def _test_spatial_locality(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试空间局部性
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 选择中心点
        center_point = torch.tensor([[0.0, 0.0, 0.5]])
        
        # 生成不同距离的点
        distances = np.linspace(0.1, 2.0, 20)
        angles = np.linspace(0, 2*np.pi, 8)
        
        influence_scores = []
        
        for distance in distances:
            distance_influences = []
            
            for angle in angles:
                # 生成测试点
                test_point = torch.tensor([[
                    distance * np.cos(angle),
                    distance * np.sin(angle),
                    0.5
                ]])
                
                # 计算中心点和测试点的导数
                center_derivatives = self._compute_derivatives(model, center_point)
                test_derivatives = self._compute_derivatives(model, test_point)
                
                # 计算影响强度（通过梯度相似性）
                center_grad = torch.cat([center_derivatives['u_x'], center_derivatives['u_y']], dim=1)
                test_grad = torch.cat([test_derivatives['u_x'], test_derivatives['u_y']], dim=1)
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(center_grad, test_grad, dim=1).item()
                distance_influences.append(abs(similarity))
            
            # 平均影响强度
            avg_influence = np.mean(distance_influences)
            influence_scores.append(avg_influence)
        
        # 检查影响是否随距离衰减
        influence_scores = np.array(influence_scores)
        
        # 计算衰减趋势
        corr_result = self._statistical_test(distances, influence_scores, "correlation")
        
        # 期望负相关（距离增加，影响减少）
        proper_decay = corr_result['statistic'] < -0.3
        
        passed = proper_decay
        
        return {
            'passed': passed,
            'distances': distances,
            'influence_scores': influence_scores,
            'decay_correlation': corr_result['statistic'],
            'decay_p_value': corr_result['p_value'],
            'proper_decay': proper_decay
        }
    
    def _test_spatial_propagation(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试空间传播速度
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 创建空间网格
        x_range = np.linspace(-1, 1, 21)
        y_range = np.linspace(-1, 1, 21)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 固定时间
        t_fixed = 0.5
        
        # 构建网格输入
        grid_inputs = torch.tensor(np.stack([
            X.flatten(),
            Y.flatten(),
            np.full(X.size, t_fixed)
        ], axis=1), dtype=torch.float32)
        
        # 计算空间导数
        derivatives = self._compute_derivatives(model, grid_inputs)
        
        # 重塑为网格形状
        u_x = derivatives['u_x'].detach().numpy().reshape(X.shape)
        u_y = derivatives['u_y'].detach().numpy().reshape(X.shape)
        
        # 计算梯度幅值
        grad_magnitude = np.sqrt(u_x**2 + u_y**2)
        
        # 检查梯度的空间变化
        # 计算梯度的梯度（二阶导数的近似）
        grad_x_grad = np.gradient(u_x, axis=1)
        grad_y_grad = np.gradient(u_y, axis=0)
        
        # 计算最大梯度变化
        max_grad_change = max(np.max(np.abs(grad_x_grad)), np.max(np.abs(grad_y_grad)))
        
        # 估计隐含的传播速度
        spatial_step = x_range[1] - x_range[0]
        if max_grad_change > 0:
            implied_speed = 1.0 / (max_grad_change * spatial_step)  # 简化估计
        else:
            implied_speed = float('inf')
        
        # 检查是否超过物理极限
        max_physical_speed = self.config.ice_wave_speed
        speed_violation = implied_speed > max_physical_speed
        
        passed = not speed_violation
        
        return {
            'passed': passed,
            'grad_magnitude': grad_magnitude,
            'max_grad_change': max_grad_change,
            'implied_speed': implied_speed,
            'max_physical_speed': max_physical_speed,
            'speed_violation': speed_violation,
            'grid_x': X,
            'grid_y': Y
        }
    
    def _test_long_range_correlation(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        测试长程相关性
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 随机选择点对
        num_pairs = 100
        indices = torch.randperm(inputs.shape[0])[:num_pairs*2]
        
        points1 = inputs[indices[:num_pairs]]
        points2 = inputs[indices[num_pairs:]]
        
        # 计算距离
        distances = torch.norm(points1[:, :2] - points2[:, :2], dim=1).numpy()
        
        # 计算模型输出
        with torch.no_grad():
            outputs1 = model(points1)
            outputs2 = model(points2)
        
        if outputs1.dim() > 1:
            outputs1 = outputs1[:, 0]
        if outputs2.dim() > 1:
            outputs2 = outputs2[:, 0]
        
        values1 = outputs1.numpy()
        values2 = outputs2.numpy()
        
        # 计算相关性
        correlations = []
        for i in range(len(values1)):
            if i < len(values2):
                corr = np.corrcoef(values1[i:i+1], values2[i:i+1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        correlations = np.array(correlations)
        valid_distances = distances[:len(correlations)]
        
        # 检查相关性是否随距离衰减
        if len(correlations) > 10:
            corr_result = self._statistical_test(valid_distances, correlations, "correlation")
            proper_decay = corr_result['statistic'] < -0.2
        else:
            proper_decay = True
            corr_result = {'statistic': 0, 'p_value': 1}
        
        # 检查长程相关性是否过强
        long_range_mask = valid_distances > self.config.max_spatial_distance / 2
        if np.any(long_range_mask):
            long_range_corr = np.mean(correlations[long_range_mask])
            excessive_long_range = long_range_corr > 0.5
        else:
            long_range_corr = 0
            excessive_long_range = False
        
        passed = proper_decay and not excessive_long_range
        
        return {
            'passed': passed,
            'distances': valid_distances,
            'correlations': correlations,
            'decay_correlation': corr_result['statistic'],
            'decay_p_value': corr_result['p_value'],
            'long_range_correlation': long_range_corr,
            'excessive_long_range': excessive_long_range,
            'proper_decay': proper_decay
        }
    
    def _plot_spatial_causality(self, inputs: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制空间因果性验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # 空间局部性
            if 'distances' in result['spatial_locality']:
                distances = result['spatial_locality']['distances']
                influences = result['spatial_locality']['influence_scores']
                
                axes[0].plot(distances, influences, 'bo-', linewidth=2, markersize=6)
                axes[0].set_title('Spatial Locality: Influence vs Distance')
                axes[0].set_xlabel('Distance')
                axes[0].set_ylabel('Influence Score')
                axes[0].grid(True, alpha=0.3)
            
            # 梯度幅值
            if 'grad_magnitude' in result['spatial_propagation']:
                grad_mag = result['spatial_propagation']['grad_magnitude']
                X = result['spatial_propagation']['grid_x']
                Y = result['spatial_propagation']['grid_y']
                
                im1 = axes[1].contourf(X, Y, grad_mag, levels=20, cmap='viridis')
                axes[1].set_title('Gradient Magnitude')
                axes[1].set_xlabel('X')
                axes[1].set_ylabel('Y')
                plt.colorbar(im1, ax=axes[1])
            
            # 长程相关性
            if 'distances' in result['long_range_correlation']:
                distances = result['long_range_correlation']['distances']
                correlations = result['long_range_correlation']['correlations']
                
                axes[2].scatter(distances, correlations, alpha=0.6)
                axes[2].set_title('Long-Range Correlation vs Distance')
                axes[2].set_xlabel('Distance')
                axes[2].set_ylabel('Correlation')
                axes[2].grid(True, alpha=0.3)
            
            # 验证结果总结
            results_text = [
                f"Spatial Locality: {'PASS' if result['spatial_locality']['passed'] else 'FAIL'}",
                f"Spatial Propagation: {'PASS' if result['spatial_propagation']['passed'] else 'FAIL'}",
                f"Long-Range Correlation: {'PASS' if result['long_range_correlation']['passed'] else 'FAIL'}",
                f"Overall: {'PASS' if result['passed'] else 'FAIL'}"
            ]
            
            axes[3].text(0.1, 0.5, '\n'.join(results_text), 
                        transform=axes[3].transAxes, fontsize=12,
                        verticalalignment='center')
            axes[3].set_title('Validation Results')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "spatial_causality.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制空间因果性图失败: {e}")

class GrangerCausalityValidator(CausalityValidatorBase):
    """格兰杰因果性验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证格兰杰因果性
        
        使用格兰杰因果性检验分析变量间的因果关系
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        
        # 生成时间序列数据
        time_series_data = self._generate_time_series_data(model, inputs)
        
        # 执行格兰杰因果性检验
        granger_results = self._perform_granger_test(time_series_data)
        
        # 分析因果网络
        network_results = self._analyze_causal_network(granger_results)
        
        # 检查因果性合理性
        reasonableness_results = self._check_causality_reasonableness(granger_results)
        
        # 综合结果
        overall_passed = reasonableness_results['passed']
        
        result = {
            'passed': overall_passed,
            'granger_tests': granger_results,
            'causal_network': network_results,
            'reasonableness': reasonableness_results,
            'validation_type': 'granger_causality'
        }
        
        # 记录日志
        status = "PASSED" if overall_passed else "FAILED"
        self.logger.info(f"格兰杰因果性验证: {status}")
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_granger_causality(result)
        
        return result
    
    def _generate_time_series_data(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        生成时间序列数据
        
        Args:
            model: 模型
            inputs: 输入数据
            
        Returns:
            Dict[str, np.ndarray]: 时间序列数据
        """
        # 选择几个固定的空间位置
        spatial_locations = [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [-0.5, 0.0]
        ]
        
        # 生成时间序列
        t_min, t_max = self.config.temporal_domain
        time_points = torch.linspace(t_min, t_max, self.config.temporal_window)
        
        time_series = {}
        
        for i, (x, y) in enumerate(spatial_locations):
            series_inputs = torch.stack([
                torch.full_like(time_points, x),
                torch.full_like(time_points, y),
                time_points
            ], dim=1)
            
            with torch.no_grad():
                outputs = model(series_inputs)
            
            # 如果是多输出，取所有输出
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                for j in range(outputs.shape[1]):
                    series_name = f"location_{i}_output_{j}"
                    time_series[series_name] = outputs[:, j].numpy()
            else:
                series_name = f"location_{i}"
                if outputs.dim() > 1:
                    time_series[series_name] = outputs[:, 0].numpy()
                else:
                    time_series[series_name] = outputs.numpy()
        
        return time_series
    
    def _perform_granger_test(self, time_series_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        执行格兰杰因果性检验
        
        Args:
            time_series_data: 时间序列数据
            
        Returns:
            Dict[str, Any]: 格兰杰检验结果
        """
        series_names = list(time_series_data.keys())
        granger_results = {}
        
        # 两两检验
        for i, series1_name in enumerate(series_names):
            for j, series2_name in enumerate(series_names):
                if i != j:
                    series1 = time_series_data[series1_name]
                    series2 = time_series_data[series2_name]
                    
                    # 确保序列长度足够
                    min_length = min(len(series1), len(series2))
                    if min_length < self.config.min_samples_for_test:
                        continue
                    
                    series1 = series1[:min_length]
                    series2 = series2[:min_length]
                    
                    # 构建数据矩阵
                    data = np.column_stack([series2, series1])  # [被解释变量, 解释变量]
                    
                    try:
                        # 执行格兰杰因果性检验
                        test_result = grangercausalitytests(
                            data, 
                            maxlag=self.config.granger_max_lags, 
                            verbose=False
                        )
                        
                        # 提取p值
                        p_values = []
                        for lag in range(1, self.config.granger_max_lags + 1):
                            if lag in test_result:
                                # 使用F检验的p值
                                p_val = test_result[lag][0]['ssr_ftest'][1]
                                p_values.append(p_val)
                        
                        # 取最小p值
                        min_p_value = min(p_values) if p_values else 1.0
                        
                        # 判断是否存在因果关系
                        is_causal = min_p_value < self.config.significance_level
                        
                        granger_results[f"{series1_name}_causes_{series2_name}"] = {
                            'p_value': min_p_value,
                            'is_causal': is_causal,
                            'all_p_values': p_values
                        }
                    
                    except Exception as e:
                        self.logger.warning(f"格兰杰因果性检验失败 {series1_name} -> {series2_name}: {e}")
                        granger_results[f"{series1_name}_causes_{series2_name}"] = {
                            'p_value': 1.0,
                            'is_causal': False,
                            'error': str(e)
                        }
        
        return granger_results
    
    def _analyze_causal_network(self, granger_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析因果网络
        
        Args:
            granger_results: 格兰杰检验结果
            
        Returns:
            Dict[str, Any]: 网络分析结果
        """
        # 构建因果网络图
        G = nx.DiGraph()
        
        # 提取节点
        nodes = set()
        for key in granger_results.keys():
            if '_causes_' in key:
                parts = key.split('_causes_')
                nodes.add(parts[0])
                nodes.add(parts[1])
        
        G.add_nodes_from(nodes)
        
        # 添加边
        causal_edges = []
        for key, result in granger_results.items():
            if '_causes_' in key and result['is_causal']:
                parts = key.split('_causes_')
                source, target = parts[0], parts[1]
                weight = 1 - result['p_value']  # 权重为1-p值
                G.add_edge(source, target, weight=weight)
                causal_edges.append((source, target, weight))
        
        # 网络分析
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0
        
        # 检查循环
        try:
            cycles = list(nx.simple_cycles(G))
            has_cycles = len(cycles) > 0
        except:
            cycles = []
            has_cycles = False
        
        # 计算中心性
        try:
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
        except:
            in_degree_centrality = {}
            out_degree_centrality = {}
        
        return {
            'graph': G,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'causal_edges': causal_edges,
            'cycles': cycles,
            'has_cycles': has_cycles,
            'in_degree_centrality': in_degree_centrality,
            'out_degree_centrality': out_degree_centrality
        }
    
    def _check_causality_reasonableness(self, granger_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查因果性合理性
        
        Args:
            granger_results: 格兰杰检验结果
            
        Returns:
            Dict[str, Any]: 合理性检查结果
        """
        # 统计显著因果关系数量
        total_tests = len(granger_results)
        significant_causalities = sum(1 for r in granger_results.values() if r['is_causal'])
        
        # 计算显著率
        significance_rate = significant_causalities / max(total_tests, 1)
        
        # 检查是否过多或过少显著因果关系
        # 期望在物理系统中有适度的因果关系
        reasonable_rate = 0.1 <= significance_rate <= 0.5
        
        # 检查双向因果关系
        bidirectional_count = 0
        for key1, result1 in granger_results.items():
            if '_causes_' in key1 and result1['is_causal']:
                parts = key1.split('_causes_')
                reverse_key = f"{parts[1]}_causes_{parts[0]}"
                if reverse_key in granger_results and granger_results[reverse_key]['is_causal']:
                    bidirectional_count += 1
        
        bidirectional_rate = bidirectional_count / max(significant_causalities, 1)
        
        # 合理性判断
        passed = reasonable_rate and bidirectional_rate < 0.8  # 不应有太多双向因果
        
        return {
            'passed': passed,
            'total_tests': total_tests,
            'significant_causalities': significant_causalities,
            'significance_rate': significance_rate,
            'reasonable_rate': reasonable_rate,
            'bidirectional_count': bidirectional_count,
            'bidirectional_rate': bidirectional_rate
        }
    
    def _plot_granger_causality(self, result: Dict[str, Any]) -> None:
        """绘制格兰杰因果性验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # 因果网络图
            if 'graph' in result['causal_network']:
                G = result['causal_network']['graph']
                
                if G.number_of_nodes() > 0:
                    pos = nx.spring_layout(G)
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(G, pos, ax=axes[0], 
                                         node_color='lightblue', 
                                         node_size=1000)
                    
                    # 绘制边
                    nx.draw_networkx_edges(G, pos, ax=axes[0], 
                                         edge_color='red', 
                                         arrows=True, 
                                         arrowsize=20)
                    
                    # 绘制标签
                    nx.draw_networkx_labels(G, pos, ax=axes[0], 
                                          font_size=8)
                    
                    axes[0].set_title('Causal Network')
                    axes[0].axis('off')
            
            # p值分布
            granger_results = result['granger_tests']
            p_values = [r['p_value'] for r in granger_results.values() if 'p_value' in r]
            
            if p_values:
                axes[1].hist(p_values, bins=20, alpha=0.7, edgecolor='black')
                axes[1].axvline(self.config.significance_level, color='red', 
                              linestyle='--', label=f'α = {self.config.significance_level}')
                axes[1].set_title('P-value Distribution')
                axes[1].set_xlabel('P-value')
                axes[1].set_ylabel('Frequency')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # 因果关系矩阵
            if granger_results:
                # 提取所有变量名
                variables = set()
                for key in granger_results.keys():
                    if '_causes_' in key:
                        parts = key.split('_causes_')
                        variables.add(parts[0])
                        variables.add(parts[1])
                
                variables = sorted(list(variables))
                n_vars = len(variables)
                
                if n_vars > 1:
                    causality_matrix = np.zeros((n_vars, n_vars))
                    
                    for i, var1 in enumerate(variables):
                        for j, var2 in enumerate(variables):
                            key = f"{var1}_causes_{var2}"
                            if key in granger_results:
                                if granger_results[key]['is_causal']:
                                    causality_matrix[i, j] = 1 - granger_results[key]['p_value']
                    
                    im = axes[2].imshow(causality_matrix, cmap='Reds', vmin=0, vmax=1)
                    axes[2].set_title('Causality Matrix')
                    axes[2].set_xticks(range(n_vars))
                    axes[2].set_yticks(range(n_vars))
                    axes[2].set_xticklabels(variables, rotation=45)
                    axes[2].set_yticklabels(variables)
                    plt.colorbar(im, ax=axes[2])
            
            # 验证结果总结
            reasonableness = result['reasonableness']
            network = result['causal_network']
            
            results_text = [
                f"Total Tests: {reasonableness['total_tests']}",
                f"Significant Causalities: {reasonableness['significant_causalities']}",
                f"Significance Rate: {reasonableness['significance_rate']:.3f}",
                f"Network Density: {network['density']:.3f}",
                f"Has Cycles: {network['has_cycles']}",
                f"Overall: {'PASS' if result['passed'] else 'FAIL'}"
            ]
            
            axes[3].text(0.1, 0.5, '\n'.join(results_text), 
                        transform=axes[3].transAxes, fontsize=10,
                        verticalalignment='center')
            axes[3].set_title('Validation Summary')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "granger_causality.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制格兰杰因果性图失败: {e}")

class CausalityValidator:
    """
    因果性验证器
    
    整合所有因果性验证功能
    """
    
    def __init__(self, config: CausalityConfig = None):
        """
        初始化验证器
        
        Args:
            config: 配置
        """
        if config is None:
            config = CausalityConfig()
        
        self.config = config
        self.validators = {
            CausalityType.TEMPORAL: TemporalCausalityValidator(config),
            CausalityType.SPATIAL: SpatialCausalityValidator(config),
            CausalityType.GRANGER: GrangerCausalityValidator(config)
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate_all(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证所有因果性
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for causality_type, validator in self.validators.items():
            try:
                result = validator.validate(model, test_data)
                results[causality_type.value] = result
                
                if not result.get('passed', False):
                    overall_passed = False
                    
            except Exception as e:
                self.logger.error(f"验证 {causality_type.value} 因果性失败: {e}")
                results[causality_type.value] = {
                    'passed': False,
                    'error': str(e)
                }
                overall_passed = False
        
        # 综合结果
        summary = {
            'overall_passed': overall_passed,
            'num_causalities_tested': len(self.validators),
            'num_causalities_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'tolerance': self.config.tolerance,
            'results': results
        }
        
        self.logger.info(
            f"因果性验证完成: {summary['num_causalities_passed']}/{summary['num_causalities_tested']} 通过, "
            f"总体结果: {'PASSED' if overall_passed else 'FAILED'}"
        )
        
        return summary
    
    def validate_specific(self, model: nn.Module, causality_types: List[CausalityType], 
                         test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证特定因果性
        
        Args:
            model: 模型
            causality_types: 要验证的因果性类型列表
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for causality_type in causality_types:
            if causality_type in self.validators:
                try:
                    result = self.validators[causality_type].validate(model, test_data)
                    results[causality_type.value] = result
                    
                    if not result.get('passed', False):
                        overall_passed = False
                        
                except Exception as e:
                    self.logger.error(f"验证 {causality_type.value} 因果性失败: {e}")
                    results[causality_type.value] = {
                        'passed': False,
                        'error': str(e)
                    }
                    overall_passed = False
            else:
                self.logger.warning(f"不支持的因果性类型: {causality_type.value}")
        
        summary = {
            'overall_passed': overall_passed,
            'num_causalities_tested': len(causality_types),
            'num_causalities_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'results': results
        }
        
        return summary
    
    def generate_report(self, validation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "=" * 60,
            "因果性验证报告",
            "=" * 60,
            f"验证时间: {torch.datetime.now()}",
            f"容差设置: {self.config.tolerance:.2e}",
            f"显著性水平: {self.config.significance_level}",
            "",
            "总体结果:",
            f"  状态: {'通过' if validation_results['overall_passed'] else '失败'}",
            f"  通过率: {validation_results['num_causalities_passed']}/{validation_results['num_causalities_tested']}",
            ""
        ]
        
        # 详细结果
        report_lines.append("详细结果:")
        for causality_name, result in validation_results['results'].items():
            if 'error' in result:
                report_lines.extend([
                    f"  {causality_name.upper()} 因果性:",
                    f"    状态: 错误",
                    f"    错误信息: {result['error']}",
                    ""
                ])
            else:
                status = "通过" if result['passed'] else "失败"
                report_lines.extend([
                    f"  {causality_name.upper()} 因果性:",
                    f"    状态: {status}",
                    ""
                ])
                
                # 添加特定类型的详细信息
                if causality_name == 'temporal':
                    if 'time_series_causality' in result:
                        tsc = result['time_series_causality']
                        report_lines.extend([
                            f"    时间序列因果性: {'通过' if tsc['passed'] else '失败'}",
                            f"    违反率: {tsc['violation_rate']:.3f}"
                        ])
                    
                    if 'information_propagation' in result:
                        ip = result['information_propagation']
                        report_lines.extend([
                            f"    信息传播: {'通过' if ip['passed'] else '失败'}",
                            f"    隐含速度: {ip['implied_speed']:.2e} m/s"
                        ])
                
                elif causality_name == 'spatial':
                    if 'spatial_locality' in result:
                        sl = result['spatial_locality']
                        report_lines.extend([
                            f"    空间局部性: {'通过' if sl['passed'] else '失败'}",
                            f"    衰减相关性: {sl['decay_correlation']:.3f}"
                        ])
                
                elif causality_name == 'granger':
                    if 'reasonableness' in result:
                        r = result['reasonableness']
                        report_lines.extend([
                            f"    显著因果关系: {r['significant_causalities']}/{r['total_tests']}",
                            f"    显著率: {r['significance_rate']:.3f}"
                        ])
                
                report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"验证报告已保存到: {save_path}")
            except Exception as e:
                self.logger.error(f"保存验证报告失败: {e}")
        
        return report_content

def create_causality_validator(config: CausalityConfig = None) -> CausalityValidator:
    """
    创建因果性验证器
    
    Args:
        config: 配置
        
    Returns:
        CausalityValidator: 验证器实例
    """
    return CausalityValidator(config)

if __name__ == "__main__":
    # 测试因果性验证器
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),  # 输入: [x, y, t]
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 2)  # 输出: [温度, 速度]
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # 创建配置
    config = CausalityConfig(
        tolerance=1e-4,
        num_test_points=500,
        temporal_window=30,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_causality_validator(config)
    
    print("=== 因果性验证器测试 ===")
    
    # 验证所有因果性
    results = validator.validate_all(model)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    # 验证特定因果性
    print("\n=== 验证特定因果性 ===")
    specific_results = validator.validate_specific(
        model, 
        [CausalityType.TEMPORAL, CausalityType.SPATIAL]
    )
    
    print(f"特定验证结果: {specific_results['num_causalities_passed']}/{specific_results['num_causalities_tested']} 通过")
    
    print("\n因果性验证器测试完成！")