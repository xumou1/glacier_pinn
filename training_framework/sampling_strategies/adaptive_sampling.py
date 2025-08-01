#!/usr/bin/env python3
"""
自适应采样策略模块

实现各种自适应采样策略用于PINNs训练，包括：
- 基础采样策略
- 残差自适应采样
- 梯度自适应采样
- 不确定性采样
- 多尺度采样
- 时空自适应采样

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

@dataclass
class SamplingConfig:
    """
    采样配置类
    """
    n_samples: int = 1000
    domain_bounds: List[Tuple[float, float]] = None
    adaptive_rate: float = 0.1
    refinement_threshold: float = 1e-3
    max_refinement_levels: int = 5
    batch_size: int = 100
    seed: int = 42

class BaseSampler(ABC):
    """
    基础采样器
    
    所有采样策略的基类
    """
    
    def __init__(self, config: SamplingConfig):
        """
        初始化基础采样器
        
        Args:
            config: 采样配置
        """
        self.config = config
        self.samples_history = []
        self.metrics_history = []
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
    @abstractmethod
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        生成采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 新的采样点
        """
        pass
    
    def update_samples(self, new_samples: torch.Tensor, metrics: Dict[str, float] = None):
        """
        更新采样历史
        
        Args:
            new_samples: 新采样点
            metrics: 采样指标
        """
        self.samples_history.append(new_samples.clone())
        if metrics:
            self.metrics_history.append(metrics)
    
    def get_sample_density(self, samples: torch.Tensor, 
                          grid_resolution: int = 50) -> torch.Tensor:
        """
        计算采样密度
        
        Args:
            samples: 采样点
            grid_resolution: 网格分辨率
            
        Returns:
            torch.Tensor: 采样密度
        """
        if samples.shape[1] != 2:
            raise ValueError("仅支持2D采样密度计算")
        
        # 创建网格
        x_min, x_max = samples[:, 0].min(), samples[:, 0].max()
        y_min, y_max = samples[:, 1].min(), samples[:, 1].max()
        
        x_grid = torch.linspace(x_min, x_max, grid_resolution)
        y_grid = torch.linspace(y_min, y_max, grid_resolution)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        
        # 计算密度
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        distances = torch.cdist(grid_points, samples)
        
        # 使用高斯核估计密度
        bandwidth = 0.1
        density = torch.exp(-distances**2 / (2 * bandwidth**2)).sum(dim=1)
        density = density.reshape(grid_resolution, grid_resolution)
        
        return density

class UniformSampler(BaseSampler):
    """
    均匀采样器
    
    在指定域内均匀采样
    """
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        生成均匀采样点
        
        Args:
            model: 神经网络模型（未使用）
            current_samples: 当前采样点（未使用）
            
        Returns:
            torch.Tensor: 均匀采样点
        """
        if self.config.domain_bounds is None:
            raise ValueError("均匀采样需要指定域边界")
        
        n_dims = len(self.config.domain_bounds)
        samples = torch.zeros(self.config.n_samples, n_dims)
        
        for i, (low, high) in enumerate(self.config.domain_bounds):
            samples[:, i] = torch.rand(self.config.n_samples) * (high - low) + low
        
        return samples

class LatinHypercubeSampler(BaseSampler):
    """
    拉丁超立方采样器
    
    使用拉丁超立方采样策略
    """
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        生成拉丁超立方采样点
        
        Args:
            model: 神经网络模型（未使用）
            current_samples: 当前采样点（未使用）
            
        Returns:
            torch.Tensor: 拉丁超立方采样点
        """
        if self.config.domain_bounds is None:
            raise ValueError("拉丁超立方采样需要指定域边界")
        
        n_dims = len(self.config.domain_bounds)
        n_samples = self.config.n_samples
        
        # 生成拉丁超立方采样
        samples = torch.zeros(n_samples, n_dims)
        
        for i in range(n_dims):
            # 创建等间隔的区间
            intervals = torch.linspace(0, 1, n_samples + 1)
            # 在每个区间内随机采样
            random_offsets = torch.rand(n_samples)
            lhs_samples = intervals[:-1] + random_offsets * (intervals[1] - intervals[0])
            
            # 随机排列
            lhs_samples = lhs_samples[torch.randperm(n_samples)]
            
            # 映射到实际域
            low, high = self.config.domain_bounds[i]
            samples[:, i] = lhs_samples * (high - low) + low
        
        return samples

class ResidualAdaptiveSampler(BaseSampler):
    """
    残差自适应采样器
    
    基于PDE残差进行自适应采样
    """
    
    def __init__(self, config: SamplingConfig, pde_residual_func: Callable):
        """
        初始化残差自适应采样器
        
        Args:
            config: 采样配置
            pde_residual_func: PDE残差函数
        """
        super().__init__(config)
        self.pde_residual_func = pde_residual_func
        self.residual_history = []
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        基于残差生成自适应采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 新的采样点
        """
        if model is None or current_samples is None:
            # 初始采样使用拉丁超立方
            lhs_sampler = LatinHypercubeSampler(self.config)
            return lhs_sampler.generate_samples()
        
        # 计算当前采样点的残差
        with torch.no_grad():
            current_samples.requires_grad_(True)
            predictions = model(current_samples)
            residuals = self.pde_residual_func(current_samples, predictions)
            residual_magnitudes = torch.norm(residuals, dim=-1)
        
        self.residual_history.append(residual_magnitudes.clone())
        
        # 基于残差进行自适应采样
        new_samples = self._adaptive_refinement(current_samples, residual_magnitudes)
        
        return new_samples
    
    def _adaptive_refinement(self, samples: torch.Tensor, 
                           residuals: torch.Tensor) -> torch.Tensor:
        """
        基于残差进行自适应细化
        
        Args:
            samples: 当前采样点
            residuals: 残差大小
            
        Returns:
            torch.Tensor: 细化后的采样点
        """
        # 找到高残差区域
        threshold = torch.quantile(residuals, 1 - self.config.adaptive_rate)
        high_residual_mask = residuals > threshold
        high_residual_points = samples[high_residual_mask]
        
        if len(high_residual_points) == 0:
            # 如果没有高残差点，使用均匀采样
            uniform_sampler = UniformSampler(self.config)
            return uniform_sampler.generate_samples()
        
        # 在高残差点周围生成新采样点
        n_new_samples = min(self.config.n_samples, len(high_residual_points) * 5)
        new_samples = torch.zeros(n_new_samples, samples.shape[1])
        
        for i in range(n_new_samples):
            # 随机选择一个高残差点
            center_idx = torch.randint(0, len(high_residual_points), (1,))
            center = high_residual_points[center_idx]
            
            # 在其周围生成新点
            noise_scale = 0.1  # 可调参数
            noise = torch.randn_like(center) * noise_scale
            new_point = center + noise
            
            # 确保新点在域内
            if self.config.domain_bounds:
                for j, (low, high) in enumerate(self.config.domain_bounds):
                    new_point[:, j] = torch.clamp(new_point[:, j], low, high)
            
            new_samples[i] = new_point
        
        return new_samples

class GradientAdaptiveSampler(BaseSampler):
    """
    梯度自适应采样器
    
    基于梯度信息进行自适应采样
    """
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        基于梯度生成自适应采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 新的采样点
        """
        if model is None or current_samples is None:
            # 初始采样
            lhs_sampler = LatinHypercubeSampler(self.config)
            return lhs_sampler.generate_samples()
        
        # 计算梯度
        current_samples.requires_grad_(True)
        predictions = model(current_samples)
        
        # 计算输出对输入的梯度
        gradients = []
        for i in range(predictions.shape[1]):
            grad = torch.autograd.grad(
                predictions[:, i].sum(), current_samples,
                create_graph=False, retain_graph=True
            )[0]
            gradients.append(grad)
        
        # 计算梯度范数
        gradient_norms = torch.stack([torch.norm(g, dim=1) for g in gradients], dim=1)
        total_gradient_norm = torch.norm(gradient_norms, dim=1)
        
        # 基于梯度进行采样
        new_samples = self._gradient_based_refinement(current_samples, total_gradient_norm)
        
        return new_samples
    
    def _gradient_based_refinement(self, samples: torch.Tensor, 
                                  gradient_norms: torch.Tensor) -> torch.Tensor:
        """
        基于梯度进行细化
        
        Args:
            samples: 当前采样点
            gradient_norms: 梯度范数
            
        Returns:
            torch.Tensor: 细化后的采样点
        """
        # 找到高梯度区域
        threshold = torch.quantile(gradient_norms, 1 - self.config.adaptive_rate)
        high_gradient_mask = gradient_norms > threshold
        high_gradient_points = samples[high_gradient_mask]
        
        if len(high_gradient_points) == 0:
            uniform_sampler = UniformSampler(self.config)
            return uniform_sampler.generate_samples()
        
        # 在高梯度点周围生成新采样点
        n_new_samples = min(self.config.n_samples, len(high_gradient_points) * 3)
        new_samples = torch.zeros(n_new_samples, samples.shape[1])
        
        for i in range(n_new_samples):
            center_idx = torch.randint(0, len(high_gradient_points), (1,))
            center = high_gradient_points[center_idx]
            
            # 在梯度方向和垂直方向采样
            noise_scale = 0.05
            noise = torch.randn_like(center) * noise_scale
            new_point = center + noise
            
            # 域约束
            if self.config.domain_bounds:
                for j, (low, high) in enumerate(self.config.domain_bounds):
                    new_point[:, j] = torch.clamp(new_point[:, j], low, high)
            
            new_samples[i] = new_point
        
        return new_samples

class UncertaintySampler(BaseSampler):
    """
    不确定性采样器
    
    基于模型不确定性进行采样
    """
    
    def __init__(self, config: SamplingConfig, uncertainty_estimator: Callable = None):
        """
        初始化不确定性采样器
        
        Args:
            config: 采样配置
            uncertainty_estimator: 不确定性估计函数
        """
        super().__init__(config)
        self.uncertainty_estimator = uncertainty_estimator or self._default_uncertainty
    
    def _default_uncertainty(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        默认不确定性估计（使用MC Dropout）
        
        Args:
            model: 模型
            x: 输入
            
        Returns:
            torch.Tensor: 不确定性估计
        """
        model.train()  # 启用dropout
        n_samples = 10
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        uncertainty = torch.std(predictions, dim=0).mean(dim=1)
        
        return uncertainty
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        基于不确定性生成采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 新的采样点
        """
        if model is None or current_samples is None:
            lhs_sampler = LatinHypercubeSampler(self.config)
            return lhs_sampler.generate_samples()
        
        # 计算不确定性
        uncertainty = self.uncertainty_estimator(model, current_samples)
        
        # 基于不确定性进行采样
        new_samples = self._uncertainty_based_refinement(current_samples, uncertainty)
        
        return new_samples
    
    def _uncertainty_based_refinement(self, samples: torch.Tensor, 
                                     uncertainty: torch.Tensor) -> torch.Tensor:
        """
        基于不确定性进行细化
        
        Args:
            samples: 当前采样点
            uncertainty: 不确定性
            
        Returns:
            torch.Tensor: 细化后的采样点
        """
        # 找到高不确定性区域
        threshold = torch.quantile(uncertainty, 1 - self.config.adaptive_rate)
        high_uncertainty_mask = uncertainty > threshold
        high_uncertainty_points = samples[high_uncertainty_mask]
        
        if len(high_uncertainty_points) == 0:
            uniform_sampler = UniformSampler(self.config)
            return uniform_sampler.generate_samples()
        
        # 在高不确定性点周围生成新采样点
        n_new_samples = min(self.config.n_samples, len(high_uncertainty_points) * 4)
        new_samples = torch.zeros(n_new_samples, samples.shape[1])
        
        for i in range(n_new_samples):
            center_idx = torch.randint(0, len(high_uncertainty_points), (1,))
            center = high_uncertainty_points[center_idx]
            
            noise_scale = 0.08
            noise = torch.randn_like(center) * noise_scale
            new_point = center + noise
            
            # 域约束
            if self.config.domain_bounds:
                for j, (low, high) in enumerate(self.config.domain_bounds):
                    new_point[:, j] = torch.clamp(new_point[:, j], low, high)
            
            new_samples[i] = new_point
        
        return new_samples

class MultiScaleSampler(BaseSampler):
    """
    多尺度采样器
    
    在不同尺度上进行采样
    """
    
    def __init__(self, config: SamplingConfig, scales: List[float] = None):
        """
        初始化多尺度采样器
        
        Args:
            config: 采样配置
            scales: 尺度列表
        """
        super().__init__(config)
        self.scales = scales or [1.0, 0.5, 0.25, 0.125]
        self.current_scale_idx = 0
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        生成多尺度采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 多尺度采样点
        """
        all_samples = []
        
        for scale in self.scales:
            # 为每个尺度生成采样点
            scale_samples = self._generate_scale_samples(scale)
            all_samples.append(scale_samples)
        
        # 合并所有尺度的采样点
        combined_samples = torch.cat(all_samples, dim=0)
        
        # 如果超过了目标数量，随机选择
        if len(combined_samples) > self.config.n_samples:
            indices = torch.randperm(len(combined_samples))[:self.config.n_samples]
            combined_samples = combined_samples[indices]
        
        return combined_samples
    
    def _generate_scale_samples(self, scale: float) -> torch.Tensor:
        """
        为特定尺度生成采样点
        
        Args:
            scale: 尺度因子
            
        Returns:
            torch.Tensor: 该尺度的采样点
        """
        if self.config.domain_bounds is None:
            raise ValueError("多尺度采样需要指定域边界")
        
        n_dims = len(self.config.domain_bounds)
        n_scale_samples = self.config.n_samples // len(self.scales)
        
        # 调整域边界
        scaled_bounds = []
        for low, high in self.config.domain_bounds:
            center = (low + high) / 2
            half_range = (high - low) / 2 * scale
            scaled_bounds.append((center - half_range, center + half_range))
        
        # 生成该尺度的采样点
        samples = torch.zeros(n_scale_samples, n_dims)
        for i, (low, high) in enumerate(scaled_bounds):
            samples[:, i] = torch.rand(n_scale_samples) * (high - low) + low
        
        return samples

class SpatioTemporalSampler(BaseSampler):
    """
    时空自适应采样器
    
    专门用于时空问题的采样
    """
    
    def __init__(self, config: SamplingConfig, 
                 spatial_dims: int = 2, temporal_weight: float = 0.3):
        """
        初始化时空采样器
        
        Args:
            config: 采样配置
            spatial_dims: 空间维度数
            temporal_weight: 时间维度权重
        """
        super().__init__(config)
        self.spatial_dims = spatial_dims
        self.temporal_weight = temporal_weight
    
    def generate_samples(self, model: nn.Module = None, 
                        current_samples: torch.Tensor = None) -> torch.Tensor:
        """
        生成时空采样点
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            torch.Tensor: 时空采样点
        """
        if self.config.domain_bounds is None:
            raise ValueError("时空采样需要指定域边界")
        
        # 分别处理空间和时间维度
        spatial_samples = self._generate_spatial_samples()
        temporal_samples = self._generate_temporal_samples()
        
        # 组合时空采样点
        spatiotemporal_samples = self._combine_spatiotemporal(spatial_samples, temporal_samples)
        
        return spatiotemporal_samples
    
    def _generate_spatial_samples(self) -> torch.Tensor:
        """
        生成空间采样点
        
        Returns:
            torch.Tensor: 空间采样点
        """
        spatial_bounds = self.config.domain_bounds[:self.spatial_dims]
        n_spatial_samples = int(self.config.n_samples * (1 - self.temporal_weight))
        
        # 使用拉丁超立方采样
        spatial_config = SamplingConfig(
            n_samples=n_spatial_samples,
            domain_bounds=spatial_bounds,
            seed=self.config.seed
        )
        lhs_sampler = LatinHypercubeSampler(spatial_config)
        
        return lhs_sampler.generate_samples()
    
    def _generate_temporal_samples(self) -> torch.Tensor:
        """
        生成时间采样点
        
        Returns:
            torch.Tensor: 时间采样点
        """
        temporal_bounds = self.config.domain_bounds[self.spatial_dims:]
        n_temporal_samples = int(self.config.n_samples * self.temporal_weight)
        
        temporal_samples = torch.zeros(n_temporal_samples, len(temporal_bounds))
        
        for i, (low, high) in enumerate(temporal_bounds):
            # 时间维度使用更密集的采样
            temporal_samples[:, i] = torch.rand(n_temporal_samples) * (high - low) + low
        
        return temporal_samples
    
    def _combine_spatiotemporal(self, spatial_samples: torch.Tensor, 
                               temporal_samples: torch.Tensor) -> torch.Tensor:
        """
        组合时空采样点
        
        Args:
            spatial_samples: 空间采样点
            temporal_samples: 时间采样点
            
        Returns:
            torch.Tensor: 组合的时空采样点
        """
        n_spatial = len(spatial_samples)
        n_temporal = len(temporal_samples)
        
        # 创建笛卡尔积
        spatiotemporal_samples = []
        
        for i in range(min(self.config.n_samples, n_spatial * n_temporal)):
            spatial_idx = i % n_spatial
            temporal_idx = (i // n_spatial) % n_temporal
            
            combined_point = torch.cat([
                spatial_samples[spatial_idx],
                temporal_samples[temporal_idx]
            ])
            spatiotemporal_samples.append(combined_point)
        
        return torch.stack(spatiotemporal_samples)

class AdaptiveSamplingManager:
    """
    自适应采样管理器
    
    管理和协调不同的采样策略
    """
    
    def __init__(self, config: SamplingConfig):
        """
        初始化采样管理器
        
        Args:
            config: 采样配置
        """
        self.config = config
        self.samplers = {}
        self.current_sampler = None
        self.sampling_history = []
        
        # 注册默认采样器
        self._register_default_samplers()
    
    def _register_default_samplers(self):
        """
        注册默认采样器
        """
        self.samplers['uniform'] = UniformSampler(self.config)
        self.samplers['lhs'] = LatinHypercubeSampler(self.config)
        self.samplers['multiscale'] = MultiScaleSampler(self.config)
        self.samplers['spatiotemporal'] = SpatioTemporalSampler(self.config)
    
    def register_sampler(self, name: str, sampler: BaseSampler):
        """
        注册新的采样器
        
        Args:
            name: 采样器名称
            sampler: 采样器实例
        """
        self.samplers[name] = sampler
    
    def set_active_sampler(self, name: str):
        """
        设置活跃采样器
        
        Args:
            name: 采样器名称
        """
        if name not in self.samplers:
            raise ValueError(f"未知的采样器: {name}")
        
        self.current_sampler = self.samplers[name]
    
    def adaptive_sample(self, model: nn.Module = None, 
                       current_samples: torch.Tensor = None,
                       strategy: str = 'auto') -> torch.Tensor:
        """
        自适应采样
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            strategy: 采样策略
            
        Returns:
            torch.Tensor: 新的采样点
        """
        if strategy == 'auto':
            # 自动选择采样策略
            strategy = self._select_optimal_strategy(model, current_samples)
        
        if strategy not in self.samplers:
            raise ValueError(f"未知的采样策略: {strategy}")
        
        sampler = self.samplers[strategy]
        new_samples = sampler.generate_samples(model, current_samples)
        
        # 记录采样历史
        self.sampling_history.append({
            'strategy': strategy,
            'n_samples': len(new_samples),
            'samples': new_samples.clone()
        })
        
        return new_samples
    
    def _select_optimal_strategy(self, model: nn.Module = None, 
                                current_samples: torch.Tensor = None) -> str:
        """
        自动选择最优采样策略
        
        Args:
            model: 神经网络模型
            current_samples: 当前采样点
            
        Returns:
            str: 最优策略名称
        """
        if model is None or current_samples is None:
            return 'lhs'  # 初始采样使用拉丁超立方
        
        # 简单的策略选择逻辑
        n_history = len(self.sampling_history)
        
        if n_history < 5:
            return 'lhs'
        elif n_history < 10:
            return 'multiscale'
        else:
            # 根据问题类型选择
            if current_samples.shape[1] > 2:  # 高维问题
                return 'spatiotemporal'
            else:
                return 'uniform'
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """
        获取采样统计信息
        
        Returns:
            Dict: 采样统计
        """
        if not self.sampling_history:
            return {}
        
        strategies_used = [entry['strategy'] for entry in self.sampling_history]
        strategy_counts = {}
        for strategy in strategies_used:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total_samples = sum(entry['n_samples'] for entry in self.sampling_history)
        
        return {
            'total_sampling_rounds': len(self.sampling_history),
            'total_samples_generated': total_samples,
            'strategy_usage': strategy_counts,
            'average_samples_per_round': total_samples / len(self.sampling_history)
        }

if __name__ == "__main__":
    # 测试采样策略
    torch.manual_seed(42)
    
    # 配置
    config = SamplingConfig(
        n_samples=500,
        domain_bounds=[(-1, 1), (-1, 1), (0, 1)],  # x, y, t
        adaptive_rate=0.2
    )
    
    # 测试不同采样器
    print("测试采样策略...")
    
    # 均匀采样
    uniform_sampler = UniformSampler(config)
    uniform_samples = uniform_sampler.generate_samples()
    print(f"均匀采样: {uniform_samples.shape}")
    
    # 拉丁超立方采样
    lhs_sampler = LatinHypercubeSampler(config)
    lhs_samples = lhs_sampler.generate_samples()
    print(f"拉丁超立方采样: {lhs_samples.shape}")
    
    # 多尺度采样
    multiscale_sampler = MultiScaleSampler(config)
    multiscale_samples = multiscale_sampler.generate_samples()
    print(f"多尺度采样: {multiscale_samples.shape}")
    
    # 时空采样
    spatiotemporal_sampler = SpatioTemporalSampler(config, spatial_dims=2)
    st_samples = spatiotemporal_sampler.generate_samples()
    print(f"时空采样: {st_samples.shape}")
    
    # 测试自适应采样管理器
    print("\n测试自适应采样管理器...")
    manager = AdaptiveSamplingManager(config)
    
    # 创建简单模型用于测试
    model = nn.Sequential(
        nn.Linear(3, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )
    
    # 模拟自适应采样过程
    current_samples = None
    for i in range(5):
        new_samples = manager.adaptive_sample(model, current_samples)
        current_samples = new_samples
        print(f"轮次 {i}: 生成 {len(new_samples)} 个采样点")
    
    # 获取统计信息
    stats = manager.get_sampling_statistics()
    print("\n采样统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试残差自适应采样
    print("\n测试残差自适应采样...")
    
    def simple_pde_residual(x, u):
        """简单的PDE残差函数"""
        # 简化的热方程残差
        return torch.randn_like(u)  # 模拟残差
    
    residual_sampler = ResidualAdaptiveSampler(config, simple_pde_residual)
    
    # 初始采样
    initial_samples = residual_sampler.generate_samples()
    print(f"初始采样: {initial_samples.shape}")
    
    # 自适应采样
    adaptive_samples = residual_sampler.generate_samples(model, initial_samples)
    print(f"自适应采样: {adaptive_samples.shape}")
    
    print("\n采样策略测试完成！")