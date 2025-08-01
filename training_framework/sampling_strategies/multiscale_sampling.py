#!/usr/bin/env python3
"""
多尺度采样策略模块

实现多时空尺度的采样策略，确保PINNs模型能够捕获从短期动态到长期趋势
的多尺度冰川演化过程。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class MultiscaleSampler(ABC):
    """
    多尺度采样器基类
    
    设计多时空尺度的采样策略，确保模型能够同时学习：
    - 空间尺度：从局部冰川特征到区域模式
    - 时间尺度：从季节变化到长期趋势
    """
    
    def __init__(self, 
                 domain_bounds: Dict[str, Tuple[float, float]],
                 scale_hierarchy: Dict[str, List[float]]):
        """
        初始化多尺度采样器
        
        Args:
            domain_bounds: 空间-时间域边界
            scale_hierarchy: 尺度层次结构
                {
                    'spatial_scales': [1.0, 5.0, 25.0, 100.0],  # km
                    'temporal_scales': [0.1, 1.0, 5.0, 20.0]     # years
                }
        """
        self.domain_bounds = domain_bounds
        self.scale_hierarchy = scale_hierarchy
        self.spatial_scales = scale_hierarchy.get('spatial_scales', [1.0, 10.0, 50.0])
        self.temporal_scales = scale_hierarchy.get('temporal_scales', [0.1, 1.0, 10.0])
        
    @abstractmethod
    def sample_points(self, n_points: int, **kwargs) -> Dict[str, jnp.ndarray]:
        """
        生成多尺度采样点
        
        Args:
            n_points: 采样点数量
            **kwargs: 额外参数
            
        Returns:
            采样点字典 {'x': array, 'y': array, 't': array, 'scale_info': dict}
        """
        pass
        
    def _allocate_points_by_scale(self, 
                                 n_points: int,
                                 allocation_strategy: str = 'logarithmic') -> Dict[str, int]:
        """
        按尺度分配采样点数量
        
        Args:
            n_points: 总采样点数量
            allocation_strategy: 分配策略 ('equal', 'logarithmic', 'adaptive')
            
        Returns:
            各尺度的采样点数量
        """
        n_scales = len(self.spatial_scales)
        
        if allocation_strategy == 'equal':
            points_per_scale = n_points // n_scales
            return {f'scale_{i}': points_per_scale for i in range(n_scales)}
            
        elif allocation_strategy == 'logarithmic':
            # 较小尺度分配更多点
            weights = [1.0 / (i + 1) for i in range(n_scales)]
            total_weight = sum(weights)
            allocation = {}
            
            for i, weight in enumerate(weights):
                allocation[f'scale_{i}'] = int(n_points * weight / total_weight)
                
            return allocation
            
        elif allocation_strategy == 'adaptive':
            # TODO: 实现自适应分配策略
            return self._allocate_points_by_scale(n_points, 'logarithmic')
            
        else:
            raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")

class HierarchicalSpatialSampler(MultiscaleSampler):
    """
    分层空间采样器
    
    实现分层的空间采样策略，从细粒度的局部特征到粗粒度的区域模式，
    确保模型能够捕获不同空间尺度的冰川动力学过程。
    """
    
    def __init__(self, 
                 domain_bounds: Dict[str, Tuple[float, float]],
                 scale_hierarchy: Dict[str, List[float]],
                 clustering_centers: Optional[jnp.ndarray] = None):
        super().__init__(domain_bounds, scale_hierarchy)
        self.clustering_centers = clustering_centers
        
    def sample_points(self, 
                     n_points: int,
                     allocation_strategy: str = 'logarithmic',
                     **kwargs) -> Dict[str, jnp.ndarray]:
        """
        生成分层空间采样点
        
        Args:
            n_points: 采样点数量
            allocation_strategy: 点分配策略
            **kwargs: 额外参数
            
        Returns:
            采样点字典
        """
        # 按尺度分配采样点
        point_allocation = self._allocate_points_by_scale(n_points, allocation_strategy)
        
        all_points = {'x': [], 'y': [], 't': [], 'spatial_scale': []}
        
        for scale_idx, (scale_name, n_scale_points) in enumerate(point_allocation.items()):
            if n_scale_points > 0:
                scale_value = self.spatial_scales[scale_idx]
                
                # 为当前尺度生成采样点
                scale_points = self._sample_at_spatial_scale(
                    scale_value, n_scale_points
                )
                
                all_points['x'].append(scale_points['x'])
                all_points['y'].append(scale_points['y'])
                all_points['t'].append(scale_points['t'])
                all_points['spatial_scale'].append(
                    jnp.full(n_scale_points, scale_value)
                )
                
        # 合并所有尺度的采样点
        return {
            'x': jnp.concatenate(all_points['x']) if all_points['x'] else jnp.array([]),
            'y': jnp.concatenate(all_points['y']) if all_points['y'] else jnp.array([]),
            't': jnp.concatenate(all_points['t']) if all_points['t'] else jnp.array([]),
            'spatial_scale': jnp.concatenate(all_points['spatial_scale']) if all_points['spatial_scale'] else jnp.array([])
        }
        
    def _sample_at_spatial_scale(self, 
                                scale: float,
                                n_points: int) -> Dict[str, jnp.ndarray]:
        """
        在特定空间尺度采样
        
        Args:
            scale: 空间尺度 (km)
            n_points: 采样点数量
            
        Returns:
            采样点字典
        """
        if scale <= 1.0:
            # 细尺度：密集局部采样
            return self._fine_scale_sampling(n_points)
        elif scale <= 10.0:
            # 中尺度：聚类采样
            return self._medium_scale_sampling(n_points)
        else:
            # 粗尺度：区域代表性采样
            return self._coarse_scale_sampling(n_points)
            
    def _fine_scale_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        细尺度采样：关注局部特征
        """
        # 在关键区域（如冰川边界、汇流区）进行密集采样
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # 生成聚集在特定区域的采样点
        if self.clustering_centers is not None:
            # 围绕聚类中心采样
            center_idx = jax.random.choice(
                keys[0], len(self.clustering_centers), shape=(n_points,)
            )
            centers = self.clustering_centers[center_idx]
            
            # 在中心周围添加噪声
            noise_scale = 0.5  # km
            noise = jax.random.normal(keys[1], (n_points, 2)) * noise_scale
            
            x = jnp.clip(
                centers[:, 0] + noise[:, 0],
                self.domain_bounds['x'][0],
                self.domain_bounds['x'][1]
            )
            y = jnp.clip(
                centers[:, 1] + noise[:, 1],
                self.domain_bounds['y'][0],
                self.domain_bounds['y'][1]
            )
        else:
            # 均匀采样作为后备
            x = jax.random.uniform(
                keys[0], (n_points,),
                minval=self.domain_bounds['x'][0],
                maxval=self.domain_bounds['x'][1]
            )
            y = jax.random.uniform(
                keys[1], (n_points,),
                minval=self.domain_bounds['y'][0],
                maxval=self.domain_bounds['y'][1]
            )
            
        t = jax.random.uniform(
            keys[2], (n_points,),
            minval=self.domain_bounds['t'][0],
            maxval=self.domain_bounds['t'][1]
        )
        
        return {'x': x, 'y': y, 't': t}
        
    def _medium_scale_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        中尺度采样：平衡局部和区域特征
        """
        # 使用分层采样策略
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)
        
        # 将域分成网格，在每个网格内采样
        n_grid_x = int(np.sqrt(n_points / 4))
        n_grid_y = int(np.sqrt(n_points / 4))
        
        x_grid = jnp.linspace(
            self.domain_bounds['x'][0], 
            self.domain_bounds['x'][1], 
            n_grid_x + 1
        )
        y_grid = jnp.linspace(
            self.domain_bounds['y'][0],
            self.domain_bounds['y'][1],
            n_grid_y + 1
        )
        
        x_points = []
        y_points = []
        
        for i in range(n_grid_x):
            for j in range(n_grid_y):
                # 在每个网格单元内随机采样
                n_cell_points = max(1, n_points // (n_grid_x * n_grid_y))
                
                x_cell = jax.random.uniform(
                    keys[0], (n_cell_points,),
                    minval=x_grid[i],
                    maxval=x_grid[i + 1]
                )
                y_cell = jax.random.uniform(
                    keys[1], (n_cell_points,),
                    minval=y_grid[j],
                    maxval=y_grid[j + 1]
                )
                
                x_points.append(x_cell)
                y_points.append(y_cell)
                
        x = jnp.concatenate(x_points)[:n_points]
        y = jnp.concatenate(y_points)[:n_points]
        
        t = jax.random.uniform(
            keys[2], (len(x),),
            minval=self.domain_bounds['t'][0],
            maxval=self.domain_bounds['t'][1]
        )
        
        return {'x': x, 'y': y, 't': t}
        
    def _coarse_scale_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        粗尺度采样：关注区域代表性
        """
        # 使用拉丁超立方采样确保良好的空间覆盖
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)
        
        # 简化的拉丁超立方采样
        x_indices = jax.random.permutation(keys[0], n_points)
        y_indices = jax.random.permutation(keys[1], n_points)
        
        x = self.domain_bounds['x'][0] + (x_indices + jax.random.uniform(keys[0], (n_points,))) / n_points * (
            self.domain_bounds['x'][1] - self.domain_bounds['x'][0]
        )
        y = self.domain_bounds['y'][0] + (y_indices + jax.random.uniform(keys[1], (n_points,))) / n_points * (
            self.domain_bounds['y'][1] - self.domain_bounds['y'][0]
        )
        
        t = jax.random.uniform(
            keys[2], (n_points,),
            minval=self.domain_bounds['t'][0],
            maxval=self.domain_bounds['t'][1]
        )
        
        return {'x': x, 'y': y, 't': t}

class TemporalMultiscaleSampler(MultiscaleSampler):
    """
    时间多尺度采样器
    
    实现多时间尺度的采样策略，从短期季节变化到长期气候趋势，
    确保模型能够捕获不同时间尺度的冰川演化过程。
    """
    
    def sample_points(self, 
                     n_points: int,
                     temporal_focus: str = 'balanced',
                     **kwargs) -> Dict[str, jnp.ndarray]:
        """
        生成时间多尺度采样点
        
        Args:
            n_points: 采样点数量
            temporal_focus: 时间焦点 ('short_term', 'long_term', 'balanced')
            **kwargs: 额外参数
            
        Returns:
            采样点字典
        """
        if temporal_focus == 'short_term':
            return self._short_term_focused_sampling(n_points)
        elif temporal_focus == 'long_term':
            return self._long_term_focused_sampling(n_points)
        else:  # balanced
            return self._balanced_temporal_sampling(n_points)
            
    def _short_term_focused_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        短期聚焦采样：关注季节和年际变化
        """
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # 在最近几年内密集采样
        recent_years = 5.0  # 最近5年
        t_max = self.domain_bounds['t'][1]
        t_min = max(self.domain_bounds['t'][0], t_max - recent_years)
        
        # 空间均匀采样
        x = jax.random.uniform(
            keys[0], (n_points,),
            minval=self.domain_bounds['x'][0],
            maxval=self.domain_bounds['x'][1]
        )
        y = jax.random.uniform(
            keys[1], (n_points,),
            minval=self.domain_bounds['y'][0],
            maxval=self.domain_bounds['y'][1]
        )
        
        # 时间聚焦在短期
        t = jax.random.uniform(
            keys[2], (n_points,),
            minval=t_min,
            maxval=t_max
        )
        
        return {'x': x, 'y': y, 't': t}
        
    def _long_term_focused_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        长期聚焦采样：关注长期趋势
        """
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)
        
        # 在整个时间域内采样，但偏向早期时间点
        x = jax.random.uniform(
            keys[0], (n_points,),
            minval=self.domain_bounds['x'][0],
            maxval=self.domain_bounds['x'][1]
        )
        y = jax.random.uniform(
            keys[1], (n_points,),
            minval=self.domain_bounds['y'][0],
            maxval=self.domain_bounds['y'][1]
        )
        
        # 使用beta分布偏向早期时间点
        t_uniform = jax.random.beta(keys[2], 0.5, 2.0, (n_points,))
        t = self.domain_bounds['t'][0] + t_uniform * (
            self.domain_bounds['t'][1] - self.domain_bounds['t'][0]
        )
        
        return {'x': x, 'y': y, 't': t}
        
    def _balanced_temporal_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        平衡时间采样：均衡短期和长期
        """
        # 分配点数给不同时间尺度
        n_short = n_points // 3
        n_medium = n_points // 3
        n_long = n_points - n_short - n_medium
        
        # 短期采样
        short_points = self._short_term_focused_sampling(n_short)
        
        # 中期采样
        medium_points = self._medium_term_sampling(n_medium)
        
        # 长期采样
        long_points = self._long_term_focused_sampling(n_long)
        
        # 合并采样点
        return {
            'x': jnp.concatenate([short_points['x'], medium_points['x'], long_points['x']]),
            'y': jnp.concatenate([short_points['y'], medium_points['y'], long_points['y']]),
            't': jnp.concatenate([short_points['t'], medium_points['t'], long_points['t']])
        }
        
    def _medium_term_sampling(self, n_points: int) -> Dict[str, jnp.ndarray]:
        """
        中期采样：关注年代际变化
        """
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)
        
        x = jax.random.uniform(
            keys[0], (n_points,),
            minval=self.domain_bounds['x'][0],
            maxval=self.domain_bounds['x'][1]
        )
        y = jax.random.uniform(
            keys[1], (n_points,),
            minval=self.domain_bounds['y'][0],
            maxval=self.domain_bounds['y'][1]
        )
        t = jax.random.uniform(
            keys[2], (n_points,),
            minval=self.domain_bounds['t'][0],
            maxval=self.domain_bounds['t'][1]
        )
        
        return {'x': x, 'y': y, 't': t}

class AdaptiveMultiscaleSampler(MultiscaleSampler):
    """
    自适应多尺度采样器
    
    根据模型训练过程中的表现动态调整采样策略，
    在模型表现较差的尺度增加采样密度。
    """
    
    def __init__(self, 
                 domain_bounds: Dict[str, Tuple[float, float]],
                 scale_hierarchy: Dict[str, List[float]],
                 adaptation_rate: float = 0.1):
        super().__init__(domain_bounds, scale_hierarchy)
        self.adaptation_rate = adaptation_rate
        self.scale_performance_history = []
        
    def sample_points(self, 
                     n_points: int,
                     performance_feedback: Optional[Dict[str, float]] = None,
                     **kwargs) -> Dict[str, jnp.ndarray]:
        """
        生成自适应多尺度采样点
        
        Args:
            n_points: 采样点数量
            performance_feedback: 各尺度的性能反馈
            **kwargs: 额外参数
            
        Returns:
            采样点字典
        """
        if performance_feedback is not None:
            self.scale_performance_history.append(performance_feedback)
            
        # 计算自适应权重
        adaptive_weights = self._compute_adaptive_weights()
        
        # 基于自适应权重分配采样点
        point_allocation = self._adaptive_point_allocation(n_points, adaptive_weights)
        
        # 生成采样点
        return self._generate_adaptive_samples(point_allocation)
        
    def _compute_adaptive_weights(self) -> Dict[str, float]:
        """
        计算自适应权重
        
        Returns:
            各尺度的自适应权重
        """
        if not self.scale_performance_history:
            # 如果没有历史数据，使用均匀权重
            n_scales = len(self.spatial_scales)
            return {f'scale_{i}': 1.0 / n_scales for i in range(n_scales)}
            
        # 基于最近的性能反馈计算权重
        recent_performance = self.scale_performance_history[-1]
        
        # 性能越差，权重越高（需要更多采样）
        weights = {}
        total_inverse_performance = 0.0
        
        for scale_name, performance in recent_performance.items():
            inverse_performance = 1.0 / (performance + 1e-8)
            weights[scale_name] = inverse_performance
            total_inverse_performance += inverse_performance
            
        # 归一化权重
        for scale_name in weights:
            weights[scale_name] /= total_inverse_performance
            
        return weights
        
    def _adaptive_point_allocation(self, 
                                  n_points: int,
                                  weights: Dict[str, float]) -> Dict[str, int]:
        """
        自适应点分配
        
        Args:
            n_points: 总采样点数量
            weights: 各尺度权重
            
        Returns:
            各尺度的采样点数量
        """
        allocation = {}
        
        for scale_name, weight in weights.items():
            allocation[scale_name] = int(n_points * weight)
            
        # 确保总数正确
        total_allocated = sum(allocation.values())
        if total_allocated < n_points:
            # 将剩余点分配给权重最高的尺度
            max_weight_scale = max(weights.keys(), key=lambda k: weights[k])
            allocation[max_weight_scale] += n_points - total_allocated
            
        return allocation
        
    def _generate_adaptive_samples(self, 
                                  point_allocation: Dict[str, int]) -> Dict[str, jnp.ndarray]:
        """
        生成自适应采样点
        
        Args:
            point_allocation: 点分配方案
            
        Returns:
            采样点字典
        """
        # TODO: 实现更复杂的自适应采样逻辑
        # 这里使用简单的均匀采样作为占位符
        total_points = sum(point_allocation.values())
        
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        x = jax.random.uniform(
            keys[0], (total_points,),
            minval=self.domain_bounds['x'][0],
            maxval=self.domain_bounds['x'][1]
        )
        y = jax.random.uniform(
            keys[1], (total_points,),
            minval=self.domain_bounds['y'][0],
            maxval=self.domain_bounds['y'][1]
        )
        t = jax.random.uniform(
            keys[2], (total_points,),
            minval=self.domain_bounds['t'][0],
            maxval=self.domain_bounds['t'][1]
        )
        
        return {'x': x, 'y': y, 't': t}

def create_multiscale_sampler(sampler_type: str,
                             domain_bounds: Dict[str, Tuple[float, float]],
                             scale_hierarchy: Dict[str, List[float]],
                             **kwargs) -> MultiscaleSampler:
    """
    工厂函数：创建多尺度采样器
    
    Args:
        sampler_type: 采样器类型 ('spatial', 'temporal', 'adaptive')
        domain_bounds: 域边界
        scale_hierarchy: 尺度层次结构
        **kwargs: 额外参数
        
    Returns:
        多尺度采样器实例
    """
    if sampler_type == 'spatial':
        return HierarchicalSpatialSampler(domain_bounds, scale_hierarchy, **kwargs)
    elif sampler_type == 'temporal':
        return TemporalMultiscaleSampler(domain_bounds, scale_hierarchy, **kwargs)
    elif sampler_type == 'adaptive':
        return AdaptiveMultiscaleSampler(domain_bounds, scale_hierarchy, **kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

if __name__ == "__main__":
    # 测试代码
    domain_bounds = {
        'x': (0.0, 100.0),
        'y': (0.0, 100.0),
        't': (0.0, 20.0)
    }
    
    scale_hierarchy = {
        'spatial_scales': [1.0, 5.0, 25.0],
        'temporal_scales': [0.1, 1.0, 5.0]
    }
    
    # 测试分层空间采样器
    sampler = create_multiscale_sampler(
        'spatial', domain_bounds, scale_hierarchy
    )
    points = sampler.sample_points(1000)
    
    print(f"Generated {len(points['x'])} multiscale sampling points")
    print(f"X range: [{points['x'].min():.2f}, {points['x'].max():.2f}]")
    print(f"Y range: [{points['y'].min():.2f}, {points['y'].max():.2f}]")
    print(f"T range: [{points['t'].min():.2f}, {points['t'].max():.2f}]")
    
    if 'spatial_scale' in points:
        print(f"Spatial scales used: {jnp.unique(points['spatial_scale'])}")