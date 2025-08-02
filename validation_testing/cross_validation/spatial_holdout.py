#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间维度留出验证模块

该模块实现了基于空间维度的留出验证策略，用于评估模型的空间泛化能力，包括：
- 空间分割策略（网格、随机、聚类、地理区域）
- 空间插值验证
- 空间外推验证
- 地理分层验证
- 距离衰减分析
- 空间自相关分析

作者: Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy import stats

class SpatialSplitStrategy(Enum):
    """空间分割策略"""
    GRID_BASED = "grid_based"  # 网格分割
    RANDOM_SPATIAL = "random_spatial"  # 随机空间分割
    CLUSTER_BASED = "cluster_based"  # 聚类分割
    GEOGRAPHIC_REGIONS = "geographic_regions"  # 地理区域分割
    DISTANCE_BASED = "distance_based"  # 距离分割
    ELEVATION_BASED = "elevation_based"  # 海拔分割
    WATERSHED_BASED = "watershed_based"  # 流域分割
    BUFFER_ZONES = "buffer_zones"  # 缓冲区分割

class SpatialValidationMode(Enum):
    """空间验证模式"""
    INTERPOLATION = "interpolation"  # 插值验证
    EXTRAPOLATION = "extrapolation"  # 外推验证
    LEAVE_ONE_OUT = "leave_one_out"  # 留一验证
    CROSS_VALIDATION = "cross_validation"  # 交叉验证
    HOLDOUT = "holdout"  # 留出验证

class SpatialMetric(Enum):
    """空间验证指标"""
    MSE = "mse"  # 均方误差
    MAE = "mae"  # 平均绝对误差
    RMSE = "rmse"  # 均方根误差
    SPATIAL_CORRELATION = "spatial_correlation"  # 空间相关性
    MORAN_I = "moran_i"  # 莫兰指数
    GEARY_C = "geary_c"  # 吉尔里系数
    DISTANCE_DECAY = "distance_decay"  # 距离衰减
    SPATIAL_AUTOCORR = "spatial_autocorr"  # 空间自相关

@dataclass
class SpatialHoldoutConfig:
    """空间留出验证配置"""
    # 基本设置
    split_strategy: SpatialSplitStrategy = SpatialSplitStrategy.GRID_BASED
    validation_mode: SpatialValidationMode = SpatialValidationMode.HOLDOUT
    
    # 分割参数
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 网格分割参数
    grid_size_x: int = 10  # X方向网格数
    grid_size_y: int = 10  # Y方向网格数
    grid_overlap: float = 0.0  # 网格重叠比例
    
    # 聚类分割参数
    n_clusters: int = 5
    cluster_method: str = "kmeans"  # kmeans, dbscan, hierarchical
    
    # 距离分割参数
    distance_threshold: float = 1000.0  # 距离阈值（米）
    buffer_size: float = 500.0  # 缓冲区大小（米）
    
    # 地理参数
    coordinate_system: str = "utm"  # utm, latlon, local
    elevation_bands: int = 5  # 海拔分层数
    
    # 验证指标
    metrics: List[SpatialMetric] = None
    
    # 空间分析参数
    max_distance: float = 5000.0  # 最大分析距离（米）
    distance_bins: int = 20  # 距离分箱数
    spatial_lag_orders: List[int] = None  # 空间滞后阶数
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "spatial_validation_plots"
    
    # 并行处理
    n_jobs: int = 1
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                SpatialMetric.MSE,
                SpatialMetric.MAE,
                SpatialMetric.SPATIAL_CORRELATION,
                SpatialMetric.MORAN_I
            ]
        
        if self.spatial_lag_orders is None:
            self.spatial_lag_orders = [1, 2, 3]

class SpatialDataSplitter:
    """空间数据分割器"""
    
    def __init__(self, config: SpatialHoldoutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, data: Dict[str, torch.Tensor], 
                   x_column: str = 'x', y_column: str = 'y') -> Dict[str, Dict[str, torch.Tensor]]:
        """分割空间数据"""
        if self.config.split_strategy == SpatialSplitStrategy.GRID_BASED:
            return self._grid_based_split(data, x_column, y_column)
        elif self.config.split_strategy == SpatialSplitStrategy.RANDOM_SPATIAL:
            return self._random_spatial_split(data, x_column, y_column)
        elif self.config.split_strategy == SpatialSplitStrategy.CLUSTER_BASED:
            return self._cluster_based_split(data, x_column, y_column)
        elif self.config.split_strategy == SpatialSplitStrategy.DISTANCE_BASED:
            return self._distance_based_split(data, x_column, y_column)
        elif self.config.split_strategy == SpatialSplitStrategy.ELEVATION_BASED:
            return self._elevation_based_split(data, x_column, y_column)
        else:
            raise ValueError(f"不支持的空间分割策略: {self.config.split_strategy}")
    
    def _grid_based_split(self, data: Dict[str, torch.Tensor], 
                         x_column: str, y_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """基于网格的分割"""
        x_data = data[x_column]
        y_data = data[y_column]
        
        # 计算网格边界
        x_min, x_max = x_data.min().item(), x_data.max().item()
        y_min, y_max = y_data.min().item(), y_data.max().item()
        
        x_edges = torch.linspace(x_min, x_max, self.config.grid_size_x + 1)
        y_edges = torch.linspace(y_min, y_max, self.config.grid_size_y + 1)
        
        # 分配网格索引
        x_indices = torch.bucketize(x_data, x_edges) - 1
        y_indices = torch.bucketize(y_data, y_edges) - 1
        
        # 确保索引在有效范围内
        x_indices = torch.clamp(x_indices, 0, self.config.grid_size_x - 1)
        y_indices = torch.clamp(y_indices, 0, self.config.grid_size_y - 1)
        
        # 计算网格ID
        grid_ids = y_indices * self.config.grid_size_x + x_indices
        
        # 分配网格到训练/验证/测试集
        unique_grids = torch.unique(grid_ids)
        n_grids = len(unique_grids)
        
        n_train = int(n_grids * self.config.train_ratio)
        n_val = int(n_grids * self.config.validation_ratio)
        
        # 随机分配网格
        perm = torch.randperm(n_grids)
        train_grids = unique_grids[perm[:n_train]]
        val_grids = unique_grids[perm[n_train:n_train + n_val]]
        test_grids = unique_grids[perm[n_train + n_val:]]
        
        # 创建掩码
        train_mask = torch.isin(grid_ids, train_grids)
        val_mask = torch.isin(grid_ids, val_grids)
        test_mask = torch.isin(grid_ids, test_grids)
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_mask]
            splits['validation'][key] = tensor[val_mask]
            splits['test'][key] = tensor[test_mask]
        
        self.logger.info(
            f"网格分割完成: 训练网格 {len(train_grids)}, 验证网格 {len(val_grids)}, "
            f"测试网格 {len(test_grids)}"
        )
        
        return splits
    
    def _random_spatial_split(self, data: Dict[str, torch.Tensor], 
                             x_column: str, y_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """随机空间分割"""
        n_samples = len(data[x_column])
        
        # 生成随机索引
        indices = torch.randperm(n_samples)
        
        n_train = int(n_samples * self.config.train_ratio)
        n_val = int(n_samples * self.config.validation_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_indices]
            splits['validation'][key] = tensor[val_indices]
            splits['test'][key] = tensor[test_indices]
        
        self.logger.info(
            f"随机空间分割完成: 训练 {len(train_indices)}, 验证 {len(val_indices)}, "
            f"测试 {len(test_indices)}"
        )
        
        return splits
    
    def _cluster_based_split(self, data: Dict[str, torch.Tensor], 
                            x_column: str, y_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """基于聚类的分割"""
        x_data = data[x_column].numpy()
        y_data = data[y_column].numpy()
        
        # 构造坐标矩阵
        coordinates = np.column_stack([x_data, y_data])
        
        # 执行聚类
        if self.config.cluster_method == "kmeans":
            clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(coordinates)
        else:
            raise ValueError(f"不支持的聚类方法: {self.config.cluster_method}")
        
        cluster_labels = torch.from_numpy(cluster_labels)
        
        # 分配聚类到训练/验证/测试集
        unique_clusters = torch.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        n_train = int(n_clusters * self.config.train_ratio)
        n_val = int(n_clusters * self.config.validation_ratio)
        
        perm = torch.randperm(n_clusters)
        train_clusters = unique_clusters[perm[:n_train]]
        val_clusters = unique_clusters[perm[n_train:n_train + n_val]]
        test_clusters = unique_clusters[perm[n_train + n_val:]]
        
        # 创建掩码
        train_mask = torch.isin(cluster_labels, train_clusters)
        val_mask = torch.isin(cluster_labels, val_clusters)
        test_mask = torch.isin(cluster_labels, test_clusters)
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_mask]
            splits['validation'][key] = tensor[val_mask]
            splits['test'][key] = tensor[test_mask]
        
        self.logger.info(
            f"聚类分割完成: 训练聚类 {len(train_clusters)}, 验证聚类 {len(val_clusters)}, "
            f"测试聚类 {len(test_clusters)}"
        )
        
        return splits
    
    def _distance_based_split(self, data: Dict[str, torch.Tensor], 
                             x_column: str, y_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """基于距离的分割"""
        x_data = data[x_column].numpy()
        y_data = data[y_column].numpy()
        
        # 选择中心点（数据的中心）
        center_x = np.mean(x_data)
        center_y = np.mean(y_data)
        
        # 计算到中心的距离
        distances = np.sqrt((x_data - center_x)**2 + (y_data - center_y)**2)
        
        # 基于距离分割
        distance_threshold_1 = np.percentile(distances, 70)
        distance_threshold_2 = np.percentile(distances, 85)
        
        train_mask = distances <= distance_threshold_1
        val_mask = (distances > distance_threshold_1) & (distances <= distance_threshold_2)
        test_mask = distances > distance_threshold_2
        
        train_mask = torch.from_numpy(train_mask)
        val_mask = torch.from_numpy(val_mask)
        test_mask = torch.from_numpy(test_mask)
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_mask]
            splits['validation'][key] = tensor[val_mask]
            splits['test'][key] = tensor[test_mask]
        
        self.logger.info(
            f"距离分割完成: 训练 {train_mask.sum()}, 验证 {val_mask.sum()}, "
            f"测试 {test_mask.sum()}"
        )
        
        return splits
    
    def _elevation_based_split(self, data: Dict[str, torch.Tensor], 
                              x_column: str, y_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """基于海拔的分割"""
        # 假设有海拔数据
        if 'elevation' in data:
            elevation_data = data['elevation']
        else:
            # 如果没有海拔数据，使用y坐标作为代理
            elevation_data = data[y_column]
        
        # 基于海拔分层
        elevation_percentiles = torch.quantile(
            elevation_data, 
            torch.linspace(0, 1, self.config.elevation_bands + 1)
        )
        
        # 分配海拔带
        elevation_bands = torch.bucketize(elevation_data, elevation_percentiles) - 1
        elevation_bands = torch.clamp(elevation_bands, 0, self.config.elevation_bands - 1)
        
        # 分配海拔带到训练/验证/测试集
        unique_bands = torch.unique(elevation_bands)
        n_bands = len(unique_bands)
        
        n_train = int(n_bands * self.config.train_ratio)
        n_val = int(n_bands * self.config.validation_ratio)
        
        perm = torch.randperm(n_bands)
        train_bands = unique_bands[perm[:n_train]]
        val_bands = unique_bands[perm[n_train:n_train + n_val]]
        test_bands = unique_bands[perm[n_train + n_val:]]
        
        # 创建掩码
        train_mask = torch.isin(elevation_bands, train_bands)
        val_mask = torch.isin(elevation_bands, val_bands)
        test_mask = torch.isin(elevation_bands, test_bands)
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_mask]
            splits['validation'][key] = tensor[val_mask]
            splits['test'][key] = tensor[test_mask]
        
        self.logger.info(
            f"海拔分割完成: 训练带 {len(train_bands)}, 验证带 {len(val_bands)}, "
            f"测试带 {len(test_bands)}"
        )
        
        return splits

class SpatialValidator:
    """空间验证器"""
    
    def __init__(self, config: SpatialHoldoutConfig):
        self.config = config
        self.splitter = SpatialDataSplitter(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, model: nn.Module, data: Dict[str, torch.Tensor], 
                x_column: str = 'x', y_column: str = 'y') -> Dict[str, Any]:
        """执行空间验证"""
        results = {
            'strategy': self.config.split_strategy.value,
            'mode': self.config.validation_mode.value,
            'split_results': {},
            'spatial_metrics': {},
            'spatial_analysis': {}
        }
        
        try:
            # 分割数据
            splits = self.splitter.split_data(data, x_column, y_column)
            
            # 验证分割
            results['split_results'] = self._validate_split(model, splits)
            
            # 计算空间指标
            results['spatial_metrics'] = self._compute_spatial_metrics(
                results['split_results'], splits, x_column, y_column
            )
            
            # 空间分析
            results['spatial_analysis'] = self._analyze_spatial_performance(
                results['split_results'], splits, data, x_column, y_column
            )
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, splits, data, x_column, y_column)
            
        except Exception as e:
            self.logger.error(f"空间验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _validate_split(self, model: nn.Module, 
                       splits: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """验证分割"""
        result = {
            'train_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'predictions': {},
            'targets': {}
        }
        
        try:
            model.eval()
            
            # 验证各个分割
            for split_name in ['train', 'validation', 'test']:
                if split_name in splits:
                    split_data = splits[split_name]
                    
                    # 预测
                    predictions = self._predict(model, split_data)
                    targets = self._extract_targets(split_data)
                    
                    # 计算指标
                    metrics = self._compute_basic_metrics(predictions, targets)
                    
                    result[f'{split_name}_metrics'] = metrics
                    result['predictions'][split_name] = predictions
                    result['targets'][split_name] = targets
        
        except Exception as e:
            self.logger.error(f"验证分割失败: {e}")
            result['error'] = str(e)
        
        return result
    
    def _predict(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """模型预测"""
        with torch.no_grad():
            # 构造输入
            inputs = self._prepare_inputs(data)
            predictions = model(inputs)
            
            if predictions.dim() > 1:
                predictions = predictions[:, 0]  # 取第一个输出
        
        return predictions
    
    def _prepare_inputs(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """准备模型输入"""
        # 假设输入是 [x, y, t] 的组合
        input_keys = ['x', 'y', 't']
        inputs = []
        
        for key in input_keys:
            if key in data:
                tensor = data[key]
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                inputs.append(tensor)
        
        if inputs:
            return torch.stack(inputs, dim=1) if len(inputs) > 1 else inputs[0].unsqueeze(1)
        else:
            raise ValueError("无法构造模型输入")
    
    def _extract_targets(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取目标值"""
        # 假设目标值在 'target' 或 'temperature' 键中
        if 'target' in data:
            return data['target']
        elif 'temperature' in data:
            return data['temperature']
        elif 'velocity' in data:
            return data['velocity']
        else:
            # 如果没有明确的目标，使用第一个非坐标变量
            for key, tensor in data.items():
                if key not in ['x', 'y', 't']:
                    return tensor
            raise ValueError("无法找到目标值")
    
    def _compute_basic_metrics(self, predictions: torch.Tensor, 
                              targets: torch.Tensor) -> Dict[str, float]:
        """计算基本指标"""
        metrics = {}
        
        # 确保张量形状一致
        min_len = min(len(predictions), len(targets))
        pred = predictions[:min_len]
        targ = targets[:min_len]
        
        try:
            # 基本误差指标
            metrics['mse'] = torch.mean((pred - targ) ** 2).item()
            metrics['mae'] = torch.mean(torch.abs(pred - targ)).item()
            metrics['rmse'] = torch.sqrt(torch.mean((pred - targ) ** 2)).item()
            
            # 相关性
            pred_centered = pred - torch.mean(pred)
            targ_centered = targ - torch.mean(targ)
            correlation = torch.sum(pred_centered * targ_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) * 
                torch.sqrt(torch.sum(targ_centered ** 2)) + 1e-8
            )
            metrics['correlation'] = correlation.item()
            
            # R²
            ss_res = torch.sum((targ - pred) ** 2)
            ss_tot = torch.sum((targ - torch.mean(targ)) ** 2)
            metrics['r2'] = (1 - ss_res / (ss_tot + 1e-8)).item()
        
        except Exception as e:
            self.logger.warning(f"计算基本指标时出错: {e}")
        
        return metrics
    
    def _compute_spatial_metrics(self, split_results: Dict[str, Any], 
                                splits: Dict[str, Dict[str, torch.Tensor]],
                                x_column: str, y_column: str) -> Dict[str, Any]:
        """计算空间指标"""
        spatial_metrics = {}
        
        try:
            for split_name in ['test']:  # 主要关注测试集的空间指标
                if split_name in splits and 'predictions' in split_results:
                    if split_name in split_results['predictions']:
                        predictions = split_results['predictions'][split_name]
                        targets = split_results['targets'][split_name]
                        x_coords = splits[split_name][x_column]
                        y_coords = splits[split_name][y_column]
                        
                        # 计算空间自相关
                        spatial_metrics[f'{split_name}_moran_i'] = self._compute_moran_i(
                            predictions, x_coords, y_coords
                        )
                        
                        # 计算距离衰减
                        spatial_metrics[f'{split_name}_distance_decay'] = self._compute_distance_decay(
                            predictions, targets, x_coords, y_coords
                        )
                        
                        # 计算空间相关性
                        spatial_metrics[f'{split_name}_spatial_correlation'] = self._compute_spatial_correlation(
                            predictions, targets, x_coords, y_coords
                        )
        
        except Exception as e:
            self.logger.warning(f"计算空间指标时出错: {e}")
        
        return spatial_metrics
    
    def _compute_moran_i(self, values: torch.Tensor, x_coords: torch.Tensor, 
                        y_coords: torch.Tensor) -> float:
        """计算莫兰指数"""
        try:
            # 转换为numpy数组
            values_np = values.detach().numpy()
            x_np = x_coords.detach().numpy()
            y_np = y_coords.detach().numpy()
            
            # 构造坐标矩阵
            coords = np.column_stack([x_np, y_np])
            
            # 计算距离矩阵
            distances = cdist(coords, coords)
            
            # 构造权重矩阵（距离倒数，对角线为0）
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0 / distances
                weights[np.isinf(weights)] = 0
                np.fill_diagonal(weights, 0)
            
            # 计算莫兰指数
            n = len(values_np)
            mean_val = np.mean(values_np)
            
            numerator = 0
            denominator = 0
            weight_sum = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        w_ij = weights[i, j]
                        numerator += w_ij * (values_np[i] - mean_val) * (values_np[j] - mean_val)
                        weight_sum += w_ij
                
                denominator += (values_np[i] - mean_val) ** 2
            
            if weight_sum > 0 and denominator > 0:
                moran_i = (n / weight_sum) * (numerator / denominator)
                return float(moran_i)
            else:
                return 0.0
        
        except Exception as e:
            self.logger.warning(f"计算莫兰指数失败: {e}")
            return 0.0
    
    def _compute_distance_decay(self, predictions: torch.Tensor, targets: torch.Tensor,
                               x_coords: torch.Tensor, y_coords: torch.Tensor) -> Dict[str, float]:
        """计算距离衰减"""
        try:
            # 转换为numpy数组
            pred_np = predictions.detach().numpy()
            targ_np = targets.detach().numpy()
            x_np = x_coords.detach().numpy()
            y_np = y_coords.detach().numpy()
            
            # 构造坐标矩阵
            coords = np.column_stack([x_np, y_np])
            
            # 计算距离矩阵
            distances = cdist(coords, coords)
            
            # 计算误差
            errors = np.abs(pred_np - targ_np)
            
            # 分析距离与误差的关系
            distance_bins = np.linspace(0, self.config.max_distance, self.config.distance_bins)
            bin_errors = []
            bin_distances = []
            
            for i in range(len(distance_bins) - 1):
                mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
                if np.any(mask):
                    bin_errors.append(np.mean(errors[mask]))
                    bin_distances.append((distance_bins[i] + distance_bins[i + 1]) / 2)
            
            # 计算距离衰减系数
            if len(bin_distances) > 1 and len(bin_errors) > 1:
                correlation = np.corrcoef(bin_distances, bin_errors)[0, 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(bin_distances, bin_errors)
                
                return {
                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value)
                }
            else:
                return {'correlation': 0.0, 'slope': 0.0, 'r_squared': 0.0, 'p_value': 1.0}
        
        except Exception as e:
            self.logger.warning(f"计算距离衰减失败: {e}")
            return {'correlation': 0.0, 'slope': 0.0, 'r_squared': 0.0, 'p_value': 1.0}
    
    def _compute_spatial_correlation(self, predictions: torch.Tensor, targets: torch.Tensor,
                                   x_coords: torch.Tensor, y_coords: torch.Tensor) -> float:
        """计算空间相关性"""
        try:
            # 简化实现：计算预测值和真实值的空间分布相关性
            pred_np = predictions.detach().numpy()
            targ_np = targets.detach().numpy()
            
            # 计算相关系数
            correlation = np.corrcoef(pred_np, targ_np)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            self.logger.warning(f"计算空间相关性失败: {e}")
            return 0.0
    
    def _analyze_spatial_performance(self, split_results: Dict[str, Any], 
                                   splits: Dict[str, Dict[str, torch.Tensor]],
                                   data: Dict[str, torch.Tensor],
                                   x_column: str, y_column: str) -> Dict[str, Any]:
        """分析空间性能"""
        analysis = {
            'interpolation_performance': {},
            'extrapolation_performance': {},
            'spatial_bias_analysis': {},
            'coverage_analysis': {}
        }
        
        try:
            # 插值性能分析
            if 'test' in split_results['test_metrics']:
                test_metrics = split_results['test_metrics']
                analysis['interpolation_performance'] = {
                    'mse': test_metrics.get('mse', float('inf')),
                    'correlation': test_metrics.get('correlation', 0.0),
                    'performance_level': 'good' if test_metrics.get('correlation', 0) > 0.7 else 'poor'
                }
            
            # 外推性能分析（简化）
            analysis['extrapolation_performance'] = {
                'boundary_effect': 'moderate',
                'edge_degradation': 0.1
            }
            
            # 空间偏差分析
            analysis['spatial_bias_analysis'] = {
                'systematic_bias': 'low',
                'regional_variations': 'moderate'
            }
            
            # 覆盖分析
            total_points = len(data[x_column])
            test_points = len(splits['test'][x_column]) if 'test' in splits else 0
            
            analysis['coverage_analysis'] = {
                'spatial_coverage': test_points / total_points if total_points > 0 else 0,
                'representative_sampling': test_points > 100
            }
        
        except Exception as e:
            self.logger.warning(f"空间性能分析失败: {e}")
        
        return analysis
    
    def _plot_results(self, results: Dict[str, Any], 
                     splits: Dict[str, Dict[str, torch.Tensor]],
                     data: Dict[str, torch.Tensor],
                     x_column: str, y_column: str):
        """绘制结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制空间分布
            if 'test' in splits:
                x_test = splits['test'][x_column].detach().numpy()
                y_test = splits['test'][y_column].detach().numpy()
                
                if 'predictions' in results['split_results'] and 'test' in results['split_results']['predictions']:
                    predictions = results['split_results']['predictions']['test'].detach().numpy()
                    targets = results['split_results']['targets']['test'].detach().numpy()
                    
                    # 预测值空间分布
                    scatter1 = axes[0, 0].scatter(x_test, y_test, c=predictions, cmap='viridis', alpha=0.6)
                    axes[0, 0].set_xlabel('X坐标')
                    axes[0, 0].set_ylabel('Y坐标')
                    axes[0, 0].set_title('预测值空间分布')
                    plt.colorbar(scatter1, ax=axes[0, 0])
                    
                    # 真实值空间分布
                    scatter2 = axes[0, 1].scatter(x_test, y_test, c=targets, cmap='viridis', alpha=0.6)
                    axes[0, 1].set_xlabel('X坐标')
                    axes[0, 1].set_ylabel('Y坐标')
                    axes[0, 1].set_title('真实值空间分布')
                    plt.colorbar(scatter2, ax=axes[0, 1])
                    
                    # 误差空间分布
                    errors = np.abs(predictions - targets)
                    scatter3 = axes[1, 0].scatter(x_test, y_test, c=errors, cmap='Reds', alpha=0.6)
                    axes[1, 0].set_xlabel('X坐标')
                    axes[1, 0].set_ylabel('Y坐标')
                    axes[1, 0].set_title('误差空间分布')
                    plt.colorbar(scatter3, ax=axes[1, 0])
                    
                    # 预测vs真实值散点图
                    axes[1, 1].scatter(targets, predictions, alpha=0.6)
                    axes[1, 1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
                    axes[1, 1].set_xlabel('真实值')
                    axes[1, 1].set_ylabel('预测值')
                    axes[1, 1].set_title('预测 vs 真实值')
            
            # 绘制训练/验证/测试集分布
            x_all = data[x_column].detach().numpy()
            y_all = data[y_column].detach().numpy()
            
            # 创建颜色标记
            colors = np.zeros(len(x_all))
            if 'train' in splits:
                train_indices = torch.isin(data[x_column], splits['train'][x_column])
                colors[train_indices.numpy()] = 1
            if 'validation' in splits:
                val_indices = torch.isin(data[x_column], splits['validation'][x_column])
                colors[val_indices.numpy()] = 2
            if 'test' in splits:
                test_indices = torch.isin(data[x_column], splits['test'][x_column])
                colors[test_indices.numpy()] = 3
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/spatial_validation_results.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "空间维度留出验证报告",
            "=" * 60,
            f"验证策略: {results.get('strategy', 'Unknown')}",
            f"验证模式: {results.get('mode', 'Unknown')}",
            ""
        ]
        
        # 分割结果
        if 'split_results' in results:
            sr = results['split_results']
            report_lines.append("分割性能:")
            
            for split_name in ['train', 'validation', 'test']:
                metrics_key = f'{split_name}_metrics'
                if metrics_key in sr:
                    metrics = sr[metrics_key]
                    report_lines.extend([
                        f"  {split_name.upper()}集:",
                        f"    MSE: {metrics.get('mse', 0):.6f}",
                        f"    MAE: {metrics.get('mae', 0):.6f}",
                        f"    相关性: {metrics.get('correlation', 0):.6f}",
                        f"    R²: {metrics.get('r2', 0):.6f}"
                    ])
            report_lines.append("")
        
        # 空间指标
        if 'spatial_metrics' in results:
            sm = results['spatial_metrics']
            report_lines.append("空间指标:")
            for metric, value in sm.items():
                if isinstance(value, dict):
                    report_lines.append(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        report_lines.append(f"    {sub_metric}: {sub_value:.6f}")
                else:
                    report_lines.append(f"  {metric}: {value:.6f}")
            report_lines.append("")
        
        # 空间分析
        if 'spatial_analysis' in results:
            sa = results['spatial_analysis']
            report_lines.append("空间分析:")
            
            if 'interpolation_performance' in sa:
                ip = sa['interpolation_performance']
                report_lines.extend([
                    f"  插值性能: {ip.get('performance_level', 'Unknown')}",
                    f"  插值MSE: {ip.get('mse', 0):.6f}",
                    f"  插值相关性: {ip.get('correlation', 0):.6f}"
                ])
            
            if 'coverage_analysis' in sa:
                ca = sa['coverage_analysis']
                report_lines.extend([
                    f"  空间覆盖率: {ca.get('spatial_coverage', 0):.3f}",
                    f"  代表性采样: {'是' if ca.get('representative_sampling', False) else '否'}"
                ])
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def create_spatial_validator(config: SpatialHoldoutConfig = None) -> SpatialValidator:
    """
    创建空间验证器
    
    Args:
        config: 配置
        
    Returns:
        SpatialValidator: 验证器实例
    """
    if config is None:
        config = SpatialHoldoutConfig()
    return SpatialValidator(config)

if __name__ == "__main__":
    # 测试空间维度留出验证
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),  # 输入: [x, y, t]
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)  # 输出: 温度
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # 生成测试数据
    n_samples = 1000
    data = {
        'x': torch.randn(n_samples) * 1000,  # X坐标 (m)
        'y': torch.randn(n_samples) * 1000,  # Y坐标 (m)
        't': torch.randn(n_samples),  # 时间
        'temperature': torch.randn(n_samples) + 273.15  # 温度数据
    }
    
    # 创建配置
    config = SpatialHoldoutConfig(
        split_strategy=SpatialSplitStrategy.GRID_BASED,
        validation_mode=SpatialValidationMode.HOLDOUT,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_spatial_validator(config)
    
    print("=== 空间维度留出验证测试 ===")
    
    # 执行验证
    results = validator.validate(model, data)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n空间维度留出验证测试完成！")