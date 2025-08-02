#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度指标模块

该模块实现了用于评估模型预测精度的各种指标，包括传统统计指标、
空间精度指标、时间精度指标和专门针对冰川建模的精度指标。

主要功能:
- 基础统计精度指标
- 空间精度评估
- 时间精度评估
- 分布式精度指标
- 相对精度指标
- 分层精度分析

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
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyMetricType(Enum):
    """精度指标类型"""
    # 基础统计指标
    MSE = "mse"                    # 均方误差
    RMSE = "rmse"                  # 均方根误差
    MAE = "mae"                    # 平均绝对误差
    MAPE = "mape"                  # 平均绝对百分比误差
    R2 = "r2"                      # 决定系数
    CORRELATION = "correlation"    # 相关系数
    
    # 相对误差指标
    RELATIVE_MSE = "relative_mse"  # 相对均方误差
    RELATIVE_MAE = "relative_mae"  # 相对平均绝对误差
    NORMALIZED_RMSE = "nrmse"      # 归一化均方根误差
    
    # 分布指标
    BIAS = "bias"                  # 偏差
    VARIANCE = "variance"          # 方差
    SKEWNESS = "skewness"          # 偏度
    KURTOSIS = "kurtosis"          # 峰度
    
    # 百分位指标
    MEDIAN_ERROR = "median_error"  # 中位数误差
    Q95_ERROR = "q95_error"        # 95%分位数误差
    IQR_ERROR = "iqr_error"        # 四分位距误差
    
    # 空间指标
    SPATIAL_CORRELATION = "spatial_correlation"  # 空间相关性
    SPATIAL_RMSE = "spatial_rmse"                # 空间RMSE
    
    # 时间指标
    TEMPORAL_CORRELATION = "temporal_correlation"  # 时间相关性
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # 时间一致性

class AggregationMethod(Enum):
    """聚合方法"""
    MEAN = "mean"          # 平均值
    MEDIAN = "median"      # 中位数
    WEIGHTED = "weighted"  # 加权平均
    ROBUST = "robust"      # 鲁棒统计

class ErrorDistribution(Enum):
    """误差分布类型"""
    NORMAL = "normal"          # 正态分布
    LOGNORMAL = "lognormal"    # 对数正态分布
    EXPONENTIAL = "exponential"  # 指数分布
    UNIFORM = "uniform"        # 均匀分布
    UNKNOWN = "unknown"        # 未知分布

@dataclass
class AccuracyConfig:
    """精度评估配置"""
    # 基本设置
    metrics: List[AccuracyMetricType] = field(default_factory=lambda: [
        AccuracyMetricType.RMSE, AccuracyMetricType.MAE, 
        AccuracyMetricType.R2, AccuracyMetricType.CORRELATION
    ])
    
    # 聚合设置
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    enable_weighted_metrics: bool = True
    
    # 分层分析
    enable_stratified_analysis: bool = True
    stratification_variables: List[str] = field(default_factory=lambda: ['elevation', 'slope', 'aspect'])
    stratification_bins: int = 5
    
    # 空间分析
    enable_spatial_analysis: bool = True
    spatial_resolution: float = 1000.0  # 米
    spatial_lag_distances: List[float] = field(default_factory=lambda: [1000, 2000, 5000])  # 米
    
    # 时间分析
    enable_temporal_analysis: bool = True
    temporal_windows: List[int] = field(default_factory=lambda: [30, 90, 365])  # 天
    
    # 统计设置
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    enable_bootstrap: bool = True
    
    # 异常值处理
    outlier_threshold: float = 3.0  # 标准差倍数
    remove_outliers: bool = False
    
    # 相对误差设置
    relative_error_threshold: float = 0.01  # 避免除零的最小值
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./accuracy_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

@dataclass
class AccuracyResult:
    """精度评估结果"""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasicMetrics:
    """基础精度指标计算器"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """均方误差"""
        if weights is not None:
            return np.average((y_true - y_pred) ** 2, weights=weights)
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """均方根误差"""
        return np.sqrt(BasicMetrics.mse(y_true, y_pred, weights))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """平均绝对误差"""
        if weights is not None:
            return np.average(np.abs(y_true - y_pred), weights=weights)
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, 
             threshold: float = 0.01, weights: Optional[np.ndarray] = None) -> float:
        """平均绝对百分比误差"""
        # 避免除零
        mask = np.abs(y_true) > threshold
        if not np.any(mask):
            return np.inf
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
        
        if weights is not None:
            weights_filtered = weights[mask]
            return np.average(ape, weights=weights_filtered)
        return np.mean(ape)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """决定系数"""
        if weights is not None:
            # 加权R²计算
            y_mean = np.average(y_true, weights=weights)
            ss_tot = np.sum(weights * (y_true - y_mean) ** 2)
            ss_res = np.sum(weights * (y_true - y_pred) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """相关系数"""
        if weights is not None:
            # 加权相关系数
            cov_matrix = np.cov(y_true, y_pred, aweights=weights)
            return cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """偏差"""
        if weights is not None:
            return np.average(y_pred - y_true, weights=weights)
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, 
                       normalization: str = 'range', weights: Optional[np.ndarray] = None) -> float:
        """归一化均方根误差"""
        rmse_val = BasicMetrics.rmse(y_true, y_pred, weights)
        
        if normalization == 'range':
            norm_factor = np.max(y_true) - np.min(y_true)
        elif normalization == 'mean':
            norm_factor = np.mean(y_true)
        elif normalization == 'std':
            norm_factor = np.std(y_true)
        else:
            norm_factor = 1.0
        
        return rmse_val / norm_factor if norm_factor > 0 else np.inf

class DistributionMetrics:
    """分布相关指标"""
    
    @staticmethod
    def error_variance(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """误差方差"""
        errors = y_pred - y_true
        if weights is not None:
            return np.average((errors - np.average(errors, weights=weights)) ** 2, weights=weights)
        return np.var(errors)
    
    @staticmethod
    def error_skewness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """误差偏度"""
        errors = y_pred - y_true
        return stats.skew(errors)
    
    @staticmethod
    def error_kurtosis(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """误差峰度"""
        errors = y_pred - y_true
        return stats.kurtosis(errors)
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """中位数绝对误差"""
        return median_absolute_error(y_true, y_pred)
    
    @staticmethod
    def quantile_error(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.95) -> float:
        """分位数误差"""
        errors = np.abs(y_true - y_pred)
        return np.quantile(errors, quantile)
    
    @staticmethod
    def interquartile_range_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """四分位距误差"""
        errors = np.abs(y_true - y_pred)
        q75 = np.quantile(errors, 0.75)
        q25 = np.quantile(errors, 0.25)
        return q75 - q25

class SpatialMetrics:
    """空间精度指标"""
    
    def __init__(self, coordinates: np.ndarray):
        """
        初始化空间指标计算器
        
        Args:
            coordinates: 空间坐标数组，形状为(n_points, 2) [lat, lon]
        """
        self.coordinates = coordinates
    
    def spatial_autocorrelation(self, errors: np.ndarray, distance_threshold: float = 1000.0) -> float:
        """空间自相关性(Moran's I)"""
        try:
            from scipy.spatial.distance import pdist, squareform
            
            # 计算距离矩阵
            distances = squareform(pdist(self.coordinates))
            
            # 创建权重矩阵
            weights = (distances <= distance_threshold).astype(float)
            np.fill_diagonal(weights, 0)
            
            # 计算Moran's I
            n = len(errors)
            mean_error = np.mean(errors)
            
            numerator = 0
            denominator = 0
            weight_sum = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        w_ij = weights[i, j]
                        numerator += w_ij * (errors[i] - mean_error) * (errors[j] - mean_error)
                        weight_sum += w_ij
                
                denominator += (errors[i] - mean_error) ** 2
            
            if weight_sum > 0 and denominator > 0:
                moran_i = (n / weight_sum) * (numerator / denominator)
                return moran_i
            else:
                return 0.0
        
        except Exception as e:
            logger.warning(f"空间自相关计算失败: {e}")
            return 0.0
    
    def spatial_rmse_by_distance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                distance_bins: List[float]) -> Dict[str, float]:
        """按距离分层的空间RMSE"""
        try:
            from scipy.spatial.distance import pdist, squareform
            
            distances = squareform(pdist(self.coordinates))
            errors = (y_true - y_pred) ** 2
            
            spatial_rmse = {}
            
            for i, max_dist in enumerate(distance_bins):
                min_dist = distance_bins[i-1] if i > 0 else 0
                
                # 找到在距离范围内的点对
                mask = (distances >= min_dist) & (distances < max_dist)
                
                if np.any(mask):
                    # 计算该距离范围内的平均误差
                    range_errors = []
                    for j in range(len(errors)):
                        for k in range(j+1, len(errors)):
                            if mask[j, k]:
                                range_errors.extend([errors[j], errors[k]])
                    
                    if range_errors:
                        spatial_rmse[f"{min_dist}-{max_dist}m"] = np.sqrt(np.mean(range_errors))
            
            return spatial_rmse
        
        except Exception as e:
            logger.warning(f"空间RMSE计算失败: {e}")
            return {}

class TemporalMetrics:
    """时间精度指标"""
    
    def __init__(self, timestamps: np.ndarray):
        """
        初始化时间指标计算器
        
        Args:
            timestamps: 时间戳数组
        """
        self.timestamps = timestamps
    
    def temporal_correlation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           window_size: int = 30) -> float:
        """时间相关性"""
        try:
            # 按时间排序
            sorted_indices = np.argsort(self.timestamps)
            y_true_sorted = y_true[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]
            
            # 计算滑动窗口相关性
            correlations = []
            
            for i in range(len(y_true_sorted) - window_size + 1):
                window_true = y_true_sorted[i:i+window_size]
                window_pred = y_pred_sorted[i:i+window_size]
                
                if np.std(window_true) > 0 and np.std(window_pred) > 0:
                    corr = np.corrcoef(window_true, window_pred)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
        
        except Exception as e:
            logger.warning(f"时间相关性计算失败: {e}")
            return 0.0
    
    def temporal_consistency(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """时间一致性"""
        try:
            # 按时间排序
            sorted_indices = np.argsort(self.timestamps)
            y_true_sorted = y_true[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]
            
            # 计算时间差分
            true_diff = np.diff(y_true_sorted)
            pred_diff = np.diff(y_pred_sorted)
            
            # 计算差分的相关性
            if len(true_diff) > 1 and np.std(true_diff) > 0 and np.std(pred_diff) > 0:
                consistency = np.corrcoef(true_diff, pred_diff)[0, 1]
                return consistency if not np.isnan(consistency) else 0.0
            else:
                return 0.0
        
        except Exception as e:
            logger.warning(f"时间一致性计算失败: {e}")
            return 0.0

class AccuracyAnalyzer:
    """精度分析器"""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       weights: Optional[np.ndarray] = None,
                       coordinates: Optional[np.ndarray] = None,
                       timestamps: Optional[np.ndarray] = None,
                       stratification_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, AccuracyResult]:
        """计算精度指标"""
        try:
            # 数据验证
            if len(y_true) != len(y_pred):
                raise ValueError("真实值和预测值长度不匹配")
            
            # 移除无效值
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            
            if not np.any(valid_mask):
                raise ValueError("没有有效的数据点")
            
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            
            if weights is not None:
                weights_clean = weights[valid_mask]
            else:
                weights_clean = None
            
            # 异常值处理
            if self.config.remove_outliers:
                outlier_mask = self._detect_outliers(y_true_clean, y_pred_clean)
                y_true_clean = y_true_clean[~outlier_mask]
                y_pred_clean = y_pred_clean[~outlier_mask]
                if weights_clean is not None:
                    weights_clean = weights_clean[~outlier_mask]
            
            results = {}
            
            # 计算基础指标
            for metric_type in self.config.metrics:
                result = self._compute_single_metric(
                    metric_type, y_true_clean, y_pred_clean, weights_clean,
                    coordinates, timestamps
                )
                if result is not None:
                    results[metric_type.value] = result
            
            # 分层分析
            if self.config.enable_stratified_analysis and stratification_data:
                stratified_results = self._compute_stratified_metrics(
                    y_true_clean, y_pred_clean, weights_clean, stratification_data
                )
                results.update(stratified_results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"精度指标计算失败: {e}")
            return {}
    
    def _compute_single_metric(self, 
                              metric_type: AccuracyMetricType,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              coordinates: Optional[np.ndarray] = None,
                              timestamps: Optional[np.ndarray] = None) -> Optional[AccuracyResult]:
        """计算单个指标"""
        try:
            value = None
            metadata = {}
            
            # 基础统计指标
            if metric_type == AccuracyMetricType.MSE:
                value = BasicMetrics.mse(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.RMSE:
                value = BasicMetrics.rmse(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.MAE:
                value = BasicMetrics.mae(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.MAPE:
                value = BasicMetrics.mape(y_true, y_pred, self.config.relative_error_threshold, weights)
            elif metric_type == AccuracyMetricType.R2:
                value = BasicMetrics.r2(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.CORRELATION:
                value = BasicMetrics.correlation(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.BIAS:
                value = BasicMetrics.bias(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.NORMALIZED_RMSE:
                value = BasicMetrics.normalized_rmse(y_true, y_pred, 'range', weights)
            
            # 分布指标
            elif metric_type == AccuracyMetricType.VARIANCE:
                value = DistributionMetrics.error_variance(y_true, y_pred, weights)
            elif metric_type == AccuracyMetricType.SKEWNESS:
                value = DistributionMetrics.error_skewness(y_true, y_pred)
            elif metric_type == AccuracyMetricType.KURTOSIS:
                value = DistributionMetrics.error_kurtosis(y_true, y_pred)
            elif metric_type == AccuracyMetricType.MEDIAN_ERROR:
                value = DistributionMetrics.median_absolute_error(y_true, y_pred)
            elif metric_type == AccuracyMetricType.Q95_ERROR:
                value = DistributionMetrics.quantile_error(y_true, y_pred, 0.95)
            elif metric_type == AccuracyMetricType.IQR_ERROR:
                value = DistributionMetrics.interquartile_range_error(y_true, y_pred)
            
            # 空间指标
            elif metric_type == AccuracyMetricType.SPATIAL_CORRELATION and coordinates is not None:
                spatial_metrics = SpatialMetrics(coordinates)
                errors = y_pred - y_true
                value = spatial_metrics.spatial_autocorrelation(errors)
                metadata['spatial_lag'] = self.config.spatial_lag_distances[0]
            
            # 时间指标
            elif metric_type == AccuracyMetricType.TEMPORAL_CORRELATION and timestamps is not None:
                temporal_metrics = TemporalMetrics(timestamps)
                value = temporal_metrics.temporal_correlation(y_true, y_pred, self.config.temporal_windows[0])
            elif metric_type == AccuracyMetricType.TEMPORAL_CONSISTENCY and timestamps is not None:
                temporal_metrics = TemporalMetrics(timestamps)
                value = temporal_metrics.temporal_consistency(y_true, y_pred)
            
            if value is not None:
                # 计算置信区间
                confidence_interval = None
                if self.config.enable_bootstrap:
                    confidence_interval = self._bootstrap_confidence_interval(
                        metric_type, y_true, y_pred, weights
                    )
                
                return AccuracyResult(
                    metric_name=metric_type.value,
                    value=value,
                    confidence_interval=confidence_interval,
                    sample_size=len(y_true),
                    metadata=metadata
                )
            
            return None
        
        except Exception as e:
            self.logger.warning(f"指标 {metric_type.value} 计算失败: {e}")
            return None
    
    def _detect_outliers(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """检测异常值"""
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        outlier_mask = np.abs(errors - mean_error) > self.config.outlier_threshold * std_error
        return outlier_mask
    
    def _bootstrap_confidence_interval(self, 
                                     metric_type: AccuracyMetricType,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Bootstrap置信区间"""
        try:
            n_samples = len(y_true)
            bootstrap_values = []
            
            for _ in range(self.config.bootstrap_samples):
                # 重采样
                indices = np.random.choice(n_samples, n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                weights_boot = weights[indices] if weights is not None else None
                
                # 计算指标
                if metric_type == AccuracyMetricType.RMSE:
                    value = BasicMetrics.rmse(y_true_boot, y_pred_boot, weights_boot)
                elif metric_type == AccuracyMetricType.MAE:
                    value = BasicMetrics.mae(y_true_boot, y_pred_boot, weights_boot)
                elif metric_type == AccuracyMetricType.R2:
                    value = BasicMetrics.r2(y_true_boot, y_pred_boot, weights_boot)
                elif metric_type == AccuracyMetricType.CORRELATION:
                    value = BasicMetrics.correlation(y_true_boot, y_pred_boot, weights_boot)
                else:
                    continue
                
                if not np.isnan(value):
                    bootstrap_values.append(value)
            
            if bootstrap_values:
                alpha = 1 - self.config.confidence_level
                lower = np.percentile(bootstrap_values, 100 * alpha / 2)
                upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
                return (lower, upper)
            
            return None
        
        except Exception as e:
            self.logger.warning(f"Bootstrap置信区间计算失败: {e}")
            return None
    
    def _compute_stratified_metrics(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  weights: Optional[np.ndarray],
                                  stratification_data: Dict[str, np.ndarray]) -> Dict[str, AccuracyResult]:
        """计算分层指标"""
        stratified_results = {}
        
        for var_name, var_data in stratification_data.items():
            if var_name in self.config.stratification_variables:
                try:
                    # 创建分层
                    bins = np.linspace(np.min(var_data), np.max(var_data), self.config.stratification_bins + 1)
                    bin_indices = np.digitize(var_data, bins) - 1
                    
                    for bin_idx in range(self.config.stratification_bins):
                        mask = bin_indices == bin_idx
                        
                        if np.sum(mask) > 10:  # 确保有足够的样本
                            y_true_bin = y_true[mask]
                            y_pred_bin = y_pred[mask]
                            weights_bin = weights[mask] if weights is not None else None
                            
                            # 计算主要指标
                            rmse = BasicMetrics.rmse(y_true_bin, y_pred_bin, weights_bin)
                            mae = BasicMetrics.mae(y_true_bin, y_pred_bin, weights_bin)
                            r2 = BasicMetrics.r2(y_true_bin, y_pred_bin, weights_bin)
                            
                            bin_range = f"{bins[bin_idx]:.2f}-{bins[bin_idx+1]:.2f}"
                            
                            stratified_results[f"{var_name}_{bin_range}_rmse"] = AccuracyResult(
                                metric_name=f"{var_name}_{bin_range}_rmse",
                                value=rmse,
                                sample_size=np.sum(mask),
                                metadata={'variable': var_name, 'bin_range': bin_range}
                            )
                            
                            stratified_results[f"{var_name}_{bin_range}_mae"] = AccuracyResult(
                                metric_name=f"{var_name}_{bin_range}_mae",
                                value=mae,
                                sample_size=np.sum(mask),
                                metadata={'variable': var_name, 'bin_range': bin_range}
                            )
                            
                            stratified_results[f"{var_name}_{bin_range}_r2"] = AccuracyResult(
                                metric_name=f"{var_name}_{bin_range}_r2",
                                value=r2,
                                sample_size=np.sum(mask),
                                metadata={'variable': var_name, 'bin_range': bin_range}
                            )
                
                except Exception as e:
                    self.logger.warning(f"分层分析失败 {var_name}: {e}")
        
        return stratified_results
    
    def plot_results(self, 
                    results: Dict[str, AccuracyResult],
                    y_true: np.ndarray,
                    y_pred: np.ndarray):
        """绘制精度分析结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('精度分析结果', fontsize=16, fontweight='bold')
            
            # 散点图
            ax = axes[0, 0]
            ax.scatter(y_true, y_pred, alpha=0.6)
            
            # 添加1:1线
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            
            # 添加R²信息
            if 'r2' in results:
                ax.set_title(f"预测 vs 真实值 (R² = {results['r2'].value:.3f})")
            else:
                ax.set_title('预测 vs 真实值')
            ax.grid(True, alpha=0.3)
            
            # 残差图
            ax = axes[0, 1]
            residuals = y_pred - y_true
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('预测值')
            ax.set_ylabel('残差')
            ax.set_title('残差图')
            ax.grid(True, alpha=0.3)
            
            # 残差直方图
            ax = axes[1, 0]
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('残差')
            ax.set_ylabel('频次')
            ax.set_title('残差分布')
            ax.grid(True, alpha=0.3)
            
            # 指标汇总
            ax = axes[1, 1]
            ax.axis('off')
            
            # 创建指标表格
            metric_text = "主要精度指标:\n\n"
            
            key_metrics = ['rmse', 'mae', 'r2', 'correlation', 'bias']
            for metric in key_metrics:
                if metric in results:
                    result = results[metric]
                    if result.confidence_interval:
                        ci_text = f" [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
                    else:
                        ci_text = ""
                    metric_text += f"{metric.upper()}: {result.value:.4f}{ci_text}\n"
            
            ax.text(0.1, 0.9, metric_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/accuracy_analysis.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, AccuracyResult],
                       y_true: np.ndarray,
                       y_pred: np.ndarray) -> str:
        """生成精度分析报告"""
        from datetime import datetime
        
        report_lines = [
            "="*80,
            "精度分析报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"样本数量: {len(y_true)}",
            "",
            "1. 基础统计指标",
            "-"*40
        ]
        
        # 基础指标
        basic_metrics = ['rmse', 'mae', 'mape', 'r2', 'correlation', 'bias']
        for metric in basic_metrics:
            if metric in results:
                result = results[metric]
                if result.confidence_interval:
                    ci_text = f" (95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}])"
                else:
                    ci_text = ""
                report_lines.append(f"{metric.upper()}: {result.value:.6f}{ci_text}")
        
        # 分布指标
        dist_metrics = ['variance', 'skewness', 'kurtosis', 'median_error', 'q95_error']
        if any(metric in results for metric in dist_metrics):
            report_lines.extend([
                "",
                "2. 分布特征指标",
                "-"*40
            ])
            
            for metric in dist_metrics:
                if metric in results:
                    result = results[metric]
                    report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 空间指标
        spatial_metrics = [key for key in results.keys() if 'spatial' in key]
        if spatial_metrics:
            report_lines.extend([
                "",
                "3. 空间精度指标",
                "-"*40
            ])
            
            for metric in spatial_metrics:
                result = results[metric]
                report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 时间指标
        temporal_metrics = [key for key in results.keys() if 'temporal' in key]
        if temporal_metrics:
            report_lines.extend([
                "",
                "4. 时间精度指标",
                "-"*40
            ])
            
            for metric in temporal_metrics:
                result = results[metric]
                report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 分层分析结果
        stratified_metrics = [key for key in results.keys() if any(var in key for var in self.config.stratification_variables)]
        if stratified_metrics:
            report_lines.extend([
                "",
                "5. 分层精度分析",
                "-"*40
            ])
            
            # 按变量分组
            for var in self.config.stratification_variables:
                var_metrics = [key for key in stratified_metrics if var in key]
                if var_metrics:
                    report_lines.append(f"\n{var.upper()}:")
                    for metric in sorted(var_metrics):
                        result = results[metric]
                        report_lines.append(f"  {metric}: {result.value:.6f} (n={result.sample_size})")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_accuracy_analyzer(config: Optional[AccuracyConfig] = None) -> AccuracyAnalyzer:
    """创建精度分析器"""
    if config is None:
        config = AccuracyConfig()
    
    return AccuracyAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    print("开始精度指标测试...")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 真实值
    y_true = np.random.normal(100, 20, n_samples)
    
    # 预测值(添加一些误差和偏差)
    noise = np.random.normal(0, 5, n_samples)
    bias = 2.0
    y_pred = y_true + noise + bias
    
    # 权重
    weights = np.random.uniform(0.5, 1.5, n_samples)
    
    # 坐标
    coordinates = np.random.uniform(-1, 1, (n_samples, 2))
    
    # 时间戳
    timestamps = np.sort(np.random.uniform(0, 365, n_samples))
    
    # 分层数据
    stratification_data = {
        'elevation': np.random.uniform(1000, 5000, n_samples),
        'slope': np.random.uniform(0, 45, n_samples)
    }
    
    # 创建配置
    config = AccuracyConfig(
        metrics=[
            AccuracyMetricType.RMSE,
            AccuracyMetricType.MAE,
            AccuracyMetricType.R2,
            AccuracyMetricType.CORRELATION,
            AccuracyMetricType.BIAS,
            AccuracyMetricType.MAPE,
            AccuracyMetricType.VARIANCE,
            AccuracyMetricType.SKEWNESS,
            AccuracyMetricType.SPATIAL_CORRELATION,
            AccuracyMetricType.TEMPORAL_CORRELATION
        ],
        enable_stratified_analysis=True,
        enable_spatial_analysis=True,
        enable_temporal_analysis=True,
        enable_bootstrap=True,
        enable_plotting=True,
        verbose=True
    )
    
    # 创建分析器
    analyzer = create_accuracy_analyzer(config)
    
    # 计算指标
    results = analyzer.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        weights=weights,
        coordinates=coordinates,
        timestamps=timestamps,
        stratification_data=stratification_data
    )
    
    # 打印结果
    print("\n精度指标计算完成！")
    print(f"计算了 {len(results)} 个指标")
    
    # 生成报告
    report = analyzer.generate_report(results, y_true, y_pred)
    print("\n" + report)
    
    # 绘制结果
    analyzer.plot_results(results, y_true, y_pred)
    
    print("\n精度指标测试完成！")