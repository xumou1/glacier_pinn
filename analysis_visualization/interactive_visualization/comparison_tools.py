#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比工具模块

该模块提供交互式的冰川数据对比分析工具，包括：
- 多模型对比
- 时间序列对比
- 空间分布对比
- 统计指标对比
- 交互式可视化

作者: 冰川研究团队
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComparisonType(Enum):
    """对比类型"""
    MODEL_COMPARISON = "model_comparison"  # 模型对比
    TIME_SERIES = "time_series"  # 时间序列对比
    SPATIAL_DISTRIBUTION = "spatial_distribution"  # 空间分布对比
    STATISTICAL_METRICS = "statistical_metrics"  # 统计指标对比
    PERFORMANCE_ANALYSIS = "performance_analysis"  # 性能分析对比
    UNCERTAINTY_COMPARISON = "uncertainty_comparison"  # 不确定性对比
    SCENARIO_COMPARISON = "scenario_comparison"  # 情景对比
    MULTI_VARIABLE = "multi_variable"  # 多变量对比

class MetricType(Enum):
    """指标类型"""
    RMSE = "rmse"  # 均方根误差
    MAE = "mae"  # 平均绝对误差
    R2 = "r2"  # 决定系数
    CORRELATION = "correlation"  # 相关系数
    BIAS = "bias"  # 偏差
    NSE = "nse"  # Nash-Sutcliffe效率
    KGE = "kge"  # Kling-Gupta效率
    PBIAS = "pbias"  # 百分比偏差
    RSR = "rsr"  # 标准化均方根误差
    IOA = "ioa"  # 一致性指数

class VisualizationType(Enum):
    """可视化类型"""
    LINE_PLOT = "line_plot"  # 线图
    SCATTER_PLOT = "scatter_plot"  # 散点图
    BAR_CHART = "bar_chart"  # 柱状图
    HEATMAP = "heatmap"  # 热力图
    BOX_PLOT = "box_plot"  # 箱线图
    VIOLIN_PLOT = "violin_plot"  # 小提琴图
    RADAR_CHART = "radar_chart"  # 雷达图
    PARALLEL_COORDINATES = "parallel_coordinates"  # 平行坐标图
    CONTOUR_PLOT = "contour_plot"  # 等高线图
    SURFACE_PLOT = "surface_plot"  # 三维表面图
    ANIMATION = "animation"  # 动画
    INTERACTIVE_MAP = "interactive_map"  # 交互式地图

class AggregationType(Enum):
    """聚合类型"""
    MEAN = "mean"  # 平均值
    MEDIAN = "median"  # 中位数
    SUM = "sum"  # 求和
    MIN = "min"  # 最小值
    MAX = "max"  # 最大值
    STD = "std"  # 标准差
    VAR = "var"  # 方差
    PERCENTILE = "percentile"  # 百分位数
    RANGE = "range"  # 范围
    IQR = "iqr"  # 四分位距

@dataclass
class ComparisonData:
    """对比数据类"""
    name: str  # 数据名称
    data: np.ndarray  # 数据数组
    time: Optional[np.ndarray] = None  # 时间序列
    coordinates: Optional[Tuple[np.ndarray, np.ndarray]] = None  # 空间坐标
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    uncertainty: Optional[np.ndarray] = None  # 不确定性
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.data, list):
            self.data = np.array(self.data)
        
        if self.time is not None and isinstance(self.time, list):
            self.time = np.array(self.time)
        
        if self.uncertainty is not None and isinstance(self.uncertainty, list):
            self.uncertainty = np.array(self.uncertainty)

@dataclass
class ComparisonConfig:
    """对比配置类"""
    comparison_types: List[ComparisonType] = field(default_factory=lambda: [ComparisonType.MODEL_COMPARISON])
    metrics: List[MetricType] = field(default_factory=lambda: [MetricType.RMSE, MetricType.R2])
    visualization_types: List[VisualizationType] = field(default_factory=lambda: [VisualizationType.LINE_PLOT])
    aggregation_type: AggregationType = AggregationType.MEAN
    
    # 可视化参数
    figure_size: Tuple[int, int] = (12, 8)
    color_palette: str = "Set1"
    style: str = "whitegrid"
    dpi: int = 300
    
    # 统计参数
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # 输出参数
    save_results: bool = True
    output_dir: str = "./comparison_results"
    output_format: str = "png"
    interactive: bool = True
    
    # 过滤参数
    time_range: Optional[Tuple[datetime, datetime]] = None
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, max_x, min_y, max_y)
    value_range: Optional[Tuple[float, float]] = None
    
    # 高级参数
    normalize_data: bool = False
    remove_outliers: bool = False
    outlier_threshold: float = 3.0  # 标准差倍数
    interpolation_method: str = "linear"
    smoothing_window: Optional[int] = None

class MetricCalculator:
    """指标计算器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_metrics(self, observed: np.ndarray, predicted: np.ndarray, 
                         metrics: Optional[List[MetricType]] = None) -> Dict[str, float]:
        """计算评估指标"""
        if metrics is None:
            metrics = self.config.metrics
        
        # 移除NaN值
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        obs = observed[mask]
        pred = predicted[mask]
        
        if len(obs) == 0:
            return {metric.value: np.nan for metric in metrics}
        
        results = {}
        
        for metric in metrics:
            try:
                if metric == MetricType.RMSE:
                    results[metric.value] = np.sqrt(mean_squared_error(obs, pred))
                elif metric == MetricType.MAE:
                    results[metric.value] = mean_absolute_error(obs, pred)
                elif metric == MetricType.R2:
                    results[metric.value] = r2_score(obs, pred)
                elif metric == MetricType.CORRELATION:
                    corr, _ = stats.pearsonr(obs, pred)
                    results[metric.value] = corr
                elif metric == MetricType.BIAS:
                    results[metric.value] = np.mean(pred - obs)
                elif metric == MetricType.NSE:
                    results[metric.value] = self._calculate_nse(obs, pred)
                elif metric == MetricType.KGE:
                    results[metric.value] = self._calculate_kge(obs, pred)
                elif metric == MetricType.PBIAS:
                    results[metric.value] = self._calculate_pbias(obs, pred)
                elif metric == MetricType.RSR:
                    results[metric.value] = self._calculate_rsr(obs, pred)
                elif metric == MetricType.IOA:
                    results[metric.value] = self._calculate_ioa(obs, pred)
                else:
                    results[metric.value] = np.nan
            
            except Exception as e:
                self.logger.warning(f"计算指标 {metric.value} 失败: {e}")
                results[metric.value] = np.nan
        
        return results
    
    def _calculate_nse(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算Nash-Sutcliffe效率"""
        numerator = np.sum((observed - predicted) ** 2)
        denominator = np.sum((observed - np.mean(observed)) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else np.nan
    
    def _calculate_kge(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算Kling-Gupta效率"""
        r, _ = stats.pearsonr(observed, predicted)
        alpha = np.std(predicted) / np.std(observed)
        beta = np.mean(predicted) / np.mean(observed)
        
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge
    
    def _calculate_pbias(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算百分比偏差"""
        return 100 * np.sum(predicted - observed) / np.sum(observed)
    
    def _calculate_rsr(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算标准化均方根误差"""
        rmse = np.sqrt(mean_squared_error(observed, predicted))
        std_obs = np.std(observed)
        return rmse / std_obs if std_obs != 0 else np.nan
    
    def _calculate_ioa(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算一致性指数"""
        mean_obs = np.mean(observed)
        numerator = np.sum((observed - predicted) ** 2)
        denominator = np.sum((np.abs(predicted - mean_obs) + np.abs(observed - mean_obs)) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else np.nan
    
    def calculate_confidence_intervals(self, data: np.ndarray, 
                                     confidence_level: Optional[float] = None) -> Tuple[float, float]:
        """计算置信区间"""
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        alpha = 1 - confidence_level
        lower = np.percentile(data, 100 * alpha / 2)
        upper = np.percentile(data, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def perform_statistical_test(self, data1: np.ndarray, data2: np.ndarray, 
                               test_type: str = "ttest") -> Dict[str, float]:
        """执行统计检验"""
        try:
            if test_type == "ttest":
                statistic, p_value = stats.ttest_ind(data1, data2)
            elif test_type == "wilcoxon":
                statistic, p_value = stats.wilcoxon(data1, data2)
            elif test_type == "ks":
                statistic, p_value = stats.ks_2samp(data1, data2)
            elif test_type == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(data1, data2)
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")
            
            return {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < self.config.significance_level
            }
        
        except Exception as e:
            self.logger.warning(f"统计检验失败: {e}")
            return {"statistic": np.nan, "p_value": np.nan, "significant": False}

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess_data(self, data: ComparisonData) -> ComparisonData:
        """预处理数据"""
        processed_data = ComparisonData(
            name=data.name,
            data=data.data.copy(),
            time=data.time.copy() if data.time is not None else None,
            coordinates=data.coordinates,
            metadata=data.metadata.copy(),
            uncertainty=data.uncertainty.copy() if data.uncertainty is not None else None
        )
        
        # 移除异常值
        if self.config.remove_outliers:
            processed_data.data = self._remove_outliers(processed_data.data)
        
        # 数据标准化
        if self.config.normalize_data:
            processed_data.data = self._normalize_data(processed_data.data)
        
        # 时间范围过滤
        if self.config.time_range is not None and processed_data.time is not None:
            processed_data = self._filter_by_time(processed_data)
        
        # 空间范围过滤
        if self.config.spatial_bounds is not None and processed_data.coordinates is not None:
            processed_data = self._filter_by_space(processed_data)
        
        # 数值范围过滤
        if self.config.value_range is not None:
            processed_data = self._filter_by_value(processed_data)
        
        # 平滑处理
        if self.config.smoothing_window is not None:
            processed_data.data = self._smooth_data(processed_data.data)
        
        return processed_data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """移除异常值"""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        threshold = self.config.outlier_threshold * std
        
        mask = np.abs(data - mean) <= threshold
        cleaned_data = data.copy()
        cleaned_data[~mask] = np.nan
        
        return cleaned_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """标准化数据"""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std == 0:
            return data - mean
        
        return (data - mean) / std
    
    def _filter_by_time(self, data: ComparisonData) -> ComparisonData:
        """按时间过滤"""
        start_time, end_time = self.config.time_range
        
        if isinstance(data.time[0], datetime):
            mask = (data.time >= start_time) & (data.time <= end_time)
        else:
            # 假设时间是数值格式
            start_num = start_time.timestamp() if hasattr(start_time, 'timestamp') else start_time
            end_num = end_time.timestamp() if hasattr(end_time, 'timestamp') else end_time
            mask = (data.time >= start_num) & (data.time <= end_num)
        
        filtered_data = ComparisonData(
            name=data.name,
            data=data.data[mask],
            time=data.time[mask],
            coordinates=data.coordinates,
            metadata=data.metadata,
            uncertainty=data.uncertainty[mask] if data.uncertainty is not None else None
        )
        
        return filtered_data
    
    def _filter_by_space(self, data: ComparisonData) -> ComparisonData:
        """按空间过滤"""
        min_x, max_x, min_y, max_y = self.config.spatial_bounds
        x_coords, y_coords = data.coordinates
        
        mask = ((x_coords >= min_x) & (x_coords <= max_x) & 
                (y_coords >= min_y) & (y_coords <= max_y))
        
        filtered_data = ComparisonData(
            name=data.name,
            data=data.data[mask],
            time=data.time[mask] if data.time is not None else None,
            coordinates=(x_coords[mask], y_coords[mask]),
            metadata=data.metadata,
            uncertainty=data.uncertainty[mask] if data.uncertainty is not None else None
        )
        
        return filtered_data
    
    def _filter_by_value(self, data: ComparisonData) -> ComparisonData:
        """按数值过滤"""
        min_val, max_val = self.config.value_range
        mask = (data.data >= min_val) & (data.data <= max_val)
        
        filtered_data = ComparisonData(
            name=data.name,
            data=data.data.copy(),
            time=data.time,
            coordinates=data.coordinates,
            metadata=data.metadata,
            uncertainty=data.uncertainty
        )
        
        filtered_data.data[~mask] = np.nan
        
        return filtered_data
    
    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """平滑数据"""
        try:
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(data, size=self.config.smoothing_window, mode='nearest')
        except ImportError:
            # 简单移动平均
            window = self.config.smoothing_window
            smoothed = np.convolve(data, np.ones(window)/window, mode='same')
            return smoothed
    
    def align_datasets(self, datasets: List[ComparisonData]) -> List[ComparisonData]:
        """对齐数据集"""
        if len(datasets) < 2:
            return datasets
        
        # 找到共同的时间范围
        if all(d.time is not None for d in datasets):
            return self._align_by_time(datasets)
        
        # 找到共同的空间范围
        if all(d.coordinates is not None for d in datasets):
            return self._align_by_space(datasets)
        
        # 简单的长度对齐
        min_length = min(len(d.data) for d in datasets)
        aligned_datasets = []
        
        for data in datasets:
            aligned_data = ComparisonData(
                name=data.name,
                data=data.data[:min_length],
                time=data.time[:min_length] if data.time is not None else None,
                coordinates=data.coordinates,
                metadata=data.metadata,
                uncertainty=data.uncertainty[:min_length] if data.uncertainty is not None else None
            )
            aligned_datasets.append(aligned_data)
        
        return aligned_datasets
    
    def _align_by_time(self, datasets: List[ComparisonData]) -> List[ComparisonData]:
        """按时间对齐"""
        # 找到共同时间点
        common_times = datasets[0].time
        for data in datasets[1:]:
            common_times = np.intersect1d(common_times, data.time)
        
        aligned_datasets = []
        for data in datasets:
            mask = np.isin(data.time, common_times)
            aligned_data = ComparisonData(
                name=data.name,
                data=data.data[mask],
                time=data.time[mask],
                coordinates=data.coordinates,
                metadata=data.metadata,
                uncertainty=data.uncertainty[mask] if data.uncertainty is not None else None
            )
            aligned_datasets.append(aligned_data)
        
        return aligned_datasets
    
    def _align_by_space(self, datasets: List[ComparisonData]) -> List[ComparisonData]:
        """按空间对齐"""
        # 简化实现：假设所有数据集具有相同的空间网格
        return datasets
    
    def aggregate_data(self, data: np.ndarray, 
                      aggregation_type: Optional[AggregationType] = None) -> float:
        """聚合数据"""
        if aggregation_type is None:
            aggregation_type = self.config.aggregation_type
        
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.nan
        
        if aggregation_type == AggregationType.MEAN:
            return np.mean(valid_data)
        elif aggregation_type == AggregationType.MEDIAN:
            return np.median(valid_data)
        elif aggregation_type == AggregationType.SUM:
            return np.sum(valid_data)
        elif aggregation_type == AggregationType.MIN:
            return np.min(valid_data)
        elif aggregation_type == AggregationType.MAX:
            return np.max(valid_data)
        elif aggregation_type == AggregationType.STD:
            return np.std(valid_data)
        elif aggregation_type == AggregationType.VAR:
            return np.var(valid_data)
        elif aggregation_type == AggregationType.RANGE:
            return np.max(valid_data) - np.min(valid_data)
        elif aggregation_type == AggregationType.IQR:
            return np.percentile(valid_data, 75) - np.percentile(valid_data, 25)
        else:
            return np.mean(valid_data)

class StaticVisualizer:
    """静态可视化器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置样式
        sns.set_style(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def create_line_plot(self, datasets: List[ComparisonData], 
                        output_dir: str) -> str:
        """创建线图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            for data in datasets:
                if data.time is not None:
                    x = data.time
                else:
                    x = np.arange(len(data.data))
                
                # 绘制主线
                ax.plot(x, data.data, label=data.name, linewidth=2)
                
                # 绘制不确定性区间
                if data.uncertainty is not None:
                    ax.fill_between(x, 
                                   data.data - data.uncertainty,
                                   data.data + data.uncertainty,
                                   alpha=0.3)
            
            ax.set_xlabel('时间' if datasets[0].time is not None else '索引')
            ax.set_ylabel('数值')
            ax.set_title('时间序列对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'line_plot.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"线图创建失败: {e}")
            return ""
    
    def create_scatter_plot(self, dataset1: ComparisonData, dataset2: ComparisonData,
                           output_dir: str) -> str:
        """创建散点图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 对齐数据
            min_len = min(len(dataset1.data), len(dataset2.data))
            x = dataset1.data[:min_len]
            y = dataset2.data[:min_len]
            
            # 移除NaN值
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            # 散点图
            ax.scatter(x, y, alpha=0.6, s=50)
            
            # 1:1线
            min_val = min(np.min(x), np.min(y))
            max_val = max(np.max(x), np.max(y))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1线')
            
            # 拟合线
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), 'g-', alpha=0.8, label=f'拟合线 (y={z[0]:.2f}x+{z[1]:.2f})')
            
            # 计算相关系数
            if len(x) > 1:
                corr, _ = stats.pearsonr(x, y)
                ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(dataset1.name)
            ax.set_ylabel(dataset2.name)
            ax.set_title(f'{dataset1.name} vs {dataset2.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'scatter_plot_{dataset1.name}_{dataset2.name}.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"散点图创建失败: {e}")
            return ""
    
    def create_bar_chart(self, metrics_data: Dict[str, Dict[str, float]], 
                        output_dir: str) -> str:
        """创建柱状图"""
        try:
            # 准备数据
            df_data = []
            for dataset_name, metrics in metrics_data.items():
                for metric_name, value in metrics.items():
                    df_data.append({
                        'Dataset': dataset_name,
                        'Metric': metric_name,
                        'Value': value
                    })
            
            df = pd.DataFrame(df_data)
            
            # 创建子图
            metrics = df['Metric'].unique()
            n_metrics = len(metrics)
            
            fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                metric_data = df[df['Metric'] == metric]
                
                ax = axes[i]
                bars = ax.bar(metric_data['Dataset'], metric_data['Value'])
                
                # 添加数值标签
                for bar, value in zip(bars, metric_data['Value']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_title(metric)
                ax.set_ylabel('数值')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'bar_chart.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"柱状图创建失败: {e}")
            return ""
    
    def create_heatmap(self, correlation_matrix: pd.DataFrame, 
                      output_dir: str) -> str:
        """创建热力图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 创建热力图
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title('相关性矩阵热力图')
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'heatmap.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"热力图创建失败: {e}")
            return ""
    
    def create_box_plot(self, datasets: List[ComparisonData], 
                       output_dir: str) -> str:
        """创建箱线图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 准备数据
            data_list = []
            labels = []
            
            for data in datasets:
                valid_data = data.data[~np.isnan(data.data)]
                if len(valid_data) > 0:
                    data_list.append(valid_data)
                    labels.append(data.name)
            
            # 创建箱线图
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = sns.color_palette(self.config.color_palette, len(data_list))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('数据分布箱线图')
            ax.set_ylabel('数值')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'box_plot.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"箱线图创建失败: {e}")
            return ""
    
    def create_violin_plot(self, datasets: List[ComparisonData], 
                          output_dir: str) -> str:
        """创建小提琴图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 准备数据
            df_data = []
            for data in datasets:
                valid_data = data.data[~np.isnan(data.data)]
                for value in valid_data:
                    df_data.append({'Dataset': data.name, 'Value': value})
            
            df = pd.DataFrame(df_data)
            
            # 创建小提琴图
            sns.violinplot(data=df, x='Dataset', y='Value', ax=ax)
            
            ax.set_title('数据分布小提琴图')
            ax.set_ylabel('数值')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'violin_plot.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"小提琴图创建失败: {e}")
            return ""
    
    def create_radar_chart(self, metrics_data: Dict[str, Dict[str, float]], 
                          output_dir: str) -> str:
        """创建雷达图"""
        try:
            # 准备数据
            datasets = list(metrics_data.keys())
            metrics = list(next(iter(metrics_data.values())).keys())
            
            # 标准化数据到0-1范围
            normalized_data = {}
            for metric in metrics:
                values = [metrics_data[dataset][metric] for dataset in datasets]
                min_val, max_val = min(values), max(values)
                if max_val != min_val:
                    normalized_data[metric] = [(v - min_val) / (max_val - min_val) for v in values]
                else:
                    normalized_data[metric] = [0.5] * len(values)
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=self.config.figure_size, subplot_kw=dict(projection='polar'))
            
            colors = sns.color_palette(self.config.color_palette, len(datasets))
            
            for i, dataset in enumerate(datasets):
                values = [normalized_data[metric][i] for metric in metrics]
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=dataset, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('性能指标雷达图', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'radar_chart.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"雷达图创建失败: {e}")
            return ""

class InteractiveVisualizer:
    """交互式可视化器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_interactive_line_plot(self, datasets: List[ComparisonData]) -> go.Figure:
        """创建交互式线图"""
        try:
            fig = go.Figure()
            
            for data in datasets:
                if data.time is not None:
                    x = data.time
                    x_title = '时间'
                else:
                    x = np.arange(len(data.data))
                    x_title = '索引'
                
                # 主线
                fig.add_trace(go.Scatter(
                    x=x,
                    y=data.data,
                    mode='lines',
                    name=data.name,
                    line=dict(width=2),
                    hovertemplate=f'{data.name}<br>{x_title}: %{{x}}<br>数值: %{{y:.3f}}<extra></extra>'
                ))
                
                # 不确定性区间
                if data.uncertainty is not None:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([data.data + data.uncertainty, 
                                        (data.data - data.uncertainty)[::-1]]),
                        fill='toself',
                        fillcolor=f'rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{data.name} 不确定性',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            fig.update_layout(
                title='交互式时间序列对比',
                xaxis_title=x_title,
                yaxis_title='数值',
                hovermode='x unified',
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"交互式线图创建失败: {e}")
            return go.Figure()
    
    def create_interactive_scatter_plot(self, dataset1: ComparisonData, 
                                       dataset2: ComparisonData) -> go.Figure:
        """创建交互式散点图"""
        try:
            # 对齐数据
            min_len = min(len(dataset1.data), len(dataset2.data))
            x = dataset1.data[:min_len]
            y = dataset2.data[:min_len]
            
            # 移除NaN值
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            fig = go.Figure()
            
            # 散点
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name='数据点',
                marker=dict(size=8, opacity=0.6),
                hovertemplate=f'{dataset1.name}: %{{x:.3f}}<br>{dataset2.name}: %{{y:.3f}}<extra></extra>'
            ))
            
            # 1:1线
            min_val = min(np.min(x), np.min(y))
            max_val = max(np.max(x), np.max(y))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='1:1线',
                line=dict(color='red', dash='dash'),
                hoverinfo='skip'
            ))
            
            # 拟合线
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(np.min(x), np.max(x), 100)
                y_fit = p(x_fit)
                
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name=f'拟合线 (y={z[0]:.2f}x+{z[1]:.2f})',
                    line=dict(color='green'),
                    hoverinfo='skip'
                ))
            
            # 计算统计信息
            if len(x) > 1:
                corr, _ = stats.pearsonr(x, y)
                rmse = np.sqrt(mean_squared_error(x, y))
                mae = mean_absolute_error(x, y)
                
                # 添加统计信息文本
                fig.add_annotation(
                    x=0.05, y=0.95,
                    xref='paper', yref='paper',
                    text=f'R = {corr:.3f}<br>RMSE = {rmse:.3f}<br>MAE = {mae:.3f}',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
            
            fig.update_layout(
                title=f'{dataset1.name} vs {dataset2.name}',
                xaxis_title=dataset1.name,
                yaxis_title=dataset2.name,
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"交互式散点图创建失败: {e}")
            return go.Figure()
    
    def create_interactive_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """创建交互式热力图"""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                hovertemplate='X: %{x}<br>Y: %{y}<br>相关系数: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='交互式相关性矩阵热力图',
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"交互式热力图创建失败: {e}")
            return go.Figure()
    
    def create_interactive_box_plot(self, datasets: List[ComparisonData]) -> go.Figure:
        """创建交互式箱线图"""
        try:
            fig = go.Figure()
            
            for data in datasets:
                valid_data = data.data[~np.isnan(data.data)]
                if len(valid_data) > 0:
                    fig.add_trace(go.Box(
                        y=valid_data,
                        name=data.name,
                        boxpoints='outliers',
                        hovertemplate=f'{data.name}<br>数值: %{{y:.3f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='交互式数据分布箱线图',
                yaxis_title='数值',
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"交互式箱线图创建失败: {e}")
            return go.Figure()
    
    def create_interactive_3d_surface(self, data: ComparisonData) -> go.Figure:
        """创建交互式3D表面图"""
        try:
            if data.coordinates is None:
                raise ValueError("需要空间坐标数据")
            
            x_coords, y_coords = data.coordinates
            
            # 创建网格
            x_unique = np.unique(x_coords)
            y_unique = np.unique(y_coords)
            
            if len(x_unique) * len(y_unique) != len(data.data):
                # 插值到规则网格
                from scipy.interpolate import griddata
                xi, yi = np.meshgrid(x_unique, y_unique)
                zi = griddata((x_coords, y_coords), data.data, (xi, yi), method='linear')
            else:
                # 重塑为网格
                zi = data.data.reshape(len(y_unique), len(x_unique))
                xi, yi = np.meshgrid(x_unique, y_unique)
            
            fig = go.Figure(data=[go.Surface(
                x=xi,
                y=yi,
                z=zi,
                colorscale='Viridis',
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f'3D表面图 - {data.name}',
                scene=dict(
                    xaxis_title='X坐标',
                    yaxis_title='Y坐标',
                    zaxis_title='数值'
                ),
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"3D表面图创建失败: {e}")
            return go.Figure()
    
    def create_animated_comparison(self, datasets: List[ComparisonData]) -> go.Figure:
        """创建动画对比"""
        try:
            if not all(d.time is not None for d in datasets):
                raise ValueError("所有数据集都需要时间信息")
            
            # 找到共同时间点
            common_times = datasets[0].time
            for data in datasets[1:]:
                common_times = np.intersect1d(common_times, data.time)
            
            fig = go.Figure()
            
            # 为每个时间点创建帧
            frames = []
            for i, time_point in enumerate(common_times):
                frame_data = []
                
                for data in datasets:
                    time_idx = np.where(data.time == time_point)[0]
                    if len(time_idx) > 0:
                        idx = time_idx[0]
                        frame_data.append(go.Scatter(
                            x=[time_point],
                            y=[data.data[idx]],
                            mode='markers+lines',
                            name=data.name,
                            marker=dict(size=10)
                        ))
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=str(i)
                ))
            
            # 初始帧
            for data in datasets:
                fig.add_trace(go.Scatter(
                    x=data.time,
                    y=data.data,
                    mode='lines',
                    name=data.name,
                    opacity=0.3
                ))
            
            fig.frames = frames
            
            # 添加播放控件
            fig.update_layout(
                title='动画时间序列对比',
                xaxis_title='时间',
                yaxis_title='数值',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '播放',
                            'method': 'animate',
                            'args': [None, {'frame': {'duration': 500, 'redraw': True},
                                          'fromcurrent': True}]
                        },
                        {
                            'label': '暂停',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                            'mode': 'immediate',
                                            'transition': {'duration': 0}}]
                        }
                    ]
                }],
                template='plotly_white'
            )
            
            return fig
        
        except Exception as e:
            self.logger.warning(f"动画对比创建失败: {e}")
            return go.Figure()

class ComparisonAnalyzer:
    """对比分析器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.metric_calculator = MetricCalculator(config)
        self.data_processor = DataProcessor(config)
        self.static_visualizer = StaticVisualizer(config)
        self.interactive_visualizer = InteractiveVisualizer(config)
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_datasets(self, datasets: List[ComparisonData], 
                        reference_dataset: Optional[ComparisonData] = None) -> Dict[str, Any]:
        """对比数据集"""
        try:
            self.logger.info(f"开始对比 {len(datasets)} 个数据集")
            
            # 预处理数据
            processed_datasets = [self.data_processor.preprocess_data(data) for data in datasets]
            
            # 对齐数据
            aligned_datasets = self.data_processor.align_datasets(processed_datasets)
            
            # 执行对比分析
            results = {
                'datasets': [data.name for data in aligned_datasets],
                'comparison_types': [ct.value for ct in self.config.comparison_types],
                'metrics': {},
                'statistics': {},
                'correlations': {},
                'visualizations': {},
                'summary': {}
            }
            
            # 计算指标
            if reference_dataset is not None:
                ref_processed = self.data_processor.preprocess_data(reference_dataset)
                results['metrics'] = self._calculate_metrics_vs_reference(
                    aligned_datasets, ref_processed)
            else:
                results['metrics'] = self._calculate_cross_metrics(aligned_datasets)
            
            # 统计分析
            results['statistics'] = self._perform_statistical_analysis(aligned_datasets)
            
            # 相关性分析
            results['correlations'] = self._calculate_correlations(aligned_datasets)
            
            # 生成可视化
            results['visualizations'] = self._generate_visualizations(aligned_datasets)
            
            # 生成摘要
            results['summary'] = self._generate_summary(results)
            
            # 保存结果
            if self.config.save_results:
                self._save_results(results)
            
            self.logger.info("对比分析完成")
            return results
        
        except Exception as e:
            self.logger.error(f"对比分析失败: {e}")
            return {}
    
    def _calculate_metrics_vs_reference(self, datasets: List[ComparisonData], 
                                       reference: ComparisonData) -> Dict[str, Dict[str, float]]:
        """计算相对于参考数据集的指标"""
        metrics = {}
        
        for data in datasets:
            if data.name != reference.name:
                metrics[data.name] = self.metric_calculator.calculate_metrics(
                    reference.data, data.data)
        
        return metrics
    
    def _calculate_cross_metrics(self, datasets: List[ComparisonData]) -> Dict[str, Dict[str, float]]:
        """计算交叉指标"""
        metrics = {}
        
        for i, data1 in enumerate(datasets):
            for j, data2 in enumerate(datasets):
                if i < j:  # 避免重复计算
                    pair_name = f"{data1.name}_vs_{data2.name}"
                    metrics[pair_name] = self.metric_calculator.calculate_metrics(
                        data1.data, data2.data)
        
        return metrics
    
    def _perform_statistical_analysis(self, datasets: List[ComparisonData]) -> Dict[str, Any]:
        """执行统计分析"""
        statistics = {}
        
        # 描述性统计
        for data in datasets:
            valid_data = data.data[~np.isnan(data.data)]
            if len(valid_data) > 0:
                statistics[data.name] = {
                    'mean': np.mean(valid_data),
                    'median': np.median(valid_data),
                    'std': np.std(valid_data),
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'q25': np.percentile(valid_data, 25),
                    'q75': np.percentile(valid_data, 75),
                    'skewness': stats.skew(valid_data),
                    'kurtosis': stats.kurtosis(valid_data),
                    'count': len(valid_data)
                }
                
                # 置信区间
                ci_lower, ci_upper = self.metric_calculator.calculate_confidence_intervals(valid_data)
                statistics[data.name]['ci_lower'] = ci_lower
                statistics[data.name]['ci_upper'] = ci_upper
        
        # 统计检验
        if len(datasets) >= 2:
            statistics['statistical_tests'] = {}
            for i, data1 in enumerate(datasets):
                for j, data2 in enumerate(datasets):
                    if i < j:
                        pair_name = f"{data1.name}_vs_{data2.name}"
                        
                        # t检验
                        ttest_result = self.metric_calculator.perform_statistical_test(
                            data1.data, data2.data, "ttest")
                        statistics['statistical_tests'][f"{pair_name}_ttest"] = ttest_result
                        
                        # Wilcoxon检验
                        if len(data1.data) == len(data2.data):
                            wilcoxon_result = self.metric_calculator.perform_statistical_test(
                                data1.data, data2.data, "wilcoxon")
                            statistics['statistical_tests'][f"{pair_name}_wilcoxon"] = wilcoxon_result
        
        return statistics
    
    def _calculate_correlations(self, datasets: List[ComparisonData]) -> Dict[str, Any]:
        """计算相关性"""
        correlations = {}
        
        # 创建相关性矩阵
        if len(datasets) > 1:
            # 对齐数据长度
            min_length = min(len(d.data) for d in datasets)
            data_matrix = np.array([d.data[:min_length] for d in datasets]).T
            
            # 移除包含NaN的行
            valid_rows = ~np.any(np.isnan(data_matrix), axis=1)
            clean_data = data_matrix[valid_rows]
            
            if len(clean_data) > 1:
                corr_matrix = np.corrcoef(clean_data.T)
                
                # 转换为DataFrame
                dataset_names = [d.name for d in datasets]
                correlations['matrix'] = pd.DataFrame(
                    corr_matrix, 
                    index=dataset_names, 
                    columns=dataset_names
                )
                
                # 计算显著性
                n = len(clean_data)
                correlations['significance'] = {}
                for i, name1 in enumerate(dataset_names):
                    for j, name2 in enumerate(dataset_names):
                        if i != j:
                            r = corr_matrix[i, j]
                            # t统计量
                            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                            
                            correlations['significance'][f"{name1}_vs_{name2}"] = {
                                'correlation': r,
                                'p_value': p_value,
                                'significant': p_value < self.config.significance_level
                            }
        
        return correlations
    
    def _generate_visualizations(self, datasets: List[ComparisonData]) -> Dict[str, Any]:
        """生成可视化"""
        visualizations = {
            'static': {},
            'interactive': {}
        }
        
        try:
            # 静态可视化
            for viz_type in self.config.visualization_types:
                if viz_type == VisualizationType.LINE_PLOT:
                    filepath = self.static_visualizer.create_line_plot(datasets, str(self.output_dir))
                    if filepath:
                        visualizations['static']['line_plot'] = filepath
                
                elif viz_type == VisualizationType.BOX_PLOT:
                    filepath = self.static_visualizer.create_box_plot(datasets, str(self.output_dir))
                    if filepath:
                        visualizations['static']['box_plot'] = filepath
                
                elif viz_type == VisualizationType.VIOLIN_PLOT:
                    filepath = self.static_visualizer.create_violin_plot(datasets, str(self.output_dir))
                    if filepath:
                        visualizations['static']['violin_plot'] = filepath
                
                elif viz_type == VisualizationType.SCATTER_PLOT and len(datasets) >= 2:
                    for i in range(len(datasets) - 1):
                        filepath = self.static_visualizer.create_scatter_plot(
                            datasets[i], datasets[i+1], str(self.output_dir))
                        if filepath:
                            visualizations['static'][f'scatter_plot_{i}_{i+1}'] = filepath
            
            # 交互式可视化
            if self.config.interactive:
                # 交互式线图
                fig = self.interactive_visualizer.create_interactive_line_plot(datasets)
                if fig.data:
                    visualizations['interactive']['line_plot'] = fig
                
                # 交互式箱线图
                fig = self.interactive_visualizer.create_interactive_box_plot(datasets)
                if fig.data:
                    visualizations['interactive']['box_plot'] = fig
                
                # 交互式散点图
                if len(datasets) >= 2:
                    fig = self.interactive_visualizer.create_interactive_scatter_plot(
                        datasets[0], datasets[1])
                    if fig.data:
                        visualizations['interactive']['scatter_plot'] = fig
        
        except Exception as e:
            self.logger.warning(f"可视化生成失败: {e}")
        
        return visualizations
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要"""
        summary = {
            'total_datasets': len(results['datasets']),
            'comparison_date': datetime.now().isoformat(),
            'best_performing': {},
            'worst_performing': {},
            'key_findings': []
        }
        
        try:
            # 找出最佳和最差表现的数据集
            if results['metrics']:
                metric_scores = {}
                
                for dataset_pair, metrics in results['metrics'].items():
                    for metric_name, value in metrics.items():
                        if not np.isnan(value):
                            if metric_name not in metric_scores:
                                metric_scores[metric_name] = {}
                            metric_scores[metric_name][dataset_pair] = value
                
                # 对于每个指标，找出最佳值
                for metric_name, scores in metric_scores.items():
                    if metric_name in ['r2', 'correlation', 'nse', 'kge', 'ioa']:
                        # 越大越好
                        best_pair = max(scores.items(), key=lambda x: x[1])
                        worst_pair = min(scores.items(), key=lambda x: x[1])
                    else:
                        # 越小越好
                        best_pair = min(scores.items(), key=lambda x: x[1])
                        worst_pair = max(scores.items(), key=lambda x: x[1])
                    
                    summary['best_performing'][metric_name] = {
                        'pair': best_pair[0],
                        'value': best_pair[1]
                    }
                    summary['worst_performing'][metric_name] = {
                        'pair': worst_pair[0],
                        'value': worst_pair[1]
                    }
            
            # 关键发现
            if results['correlations'] and 'significance' in results['correlations']:
                significant_correlations = [
                    pair for pair, data in results['correlations']['significance'].items()
                    if data['significant']
                ]
                if significant_correlations:
                    summary['key_findings'].append(
                        f"发现 {len(significant_correlations)} 对数据集之间存在显著相关性")
            
            if results['statistics'] and 'statistical_tests' in results['statistics']:
                significant_differences = [
                    test for test, data in results['statistics']['statistical_tests'].items()
                    if data['significant']
                ]
                if significant_differences:
                    summary['key_findings'].append(
                        f"发现 {len(significant_differences)} 对数据集之间存在显著差异")
        
        except Exception as e:
            self.logger.warning(f"摘要生成失败: {e}")
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存结果"""
        try:
            # 保存JSON结果（排除可视化对象）
            json_results = results.copy()
            if 'visualizations' in json_results:
                # 只保存静态可视化的文件路径
                json_results['visualizations'] = {
                    'static': json_results['visualizations'].get('static', {})
                }
            
            # 转换DataFrame为字典
            if 'correlations' in json_results and 'matrix' in json_results['correlations']:
                json_results['correlations']['matrix'] = \
                    json_results['correlations']['matrix'].to_dict()
            
            json_file = self.output_dir / 'comparison_results.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存交互式可视化
            if ('visualizations' in results and 
                'interactive' in results['visualizations']):
                
                for viz_name, fig in results['visualizations']['interactive'].items():
                    if hasattr(fig, 'write_html'):
                        html_file = self.output_dir / f'interactive_{viz_name}.html'
                        fig.write_html(str(html_file))
            
            self.logger.info(f"结果已保存到 {self.output_dir}")
        
        except Exception as e:
            self.logger.warning(f"结果保存失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成报告"""
        try:
            report = []
            report.append("# 冰川数据对比分析报告\n")
            report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 基本信息
            report.append("## 基本信息")
            report.append(f"- 数据集数量: {results['summary']['total_datasets']}")
            report.append(f"- 数据集名称: {', '.join(results['datasets'])}")
            report.append(f"- 对比类型: {', '.join(results['comparison_types'])}\n")
            
            # 统计摘要
            if results['statistics']:
                report.append("## 统计摘要")
                for dataset_name, stats in results['statistics'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        report.append(f"### {dataset_name}")
                        report.append(f"- 平均值: {stats['mean']:.3f}")
                        report.append(f"- 标准差: {stats['std']:.3f}")
                        report.append(f"- 最小值: {stats['min']:.3f}")
                        report.append(f"- 最大值: {stats['max']:.3f}")
                        report.append(f"- 数据点数: {stats['count']}\n")
            
            # 性能指标
            if results['metrics']:
                report.append("## 性能指标")
                for pair_name, metrics in results['metrics'].items():
                    report.append(f"### {pair_name}")
                    for metric_name, value in metrics.items():
                        if not np.isnan(value):
                            report.append(f"- {metric_name.upper()}: {value:.3f}")
                    report.append("")
            
            # 相关性分析
            if results['correlations'] and 'matrix' in results['correlations']:
                report.append("## 相关性分析")
                corr_matrix = results['correlations']['matrix']
                if isinstance(corr_matrix, pd.DataFrame):
                    report.append("相关性矩阵:")
                    report.append(corr_matrix.to_string())
                    report.append("")
            
            # 关键发现
            if results['summary']['key_findings']:
                report.append("## 关键发现")
                for finding in results['summary']['key_findings']:
                    report.append(f"- {finding}")
                report.append("")
            
            # 最佳表现
            if results['summary']['best_performing']:
                report.append("## 最佳表现")
                for metric, data in results['summary']['best_performing'].items():
                    report.append(f"- {metric.upper()}: {data['pair']} ({data['value']:.3f})")
                report.append("")
            
            report_text = "\n".join(report)
            
            # 保存报告
            if self.config.save_results:
                report_file = self.output_dir / 'comparison_report.md'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            
            return report_text
        
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            return "报告生成失败"

def create_comparison_analyzer(config: Optional[ComparisonConfig] = None) -> ComparisonAnalyzer:
    """创建对比分析器"""
    if config is None:
        config = ComparisonConfig()
    
    return ComparisonAnalyzer(config)

# Streamlit应用
def create_streamlit_app():
    """创建Streamlit应用"""
    st.set_page_config(
        page_title="冰川数据对比工具",
        page_icon="🏔️",
        layout="wide"
    )
    
    st.title("🏔️ 冰川数据对比分析工具")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("配置选项")
    
    # 对比类型选择
    comparison_types = st.sidebar.multiselect(
        "选择对比类型",
        [ct.value for ct in ComparisonType],
        default=[ComparisonType.MODEL_COMPARISON.value]
    )
    
    # 指标选择
    metrics = st.sidebar.multiselect(
        "选择评估指标",
        [mt.value for mt in MetricType],
        default=[MetricType.RMSE.value, MetricType.R2.value]
    )
    
    # 可视化类型选择
    viz_types = st.sidebar.multiselect(
        "选择可视化类型",
        [vt.value for vt in VisualizationType],
        default=[VisualizationType.LINE_PLOT.value]
    )
    
    # 高级选项
    st.sidebar.subheader("高级选项")
    normalize_data = st.sidebar.checkbox("数据标准化", False)
    remove_outliers = st.sidebar.checkbox("移除异常值", False)
    confidence_level = st.sidebar.slider("置信水平", 0.8, 0.99, 0.95, 0.01)
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("数据上传")
        
        # 文件上传
        uploaded_files = st.file_uploader(
            "上传CSV文件",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            datasets = []
            
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    
                    # 假设第一列是时间，第二列是数据
                    if len(df.columns) >= 2:
                        time_data = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                        value_data = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                        
                        dataset = ComparisonData(
                            name=file.name.replace('.csv', ''),
                            data=value_data.values,
                            time=time_data.values if not time_data.isna().all() else None
                        )
                        datasets.append(dataset)
                        
                        st.success(f"成功加载 {file.name}")
                    else:
                        st.error(f"文件 {file.name} 格式不正确")
                
                except Exception as e:
                    st.error(f"加载文件 {file.name} 失败: {e}")
            
            if datasets:
                # 创建配置
                config = ComparisonConfig(
                    comparison_types=[ComparisonType(ct) for ct in comparison_types],
                    metrics=[MetricType(mt) for mt in metrics],
                    visualization_types=[VisualizationType(vt) for vt in viz_types],
                    normalize_data=normalize_data,
                    remove_outliers=remove_outliers,
                    confidence_level=confidence_level,
                    interactive=True
                )
                
                # 创建分析器
                analyzer = create_comparison_analyzer(config)
                
                # 执行分析
                if st.button("开始分析", type="primary"):
                    with st.spinner("正在进行对比分析..."):
                        results = analyzer.compare_datasets(datasets)
                    
                    if results:
                        st.success("分析完成！")
                        
                        # 显示结果
                        st.header("分析结果")
                        
                        # 摘要
                        if results['summary']:
                            st.subheader("摘要")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("数据集数量", results['summary']['total_datasets'])
                            
                            with col2:
                                if results['summary']['key_findings']:
                                    st.metric("关键发现", len(results['summary']['key_findings']))
                            
                            with col3:
                                st.metric("分析时间", results['summary']['comparison_date'][:19])
                        
                        # 可视化
                        if 'visualizations' in results and 'interactive' in results['visualizations']:
                            st.subheader("交互式可视化")
                            
                            for viz_name, fig in results['visualizations']['interactive'].items():
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 统计表格
                        if results['statistics']:
                            st.subheader("统计摘要")
                            
                            stats_data = []
                            for dataset_name, stats in results['statistics'].items():
                                if isinstance(stats, dict) and 'mean' in stats:
                                    stats_data.append({
                                        '数据集': dataset_name,
                                        '平均值': f"{stats['mean']:.3f}",
                                        '标准差': f"{stats['std']:.3f}",
                                        '最小值': f"{stats['min']:.3f}",
                                        '最大值': f"{stats['max']:.3f}",
                                        '数据点数': stats['count']
                                    })
                            
                            if stats_data:
                                st.dataframe(pd.DataFrame(stats_data))
                        
                        # 性能指标
                        if results['metrics']:
                            st.subheader("性能指标")
                            
                            metrics_data = []
                            for pair_name, metrics in results['metrics'].items():
                                for metric_name, value in metrics.items():
                                    if not np.isnan(value):
                                        metrics_data.append({
                                            '对比对': pair_name,
                                            '指标': metric_name.upper(),
                                            '数值': f"{value:.3f}"
                                        })
                            
                            if metrics_data:
                                st.dataframe(pd.DataFrame(metrics_data))
                        
                        # 生成报告
                        report = analyzer.generate_report(results)
                        
                        st.subheader("分析报告")
                        st.markdown(report)
                        
                        # 下载按钮
                        st.download_button(
                            label="下载报告",
                            data=report,
                            file_name="glacier_comparison_report.md",
                            mime="text/markdown"
                        )
                    
                    else:
                        st.error("分析失败，请检查数据格式")
    
    with col2:
        st.header("使用说明")
        st.markdown("""
        ### 📋 使用步骤
        1. 选择对比类型和评估指标
        2. 上传CSV格式的数据文件
        3. 配置高级选项（可选）
        4. 点击"开始分析"按钮
        5. 查看分析结果和可视化
        
        ### 📁 数据格式要求
        - CSV文件格式
        - 第一列：时间（可选）
        - 第二列：数值数据
        - 支持多个文件同时上传
        
        ### 📊 支持的分析类型
        - 模型对比
        - 时间序列分析
        - 空间分布对比
        - 统计指标评估
        - 不确定性分析
        
        ### 📈 可视化类型
        - 线图：时间序列对比
        - 散点图：相关性分析
        - 箱线图：分布对比
        - 热力图：相关性矩阵
        - 雷达图：多指标对比
        """)

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 示例使用
    def main():
        """主函数"""
        print("冰川数据对比工具示例")
        
        # 创建示例数据
        np.random.seed(42)
        time_points = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # 模拟冰川厚度数据
        thickness_model1 = 100 + 10 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 2, 100)
        thickness_model2 = 98 + 12 * np.sin(np.arange(100) * 0.1 + 0.2) + np.random.normal(0, 2.5, 100)
        thickness_observed = 99 + 11 * np.sin(np.arange(100) * 0.1 + 0.1) + np.random.normal(0, 1.5, 100)
        
        # 创建数据集
        datasets = [
            ComparisonData(
                name="模型1",
                data=thickness_model1,
                time=time_points.values,
                uncertainty=np.random.uniform(1, 3, 100)
            ),
            ComparisonData(
                name="模型2",
                data=thickness_model2,
                time=time_points.values,
                uncertainty=np.random.uniform(1.5, 3.5, 100)
            ),
            ComparisonData(
                name="观测数据",
                data=thickness_observed,
                time=time_points.values,
                uncertainty=np.random.uniform(0.5, 2, 100)
            )
        ]
        
        # 创建配置
        config = ComparisonConfig(
            comparison_types=[ComparisonType.MODEL_COMPARISON, ComparisonType.TIME_SERIES],
            metrics=[MetricType.RMSE, MetricType.R2, MetricType.CORRELATION, MetricType.NSE],
            visualization_types=[
                VisualizationType.LINE_PLOT,
                VisualizationType.SCATTER_PLOT,
                VisualizationType.BOX_PLOT,
                VisualizationType.RADAR_CHART
            ],
            normalize_data=False,
            remove_outliers=True,
            confidence_level=0.95,
            save_results=True,
            output_dir="./comparison_results",
            interactive=True
        )
        
        # 创建分析器
        analyzer = create_comparison_analyzer(config)
        
        # 执行对比分析
        print("\n开始对比分析...")
        results = analyzer.compare_datasets(datasets, reference_dataset=datasets[2])  # 以观测数据为参考
        
        if results:
            print("\n=== 分析完成 ===")
            print(f"数据集数量: {results['summary']['total_datasets']}")
            print(f"关键发现数量: {len(results['summary']['key_findings'])}")
            
            # 显示最佳性能
            if results['summary']['best_performing']:
                print("\n最佳性能:")
                for metric, data in results['summary']['best_performing'].items():
                    print(f"  {metric}: {data['pair']} ({data['value']:.3f})")
            
            # 生成报告
            report = analyzer.generate_report(results)
            print("\n=== 分析报告 ===")
            print(report[:500] + "..." if len(report) > 500 else report)
            
            print(f"\n详细结果已保存到: {config.output_dir}")
        
        else:
            print("分析失败")
    
    # 运行示例
    main()