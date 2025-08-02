#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域对比分析模块

该模块实现了不同区域间冰川变化的对比分析功能，包括多区域统计对比、
空间分布对比、变化趋势对比等。

主要功能:
- 多区域统计对比
- 空间分布对比
- 变化趋势对比
- 区域特征分析
- 对比可视化
- 统计显著性检验

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonType(Enum):
    """对比类型"""
    STATISTICAL = "statistical"                    # 统计对比
    SPATIAL = "spatial"                            # 空间对比
    TEMPORAL = "temporal"                          # 时间对比
    MORPHOLOGICAL = "morphological"                # 形态对比
    CLIMATIC = "climatic"                          # 气候对比
    HYPSOMETRIC = "hypsometric"                    # 高程对比

class RegionType(Enum):
    """区域类型"""
    ADMINISTRATIVE = "administrative"              # 行政区域
    WATERSHED = "watershed"                        # 流域
    ELEVATION_ZONE = "elevation_zone"              # 高程带
    CLIMATE_ZONE = "climate_zone"                  # 气候带
    GLACIER_COMPLEX = "glacier_complex"            # 冰川群
    CUSTOM = "custom"                              # 自定义

class MetricType(Enum):
    """指标类型"""
    AREA = "area"                                  # 面积
    VOLUME = "volume"                              # 体积
    THICKNESS = "thickness"                        # 厚度
    VELOCITY = "velocity"                          # 速度
    ELEVATION = "elevation"                        # 高程
    SLOPE = "slope"                                # 坡度
    ASPECT = "aspect"                              # 坡向
    MASS_BALANCE = "mass_balance"                  # 物质平衡
    TEMPERATURE = "temperature"                    # 温度
    PRECIPITATION = "precipitation"                # 降水

class VisualizationType(Enum):
    """可视化类型"""
    BOX_PLOT = "box_plot"                          # 箱线图
    VIOLIN_PLOT = "violin_plot"                    # 小提琴图
    SCATTER_PLOT = "scatter_plot"                  # 散点图
    BAR_CHART = "bar_chart"                        # 柱状图
    HEATMAP = "heatmap"                            # 热力图
    RADAR_CHART = "radar_chart"                    # 雷达图
    MAP_COMPARISON = "map_comparison"              # 地图对比

@dataclass
class Region:
    """区域定义"""
    name: str                                      # 区域名称
    geometry: Polygon                              # 区域几何
    region_type: RegionType = RegionType.CUSTOM   # 区域类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 区域属性
    
    def __post_init__(self):
        if not isinstance(self.geometry, Polygon):
            raise ValueError("区域几何必须是Polygon类型")

@dataclass
class ComparisonConfig:
    """对比分析配置"""
    # 基本设置
    comparison_type: ComparisonType = ComparisonType.STATISTICAL
    metrics: List[MetricType] = field(default_factory=lambda: [MetricType.AREA, MetricType.THICKNESS])
    
    # 统计设置
    confidence_level: float = 0.95
    statistical_tests: List[str] = field(default_factory=lambda: ['t_test', 'anova', 'kruskal'])
    multiple_comparison_correction: str = 'bonferroni'
    
    # 空间设置
    spatial_resolution: float = 100.0  # 米
    buffer_distance: float = 1000.0    # 缓冲区距离
    
    # 时间设置
    time_aggregation: str = 'annual'   # 时间聚合方式
    trend_analysis: bool = True
    
    # 聚类设置
    enable_clustering: bool = False
    n_clusters: Optional[int] = None
    clustering_method: str = 'kmeans'
    
    # 可视化设置
    visualization_types: List[VisualizationType] = field(
        default_factory=lambda: [VisualizationType.BOX_PLOT, VisualizationType.MAP_COMPARISON]
    )
    color_palette: str = 'Set2'
    figure_size: Tuple[float, float] = (12, 8)
    
    # 输出设置
    save_results: bool = True
    output_format: str = 'png'
    output_dir: str = './regional_comparison'
    
    # 其他设置
    normalize_by_area: bool = True
    remove_outliers: bool = False
    outlier_threshold: float = 3.0

@dataclass
class RegionalData:
    """区域数据"""
    region: Region
    data: Dict[str, np.ndarray]  # 指标数据
    time: np.ndarray             # 时间序列
    coordinates: Optional[Tuple[np.ndarray, np.ndarray]] = None  # 坐标
    metadata: Dict[str, Any] = field(default_factory=dict)

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compare_regions(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """区域对比分析"""
        results = {}
        
        try:
            # 提取数据
            data_dict = self._extract_data(regional_data)
            
            # 描述性统计
            results['descriptive_stats'] = self._calculate_descriptive_stats(data_dict)
            
            # 统计检验
            results['statistical_tests'] = self._perform_statistical_tests(data_dict)
            
            # 相关性分析
            results['correlation_analysis'] = self._analyze_correlations(data_dict)
            
            # 聚类分析
            if self.config.enable_clustering:
                results['clustering'] = self._perform_clustering(data_dict)
            
            return results
        
        except Exception as e:
            self.logger.error(f"区域对比分析失败: {e}")
            return {}
    
    def _extract_data(self, regional_data: List[RegionalData]) -> Dict[str, Dict[str, np.ndarray]]:
        """提取数据"""
        data_dict = {}
        
        for region_data in regional_data:
            region_name = region_data.region.name
            data_dict[region_name] = {}
            
            for metric in self.config.metrics:
                metric_name = metric.value
                if metric_name in region_data.data:
                    data = region_data.data[metric_name]
                    
                    # 移除异常值
                    if self.config.remove_outliers:
                        data = self._remove_outliers(data)
                    
                    # 面积标准化
                    if self.config.normalize_by_area and metric != MetricType.AREA:
                        area = region_data.data.get('area', np.ones_like(data))
                        data = data / area
                    
                    data_dict[region_name][metric_name] = data
        
        return data_dict
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """移除异常值"""
        try:
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            return data[z_scores < self.config.outlier_threshold]
        except:
            return data
    
    def _calculate_descriptive_stats(self, data_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, pd.DataFrame]:
        """计算描述性统计"""
        stats_dict = {}
        
        for metric in self.config.metrics:
            metric_name = metric.value
            stats_data = []
            
            for region_name, region_data in data_dict.items():
                if metric_name in region_data:
                    data = region_data[metric_name]
                    
                    stats_row = {
                        'Region': region_name,
                        'Count': len(data),
                        'Mean': np.nanmean(data),
                        'Std': np.nanstd(data),
                        'Min': np.nanmin(data),
                        'Q25': np.nanpercentile(data, 25),
                        'Median': np.nanmedian(data),
                        'Q75': np.nanpercentile(data, 75),
                        'Max': np.nanmax(data),
                        'Skewness': stats.skew(data, nan_policy='omit'),
                        'Kurtosis': stats.kurtosis(data, nan_policy='omit')
                    }
                    stats_data.append(stats_row)
            
            stats_dict[metric_name] = pd.DataFrame(stats_data)
        
        return stats_dict
    
    def _perform_statistical_tests(self, data_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """执行统计检验"""
        test_results = {}
        
        for metric in self.config.metrics:
            metric_name = metric.value
            test_results[metric_name] = {}
            
            # 收集数据
            groups = []
            group_names = []
            
            for region_name, region_data in data_dict.items():
                if metric_name in region_data:
                    groups.append(region_data[metric_name])
                    group_names.append(region_name)
            
            if len(groups) < 2:
                continue
            
            # t检验（两组比较）
            if 't_test' in self.config.statistical_tests and len(groups) == 2:
                try:
                    statistic, p_value = stats.ttest_ind(groups[0], groups[1], nan_policy='omit')
                    test_results[metric_name]['t_test'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < (1 - self.config.confidence_level)
                    }
                except Exception as e:
                    self.logger.warning(f"t检验失败: {e}")
            
            # 方差分析（多组比较）
            if 'anova' in self.config.statistical_tests and len(groups) > 2:
                try:
                    statistic, p_value = stats.f_oneway(*groups)
                    test_results[metric_name]['anova'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < (1 - self.config.confidence_level)
                    }
                except Exception as e:
                    self.logger.warning(f"方差分析失败: {e}")
            
            # Kruskal-Wallis检验（非参数）
            if 'kruskal' in self.config.statistical_tests and len(groups) > 1:
                try:
                    statistic, p_value = stats.kruskal(*groups, nan_policy='omit')
                    test_results[metric_name]['kruskal'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < (1 - self.config.confidence_level)
                    }
                except Exception as e:
                    self.logger.warning(f"Kruskal-Wallis检验失败: {e}")
        
        return test_results
    
    def _analyze_correlations(self, data_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """分析相关性"""
        correlation_results = {}
        
        try:
            # 构建数据矩阵
            all_data = []
            region_names = []
            metric_names = [metric.value for metric in self.config.metrics]
            
            for region_name, region_data in data_dict.items():
                region_means = []
                for metric_name in metric_names:
                    if metric_name in region_data:
                        region_means.append(np.nanmean(region_data[metric_name]))
                    else:
                        region_means.append(np.nan)
                
                if not all(np.isnan(region_means)):
                    all_data.append(region_means)
                    region_names.append(region_name)
            
            if len(all_data) > 1:
                data_matrix = np.array(all_data)
                
                # 计算相关矩阵
                correlation_matrix = np.corrcoef(data_matrix.T)
                
                correlation_results['matrix'] = correlation_matrix
                correlation_results['metric_names'] = metric_names
                correlation_results['region_names'] = region_names
        
        except Exception as e:
            self.logger.warning(f"相关性分析失败: {e}")
        
        return correlation_results
    
    def _perform_clustering(self, data_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """执行聚类分析"""
        clustering_results = {}
        
        try:
            # 构建特征矩阵
            features = []
            region_names = []
            
            for region_name, region_data in data_dict.items():
                region_features = []
                for metric in self.config.metrics:
                    metric_name = metric.value
                    if metric_name in region_data:
                        # 使用统计特征
                        data = region_data[metric_name]
                        region_features.extend([
                            np.nanmean(data),
                            np.nanstd(data),
                            np.nanmedian(data)
                        ])
                    else:
                        region_features.extend([np.nan, np.nan, np.nan])
                
                if not all(np.isnan(region_features)):
                    features.append(region_features)
                    region_names.append(region_name)
            
            if len(features) > 2:
                features = np.array(features)
                
                # 标准化
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # 确定聚类数
                if self.config.n_clusters is None:
                    n_clusters = min(4, len(features) - 1)
                else:
                    n_clusters = min(self.config.n_clusters, len(features) - 1)
                
                # K-means聚类
                if self.config.clustering_method == 'kmeans':
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    
                    clustering_results['labels'] = cluster_labels
                    clustering_results['centers'] = kmeans.cluster_centers_
                    clustering_results['region_names'] = region_names
                    
                    # 计算轮廓系数
                    if len(set(cluster_labels)) > 1:
                        silhouette = silhouette_score(features_scaled, cluster_labels)
                        clustering_results['silhouette_score'] = silhouette
        
        except Exception as e:
            self.logger.warning(f"聚类分析失败: {e}")
        
        return clustering_results

class SpatialAnalyzer:
    """空间分析器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_spatial_patterns(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """分析空间模式"""
        results = {}
        
        try:
            # 空间自相关
            results['spatial_autocorrelation'] = self._calculate_spatial_autocorrelation(regional_data)
            
            # 空间聚集性
            results['spatial_clustering'] = self._analyze_spatial_clustering(regional_data)
            
            # 距离衰减
            results['distance_decay'] = self._analyze_distance_decay(regional_data)
            
            return results
        
        except Exception as e:
            self.logger.error(f"空间模式分析失败: {e}")
            return {}
    
    def _calculate_spatial_autocorrelation(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """计算空间自相关"""
        autocorr_results = {}
        
        try:
            # 计算区域中心点
            centroids = []
            for region_data in regional_data:
                centroid = region_data.region.geometry.centroid
                centroids.append([centroid.x, centroid.y])
            
            centroids = np.array(centroids)
            
            # 计算距离矩阵
            distances = pdist(centroids)
            distance_matrix = squareform(distances)
            
            # 构建权重矩阵（距离倒数）
            weights = 1.0 / (distance_matrix + 1e-10)
            np.fill_diagonal(weights, 0)
            
            # 对每个指标计算Moran's I
            for metric in self.config.metrics:
                metric_name = metric.value
                values = []
                
                for region_data in regional_data:
                    if metric_name in region_data.data:
                        values.append(np.nanmean(region_data.data[metric_name]))
                    else:
                        values.append(np.nan)
                
                values = np.array(values)
                valid_mask = ~np.isnan(values)
                
                if np.sum(valid_mask) > 3:
                    values_valid = values[valid_mask]
                    weights_valid = weights[np.ix_(valid_mask, valid_mask)]
                    
                    # 计算Moran's I
                    moran_i = self._calculate_morans_i(values_valid, weights_valid)
                    autocorr_results[metric_name] = moran_i
        
        except Exception as e:
            self.logger.warning(f"空间自相关计算失败: {e}")
        
        return autocorr_results
    
    def _calculate_morans_i(self, values: np.ndarray, weights: np.ndarray) -> float:
        """计算Moran's I指数"""
        try:
            n = len(values)
            mean_val = np.mean(values)
            
            # 标准化值
            z = values - mean_val
            
            # 计算Moran's I
            numerator = np.sum(weights * np.outer(z, z))
            denominator = np.sum(z**2)
            
            W = np.sum(weights)
            
            if denominator > 0 and W > 0:
                moran_i = (n / W) * (numerator / denominator)
                return moran_i
            else:
                return 0.0
        
        except:
            return 0.0
    
    def _analyze_spatial_clustering(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """分析空间聚集性"""
        clustering_results = {}
        
        try:
            # 使用最近邻距离分析
            centroids = []
            for region_data in regional_data:
                centroid = region_data.region.geometry.centroid
                centroids.append([centroid.x, centroid.y])
            
            centroids = np.array(centroids)
            
            # 计算最近邻距离
            distances = pdist(centroids)
            distance_matrix = squareform(distances)
            
            # 排除自身（对角线）
            np.fill_diagonal(distance_matrix, np.inf)
            
            # 最近邻距离
            nearest_distances = np.min(distance_matrix, axis=1)
            
            clustering_results['nearest_neighbor_distances'] = nearest_distances
            clustering_results['mean_nearest_distance'] = np.mean(nearest_distances)
            clustering_results['std_nearest_distance'] = np.std(nearest_distances)
        
        except Exception as e:
            self.logger.warning(f"空间聚集性分析失败: {e}")
        
        return clustering_results
    
    def _analyze_distance_decay(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """分析距离衰减"""
        decay_results = {}
        
        try:
            # 计算区域间距离和相似性
            centroids = []
            for region_data in regional_data:
                centroid = region_data.region.geometry.centroid
                centroids.append([centroid.x, centroid.y])
            
            centroids = np.array(centroids)
            distances = pdist(centroids)
            
            for metric in self.config.metrics:
                metric_name = metric.value
                values = []
                
                for region_data in regional_data:
                    if metric_name in region_data.data:
                        values.append(np.nanmean(region_data.data[metric_name]))
                    else:
                        values.append(np.nan)
                
                values = np.array(values)
                valid_mask = ~np.isnan(values)
                
                if np.sum(valid_mask) > 3:
                    values_valid = values[valid_mask]
                    
                    # 计算值的差异
                    value_differences = pdist(values_valid.reshape(-1, 1))
                    
                    # 计算相关性
                    if len(distances) == len(value_differences):
                        correlation, p_value = stats.pearsonr(distances, value_differences)
                        decay_results[metric_name] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
        
        except Exception as e:
            self.logger.warning(f"距离衰减分析失败: {e}")
        
        return decay_results

class ComparisonVisualizer:
    """对比可视化器"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_comparison_plots(self, 
                               regional_data: List[RegionalData],
                               analysis_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """创建对比图表"""
        plots = {}
        
        try:
            for viz_type in self.config.visualization_types:
                if viz_type == VisualizationType.BOX_PLOT:
                    plots['box_plot'] = self._create_box_plot(regional_data)
                
                elif viz_type == VisualizationType.VIOLIN_PLOT:
                    plots['violin_plot'] = self._create_violin_plot(regional_data)
                
                elif viz_type == VisualizationType.SCATTER_PLOT:
                    plots['scatter_plot'] = self._create_scatter_plot(regional_data)
                
                elif viz_type == VisualizationType.BAR_CHART:
                    plots['bar_chart'] = self._create_bar_chart(regional_data)
                
                elif viz_type == VisualizationType.HEATMAP:
                    if 'correlation_analysis' in analysis_results:
                        plots['heatmap'] = self._create_heatmap(analysis_results['correlation_analysis'])
                
                elif viz_type == VisualizationType.RADAR_CHART:
                    plots['radar_chart'] = self._create_radar_chart(regional_data)
                
                elif viz_type == VisualizationType.MAP_COMPARISON:
                    plots['map_comparison'] = self._create_map_comparison(regional_data)
            
            return plots
        
        except Exception as e:
            self.logger.error(f"对比图表创建失败: {e}")
            return {}
    
    def _create_box_plot(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建箱线图"""
        try:
            n_metrics = len(self.config.metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(self.config.metrics):
                metric_name = metric.value
                
                # 收集数据
                plot_data = []
                labels = []
                
                for region_data in regional_data:
                    if metric_name in region_data.data:
                        data = region_data.data[metric_name]
                        if len(data) > 0:
                            plot_data.append(data)
                            labels.append(region_data.region.name)
                
                if plot_data:
                    axes[i].boxplot(plot_data, labels=labels)
                    axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"箱线图创建失败: {e}")
            return plt.figure()
    
    def _create_violin_plot(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建小提琴图"""
        try:
            # 准备数据
            plot_data = []
            
            for region_data in regional_data:
                for metric in self.config.metrics:
                    metric_name = metric.value
                    if metric_name in region_data.data:
                        data = region_data.data[metric_name]
                        for value in data:
                            if not np.isnan(value):
                                plot_data.append({
                                    'Region': region_data.region.name,
                                    'Metric': metric_name,
                                    'Value': value
                                })
            
            if plot_data:
                df = pd.DataFrame(plot_data)
                
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                sns.violinplot(data=df, x='Metric', y='Value', hue='Region', ax=ax)
                ax.set_title('区域指标分布对比')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
            else:
                return plt.figure()
        
        except Exception as e:
            self.logger.error(f"小提琴图创建失败: {e}")
            return plt.figure()
    
    def _create_scatter_plot(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建散点图"""
        try:
            if len(self.config.metrics) < 2:
                return plt.figure()
            
            metric1 = self.config.metrics[0].value
            metric2 = self.config.metrics[1].value
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(regional_data)))
            
            for i, region_data in enumerate(regional_data):
                if metric1 in region_data.data and metric2 in region_data.data:
                    x_data = region_data.data[metric1]
                    y_data = region_data.data[metric2]
                    
                    # 确保数据长度一致
                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]
                    
                    # 移除NaN值
                    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                    x_data = x_data[valid_mask]
                    y_data = y_data[valid_mask]
                    
                    if len(x_data) > 0:
                        ax.scatter(x_data, y_data, 
                                 color=colors[i], 
                                 label=region_data.region.name,
                                 alpha=0.7)
            
            ax.set_xlabel(metric1.replace('_', ' ').title())
            ax.set_ylabel(metric2.replace('_', ' ').title())
            ax.set_title(f'{metric1} vs {metric2}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"散点图创建失败: {e}")
            return plt.figure()
    
    def _create_bar_chart(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建柱状图"""
        try:
            # 计算每个区域每个指标的均值
            region_names = [rd.region.name for rd in regional_data]
            metric_names = [m.value for m in self.config.metrics]
            
            data_matrix = np.zeros((len(region_names), len(metric_names)))
            
            for i, region_data in enumerate(regional_data):
                for j, metric in enumerate(self.config.metrics):
                    metric_name = metric.value
                    if metric_name in region_data.data:
                        data_matrix[i, j] = np.nanmean(region_data.data[metric_name])
                    else:
                        data_matrix[i, j] = 0
            
            # 创建分组柱状图
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            x = np.arange(len(region_names))
            width = 0.8 / len(metric_names)
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))
            
            for j, metric_name in enumerate(metric_names):
                offset = (j - len(metric_names)/2 + 0.5) * width
                ax.bar(x + offset, data_matrix[:, j], width, 
                      label=metric_name.replace('_', ' ').title(),
                      color=colors[j])
            
            ax.set_xlabel('区域')
            ax.set_ylabel('指标值')
            ax.set_title('区域指标对比')
            ax.set_xticks(x)
            ax.set_xticklabels(region_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"柱状图创建失败: {e}")
            return plt.figure()
    
    def _create_heatmap(self, correlation_data: Dict[str, Any]) -> plt.Figure:
        """创建热力图"""
        try:
            if 'matrix' not in correlation_data:
                return plt.figure()
            
            correlation_matrix = correlation_data['matrix']
            metric_names = correlation_data.get('metric_names', [])
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # 设置标签
            if metric_names:
                ax.set_xticks(range(len(metric_names)))
                ax.set_yticks(range(len(metric_names)))
                ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45)
                ax.set_yticklabels([name.replace('_', ' ').title() for name in metric_names])
            
            # 添加数值标注
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black")
            
            ax.set_title('指标相关性矩阵')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            return fig
        
        except Exception as e:
            self.logger.error(f"热力图创建失败: {e}")
            return plt.figure()
    
    def _create_radar_chart(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建雷达图"""
        try:
            # 计算标准化的指标值
            metric_names = [m.value for m in self.config.metrics]
            region_names = [rd.region.name for rd in regional_data]
            
            # 收集数据
            data_matrix = np.zeros((len(region_names), len(metric_names)))
            
            for i, region_data in enumerate(regional_data):
                for j, metric in enumerate(self.config.metrics):
                    metric_name = metric.value
                    if metric_name in region_data.data:
                        data_matrix[i, j] = np.nanmean(region_data.data[metric_name])
            
            # 标准化到0-1范围
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data_matrix.T).T
            
            # 创建雷达图
            angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=self.config.figure_size, subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(region_names)))
            
            for i, region_name in enumerate(region_names):
                values = data_normalized[i].tolist()
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, 'o-', linewidth=2, label=region_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names])
            ax.set_ylim(0, 1)
            ax.set_title('区域指标雷达图', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"雷达图创建失败: {e}")
            return plt.figure()
    
    def _create_map_comparison(self, regional_data: List[RegionalData]) -> plt.Figure:
        """创建地图对比"""
        try:
            fig, ax = plt.subplots(
                figsize=self.config.figure_size,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            
            # 添加地理要素
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            
            # 计算地图范围
            all_bounds = []
            for region_data in regional_data:
                bounds = region_data.region.geometry.bounds
                all_bounds.append(bounds)
            
            if all_bounds:
                min_x = min(b[0] for b in all_bounds)
                min_y = min(b[1] for b in all_bounds)
                max_x = max(b[2] for b in all_bounds)
                max_y = max(b[3] for b in all_bounds)
                
                # 添加边距
                margin = 0.1
                dx = (max_x - min_x) * margin
                dy = (max_y - min_y) * margin
                
                ax.set_extent([min_x-dx, max_x+dx, min_y-dy, max_y+dy], crs=ccrs.PlateCarree())
            
            # 绘制区域
            colors = plt.cm.Set2(np.linspace(0, 1, len(regional_data)))
            
            for i, region_data in enumerate(regional_data):
                # 获取区域几何
                geometry = region_data.region.geometry
                
                if hasattr(geometry, 'exterior'):
                    # Polygon
                    coords = list(geometry.exterior.coords)
                    lons, lats = zip(*coords)
                    
                    ax.plot(lons, lats, color=colors[i], linewidth=2, 
                           label=region_data.region.name, transform=ccrs.PlateCarree())
                    ax.fill(lons, lats, color=colors[i], alpha=0.3, transform=ccrs.PlateCarree())
            
            ax.set_title('区域分布对比')
            ax.legend(loc='upper right')
            
            # 添加网格
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.xlabels_top = False
            gl.ylabels_right = False
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"地图对比创建失败: {e}")
            return plt.figure()

class RegionalComparator:
    """区域对比分析器"""
    
    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化分析器
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        self.visualizer = ComparisonVisualizer(self.config)
    
    def compare_regions(self, regional_data: List[RegionalData]) -> Dict[str, Any]:
        """执行区域对比分析"""
        results = {}
        
        try:
            self.logger.info(f"开始对比分析 {len(regional_data)} 个区域")
            
            # 统计分析
            if self.config.comparison_type in [ComparisonType.STATISTICAL, ComparisonType.TEMPORAL]:
                results['statistical_analysis'] = self.statistical_analyzer.compare_regions(regional_data)
            
            # 空间分析
            if self.config.comparison_type in [ComparisonType.SPATIAL, ComparisonType.MORPHOLOGICAL]:
                results['spatial_analysis'] = self.spatial_analyzer.analyze_spatial_patterns(regional_data)
            
            # 可视化
            results['visualizations'] = self.visualizer.create_comparison_plots(regional_data, results)
            
            # 生成报告
            results['report'] = self._generate_report(regional_data, results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"区域对比分析失败: {e}")
            return {}
    
    def _generate_report(self, regional_data: List[RegionalData], analysis_results: Dict[str, Any]) -> str:
        """生成分析报告"""
        try:
            report_lines = []
            report_lines.append("# 区域对比分析报告")
            report_lines.append("")
            
            # 基本信息
            report_lines.append("## 基本信息")
            report_lines.append(f"- 分析区域数量: {len(regional_data)}")
            report_lines.append(f"- 分析指标: {', '.join([m.value for m in self.config.metrics])}")
            report_lines.append(f"- 对比类型: {self.config.comparison_type.value}")
            report_lines.append("")
            
            # 区域列表
            report_lines.append("## 分析区域")
            for i, region_data in enumerate(regional_data, 1):
                report_lines.append(f"{i}. {region_data.region.name} ({region_data.region.region_type.value})")
            report_lines.append("")
            
            # 统计结果
            if 'statistical_analysis' in analysis_results:
                stats_results = analysis_results['statistical_analysis']
                
                if 'descriptive_stats' in stats_results:
                    report_lines.append("## 描述性统计")
                    for metric, stats_df in stats_results['descriptive_stats'].items():
                        report_lines.append(f"### {metric.replace('_', ' ').title()}")
                        report_lines.append(stats_df.to_string())
                        report_lines.append("")
                
                if 'statistical_tests' in stats_results:
                    report_lines.append("## 统计检验结果")
                    for metric, tests in stats_results['statistical_tests'].items():
                        report_lines.append(f"### {metric.replace('_', ' ').title()}")
                        for test_name, test_result in tests.items():
                            significance = "显著" if test_result.get('significant', False) else "不显著"
                            report_lines.append(f"- {test_name}: p={test_result.get('p_value', 'N/A'):.4f} ({significance})")
                        report_lines.append("")
            
            # 空间分析结果
            if 'spatial_analysis' in analysis_results:
                spatial_results = analysis_results['spatial_analysis']
                report_lines.append("## 空间分析结果")
                
                if 'spatial_autocorrelation' in spatial_results:
                    report_lines.append("### 空间自相关")
                    for metric, moran_i in spatial_results['spatial_autocorrelation'].items():
                        report_lines.append(f"- {metric}: Moran's I = {moran_i:.4f}")
                    report_lines.append("")
            
            return "\n".join(report_lines)
        
        except Exception as e:
            self.logger.warning(f"报告生成失败: {e}")
            return "报告生成失败"
    
    def save_results(self, results: Dict[str, Any]):
        """保存分析结果"""
        try:
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 保存可视化结果
            if 'visualizations' in results:
                for plot_name, fig in results['visualizations'].items():
                    if isinstance(fig, plt.Figure):
                        filepath = os.path.join(self.config.output_dir, f"{plot_name}.{self.config.output_format}")
                        fig.savefig(filepath, dpi=300, bbox_inches='tight')
                        self.logger.info(f"图表已保存: {filepath}")
            
            # 保存报告
            if 'report' in results:
                report_path = os.path.join(self.config.output_dir, 'comparison_report.md')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(results['report'])
                self.logger.info(f"报告已保存: {report_path}")
            
            # 保存统计结果
            if 'statistical_analysis' in results and 'descriptive_stats' in results['statistical_analysis']:
                stats_path = os.path.join(self.config.output_dir, 'descriptive_statistics.xlsx')
                with pd.ExcelWriter(stats_path) as writer:
                    for metric, stats_df in results['statistical_analysis']['descriptive_stats'].items():
                        stats_df.to_excel(writer, sheet_name=metric, index=False)
                self.logger.info(f"统计结果已保存: {stats_path}")
        
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")

def create_regional_comparator(config: Optional[ComparisonConfig] = None) -> RegionalComparator:
    """创建区域对比分析器"""
    return RegionalComparator(config)

if __name__ == "__main__":
    # 测试代码
    print("开始区域对比分析测试...")
    
    # 生成测试数据
    np.random.seed(42)
    
    # 创建测试区域
    regions = [
        Region(
            name="区域A",
            geometry=box(85.0, 28.0, 85.5, 28.5),
            region_type=RegionType.CUSTOM
        ),
        Region(
            name="区域B", 
            geometry=box(85.5, 28.0, 86.0, 28.5),
            region_type=RegionType.CUSTOM
        ),
        Region(
            name="区域C",
            geometry=box(85.0, 28.5, 85.5, 29.0),
            region_type=RegionType.CUSTOM
        )
    ]
    
    # 生成测试数据
    regional_data = []
    for i, region in enumerate(regions):
        # 模拟不同区域的数据特征
        n_points = 100
        
        data = {
            'area': np.random.normal(1000 + i*200, 100, n_points),
            'thickness': np.random.normal(50 + i*10, 15, n_points),
            'velocity': np.random.normal(10 + i*5, 3, n_points),
            'elevation': np.random.normal(4000 + i*500, 200, n_points)
        }
        
        time = np.arange(2000, 2020)
        
        regional_data.append(RegionalData(
            region=region,
            data=data,
            time=time
        ))
    
    # 创建对比配置
    config = ComparisonConfig(
        comparison_type=ComparisonType.STATISTICAL,
        metrics=[MetricType.AREA, MetricType.THICKNESS, MetricType.VELOCITY, MetricType.ELEVATION],
        visualization_types=[
            VisualizationType.BOX_PLOT,
            VisualizationType.VIOLIN_PLOT,
            VisualizationType.SCATTER_PLOT,
            VisualizationType.BAR_CHART,
            VisualizationType.HEATMAP,
            VisualizationType.RADAR_CHART,
            VisualizationType.MAP_COMPARISON
        ],
        enable_clustering=True
    )
    
    # 创建对比分析器
    comparator = create_regional_comparator(config)
    
    # 执行对比分析
    results = comparator.compare_regions(regional_data)
    
    print(f"\n分析完成，生成了 {len(results)} 类结果")
    
    # 显示部分结果
    if 'statistical_analysis' in results:
        print("\n统计分析结果:")
        if 'descriptive_stats' in results['statistical_analysis']:
            for metric, stats_df in results['statistical_analysis']['descriptive_stats'].items():
                print(f"\n{metric}:")
                print(stats_df.head())
    
    if 'visualizations' in results:
        print(f"\n生成了 {len(results['visualizations'])} 个可视化图表")
        # 显示第一个图表
        for plot_name, fig in results['visualizations'].items():
            if isinstance(fig, plt.Figure):
                fig.show()
                break
    
    # 保存结果
    comparator.save_results(results)
    
    print("\n区域对比分析测试完成！")