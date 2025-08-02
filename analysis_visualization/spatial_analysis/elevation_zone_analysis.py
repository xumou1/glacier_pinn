#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高程带分析模块

该模块实现了冰川高程带分析功能，包括高程分布统计、高程带变化分析、
高程梯度分析等。

主要功能:
- 高程带划分
- 高程分布统计
- 高程带变化分析
- 高程梯度分析
- 高程-气候关系
- 高程带可视化

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElevationZoneType(Enum):
    """高程带类型"""
    FIXED_INTERVAL = "fixed_interval"          # 固定间隔
    QUANTILE_BASED = "quantile_based"          # 分位数
    NATURAL_BREAKS = "natural_breaks"          # 自然断点
    CLIMATE_BASED = "climate_based"            # 气候分带
    VEGETATION_BASED = "vegetation_based"      # 植被分带
    CUSTOM = "custom"                          # 自定义

class AnalysisType(Enum):
    """分析类型"""
    DISTRIBUTION = "distribution"              # 分布分析
    CHANGE_DETECTION = "change_detection"      # 变化检测
    GRADIENT_ANALYSIS = "gradient_analysis"    # 梯度分析
    CORRELATION = "correlation"                # 相关性分析
    TREND_ANALYSIS = "trend_analysis"          # 趋势分析
    HYPSOMETRY = "hypsometry"                  # 高程曲线

class MetricType(Enum):
    """指标类型"""
    AREA = "area"                              # 面积
    VOLUME = "volume"                          # 体积
    THICKNESS = "thickness"                    # 厚度
    VELOCITY = "velocity"                      # 速度
    MASS_BALANCE = "mass_balance"              # 物质平衡
    TEMPERATURE = "temperature"                # 温度
    PRECIPITATION = "precipitation"            # 降水
    SLOPE = "slope"                            # 坡度
    ASPECT = "aspect"                          # 坡向

class VisualizationType(Enum):
    """可视化类型"""
    ELEVATION_PROFILE = "elevation_profile"    # 高程剖面
    HYPSOMETRIC_CURVE = "hypsometric_curve"    # 高程曲线
    ZONE_MAP = "zone_map"                      # 分带地图
    GRADIENT_MAP = "gradient_map"              # 梯度地图
    SCATTER_PLOT = "scatter_plot"              # 散点图
    BOX_PLOT = "box_plot"                      # 箱线图
    HEATMAP = "heatmap"                        # 热力图
    CONTOUR_PLOT = "contour_plot"              # 等高线图

@dataclass
class ElevationZone:
    """高程带定义"""
    zone_id: int                               # 带编号
    min_elevation: float                       # 最低高程
    max_elevation: float                       # 最高高程
    name: str = ""                             # 带名称
    description: str = ""                      # 描述
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    
    def __post_init__(self):
        if self.min_elevation >= self.max_elevation:
            raise ValueError("最低高程必须小于最高高程")
        
        if not self.name:
            self.name = f"Zone_{self.zone_id}_{int(self.min_elevation)}-{int(self.max_elevation)}m"

@dataclass
class ElevationConfig:
    """高程分析配置"""
    # 分带设置
    zone_type: ElevationZoneType = ElevationZoneType.FIXED_INTERVAL
    zone_interval: float = 200.0               # 固定间隔(米)
    n_zones: int = 10                          # 分带数量
    custom_breaks: Optional[List[float]] = None # 自定义断点
    
    # 分析设置
    analysis_types: List[AnalysisType] = field(
        default_factory=lambda: [AnalysisType.DISTRIBUTION, AnalysisType.GRADIENT_ANALYSIS]
    )
    metrics: List[MetricType] = field(
        default_factory=lambda: [MetricType.AREA, MetricType.THICKNESS]
    )
    
    # 统计设置
    confidence_level: float = 0.95
    trend_method: str = 'linear'               # 趋势分析方法
    smoothing_window: int = 3                  # 平滑窗口
    
    # 空间设置
    spatial_resolution: float = 100.0          # 空间分辨率(米)
    buffer_distance: float = 500.0             # 缓冲距离
    
    # 可视化设置
    visualization_types: List[VisualizationType] = field(
        default_factory=lambda: [VisualizationType.ELEVATION_PROFILE, VisualizationType.ZONE_MAP]
    )
    color_scheme: str = 'terrain'              # 颜色方案
    figure_size: Tuple[float, float] = (12, 8)
    
    # 输出设置
    save_results: bool = True
    output_format: str = 'png'
    output_dir: str = './elevation_analysis'
    
    # 其他设置
    remove_outliers: bool = False
    outlier_threshold: float = 3.0
    normalize_by_area: bool = True

@dataclass
class ElevationData:
    """高程数据"""
    # 空间坐标
    x: np.ndarray                              # x坐标
    y: np.ndarray                              # y坐标
    elevation: np.ndarray                      # 高程数据
    
    # 时间信息
    time: Optional[np.ndarray] = None          # 时间序列
    
    # 其他数据
    data: Dict[str, np.ndarray] = field(default_factory=dict)  # 其他指标数据
    
    # 元数据
    coordinate_system: str = "EPSG:4326"
    units: str = "meters"
    
    def __post_init__(self):
        # 检查数据维度一致性
        if self.elevation.shape != (len(self.y), len(self.x)):
            raise ValueError("高程数据维度与坐标不匹配")

class ElevationZoneGenerator:
    """高程带生成器"""
    
    def __init__(self, config: ElevationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_zones(self, elevation_data: ElevationData) -> List[ElevationZone]:
        """生成高程带"""
        try:
            elevation = elevation_data.elevation
            valid_elevation = elevation[~np.isnan(elevation)]
            
            if len(valid_elevation) == 0:
                raise ValueError("没有有效的高程数据")
            
            min_elev = np.min(valid_elevation)
            max_elev = np.max(valid_elevation)
            
            if self.config.zone_type == ElevationZoneType.FIXED_INTERVAL:
                return self._generate_fixed_interval_zones(min_elev, max_elev)
            
            elif self.config.zone_type == ElevationZoneType.QUANTILE_BASED:
                return self._generate_quantile_zones(valid_elevation)
            
            elif self.config.zone_type == ElevationZoneType.NATURAL_BREAKS:
                return self._generate_natural_break_zones(valid_elevation)
            
            elif self.config.zone_type == ElevationZoneType.CUSTOM:
                return self._generate_custom_zones()
            
            else:
                return self._generate_fixed_interval_zones(min_elev, max_elev)
        
        except Exception as e:
            self.logger.error(f"高程带生成失败: {e}")
            return []
    
    def _generate_fixed_interval_zones(self, min_elev: float, max_elev: float) -> List[ElevationZone]:
        """生成固定间隔高程带"""
        zones = []
        
        # 计算分带边界
        n_intervals = int(np.ceil((max_elev - min_elev) / self.config.zone_interval))
        
        for i in range(n_intervals):
            zone_min = min_elev + i * self.config.zone_interval
            zone_max = min(min_elev + (i + 1) * self.config.zone_interval, max_elev)
            
            if zone_min < zone_max:
                zone = ElevationZone(
                    zone_id=i,
                    min_elevation=zone_min,
                    max_elevation=zone_max,
                    name=f"Zone_{i}_{int(zone_min)}-{int(zone_max)}m"
                )
                zones.append(zone)
        
        return zones
    
    def _generate_quantile_zones(self, elevation_data: np.ndarray) -> List[ElevationZone]:
        """生成分位数高程带"""
        zones = []
        
        # 计算分位数
        quantiles = np.linspace(0, 1, self.config.n_zones + 1)
        breaks = np.percentile(elevation_data, quantiles * 100)
        
        for i in range(len(breaks) - 1):
            zone = ElevationZone(
                zone_id=i,
                min_elevation=breaks[i],
                max_elevation=breaks[i + 1],
                name=f"Quantile_{i}_{int(breaks[i])}-{int(breaks[i+1])}m"
            )
            zones.append(zone)
        
        return zones
    
    def _generate_natural_break_zones(self, elevation_data: np.ndarray) -> List[ElevationZone]:
        """生成自然断点高程带"""
        try:
            from sklearn.cluster import KMeans
            
            # 使用K-means聚类找自然断点
            kmeans = KMeans(n_clusters=self.config.n_zones, random_state=42)
            elevation_reshaped = elevation_data.reshape(-1, 1)
            labels = kmeans.fit_predict(elevation_reshaped)
            
            # 获取聚类中心并排序
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            
            zones = []
            for i, idx in enumerate(sorted_indices):
                cluster_data = elevation_data[labels == idx]
                zone_min = np.min(cluster_data)
                zone_max = np.max(cluster_data)
                
                zone = ElevationZone(
                    zone_id=i,
                    min_elevation=zone_min,
                    max_elevation=zone_max,
                    name=f"Natural_{i}_{int(zone_min)}-{int(zone_max)}m"
                )
                zones.append(zone)
            
            return zones
        
        except Exception as e:
            self.logger.warning(f"自然断点分带失败，使用固定间隔: {e}")
            return self._generate_fixed_interval_zones(
                np.min(elevation_data), np.max(elevation_data)
            )
    
    def _generate_custom_zones(self) -> List[ElevationZone]:
        """生成自定义高程带"""
        if not self.config.custom_breaks:
            raise ValueError("自定义分带需要提供断点")
        
        breaks = sorted(self.config.custom_breaks)
        zones = []
        
        for i in range(len(breaks) - 1):
            zone = ElevationZone(
                zone_id=i,
                min_elevation=breaks[i],
                max_elevation=breaks[i + 1],
                name=f"Custom_{i}_{int(breaks[i])}-{int(breaks[i+1])}m"
            )
            zones.append(zone)
        
        return zones

class ElevationAnalyzer:
    """高程分析器"""
    
    def __init__(self, config: ElevationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.zone_generator = ElevationZoneGenerator(config)
    
    def analyze_elevation_zones(self, elevation_data: ElevationData) -> Dict[str, Any]:
        """分析高程带"""
        results = {}
        
        try:
            # 生成高程带
            zones = self.zone_generator.generate_zones(elevation_data)
            results['zones'] = zones
            
            # 分带统计
            results['zone_statistics'] = self._calculate_zone_statistics(elevation_data, zones)
            
            # 执行各种分析
            for analysis_type in self.config.analysis_types:
                if analysis_type == AnalysisType.DISTRIBUTION:
                    results['distribution_analysis'] = self._analyze_distribution(elevation_data, zones)
                
                elif analysis_type == AnalysisType.GRADIENT_ANALYSIS:
                    results['gradient_analysis'] = self._analyze_gradients(elevation_data, zones)
                
                elif analysis_type == AnalysisType.TREND_ANALYSIS:
                    results['trend_analysis'] = self._analyze_trends(elevation_data, zones)
                
                elif analysis_type == AnalysisType.HYPSOMETRY:
                    results['hypsometric_analysis'] = self._analyze_hypsometry(elevation_data, zones)
                
                elif analysis_type == AnalysisType.CORRELATION:
                    results['correlation_analysis'] = self._analyze_correlations(elevation_data, zones)
            
            return results
        
        except Exception as e:
            self.logger.error(f"高程带分析失败: {e}")
            return {}
    
    def _calculate_zone_statistics(self, 
                                  elevation_data: ElevationData, 
                                  zones: List[ElevationZone]) -> Dict[str, pd.DataFrame]:
        """计算分带统计"""
        statistics = {}
        
        try:
            for metric in self.config.metrics:
                metric_name = metric.value
                
                if metric_name == 'area':
                    # 计算面积统计
                    stats_data = self._calculate_area_statistics(elevation_data, zones)
                elif metric_name in elevation_data.data:
                    # 计算其他指标统计
                    stats_data = self._calculate_metric_statistics(
                        elevation_data, zones, metric_name
                    )
                else:
                    continue
                
                statistics[metric_name] = pd.DataFrame(stats_data)
            
            return statistics
        
        except Exception as e:
            self.logger.warning(f"分带统计计算失败: {e}")
            return {}
    
    def _calculate_area_statistics(self, 
                                  elevation_data: ElevationData, 
                                  zones: List[ElevationZone]) -> List[Dict[str, Any]]:
        """计算面积统计"""
        stats_data = []
        
        # 计算像元面积
        dx = np.abs(elevation_data.x[1] - elevation_data.x[0]) if len(elevation_data.x) > 1 else self.config.spatial_resolution
        dy = np.abs(elevation_data.y[1] - elevation_data.y[0]) if len(elevation_data.y) > 1 else self.config.spatial_resolution
        pixel_area = dx * dy
        
        for zone in zones:
            # 创建高程掩膜
            mask = ((elevation_data.elevation >= zone.min_elevation) & 
                   (elevation_data.elevation < zone.max_elevation) &
                   (~np.isnan(elevation_data.elevation)))
            
            # 计算统计量
            pixel_count = np.sum(mask)
            total_area = pixel_count * pixel_area
            
            stats_row = {
                'zone_id': zone.zone_id,
                'zone_name': zone.name,
                'min_elevation': zone.min_elevation,
                'max_elevation': zone.max_elevation,
                'pixel_count': pixel_count,
                'area_km2': total_area / 1e6,  # 转换为平方公里
                'area_percentage': 0.0  # 稍后计算
            }
            stats_data.append(stats_row)
        
        # 计算面积百分比
        total_area = sum(row['area_km2'] for row in stats_data)
        if total_area > 0:
            for row in stats_data:
                row['area_percentage'] = (row['area_km2'] / total_area) * 100
        
        return stats_data
    
    def _calculate_metric_statistics(self, 
                                   elevation_data: ElevationData, 
                                   zones: List[ElevationZone],
                                   metric_name: str) -> List[Dict[str, Any]]:
        """计算指标统计"""
        stats_data = []
        metric_data = elevation_data.data[metric_name]
        
        for zone in zones:
            # 创建高程掩膜
            mask = ((elevation_data.elevation >= zone.min_elevation) & 
                   (elevation_data.elevation < zone.max_elevation) &
                   (~np.isnan(elevation_data.elevation)) &
                   (~np.isnan(metric_data)))
            
            if np.any(mask):
                zone_data = metric_data[mask]
                
                # 移除异常值
                if self.config.remove_outliers:
                    z_scores = np.abs(stats.zscore(zone_data))
                    zone_data = zone_data[z_scores < self.config.outlier_threshold]
                
                if len(zone_data) > 0:
                    stats_row = {
                        'zone_id': zone.zone_id,
                        'zone_name': zone.name,
                        'min_elevation': zone.min_elevation,
                        'max_elevation': zone.max_elevation,
                        'mean_elevation': np.mean(elevation_data.elevation[mask]),
                        'count': len(zone_data),
                        'mean': np.mean(zone_data),
                        'std': np.std(zone_data),
                        'min': np.min(zone_data),
                        'q25': np.percentile(zone_data, 25),
                        'median': np.median(zone_data),
                        'q75': np.percentile(zone_data, 75),
                        'max': np.max(zone_data),
                        'skewness': stats.skew(zone_data),
                        'kurtosis': stats.kurtosis(zone_data)
                    }
                    stats_data.append(stats_row)
        
        return stats_data
    
    def _analyze_distribution(self, 
                            elevation_data: ElevationData, 
                            zones: List[ElevationZone]) -> Dict[str, Any]:
        """分析分布特征"""
        distribution_results = {}
        
        try:
            # 高程分布
            elevation = elevation_data.elevation[~np.isnan(elevation_data.elevation)]
            
            distribution_results['elevation_histogram'] = {
                'bins': np.histogram(elevation, bins=50)[1],
                'counts': np.histogram(elevation, bins=50)[0]
            }
            
            # 各带分布
            zone_distributions = {}
            for zone in zones:
                mask = ((elevation_data.elevation >= zone.min_elevation) & 
                       (elevation_data.elevation < zone.max_elevation) &
                       (~np.isnan(elevation_data.elevation)))
                
                zone_elevation = elevation_data.elevation[mask]
                if len(zone_elevation) > 0:
                    zone_distributions[zone.name] = {
                        'data': zone_elevation,
                        'mean': np.mean(zone_elevation),
                        'std': np.std(zone_elevation),
                        'count': len(zone_elevation)
                    }
            
            distribution_results['zone_distributions'] = zone_distributions
            
            return distribution_results
        
        except Exception as e:
            self.logger.warning(f"分布分析失败: {e}")
            return {}
    
    def _analyze_gradients(self, 
                         elevation_data: ElevationData, 
                         zones: List[ElevationZone]) -> Dict[str, Any]:
        """分析梯度特征"""
        gradient_results = {}
        
        try:
            # 计算高程梯度
            dx = np.abs(elevation_data.x[1] - elevation_data.x[0]) if len(elevation_data.x) > 1 else self.config.spatial_resolution
            dy = np.abs(elevation_data.y[1] - elevation_data.y[0]) if len(elevation_data.y) > 1 else self.config.spatial_resolution
            
            grad_y, grad_x = np.gradient(elevation_data.elevation, dy, dx)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            aspect = np.arctan2(grad_y, grad_x) * 180 / np.pi
            
            # 各带梯度统计
            zone_gradients = {}
            for zone in zones:
                mask = ((elevation_data.elevation >= zone.min_elevation) & 
                       (elevation_data.elevation < zone.max_elevation) &
                       (~np.isnan(elevation_data.elevation)))
                
                if np.any(mask):
                    zone_slope = slope[mask]
                    zone_aspect = aspect[mask]
                    
                    # 移除NaN值
                    valid_slope = zone_slope[~np.isnan(zone_slope)]
                    valid_aspect = zone_aspect[~np.isnan(zone_aspect)]
                    
                    if len(valid_slope) > 0:
                        zone_gradients[zone.name] = {
                            'mean_slope': np.mean(valid_slope),
                            'std_slope': np.std(valid_slope),
                            'max_slope': np.max(valid_slope),
                            'mean_aspect': np.mean(valid_aspect) if len(valid_aspect) > 0 else np.nan,
                            'slope_data': valid_slope,
                            'aspect_data': valid_aspect
                        }
            
            gradient_results['zone_gradients'] = zone_gradients
            gradient_results['slope_map'] = slope
            gradient_results['aspect_map'] = aspect
            
            return gradient_results
        
        except Exception as e:
            self.logger.warning(f"梯度分析失败: {e}")
            return {}
    
    def _analyze_trends(self, 
                       elevation_data: ElevationData, 
                       zones: List[ElevationZone]) -> Dict[str, Any]:
        """分析趋势特征"""
        trend_results = {}
        
        try:
            if elevation_data.time is None:
                return {}
            
            # 对每个指标分析趋势
            for metric in self.config.metrics:
                metric_name = metric.value
                if metric_name not in elevation_data.data:
                    continue
                
                metric_data = elevation_data.data[metric_name]
                zone_trends = {}
                
                for zone in zones:
                    mask = ((elevation_data.elevation >= zone.min_elevation) & 
                           (elevation_data.elevation < zone.max_elevation) &
                           (~np.isnan(elevation_data.elevation)))
                    
                    if np.any(mask):
                        # 计算时间序列均值
                        if metric_data.ndim == 3:  # [time, y, x]
                            zone_time_series = np.nanmean(metric_data[:, mask], axis=1)
                        else:
                            zone_time_series = np.nanmean(metric_data[mask])
                            zone_time_series = np.repeat(zone_time_series, len(elevation_data.time))
                        
                        # 趋势分析
                        if len(zone_time_series) > 1:
                            trend_result = self._calculate_trend(
                                elevation_data.time, zone_time_series
                            )
                            zone_trends[zone.name] = trend_result
                
                trend_results[metric_name] = zone_trends
            
            return trend_results
        
        except Exception as e:
            self.logger.warning(f"趋势分析失败: {e}")
            return {}
    
    def _calculate_trend(self, time: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
        """计算趋势"""
        try:
            # 移除NaN值
            valid_mask = ~np.isnan(data)
            if np.sum(valid_mask) < 2:
                return {}
            
            time_valid = time[valid_mask]
            data_valid = data[valid_mask]
            
            # 线性回归
            if self.config.trend_method == 'linear':
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_valid, data_valid)
                
                return {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_error': std_err,
                    'trend_type': 'linear'
                }
            
            # 多项式回归
            elif self.config.trend_method == 'polynomial':
                poly_features = PolynomialFeatures(degree=2)
                time_poly = poly_features.fit_transform(time_valid.reshape(-1, 1))
                
                model = LinearRegression()
                model.fit(time_poly, data_valid)
                
                predicted = model.predict(time_poly)
                r2 = r2_score(data_valid, predicted)
                
                return {
                    'coefficients': model.coef_,
                    'intercept': model.intercept_,
                    'r_squared': r2,
                    'trend_type': 'polynomial'
                }
            
            else:
                return {}
        
        except Exception as e:
            self.logger.warning(f"趋势计算失败: {e}")
            return {}
    
    def _analyze_hypsometry(self, 
                          elevation_data: ElevationData, 
                          zones: List[ElevationZone]) -> Dict[str, Any]:
        """分析高程曲线"""
        hypsometric_results = {}
        
        try:
            elevation = elevation_data.elevation
            valid_elevation = elevation[~np.isnan(elevation)]
            
            if len(valid_elevation) == 0:
                return {}
            
            # 计算高程曲线
            elevation_range = np.linspace(
                np.min(valid_elevation), 
                np.max(valid_elevation), 
                100
            )
            
            cumulative_area = []
            total_pixels = len(valid_elevation)
            
            for elev in elevation_range:
                pixels_above = np.sum(valid_elevation >= elev)
                cumulative_area.append(pixels_above / total_pixels)
            
            hypsometric_results['elevation_range'] = elevation_range
            hypsometric_results['cumulative_area'] = np.array(cumulative_area)
            
            # 计算高程积分
            hypsometric_integral = np.trapz(cumulative_area, 
                                          (elevation_range - np.min(elevation_range)) / 
                                          (np.max(elevation_range) - np.min(elevation_range)))
            
            hypsometric_results['hypsometric_integral'] = hypsometric_integral
            
            # 分带高程曲线
            zone_hypsometry = {}
            for zone in zones:
                mask = ((elevation >= zone.min_elevation) & 
                       (elevation < zone.max_elevation) &
                       (~np.isnan(elevation)))
                
                if np.any(mask):
                    zone_elevation = elevation[mask]
                    zone_range = np.linspace(
                        zone.min_elevation,
                        zone.max_elevation,
                        50
                    )
                    
                    zone_cumulative = []
                    zone_total = len(zone_elevation)
                    
                    for elev in zone_range:
                        pixels_above = np.sum(zone_elevation >= elev)
                        zone_cumulative.append(pixels_above / zone_total if zone_total > 0 else 0)
                    
                    zone_hypsometry[zone.name] = {
                        'elevation_range': zone_range,
                        'cumulative_area': np.array(zone_cumulative)
                    }
            
            hypsometric_results['zone_hypsometry'] = zone_hypsometry
            
            return hypsometric_results
        
        except Exception as e:
            self.logger.warning(f"高程曲线分析失败: {e}")
            return {}
    
    def _analyze_correlations(self, 
                            elevation_data: ElevationData, 
                            zones: List[ElevationZone]) -> Dict[str, Any]:
        """分析相关性"""
        correlation_results = {}
        
        try:
            # 高程与各指标的相关性
            elevation = elevation_data.elevation
            
            for metric in self.config.metrics:
                metric_name = metric.value
                if metric_name not in elevation_data.data:
                    continue
                
                metric_data = elevation_data.data[metric_name]
                
                # 展平数据
                elev_flat = elevation.flatten()
                metric_flat = metric_data.flatten() if metric_data.ndim == 2 else np.mean(metric_data, axis=0).flatten()
                
                # 移除NaN值
                valid_mask = ~(np.isnan(elev_flat) | np.isnan(metric_flat))
                
                if np.sum(valid_mask) > 10:
                    elev_valid = elev_flat[valid_mask]
                    metric_valid = metric_flat[valid_mask]
                    
                    # 计算相关系数
                    correlation, p_value = stats.pearsonr(elev_valid, metric_valid)
                    spearman_corr, spearman_p = stats.spearmanr(elev_valid, metric_valid)
                    
                    correlation_results[metric_name] = {
                        'pearson_correlation': correlation,
                        'pearson_p_value': p_value,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'sample_size': len(elev_valid)
                    }
            
            return correlation_results
        
        except Exception as e:
            self.logger.warning(f"相关性分析失败: {e}")
            return {}

class ElevationVisualizer:
    """高程可视化器"""
    
    def __init__(self, config: ElevationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_visualizations(self, 
                            elevation_data: ElevationData,
                            analysis_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """创建可视化图表"""
        plots = {}
        
        try:
            for viz_type in self.config.visualization_types:
                if viz_type == VisualizationType.ELEVATION_PROFILE:
                    plots['elevation_profile'] = self._create_elevation_profile(
                        elevation_data, analysis_results
                    )
                
                elif viz_type == VisualizationType.HYPSOMETRIC_CURVE:
                    if 'hypsometric_analysis' in analysis_results:
                        plots['hypsometric_curve'] = self._create_hypsometric_curve(
                            analysis_results['hypsometric_analysis']
                        )
                
                elif viz_type == VisualizationType.ZONE_MAP:
                    plots['zone_map'] = self._create_zone_map(
                        elevation_data, analysis_results
                    )
                
                elif viz_type == VisualizationType.GRADIENT_MAP:
                    if 'gradient_analysis' in analysis_results:
                        plots['gradient_map'] = self._create_gradient_map(
                            elevation_data, analysis_results['gradient_analysis']
                        )
                
                elif viz_type == VisualizationType.SCATTER_PLOT:
                    if 'correlation_analysis' in analysis_results:
                        plots['scatter_plot'] = self._create_scatter_plot(
                            elevation_data, analysis_results['correlation_analysis']
                        )
                
                elif viz_type == VisualizationType.BOX_PLOT:
                    plots['box_plot'] = self._create_box_plot(
                        elevation_data, analysis_results
                    )
                
                elif viz_type == VisualizationType.HEATMAP:
                    plots['heatmap'] = self._create_heatmap(
                        elevation_data, analysis_results
                    )
            
            return plots
        
        except Exception as e:
            self.logger.error(f"可视化创建失败: {e}")
            return {}
    
    def _create_elevation_profile(self, 
                                elevation_data: ElevationData,
                                analysis_results: Dict[str, Any]) -> plt.Figure:
        """创建高程剖面图"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=self.config.figure_size)
            
            # 高程分布直方图
            elevation = elevation_data.elevation[~np.isnan(elevation_data.elevation)]
            axes[0].hist(elevation, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_xlabel('高程 (m)')
            axes[0].set_ylabel('频数')
            axes[0].set_title('高程分布直方图')
            axes[0].grid(True, alpha=0.3)
            
            # 分带统计
            if 'zone_statistics' in analysis_results and 'area' in analysis_results['zone_statistics']:
                stats_df = analysis_results['zone_statistics']['area']
                
                zone_centers = (stats_df['min_elevation'] + stats_df['max_elevation']) / 2
                axes[1].bar(zone_centers, stats_df['area_km2'], 
                           width=(stats_df['max_elevation'] - stats_df['min_elevation']) * 0.8,
                           alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1].set_xlabel('高程 (m)')
                axes[1].set_ylabel('面积 (km²)')
                axes[1].set_title('分带面积分布')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"高程剖面图创建失败: {e}")
            return plt.figure()
    
    def _create_hypsometric_curve(self, hypsometric_data: Dict[str, Any]) -> plt.Figure:
        """创建高程曲线图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 主高程曲线
            if 'elevation_range' in hypsometric_data and 'cumulative_area' in hypsometric_data:
                elevation_range = hypsometric_data['elevation_range']
                cumulative_area = hypsometric_data['cumulative_area']
                
                # 标准化高程
                normalized_elevation = (elevation_range - np.min(elevation_range)) / \
                                     (np.max(elevation_range) - np.min(elevation_range))
                
                ax.plot(cumulative_area, normalized_elevation, 
                       linewidth=3, color='blue', label='总体高程曲线')
            
            # 分带高程曲线
            if 'zone_hypsometry' in hypsometric_data:
                colors = plt.cm.viridis(np.linspace(0, 1, len(hypsometric_data['zone_hypsometry'])))
                
                for i, (zone_name, zone_data) in enumerate(hypsometric_data['zone_hypsometry'].items()):
                    zone_range = zone_data['elevation_range']
                    zone_cumulative = zone_data['cumulative_area']
                    
                    # 标准化高程
                    normalized_zone_elevation = (zone_range - np.min(zone_range)) / \
                                               (np.max(zone_range) - np.min(zone_range))
                    
                    ax.plot(zone_cumulative, normalized_zone_elevation,
                           linewidth=2, color=colors[i], alpha=0.7,
                           label=zone_name)
            
            ax.set_xlabel('累积面积比例')
            ax.set_ylabel('标准化高程')
            ax.set_title('高程曲线')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加高程积分值
            if 'hypsometric_integral' in hypsometric_data:
                integral = hypsometric_data['hypsometric_integral']
                ax.text(0.05, 0.95, f'高程积分: {integral:.3f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"高程曲线图创建失败: {e}")
            return plt.figure()
    
    def _create_zone_map(self, 
                        elevation_data: ElevationData,
                        analysis_results: Dict[str, Any]) -> plt.Figure:
        """创建分带地图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 创建分带图像
            zones = analysis_results.get('zones', [])
            zone_map = np.full_like(elevation_data.elevation, np.nan)
            
            for zone in zones:
                mask = ((elevation_data.elevation >= zone.min_elevation) & 
                       (elevation_data.elevation < zone.max_elevation))
                zone_map[mask] = zone.zone_id
            
            # 绘制分带
            im = ax.imshow(zone_map, cmap='terrain', aspect='auto')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('高程带编号')
            
            # 设置坐标轴
            if len(elevation_data.x) > 1 and len(elevation_data.y) > 1:
                ax.set_xticks(np.linspace(0, len(elevation_data.x)-1, 5))
                ax.set_yticks(np.linspace(0, len(elevation_data.y)-1, 5))
                ax.set_xticklabels([f'{elevation_data.x[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.x)-1, 5)])
                ax.set_yticklabels([f'{elevation_data.y[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.y)-1, 5)])
            
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_title('高程带分布图')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"分带地图创建失败: {e}")
            return plt.figure()
    
    def _create_gradient_map(self, 
                           elevation_data: ElevationData,
                           gradient_data: Dict[str, Any]) -> plt.Figure:
        """创建梯度地图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 坡度图
            if 'slope_map' in gradient_data:
                slope_map = gradient_data['slope_map']
                im1 = axes[0].imshow(slope_map, cmap='Reds', aspect='auto')
                axes[0].set_title('坡度分布')
                plt.colorbar(im1, ax=axes[0], label='坡度')
            
            # 坡向图
            if 'aspect_map' in gradient_data:
                aspect_map = gradient_data['aspect_map']
                im2 = axes[1].imshow(aspect_map, cmap='hsv', aspect='auto')
                axes[1].set_title('坡向分布')
                plt.colorbar(im2, ax=axes[1], label='坡向 (度)')
            
            # 设置坐标轴
            for ax in axes:
                if len(elevation_data.x) > 1 and len(elevation_data.y) > 1:
                    ax.set_xticks(np.linspace(0, len(elevation_data.x)-1, 5))
                    ax.set_yticks(np.linspace(0, len(elevation_data.y)-1, 5))
                    ax.set_xticklabels([f'{elevation_data.x[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.x)-1, 5)])
                    ax.set_yticklabels([f'{elevation_data.y[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.y)-1, 5)])
                
                ax.set_xlabel('经度')
                ax.set_ylabel('纬度')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"梯度地图创建失败: {e}")
            return plt.figure()
    
    def _create_scatter_plot(self, 
                           elevation_data: ElevationData,
                           correlation_data: Dict[str, Any]) -> plt.Figure:
        """创建散点图"""
        try:
            n_metrics = len(correlation_data)
            if n_metrics == 0:
                return plt.figure()
            
            fig, axes = plt.subplots(1, min(n_metrics, 3), figsize=(5*min(n_metrics, 3), 5))
            if n_metrics == 1:
                axes = [axes]
            
            elevation = elevation_data.elevation.flatten()
            
            for i, (metric_name, corr_data) in enumerate(list(correlation_data.items())[:3]):
                if metric_name in elevation_data.data:
                    metric_data = elevation_data.data[metric_name]
                    metric_flat = metric_data.flatten() if metric_data.ndim == 2 else np.mean(metric_data, axis=0).flatten()
                    
                    # 移除NaN值
                    valid_mask = ~(np.isnan(elevation) | np.isnan(metric_flat))
                    elev_valid = elevation[valid_mask]
                    metric_valid = metric_flat[valid_mask]
                    
                    if len(elev_valid) > 0:
                        axes[i].scatter(elev_valid, metric_valid, alpha=0.5, s=1)
                        axes[i].set_xlabel('高程 (m)')
                        axes[i].set_ylabel(metric_name.replace('_', ' ').title())
                        
                        # 添加相关系数
                        corr = corr_data.get('pearson_correlation', 0)
                        axes[i].set_title(f'{metric_name}\nr = {corr:.3f}')
                        axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"散点图创建失败: {e}")
            return plt.figure()
    
    def _create_box_plot(self, 
                        elevation_data: ElevationData,
                        analysis_results: Dict[str, Any]) -> plt.Figure:
        """创建箱线图"""
        try:
            zones = analysis_results.get('zones', [])
            if not zones:
                return plt.figure()
            
            # 选择第一个可用指标
            metric_name = None
            for metric in self.config.metrics:
                if metric.value in elevation_data.data:
                    metric_name = metric.value
                    break
            
            if not metric_name:
                return plt.figure()
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 收集各带数据
            zone_data = []
            zone_labels = []
            
            metric_data = elevation_data.data[metric_name]
            
            for zone in zones:
                mask = ((elevation_data.elevation >= zone.min_elevation) & 
                       (elevation_data.elevation < zone.max_elevation) &
                       (~np.isnan(elevation_data.elevation)))
                
                if np.any(mask):
                    if metric_data.ndim == 2:
                        zone_values = metric_data[mask]
                    else:
                        zone_values = np.mean(metric_data[:, mask], axis=0)
                    
                    zone_values = zone_values[~np.isnan(zone_values)]
                    
                    if len(zone_values) > 0:
                        zone_data.append(zone_values)
                        zone_labels.append(f'{int(zone.min_elevation)}-{int(zone.max_elevation)}m')
            
            if zone_data:
                ax.boxplot(zone_data, labels=zone_labels)
                ax.set_xlabel('高程带')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(f'{metric_name.replace("_", " ").title()}的高程带分布')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"箱线图创建失败: {e}")
            return plt.figure()
    
    def _create_heatmap(self, 
                       elevation_data: ElevationData,
                       analysis_results: Dict[str, Any]) -> plt.Figure:
        """创建热力图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 使用高程数据创建热力图
            im = ax.imshow(elevation_data.elevation, cmap=self.config.color_scheme, aspect='auto')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('高程 (m)')
            
            # 设置坐标轴
            if len(elevation_data.x) > 1 and len(elevation_data.y) > 1:
                ax.set_xticks(np.linspace(0, len(elevation_data.x)-1, 5))
                ax.set_yticks(np.linspace(0, len(elevation_data.y)-1, 5))
                ax.set_xticklabels([f'{elevation_data.x[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.x)-1, 5)])
                ax.set_yticklabels([f'{elevation_data.y[int(i)]:.2f}' for i in np.linspace(0, len(elevation_data.y)-1, 5)])
            
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_title('高程分布热力图')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"热力图创建失败: {e}")
            return plt.figure()

class ElevationZoneAnalyzer:
    """高程带分析器主类"""
    
    def __init__(self, config: Optional[ElevationConfig] = None):
        self.config = config or ElevationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.analyzer = ElevationAnalyzer(self.config)
        self.visualizer = ElevationVisualizer(self.config)
    
    def analyze(self, elevation_data: ElevationData) -> Dict[str, Any]:
        """执行高程带分析"""
        try:
            self.logger.info("开始高程带分析...")
            
            # 执行分析
            analysis_results = self.analyzer.analyze_elevation_zones(elevation_data)
            
            # 创建可视化
            visualizations = self.visualizer.create_visualizations(elevation_data, analysis_results)
            analysis_results['visualizations'] = visualizations
            
            # 生成报告
            analysis_results['report'] = self._generate_report(elevation_data, analysis_results)
            
            self.logger.info("高程带分析完成")
            return analysis_results
        
        except Exception as e:
            self.logger.error(f"高程带分析失败: {e}")
            return {}
    
    def _generate_report(self, elevation_data: ElevationData, analysis_results: Dict[str, Any]) -> str:
        """生成分析报告"""
        try:
            report_lines = []
            report_lines.append("# 高程带分析报告")
            report_lines.append("")
            
            # 基本信息
            elevation = elevation_data.elevation[~np.isnan(elevation_data.elevation)]
            report_lines.append("## 基本信息")
            report_lines.append(f"- 高程范围: {np.min(elevation):.1f} - {np.max(elevation):.1f} m")
            report_lines.append(f"- 平均高程: {np.mean(elevation):.1f} m")
            report_lines.append(f"- 高程标准差: {np.std(elevation):.1f} m")
            report_lines.append(f"- 分析指标: {', '.join([m.value for m in self.config.metrics])}")
            report_lines.append("")
            
            # 分带信息
            zones = analysis_results.get('zones', [])
            if zones:
                report_lines.append("## 高程带划分")
                report_lines.append(f"- 分带方法: {self.config.zone_type.value}")
                report_lines.append(f"- 分带数量: {len(zones)}")
                report_lines.append("")
                
                for zone in zones:
                    report_lines.append(f"### {zone.name}")
                    report_lines.append(f"- 高程范围: {zone.min_elevation:.1f} - {zone.max_elevation:.1f} m")
                    report_lines.append("")
            
            # 统计结果
            if 'zone_statistics' in analysis_results:
                report_lines.append("## 分带统计")
                
                for metric_name, stats_df in analysis_results['zone_statistics'].items():
                    report_lines.append(f"### {metric_name.replace('_', ' ').title()}")
                    report_lines.append(stats_df.to_string(index=False))
                    report_lines.append("")
            
            # 相关性分析
            if 'correlation_analysis' in analysis_results:
                report_lines.append("## 高程相关性")
                
                for metric_name, corr_data in analysis_results['correlation_analysis'].items():
                    pearson_r = corr_data.get('pearson_correlation', 0)
                    p_value = corr_data.get('pearson_p_value', 1)
                    significance = "显著" if p_value < 0.05 else "不显著"
                    
                    report_lines.append(f"- {metric_name}: r = {pearson_r:.3f} (p = {p_value:.3f}, {significance})")
                
                report_lines.append("")
            
            # 高程积分
            if 'hypsometric_analysis' in analysis_results:
                hypsometric_data = analysis_results['hypsometric_analysis']
                if 'hypsometric_integral' in hypsometric_data:
                    integral = hypsometric_data['hypsometric_integral']
                    report_lines.append("## 高程积分")
                    report_lines.append(f"- 高程积分值: {integral:.3f}")
                    
                    # 解释高程积分
                    if integral > 0.6:
                        interpretation = "年轻地貌，以高海拔区域为主"
                    elif integral > 0.4:
                        interpretation = "成熟地貌，高程分布相对均匀"
                    else:
                        interpretation = "老年地貌，以低海拔区域为主"
                    
                    report_lines.append(f"- 地貌特征: {interpretation}")
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
                report_path = os.path.join(self.config.output_dir, 'elevation_analysis_report.md')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(results['report'])
                self.logger.info(f"报告已保存: {report_path}")
            
            # 保存统计数据
            if 'zone_statistics' in results:
                for metric_name, stats_df in results['zone_statistics'].items():
                    csv_path = os.path.join(self.config.output_dir, f'{metric_name}_statistics.csv')
                    stats_df.to_csv(csv_path, index=False, encoding='utf-8')
                    self.logger.info(f"统计数据已保存: {csv_path}")
        
        except Exception as e:
            self.logger.error(f"结果保存失败: {e}")

def create_elevation_zone_analyzer(config: Optional[ElevationConfig] = None) -> ElevationZoneAnalyzer:
    """创建高程带分析器"""
    return ElevationZoneAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建测试数据
    x = np.linspace(90.0, 91.0, 100)
    y = np.linspace(28.0, 29.0, 100)
    X, Y = np.meshgrid(x, y)
    
    # 模拟高程数据（喜马拉雅山脉特征）
    elevation = 3000 + 2000 * np.exp(-((X - 90.5)**2 + (Y - 28.5)**2) / 0.1) + \
                200 * np.random.randn(*X.shape)
    
    # 模拟其他数据
    thickness = np.maximum(0, elevation - 3500 + 100 * np.random.randn(*X.shape))
    velocity = 50 * np.exp(-thickness / 100) + 10 * np.random.randn(*X.shape)
    
    # 创建高程数据对象
    elevation_data = ElevationData(
        x=x,
        y=y,
        elevation=elevation,
        data={
            'thickness': thickness,
            'velocity': velocity
        }
    )
    
    # 创建配置
    config = ElevationConfig(
        zone_type=ElevationZoneType.FIXED_INTERVAL,
        zone_interval=300.0,
        analysis_types=[
            AnalysisType.DISTRIBUTION,
            AnalysisType.GRADIENT_ANALYSIS,
            AnalysisType.HYPSOMETRY,
            AnalysisType.CORRELATION
        ],
        metrics=[MetricType.AREA, MetricType.THICKNESS, MetricType.VELOCITY],
        visualization_types=[
            VisualizationType.ELEVATION_PROFILE,
            VisualizationType.HYPSOMETRIC_CURVE,
            VisualizationType.ZONE_MAP,
            VisualizationType.GRADIENT_MAP,
            VisualizationType.SCATTER_PLOT,
            VisualizationType.BOX_PLOT
        ],
        save_results=True,
        output_dir='./test_elevation_analysis'
    )
    
    # 创建分析器
    analyzer = create_elevation_zone_analyzer(config)
    
    # 执行分析
    print("开始高程带分析...")
    results = analyzer.analyze(elevation_data)
    
    # 打印结果摘要
    if results:
        print("\n=== 分析结果摘要 ===")
        
        # 分带信息
        zones = results.get('zones', [])
        print(f"生成高程带数量: {len(zones)}")
        
        # 统计信息
        if 'zone_statistics' in results:
            for metric_name, stats_df in results['zone_statistics'].items():
                print(f"\n{metric_name.upper()}统计:")
                print(stats_df[['zone_name', 'mean', 'std', 'count']].head())
        
        # 相关性
        if 'correlation_analysis' in results:
            print("\n高程相关性:")
            for metric_name, corr_data in results['correlation_analysis'].items():
                r = corr_data.get('pearson_correlation', 0)
                p = corr_data.get('pearson_p_value', 1)
                print(f"  {metric_name}: r = {r:.3f}, p = {p:.3f}")
        
        # 高程积分
        if 'hypsometric_analysis' in results:
            hypsometric_data = results['hypsometric_analysis']
            if 'hypsometric_integral' in hypsometric_data:
                integral = hypsometric_data['hypsometric_integral']
                print(f"\n高程积分: {integral:.3f}")
        
        # 保存结果
        analyzer.save_results(results)
        print("\n结果已保存到输出目录")
    
    else:
        print("分析失败")
    
    print("高程带分析测试完成")