#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星数据验证模块

该模块实现了与卫星观测数据的对比验证，用于评估模型在卫星数据上的表现，包括：
- 多种卫星数据源支持
- 时空配准和重采样
- 光谱和几何校正
- 云掩膜和质量控制
- 多尺度验证分析
- 时间序列一致性检验

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats, ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, RegularGridInterpolator
import pandas as pd
from datetime import datetime, timedelta

class SatelliteType(Enum):
    """卫星类型"""
    LANDSAT = "landsat"  # Landsat系列
    MODIS = "modis"  # MODIS
    SENTINEL = "sentinel"  # Sentinel系列
    ASTER = "aster"  # ASTER
    SPOT = "spot"  # SPOT
    WORLDVIEW = "worldview"  # WorldView
    QUICKBIRD = "quickbird"  # QuickBird
    IKONOS = "ikonos"  # IKONOS
    GOES = "goes"  # GOES
    AVHRR = "avhrr"  # AVHRR
    UNKNOWN = "unknown"  # 未知卫星

class DataProduct(Enum):
    """数据产品类型"""
    SURFACE_TEMPERATURE = "surface_temperature"  # 地表温度
    SURFACE_REFLECTANCE = "surface_reflectance"  # 地表反射率
    NDVI = "ndvi"  # 归一化植被指数
    NDSI = "ndsi"  # 归一化雪指数
    ALBEDO = "albedo"  # 反照率
    EMISSIVITY = "emissivity"  # 发射率
    SNOW_COVER = "snow_cover"  # 雪盖
    ICE_VELOCITY = "ice_velocity"  # 冰流速
    ELEVATION = "elevation"  # 高程
    PRECIPITATION = "precipitation"  # 降水
    CLOUD_MASK = "cloud_mask"  # 云掩膜
    QUALITY_FLAGS = "quality_flags"  # 质量标记

class ProcessingLevel(Enum):
    """处理级别"""
    L1A = "l1a"  # 原始数据
    L1B = "l1b"  # 辐射校正
    L1C = "l1c"  # 几何校正
    L2A = "l2a"  # 大气校正
    L2B = "l2b"  # 地表参数
    L3 = "l3"  # 时空合成
    L4 = "l4"  # 模型产品

class QualityFlag(Enum):
    """质量标记"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    CLOUD = "cloud"  # 云覆盖
    SHADOW = "shadow"  # 阴影
    SNOW = "snow"  # 雪覆盖
    WATER = "water"  # 水体
    INVALID = "invalid"  # 无效

class ValidationScale(Enum):
    """验证尺度"""
    PIXEL = "pixel"  # 像素级
    PATCH = "patch"  # 斑块级
    REGION = "region"  # 区域级
    GLACIER = "glacier"  # 冰川级
    BASIN = "basin"  # 流域级

@dataclass
class SatelliteObservation:
    """卫星观测数据"""
    observation_id: str
    satellite_type: SatelliteType
    data_product: DataProduct
    processing_level: ProcessingLevel
    
    # 时空信息
    acquisition_time: datetime
    center_lat: float  # 中心纬度
    center_lon: float  # 中心经度
    spatial_resolution: float  # 空间分辨率 (m)
    
    # 数据数组
    data: np.ndarray  # 主要数据
    coordinates: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lat, lon)坐标
    quality_mask: Optional[np.ndarray] = None  # 质量掩膜
    cloud_mask: Optional[np.ndarray] = None  # 云掩膜
    
    # 元数据
    units: str = ""
    scale_factor: float = 1.0
    offset: float = 0.0
    fill_value: float = -9999.0
    
    # 质量信息
    overall_quality: QualityFlag = QualityFlag.UNKNOWN
    cloud_coverage: float = 0.0  # 云覆盖率 (0-1)
    data_coverage: float = 1.0  # 数据覆盖率 (0-1)
    
    # 几何信息
    projection: Optional[str] = None  # 投影信息
    geotransform: Optional[Tuple[float, ...]] = None  # 地理变换参数
    
    # 传感器信息
    sensor: Optional[str] = None
    band_info: Optional[Dict[str, Any]] = None
    
    def get_scaled_data(self) -> np.ndarray:
        """获取缩放后的数据"""
        scaled_data = self.data * self.scale_factor + self.offset
        # 处理填充值
        if self.fill_value is not None:
            scaled_data = np.where(self.data == self.fill_value, np.nan, scaled_data)
        return scaled_data
    
    def get_valid_data_mask(self) -> np.ndarray:
        """获取有效数据掩膜"""
        mask = np.ones_like(self.data, dtype=bool)
        
        # 排除填充值
        if self.fill_value is not None:
            mask &= (self.data != self.fill_value)
        
        # 排除云覆盖
        if self.cloud_mask is not None:
            mask &= ~self.cloud_mask
        
        # 应用质量掩膜
        if self.quality_mask is not None:
            mask &= self.quality_mask
        
        return mask

@dataclass
class SatelliteValidationConfig:
    """卫星验证配置"""
    # 数据源设置
    supported_satellites: List[SatelliteType] = None
    supported_products: List[DataProduct] = None
    min_processing_level: ProcessingLevel = ProcessingLevel.L2A
    
    # 质量控制
    max_cloud_coverage: float = 0.3  # 最大云覆盖率
    min_data_coverage: float = 0.8  # 最小数据覆盖率
    min_quality: QualityFlag = QualityFlag.FAIR
    
    # 时空匹配
    temporal_window: float = 24.0  # 时间窗口 (hours)
    spatial_tolerance: float = 1000.0  # 空间容差 (m)
    resampling_method: str = "bilinear"  # 重采样方法
    
    # 验证尺度
    validation_scales: List[ValidationScale] = None
    aggregation_methods: List[str] = None
    
    # 几何校正
    geometric_correction: bool = True
    coregistration_tolerance: float = 0.5  # 配准容差 (pixels)
    
    # 大气校正
    atmospheric_correction: bool = True
    aerosol_correction: bool = False
    
    # 统计分析
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # 时间序列分析
    time_series_analysis: bool = True
    trend_analysis: bool = True
    seasonality_analysis: bool = True
    
    # 异常值检测
    outlier_detection: bool = True
    outlier_threshold: float = 3.0
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "satellite_validation_plots"
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.supported_satellites is None:
            self.supported_satellites = [
                SatelliteType.LANDSAT,
                SatelliteType.MODIS,
                SatelliteType.SENTINEL
            ]
        
        if self.supported_products is None:
            self.supported_products = [
                DataProduct.SURFACE_TEMPERATURE,
                DataProduct.ALBEDO,
                DataProduct.NDVI,
                DataProduct.NDSI
            ]
        
        if self.validation_scales is None:
            self.validation_scales = [
                ValidationScale.PIXEL,
                ValidationScale.PATCH,
                ValidationScale.REGION
            ]
        
        if self.aggregation_methods is None:
            self.aggregation_methods = ["mean", "median", "std"]

class GeometricProcessor:
    """几何处理器"""
    
    def __init__(self, config: SatelliteValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def coregister_data(self, 
                       reference_data: np.ndarray,
                       target_data: np.ndarray,
                       reference_coords: Tuple[np.ndarray, np.ndarray],
                       target_coords: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """数据配准"""
        coregistration_result = {
            'registered_data': target_data.copy(),
            'transformation_matrix': np.eye(3),
            'registration_error': 0.0,
            'success': False
        }
        
        try:
            # 简化的配准实现：基于相关性的平移配准
            ref_lat, ref_lon = reference_coords
            tar_lat, tar_lon = target_coords
            
            # 计算坐标差异
            lat_diff = np.mean(ref_lat) - np.mean(tar_lat)
            lon_diff = np.mean(ref_lon) - np.mean(tar_lon)
            
            # 应用平移校正
            corrected_lat = tar_lat + lat_diff
            corrected_lon = tar_lon + lon_diff
            
            # 重采样到参考网格
            registered_data = self._resample_to_grid(
                target_data, (corrected_lat, corrected_lon), reference_coords
            )
            
            coregistration_result.update({
                'registered_data': registered_data,
                'lat_shift': lat_diff,
                'lon_shift': lon_diff,
                'success': True
            })
        
        except Exception as e:
            self.logger.warning(f"数据配准失败: {e}")
        
        return coregistration_result
    
    def _resample_to_grid(self, 
                         data: np.ndarray,
                         source_coords: Tuple[np.ndarray, np.ndarray],
                         target_coords: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """重采样到目标网格"""
        source_lat, source_lon = source_coords
        target_lat, target_lon = target_coords
        
        # 展平坐标和数据
        source_points = np.column_stack([
            source_lat.flatten(), source_lon.flatten()
        ])
        data_flat = data.flatten()
        
        # 移除无效值
        valid_mask = ~np.isnan(data_flat)
        source_points = source_points[valid_mask]
        data_flat = data_flat[valid_mask]
        
        if len(data_flat) == 0:
            return np.full_like(target_lat, np.nan)
        
        # 插值到目标网格
        target_points = np.column_stack([
            target_lat.flatten(), target_lon.flatten()
        ])
        
        try:
            interpolated = griddata(
                source_points, data_flat, target_points,
                method=self.config.resampling_method, fill_value=np.nan
            )
            return interpolated.reshape(target_lat.shape)
        except Exception as e:
            self.logger.warning(f"网格重采样失败: {e}")
            return np.full_like(target_lat, np.nan)
    
    def calculate_spatial_statistics(self, 
                                   data1: np.ndarray, 
                                   data2: np.ndarray,
                                   coordinates: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """计算空间统计量"""
        stats_result = {}
        
        try:
            # 有效数据掩膜
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            
            if not valid_mask.any():
                return stats_result
            
            valid_data1 = data1[valid_mask]
            valid_data2 = data2[valid_mask]
            
            # 基本统计量
            stats_result.update({
                'correlation': float(np.corrcoef(valid_data1, valid_data2)[0, 1]),
                'rmse': float(np.sqrt(np.mean((valid_data1 - valid_data2) ** 2))),
                'mae': float(np.mean(np.abs(valid_data1 - valid_data2))),
                'bias': float(np.mean(valid_data1 - valid_data2)),
                'valid_pixels': int(np.sum(valid_mask)),
                'total_pixels': int(data1.size)
            })
            
            # 空间自相关（简化实现）
            if len(valid_data1) > 10:
                # Moran's I 的简化计算
                residuals = valid_data1 - valid_data2
                mean_residual = np.mean(residuals)
                
                # 简化的空间权重（基于距离）
                lat, lon = coordinates
                valid_lat = lat[valid_mask]
                valid_lon = lon[valid_mask]
                
                if len(valid_lat) > 1:
                    coords = np.column_stack([valid_lat, valid_lon])
                    distances = cdist(coords, coords)
                    
                    # 避免除零
                    distances[distances == 0] = np.inf
                    weights = 1.0 / distances
                    weights[np.isinf(weights)] = 0
                    
                    # 计算Moran's I
                    n = len(residuals)
                    numerator = 0
                    denominator = 0
                    total_weight = 0
                    
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                w_ij = weights[i, j]
                                numerator += w_ij * (residuals[i] - mean_residual) * (residuals[j] - mean_residual)
                                total_weight += w_ij
                        denominator += (residuals[i] - mean_residual) ** 2
                    
                    if total_weight > 0 and denominator > 0:
                        morans_i = (n / total_weight) * (numerator / denominator)
                        stats_result['morans_i'] = float(morans_i)
        
        except Exception as e:
            self.logger.warning(f"空间统计计算失败: {e}")
        
        return stats_result

class TemporalAnalyzer:
    """时间分析器"""
    
    def __init__(self, config: SatelliteValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_time_series(self, 
                          observations: List[SatelliteObservation],
                          model_predictions: List[np.ndarray],
                          timestamps: List[datetime]) -> Dict[str, Any]:
        """分析时间序列"""
        time_series_analysis = {
            'temporal_correlation': {},
            'trend_analysis': {},
            'seasonality_analysis': {},
            'change_detection': {}
        }
        
        try:
            if len(observations) < 3:
                self.logger.warning("时间序列数据不足")
                return time_series_analysis
            
            # 提取时间序列数据
            obs_series = []
            pred_series = []
            time_series = []
            
            for i, (obs, pred, timestamp) in enumerate(zip(observations, model_predictions, timestamps)):
                try:
                    # 获取有效数据的平均值
                    obs_data = obs.get_scaled_data()
                    valid_mask = obs.get_valid_data_mask()
                    
                    if valid_mask.any():
                        obs_mean = np.nanmean(obs_data[valid_mask])
                        pred_mean = np.nanmean(pred[valid_mask])
                        
                        obs_series.append(obs_mean)
                        pred_series.append(pred_mean)
                        time_series.append(timestamp)
                except Exception as e:
                    self.logger.warning(f"处理时间序列数据 {i} 失败: {e}")
            
            if len(obs_series) < 3:
                return time_series_analysis
            
            obs_array = np.array(obs_series)
            pred_array = np.array(pred_series)
            
            # 时间相关性分析
            time_series_analysis['temporal_correlation'] = self._analyze_temporal_correlation(
                obs_array, pred_array, time_series
            )
            
            # 趋势分析
            if self.config.trend_analysis:
                time_series_analysis['trend_analysis'] = self._analyze_trends(
                    obs_array, pred_array, time_series
                )
            
            # 季节性分析
            if self.config.seasonality_analysis:
                time_series_analysis['seasonality_analysis'] = self._analyze_seasonality(
                    obs_array, pred_array, time_series
                )
            
            # 变化检测
            time_series_analysis['change_detection'] = self._detect_changes(
                obs_array, pred_array, time_series
            )
        
        except Exception as e:
            self.logger.warning(f"时间序列分析失败: {e}")
        
        return time_series_analysis
    
    def _analyze_temporal_correlation(self, 
                                    obs_series: np.ndarray,
                                    pred_series: np.ndarray,
                                    timestamps: List[datetime]) -> Dict[str, float]:
        """分析时间相关性"""
        correlation_analysis = {}
        
        try:
            # 总体相关性
            correlation_analysis['overall_correlation'] = float(np.corrcoef(obs_series, pred_series)[0, 1])
            
            # 滞后相关性分析
            max_lag = min(len(obs_series) // 4, 10)  # 最大滞后
            lag_correlations = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(obs_series, pred_series)[0, 1]
                elif lag > 0:
                    if len(obs_series) > lag:
                        corr = np.corrcoef(obs_series[:-lag], pred_series[lag:])[0, 1]
                    else:
                        corr = np.nan
                else:  # lag < 0
                    if len(pred_series) > abs(lag):
                        corr = np.corrcoef(obs_series[abs(lag):], pred_series[:lag])[0, 1]
                    else:
                        corr = np.nan
                
                if not np.isnan(corr):
                    lag_correlations.append((lag, corr))
            
            if lag_correlations:
                best_lag, best_corr = max(lag_correlations, key=lambda x: abs(x[1]))
                correlation_analysis.update({
                    'best_lag': int(best_lag),
                    'best_lag_correlation': float(best_corr)
                })
        
        except Exception as e:
            self.logger.warning(f"时间相关性分析失败: {e}")
        
        return correlation_analysis
    
    def _analyze_trends(self, 
                       obs_series: np.ndarray,
                       pred_series: np.ndarray,
                       timestamps: List[datetime]) -> Dict[str, Any]:
        """分析趋势"""
        trend_analysis = {}
        
        try:
            # 转换时间为数值
            time_numeric = np.array([(t - timestamps[0]).total_seconds() / (24 * 3600) for t in timestamps])
            
            # 观测数据趋势
            obs_slope, obs_intercept, obs_r, obs_p, obs_stderr = stats.linregress(time_numeric, obs_series)
            trend_analysis['observed_trend'] = {
                'slope': float(obs_slope),
                'intercept': float(obs_intercept),
                'r_value': float(obs_r),
                'p_value': float(obs_p),
                'std_err': float(obs_stderr),
                'significant': obs_p < self.config.significance_level
            }
            
            # 预测数据趋势
            pred_slope, pred_intercept, pred_r, pred_p, pred_stderr = stats.linregress(time_numeric, pred_series)
            trend_analysis['predicted_trend'] = {
                'slope': float(pred_slope),
                'intercept': float(pred_intercept),
                'r_value': float(pred_r),
                'p_value': float(pred_p),
                'std_err': float(pred_stderr),
                'significant': pred_p < self.config.significance_level
            }
            
            # 趋势一致性
            trend_analysis['trend_consistency'] = {
                'slope_difference': float(abs(obs_slope - pred_slope)),
                'slope_ratio': float(pred_slope / (obs_slope + 1e-8)),
                'trend_agreement': (obs_slope * pred_slope) > 0  # 同向趋势
            }
        
        except Exception as e:
            self.logger.warning(f"趋势分析失败: {e}")
        
        return trend_analysis
    
    def _analyze_seasonality(self, 
                           obs_series: np.ndarray,
                           pred_series: np.ndarray,
                           timestamps: List[datetime]) -> Dict[str, Any]:
        """分析季节性"""
        seasonality_analysis = {}
        
        try:
            # 按月份分组
            monthly_obs = {}
            monthly_pred = {}
            
            for obs, pred, timestamp in zip(obs_series, pred_series, timestamps):
                month = timestamp.month
                if month not in monthly_obs:
                    monthly_obs[month] = []
                    monthly_pred[month] = []
                monthly_obs[month].append(obs)
                monthly_pred[month].append(pred)
            
            # 计算月平均值
            monthly_means_obs = {}
            monthly_means_pred = {}
            
            for month in range(1, 13):
                if month in monthly_obs and len(monthly_obs[month]) > 0:
                    monthly_means_obs[month] = np.mean(monthly_obs[month])
                    monthly_means_pred[month] = np.mean(monthly_pred[month])
            
            if len(monthly_means_obs) > 2:
                # 季节性相关性
                months = sorted(monthly_means_obs.keys())
                seasonal_obs = [monthly_means_obs[m] for m in months]
                seasonal_pred = [monthly_means_pred[m] for m in months]
                
                seasonal_corr = np.corrcoef(seasonal_obs, seasonal_pred)[0, 1]
                seasonality_analysis['seasonal_correlation'] = float(seasonal_corr)
                
                # 季节性振幅
                obs_amplitude = np.max(seasonal_obs) - np.min(seasonal_obs)
                pred_amplitude = np.max(seasonal_pred) - np.min(seasonal_pred)
                
                seasonality_analysis['seasonal_amplitude'] = {
                    'observed': float(obs_amplitude),
                    'predicted': float(pred_amplitude),
                    'ratio': float(pred_amplitude / (obs_amplitude + 1e-8))
                }
                
                # 季节性相位
                obs_peak_month = months[np.argmax(seasonal_obs)]
                pred_peak_month = months[np.argmax(seasonal_pred)]
                
                seasonality_analysis['seasonal_phase'] = {
                    'observed_peak_month': int(obs_peak_month),
                    'predicted_peak_month': int(pred_peak_month),
                    'phase_difference': int(abs(obs_peak_month - pred_peak_month))
                }
        
        except Exception as e:
            self.logger.warning(f"季节性分析失败: {e}")
        
        return seasonality_analysis
    
    def _detect_changes(self, 
                       obs_series: np.ndarray,
                       pred_series: np.ndarray,
                       timestamps: List[datetime]) -> Dict[str, Any]:
        """检测变化"""
        change_detection = {}
        
        try:
            # 计算差值序列
            diff_series = obs_series - pred_series
            
            # 变化点检测（简化实现）
            # 使用滑动窗口检测均值变化
            window_size = max(3, len(diff_series) // 5)
            change_points = []
            
            for i in range(window_size, len(diff_series) - window_size):
                before_window = diff_series[i-window_size:i]
                after_window = diff_series[i:i+window_size]
                
                # t检验检测均值变化
                try:
                    t_stat, p_value = stats.ttest_ind(before_window, after_window)
                    if p_value < self.config.significance_level:
                        change_points.append({
                            'index': i,
                            'timestamp': timestamps[i],
                            't_statistic': float(t_stat),
                            'p_value': float(p_value)
                        })
                except:
                    continue
            
            change_detection['change_points'] = change_points
            change_detection['num_change_points'] = len(change_points)
            
            # 变化幅度统计
            if len(diff_series) > 1:
                change_detection['change_statistics'] = {
                    'mean_change': float(np.mean(np.abs(np.diff(diff_series)))),
                    'max_change': float(np.max(np.abs(np.diff(diff_series)))),
                    'std_change': float(np.std(np.diff(diff_series)))
                }
        
        except Exception as e:
            self.logger.warning(f"变化检测失败: {e}")
        
        return change_detection

class SatelliteValidator:
    """卫星验证器"""
    
    def __init__(self, config: SatelliteValidationConfig):
        self.config = config
        self.geometric_processor = GeometricProcessor(config)
        self.temporal_analyzer = TemporalAnalyzer(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, 
                model: nn.Module,
                satellite_observations: List[SatelliteObservation]) -> Dict[str, Any]:
        """执行卫星数据验证"""
        results = {
            'total_observations': len(satellite_observations),
            'filtered_observations': 0,
            'validation_results': {},
            'multi_scale_analysis': {},
            'temporal_analysis': {},
            'quality_assessment': {},
            'geometric_analysis': {}
        }
        
        try:
            # 过滤观测数据
            filtered_observations = self._filter_observations(satellite_observations)
            results['filtered_observations'] = len(filtered_observations)
            
            if not filtered_observations:
                self.logger.warning("没有符合条件的卫星观测数据")
                return results
            
            # 生成模型预测
            model_predictions = []
            valid_observations = []
            
            for obs in filtered_observations:
                try:
                    prediction = self._generate_model_prediction(model, obs)
                    if prediction is not None:
                        model_predictions.append(prediction)
                        valid_observations.append(obs)
                except Exception as e:
                    self.logger.warning(f"生成预测失败 {obs.observation_id}: {e}")
            
            if not model_predictions:
                self.logger.warning("没有成功生成的模型预测")
                return results
            
            # 基本验证分析
            results['validation_results'] = self._perform_basic_validation(
                valid_observations, model_predictions
            )
            
            # 多尺度分析
            results['multi_scale_analysis'] = self._perform_multi_scale_analysis(
                valid_observations, model_predictions
            )
            
            # 时间分析
            if self.config.time_series_analysis and len(valid_observations) > 2:
                timestamps = [obs.acquisition_time for obs in valid_observations]
                results['temporal_analysis'] = self.temporal_analyzer.analyze_time_series(
                    valid_observations, model_predictions, timestamps
                )
            
            # 质量评估
            results['quality_assessment'] = self._assess_data_quality(
                valid_observations, model_predictions
            )
            
            # 几何分析
            results['geometric_analysis'] = self._perform_geometric_analysis(
                valid_observations, model_predictions
            )
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, valid_observations, model_predictions)
        
        except Exception as e:
            self.logger.error(f"卫星验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _filter_observations(self, observations: List[SatelliteObservation]) -> List[SatelliteObservation]:
        """过滤观测数据"""
        filtered = []
        
        for obs in observations:
            # 卫星类型过滤
            if obs.satellite_type not in self.config.supported_satellites:
                continue
            
            # 数据产品过滤
            if obs.data_product not in self.config.supported_products:
                continue
            
            # 处理级别过滤
            if self._processing_level_score(obs.processing_level) < self._processing_level_score(self.config.min_processing_level):
                continue
            
            # 云覆盖过滤
            if obs.cloud_coverage > self.config.max_cloud_coverage:
                continue
            
            # 数据覆盖过滤
            if obs.data_coverage < self.config.min_data_coverage:
                continue
            
            # 质量过滤
            if self._quality_score(obs.overall_quality) < self._quality_score(self.config.min_quality):
                continue
            
            filtered.append(obs)
        
        return filtered
    
    def _processing_level_score(self, level: ProcessingLevel) -> int:
        """处理级别评分"""
        scores = {
            ProcessingLevel.L1A: 1,
            ProcessingLevel.L1B: 2,
            ProcessingLevel.L1C: 3,
            ProcessingLevel.L2A: 4,
            ProcessingLevel.L2B: 5,
            ProcessingLevel.L3: 6,
            ProcessingLevel.L4: 7
        }
        return scores.get(level, 0)
    
    def _quality_score(self, quality: QualityFlag) -> int:
        """质量评分"""
        scores = {
            QualityFlag.EXCELLENT: 5,
            QualityFlag.GOOD: 4,
            QualityFlag.FAIR: 3,
            QualityFlag.POOR: 2,
            QualityFlag.INVALID: 0
        }
        return scores.get(quality, 1)
    
    def _generate_model_prediction(self, model: nn.Module, observation: SatelliteObservation) -> Optional[np.ndarray]:
        """生成模型预测"""
        try:
            # 准备输入坐标
            if observation.coordinates is None:
                # 如果没有坐标信息，使用中心点
                lat = np.array([[observation.center_lat]])
                lon = np.array([[observation.center_lon]])
            else:
                lat, lon = observation.coordinates
            
            # 准备时间输入
            time_numeric = (observation.acquisition_time - datetime(2020, 1, 1)).total_seconds() / (24 * 3600)
            time_array = np.full_like(lat, time_numeric)
            
            # 构造模型输入
            input_coords = np.stack([lat.flatten(), lon.flatten(), time_array.flatten()], axis=1)
            input_tensor = torch.tensor(input_coords, dtype=torch.float32)
            
            # 模型预测
            with torch.no_grad():
                model.eval()
                predictions = model(input_tensor)
                
                if predictions.dim() > 1:
                    predictions = predictions[:, 0]  # 取第一个输出
                
                # 重塑为原始形状
                prediction_array = predictions.numpy().reshape(lat.shape)
                
                return prediction_array
        
        except Exception as e:
            self.logger.warning(f"模型预测生成失败: {e}")
            return None
    
    def _perform_basic_validation(self, 
                                observations: List[SatelliteObservation],
                                predictions: List[np.ndarray]) -> Dict[str, Any]:
        """执行基本验证"""
        validation_results = {
            'pixel_level': {},
            'aggregated': {},
            'by_satellite': {},
            'by_product': {}
        }
        
        try:
            all_obs_data = []
            all_pred_data = []
            satellite_types = []
            product_types = []
            
            # 收集所有像素级数据
            for obs, pred in zip(observations, predictions):
                obs_data = obs.get_scaled_data()
                valid_mask = obs.get_valid_data_mask()
                
                if valid_mask.any():
                    valid_obs = obs_data[valid_mask]
                    valid_pred = pred[valid_mask]
                    
                    all_obs_data.extend(valid_obs.flatten())
                    all_pred_data.extend(valid_pred.flatten())
                    satellite_types.extend([obs.satellite_type] * len(valid_obs.flatten()))
                    product_types.extend([obs.data_product] * len(valid_obs.flatten()))
            
            if all_obs_data:
                obs_array = np.array(all_obs_data)
                pred_array = np.array(all_pred_data)
                
                # 像素级统计
                validation_results['pixel_level'] = self._compute_validation_metrics(
                    obs_array, pred_array
                )
                
                # 聚合统计
                validation_results['aggregated'] = self._compute_aggregated_metrics(
                    observations, predictions
                )
                
                # 按卫星类型分组
                validation_results['by_satellite'] = self._group_by_attribute(
                    obs_array, pred_array, satellite_types
                )
                
                # 按产品类型分组
                validation_results['by_product'] = self._group_by_attribute(
                    obs_array, pred_array, product_types
                )
        
        except Exception as e:
            self.logger.warning(f"基本验证失败: {e}")
        
        return validation_results
    
    def _compute_validation_metrics(self, observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """计算验证指标"""
        metrics = {}
        
        try:
            # 移除无效值
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
            if not valid_mask.any():
                return metrics
            
            obs = observed[valid_mask]
            pred = predicted[valid_mask]
            
            # 基本指标
            metrics.update({
                'correlation': float(np.corrcoef(obs, pred)[0, 1]),
                'rmse': float(np.sqrt(np.mean((obs - pred) ** 2))),
                'mae': float(np.mean(np.abs(obs - pred))),
                'bias': float(np.mean(pred - obs)),
                'r_squared': float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
                'count': int(len(obs))
            })
            
            # 标准化指标
            obs_range = np.max(obs) - np.min(obs)
            if obs_range > 0:
                metrics['normalized_rmse'] = metrics['rmse'] / obs_range
                metrics['normalized_mae'] = metrics['mae'] / obs_range
            
            # 百分比指标
            obs_mean = np.mean(obs)
            if obs_mean != 0:
                metrics['percent_bias'] = 100 * metrics['bias'] / obs_mean
                metrics['percent_rmse'] = 100 * metrics['rmse'] / abs(obs_mean)
        
        except Exception as e:
            self.logger.warning(f"验证指标计算失败: {e}")
        
        return metrics
    
    def _compute_aggregated_metrics(self, 
                                  observations: List[SatelliteObservation],
                                  predictions: List[np.ndarray]) -> Dict[str, float]:
        """计算聚合指标"""
        aggregated_metrics = {}
        
        try:
            obs_means = []
            pred_means = []
            
            for obs, pred in zip(observations, predictions):
                obs_data = obs.get_scaled_data()
                valid_mask = obs.get_valid_data_mask()
                
                if valid_mask.any():
                    obs_mean = np.nanmean(obs_data[valid_mask])
                    pred_mean = np.nanmean(pred[valid_mask])
                    
                    obs_means.append(obs_mean)
                    pred_means.append(pred_mean)
            
            if obs_means:
                obs_array = np.array(obs_means)
                pred_array = np.array(pred_means)
                
                aggregated_metrics = self._compute_validation_metrics(obs_array, pred_array)
        
        except Exception as e:
            self.logger.warning(f"聚合指标计算失败: {e}")
        
        return aggregated_metrics
    
    def _group_by_attribute(self, 
                          observed: np.ndarray, 
                          predicted: np.ndarray,
                          attributes: List) -> Dict[str, Dict[str, float]]:
        """按属性分组分析"""
        grouped_results = {}
        
        try:
            unique_attributes = list(set(attributes))
            
            for attr in unique_attributes:
                attr_mask = np.array([a == attr for a in attributes])
                
                if attr_mask.any():
                    attr_obs = observed[attr_mask]
                    attr_pred = predicted[attr_mask]
                    
                    attr_metrics = self._compute_validation_metrics(attr_obs, attr_pred)
                    grouped_results[str(attr)] = attr_metrics
        
        except Exception as e:
            self.logger.warning(f"分组分析失败: {e}")
        
        return grouped_results
    
    def _perform_multi_scale_analysis(self, 
                                    observations: List[SatelliteObservation],
                                    predictions: List[np.ndarray]) -> Dict[str, Any]:
        """执行多尺度分析"""
        multi_scale_results = {}
        
        try:
            for scale in self.config.validation_scales:
                if scale == ValidationScale.PIXEL:
                    # 像素级分析已在基本验证中完成
                    continue
                elif scale == ValidationScale.PATCH:
                    multi_scale_results['patch_level'] = self._analyze_patch_level(
                        observations, predictions
                    )
                elif scale == ValidationScale.REGION:
                    multi_scale_results['region_level'] = self._analyze_region_level(
                        observations, predictions
                    )
        
        except Exception as e:
            self.logger.warning(f"多尺度分析失败: {e}")
        
        return multi_scale_results
    
    def _analyze_patch_level(self, 
                           observations: List[SatelliteObservation],
                           predictions: List[np.ndarray],
                           patch_size: int = 5) -> Dict[str, Any]:
        """斑块级分析"""
        patch_results = {'patch_statistics': [], 'aggregated_metrics': {}}
        
        try:
            all_patch_obs = []
            all_patch_pred = []
            
            for obs, pred in zip(observations, predictions):
                obs_data = obs.get_scaled_data()
                valid_mask = obs.get_valid_data_mask()
                
                if obs_data.shape[0] < patch_size or obs_data.shape[1] < patch_size:
                    continue
                
                # 分割为斑块
                for i in range(0, obs_data.shape[0] - patch_size + 1, patch_size):
                    for j in range(0, obs_data.shape[1] - patch_size + 1, patch_size):
                        patch_obs = obs_data[i:i+patch_size, j:j+patch_size]
                        patch_pred = pred[i:i+patch_size, j:j+patch_size]
                        patch_mask = valid_mask[i:i+patch_size, j:j+patch_size]
                        
                        if patch_mask.sum() > (patch_size * patch_size * 0.5):  # 至少50%有效
                            patch_obs_mean = np.nanmean(patch_obs[patch_mask])
                            patch_pred_mean = np.nanmean(patch_pred[patch_mask])
                            
                            all_patch_obs.append(patch_obs_mean)
                            all_patch_pred.append(patch_pred_mean)
            
            if all_patch_obs:
                patch_obs_array = np.array(all_patch_obs)
                patch_pred_array = np.array(all_patch_pred)
                
                patch_results['aggregated_metrics'] = self._compute_validation_metrics(
                    patch_obs_array, patch_pred_array
                )
                patch_results['num_patches'] = len(all_patch_obs)
        
        except Exception as e:
            self.logger.warning(f"斑块级分析失败: {e}")
        
        return patch_results
    
    def _analyze_region_level(self, 
                            observations: List[SatelliteObservation],
                            predictions: List[np.ndarray]) -> Dict[str, Any]:
        """区域级分析"""
        region_results = {'region_statistics': [], 'aggregated_metrics': {}}
        
        try:
            # 简化的区域分析：将每个观测作为一个区域
            region_obs = []
            region_pred = []
            
            for obs, pred in zip(observations, predictions):
                obs_data = obs.get_scaled_data()
                valid_mask = obs.get_valid_data_mask()
                
                if valid_mask.any():
                    region_obs_mean = np.nanmean(obs_data[valid_mask])
                    region_pred_mean = np.nanmean(pred[valid_mask])
                    
                    region_obs.append(region_obs_mean)
                    region_pred.append(region_pred_mean)
            
            if region_obs:
                region_obs_array = np.array(region_obs)
                region_pred_array = np.array(region_pred)
                
                region_results['aggregated_metrics'] = self._compute_validation_metrics(
                    region_obs_array, region_pred_array
                )
                region_results['num_regions'] = len(region_obs)
        
        except Exception as e:
            self.logger.warning(f"区域级分析失败: {e}")
        
        return region_results
    
    def _assess_data_quality(self, 
                           observations: List[SatelliteObservation],
                           predictions: List[np.ndarray]) -> Dict[str, Any]:
        """评估数据质量"""
        quality_assessment = {
            'quality_distribution': {},
            'cloud_impact': {},
            'spatial_coverage': {},
            'temporal_coverage': {}
        }
        
        try:
            # 质量分布
            quality_counts = {}
            for obs in observations:
                quality = obs.overall_quality
                quality_counts[quality.value] = quality_counts.get(quality.value, 0) + 1
            
            quality_assessment['quality_distribution'] = quality_counts
            
            # 云覆盖影响
            cloud_coverages = [obs.cloud_coverage for obs in observations]
            if cloud_coverages:
                quality_assessment['cloud_impact'] = {
                    'mean_cloud_coverage': float(np.mean(cloud_coverages)),
                    'max_cloud_coverage': float(np.max(cloud_coverages)),
                    'cloud_free_observations': int(sum(1 for cc in cloud_coverages if cc < 0.1))
                }
            
            # 空间覆盖
            if observations:
                lats = [obs.center_lat for obs in observations]
                lons = [obs.center_lon for obs in observations]
                
                quality_assessment['spatial_coverage'] = {
                    'lat_range': [float(np.min(lats)), float(np.max(lats))],
                    'lon_range': [float(np.min(lons)), float(np.max(lons))],
                    'spatial_extent': float((np.max(lats) - np.min(lats)) * (np.max(lons) - np.min(lons)))
                }
            
            # 时间覆盖
            timestamps = [obs.acquisition_time for obs in observations]
            if timestamps:
                time_span = (max(timestamps) - min(timestamps)).total_seconds() / (24 * 3600)  # 天数
                quality_assessment['temporal_coverage'] = {
                    'start_date': min(timestamps).isoformat(),
                    'end_date': max(timestamps).isoformat(),
                    'time_span_days': float(time_span),
                    'observation_frequency': len(timestamps) / (time_span + 1)
                }
        
        except Exception as e:
            self.logger.warning(f"数据质量评估失败: {e}")
        
        return quality_assessment
    
    def _perform_geometric_analysis(self, 
                                  observations: List[SatelliteObservation],
                                  predictions: List[np.ndarray]) -> Dict[str, Any]:
        """执行几何分析"""
        geometric_analysis = {
            'spatial_statistics': {},
            'resolution_analysis': {},
            'coregistration_assessment': {}
        }
        
        try:
            # 空间分辨率分析
            resolutions = [obs.spatial_resolution for obs in observations]
            if resolutions:
                geometric_analysis['resolution_analysis'] = {
                    'mean_resolution': float(np.mean(resolutions)),
                    'min_resolution': float(np.min(resolutions)),
                    'max_resolution': float(np.max(resolutions)),
                    'resolution_std': float(np.std(resolutions))
                }
            
            # 空间统计（使用第一个观测作为示例）
            if observations and predictions:
                obs = observations[0]
                pred = predictions[0]
                
                if obs.coordinates is not None:
                    spatial_stats = self.geometric_processor.calculate_spatial_statistics(
                        obs.get_scaled_data(), pred, obs.coordinates
                    )
                    geometric_analysis['spatial_statistics'] = spatial_stats
        
        except Exception as e:
            self.logger.warning(f"几何分析失败: {e}")
        
        return geometric_analysis
    
    def _plot_results(self, 
                     results: Dict[str, Any],
                     observations: List[SatelliteObservation],
                     predictions: List[np.ndarray]):
        """绘制结果"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 收集数据用于绘图
            all_obs_data = []
            all_pred_data = []
            
            for obs, pred in zip(observations, predictions):
                obs_data = obs.get_scaled_data()
                valid_mask = obs.get_valid_data_mask()
                
                if valid_mask.any():
                    all_obs_data.extend(obs_data[valid_mask].flatten())
                    all_pred_data.extend(pred[valid_mask].flatten())
            
            if all_obs_data:
                obs_array = np.array(all_obs_data)
                pred_array = np.array(all_pred_data)
                
                # 散点图
                axes[0, 0].scatter(obs_array, pred_array, alpha=0.5, s=1)
                min_val = min(np.min(obs_array), np.min(pred_array))
                max_val = max(np.max(obs_array), np.max(pred_array))
                axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1线')
                axes[0, 0].set_xlabel('卫星观测值')
                axes[0, 0].set_ylabel('模型预测值')
                axes[0, 0].set_title('预测值 vs 观测值')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # 残差图
                residuals = pred_array - obs_array
                axes[0, 1].scatter(obs_array, residuals, alpha=0.5, s=1)
                axes[0, 1].axhline(y=0, color='r', linestyle='--')
                axes[0, 1].set_xlabel('卫星观测值')
                axes[0, 1].set_ylabel('残差')
                axes[0, 1].set_title('残差分布')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 残差直方图
                axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 2].axvline(x=0, color='r', linestyle='--')
                axes[0, 2].set_xlabel('残差')
                axes[0, 2].set_ylabel('频数')
                axes[0, 2].set_title('残差直方图')
                axes[0, 2].grid(True, alpha=0.3)
            
            # 时间序列图（如果有时间分析结果）
            if 'temporal_analysis' in results and 'temporal_correlation' in results['temporal_analysis']:
                timestamps = [obs.acquisition_time for obs in observations]
                obs_means = []
                pred_means = []
                
                for obs, pred in zip(observations, predictions):
                    obs_data = obs.get_scaled_data()
                    valid_mask = obs.get_valid_data_mask()
                    
                    if valid_mask.any():
                        obs_means.append(np.nanmean(obs_data[valid_mask]))
                        pred_means.append(np.nanmean(pred[valid_mask]))
                    else:
                        obs_means.append(np.nan)
                        pred_means.append(np.nan)
                
                axes[1, 0].plot(timestamps, obs_means, 'b-o', label='观测值', markersize=3)
                axes[1, 0].plot(timestamps, pred_means, 'r-s', label='预测值', markersize=3)
                axes[1, 0].set_xlabel('时间')
                axes[1, 0].set_ylabel('平均值')
                axes[1, 0].set_title('时间序列对比')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # 按卫星类型的性能对比
            if 'validation_results' in results and 'by_satellite' in results['validation_results']:
                by_satellite = results['validation_results']['by_satellite']
                
                satellite_names = list(by_satellite.keys())
                correlations = [metrics.get('correlation', 0) for metrics in by_satellite.values()]
                rmse_values = [metrics.get('rmse', 0) for metrics in by_satellite.values()]
                
                x_pos = np.arange(len(satellite_names))
                
                axes[1, 1].bar(x_pos, correlations, alpha=0.7)
                axes[1, 1].set_xlabel('卫星类型')
                axes[1, 1].set_ylabel('相关系数')
                axes[1, 1].set_title('各卫星类型性能对比')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(satellite_names, rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            # 质量分布
            if 'quality_assessment' in results and 'quality_distribution' in results['quality_assessment']:
                quality_dist = results['quality_assessment']['quality_distribution']
                
                qualities = list(quality_dist.keys())
                counts = list(quality_dist.values())
                
                axes[1, 2].pie(counts, labels=qualities, autopct='%1.1f%%')
                axes[1, 2].set_title('数据质量分布')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/satellite_validation_results.png", dpi=300, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       observations: List[SatelliteObservation]) -> str:
        """生成验证报告"""
        report_lines = [
            "="*80,
            "卫星数据验证报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. 数据概览",
            "-"*40,
            f"总观测数量: {results.get('total_observations', 0)}",
            f"过滤后观测数量: {results.get('filtered_observations', 0)}",
        ]
        
        # 验证结果
        if 'validation_results' in results:
            validation = results['validation_results']
            
            report_lines.extend([
                "",
                "2. 验证结果",
                "-"*40
            ])
            
            if 'pixel_level' in validation:
                pixel_metrics = validation['pixel_level']
                report_lines.extend([
                    "像素级验证:",
                    f"  相关系数: {pixel_metrics.get('correlation', 'N/A'):.4f}",
                    f"  RMSE: {pixel_metrics.get('rmse', 'N/A'):.4f}",
                    f"  MAE: {pixel_metrics.get('mae', 'N/A'):.4f}",
                    f"  偏差: {pixel_metrics.get('bias', 'N/A'):.4f}",
                    f"  R²: {pixel_metrics.get('r_squared', 'N/A'):.4f}",
                    f"  有效像素数: {pixel_metrics.get('count', 'N/A')}"
                ])
        
        # 时间分析
        if 'temporal_analysis' in results:
            temporal = results['temporal_analysis']
            
            report_lines.extend([
                "",
                "3. 时间分析",
                "-"*40
            ])
            
            if 'temporal_correlation' in temporal:
                temp_corr = temporal['temporal_correlation']
                report_lines.extend([
                    "时间相关性:",
                    f"  总体相关性: {temp_corr.get('overall_correlation', 'N/A'):.4f}",
                    f"  最佳滞后: {temp_corr.get('best_lag', 'N/A')}",
                    f"  最佳滞后相关性: {temp_corr.get('best_lag_correlation', 'N/A'):.4f}"
                ])
        
        # 质量评估
        if 'quality_assessment' in results:
            quality = results['quality_assessment']
            
            report_lines.extend([
                "",
                "4. 质量评估",
                "-"*40
            ])
            
            if 'cloud_impact' in quality:
                cloud = quality['cloud_impact']
                report_lines.extend([
                    "云覆盖影响:",
                    f"  平均云覆盖率: {cloud.get('mean_cloud_coverage', 'N/A'):.2%}",
                    f"  最大云覆盖率: {cloud.get('max_cloud_coverage', 'N/A'):.2%}",
                    f"  无云观测数: {cloud.get('cloud_free_observations', 'N/A')}"
                ])
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_satellite_validator(config: Optional[SatelliteValidationConfig] = None) -> SatelliteValidator:
    """创建卫星验证器"""
    if config is None:
        config = SatelliteValidationConfig()
    
    return SatelliteValidator(config)

if __name__ == "__main__":
    # 测试代码
    import torch.nn as nn
    
    # 创建简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 创建测试数据
    test_observations = [
        SatelliteObservation(
            observation_id="test_001",
            satellite_type=SatelliteType.LANDSAT,
            data_product=DataProduct.SURFACE_TEMPERATURE,
            processing_level=ProcessingLevel.L2A,
            acquisition_time=datetime(2023, 6, 15, 10, 30),
            center_lat=30.0,
            center_lon=90.0,
            spatial_resolution=30.0,
            data=np.random.normal(280, 10, (10, 10)),
            coordinates=(np.random.uniform(29.9, 30.1, (10, 10)), 
                        np.random.uniform(89.9, 90.1, (10, 10))),
            units="K",
            overall_quality=QualityFlag.GOOD,
            cloud_coverage=0.1,
            data_coverage=0.95
        ),
        SatelliteObservation(
            observation_id="test_002",
            satellite_type=SatelliteType.MODIS,
            data_product=DataProduct.SURFACE_TEMPERATURE,
            processing_level=ProcessingLevel.L2A,
            acquisition_time=datetime(2023, 6, 16, 11, 0),
            center_lat=30.1,
            center_lon=90.1,
            spatial_resolution=1000.0,
            data=np.random.normal(285, 8, (5, 5)),
            coordinates=(np.random.uniform(30.0, 30.2, (5, 5)), 
                        np.random.uniform(90.0, 90.2, (5, 5))),
            units="K",
            overall_quality=QualityFlag.EXCELLENT,
            cloud_coverage=0.05,
            data_coverage=0.98
        )
    ]
    
    # 创建验证器
    config = SatelliteValidationConfig(
        enable_plotting=True,
        save_plots=False,
        verbose=True
    )
    
    validator = create_satellite_validator(config)
    
    # 创建测试模型
    model = TestModel()
    
    # 执行验证
    print("开始卫星数据验证...")
    results = validator.validate(model, test_observations)
    
    # 生成报告
    report = validator.generate_report(results, test_observations)
    print(report)
    
    print("\n卫星数据验证完成！")