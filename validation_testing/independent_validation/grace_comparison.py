#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRACE数据对比验证模块

该模块实现了与GRACE重力卫星数据的对比验证功能，用于验证模型在质量变化预测方面的准确性。
支持多种GRACE数据产品、时间序列分析、空间分析和不确定性评估。

主要功能:
- GRACE数据处理和质量控制
- 时间序列对比分析
- 空间分布对比
- 趋势分析和季节性分析
- 不确定性量化
- 多尺度验证

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats, signal
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRACEProduct(Enum):
    """GRACE数据产品类型"""
    CSR_RL06 = "CSR_RL06"  # CSR RL06球谐系数
    GFZ_RL06 = "GFZ_RL06"  # GFZ RL06球谐系数
    JPL_RL06 = "JPL_RL06"  # JPL RL06球谐系数
    MASCON_CSR = "MASCON_CSR"  # CSR质量集中解
    MASCON_JPL = "MASCON_JPL"  # JPL质量集中解
    MASCON_GSFC = "MASCON_GSFC"  # GSFC质量集中解
    TELLUS = "TELLUS"  # TELLUS格网产品

class ProcessingLevel(Enum):
    """数据处理级别"""
    L1B = "L1B"  # 原始观测数据
    L2 = "L2"   # 球谐系数
    L3 = "L3"   # 格网产品
    L4 = "L4"   # 高级产品

class DataType(Enum):
    """数据类型"""
    TOTAL_WATER_STORAGE = "total_water_storage"  # 总水储量
    GROUNDWATER = "groundwater"  # 地下水
    SURFACE_WATER = "surface_water"  # 地表水
    SOIL_MOISTURE = "soil_moisture"  # 土壤水分
    ICE_MASS = "ice_mass"  # 冰质量
    EQUIVALENT_WATER_HEIGHT = "equivalent_water_height"  # 等效水高

class QualityFlag(Enum):
    """数据质量标志"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"

class ValidationScale(Enum):
    """验证尺度"""
    PIXEL = "pixel"  # 像素级
    BASIN = "basin"  # 流域级
    REGIONAL = "regional"  # 区域级
    GLOBAL = "global"  # 全球级

class AnalysisType(Enum):
    """分析类型"""
    TIME_SERIES = "time_series"  # 时间序列
    TREND = "trend"  # 趋势分析
    SEASONAL = "seasonal"  # 季节性分析
    ANOMALY = "anomaly"  # 异常分析
    CORRELATION = "correlation"  # 相关性分析

@dataclass
class GRACEObservation:
    """GRACE观测数据"""
    observation_id: str
    product_type: GRACEProduct
    processing_level: ProcessingLevel
    data_type: DataType
    acquisition_time: datetime
    time_span: Tuple[datetime, datetime]  # 数据时间跨度
    spatial_extent: Tuple[float, float, float, float]  # (min_lat, max_lat, min_lon, max_lon)
    spatial_resolution: float  # 空间分辨率(度)
    data: np.ndarray  # 数据数组
    coordinates: Tuple[np.ndarray, np.ndarray]  # (lat, lon)
    units: str
    uncertainty: Optional[np.ndarray] = None  # 不确定性
    quality_flag: QualityFlag = QualityFlag.GOOD
    data_coverage: float = 1.0  # 数据覆盖率
    processing_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GRACEComparisonConfig:
    """GRACE对比配置"""
    # 基本设置
    temporal_tolerance: float = 15.0  # 时间容差(天)
    spatial_tolerance: float = 0.5  # 空间容差(度)
    min_data_coverage: float = 0.8  # 最小数据覆盖率
    
    # 质量控制
    enable_quality_filter: bool = True
    min_quality: QualityFlag = QualityFlag.FAIR
    outlier_threshold: float = 3.0  # 异常值阈值(标准差倍数)
    
    # 分析设置
    enable_trend_analysis: bool = True
    enable_seasonal_analysis: bool = True
    enable_anomaly_detection: bool = True
    detrend_method: str = "linear"  # 去趋势方法
    seasonal_method: str = "fourier"  # 季节性分析方法
    
    # 插值设置
    interpolation_method: str = "linear"  # 插值方法
    max_interpolation_gap: int = 3  # 最大插值间隔(月)
    
    # 验证设置
    validation_scales: List[ValidationScale] = field(default_factory=lambda: [ValidationScale.PIXEL, ValidationScale.REGIONAL])
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.TIME_SERIES, AnalysisType.TREND])
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./grace_validation_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

class TimeSeriesProcessor:
    """时间序列处理器"""
    
    def __init__(self, config: GRACEComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def align_time_series(self, 
                         model_data: np.ndarray,
                         model_times: List[datetime],
                         grace_data: np.ndarray,
                         grace_times: List[datetime]) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """对齐时间序列"""
        try:
            # 找到共同时间范围
            start_time = max(min(model_times), min(grace_times))
            end_time = min(max(model_times), max(grace_times))
            
            # 创建对齐的时间序列
            aligned_times = []
            aligned_model = []
            aligned_grace = []
            
            for grace_time, grace_val in zip(grace_times, grace_data):
                if start_time <= grace_time <= end_time:
                    # 找到最近的模型时间点
                    time_diffs = [abs((mt - grace_time).total_seconds()) for mt in model_times]
                    min_diff_idx = np.argmin(time_diffs)
                    min_diff_days = time_diffs[min_diff_idx] / (24 * 3600)
                    
                    if min_diff_days <= self.config.temporal_tolerance:
                        aligned_times.append(grace_time)
                        aligned_model.append(model_data[min_diff_idx])
                        aligned_grace.append(grace_val)
            
            return np.array(aligned_model), np.array(aligned_grace), aligned_times
        
        except Exception as e:
            self.logger.error(f"时间序列对齐失败: {e}")
            return np.array([]), np.array([]), []
    
    def interpolate_gaps(self, 
                        data: np.ndarray, 
                        times: List[datetime]) -> Tuple[np.ndarray, List[datetime]]:
        """插值填补缺失数据"""
        try:
            if len(data) == 0:
                return data, times
            
            # 检测缺失值
            valid_mask = ~np.isnan(data)
            if np.all(valid_mask):
                return data, times
            
            # 插值
            if self.config.interpolation_method == "linear":
                interpolated = np.interp(
                    np.arange(len(data)),
                    np.where(valid_mask)[0],
                    data[valid_mask]
                )
            else:
                # 其他插值方法可以在这里添加
                interpolated = data.copy()
            
            return interpolated, times
        
        except Exception as e:
            self.logger.error(f"数据插值失败: {e}")
            return data, times
    
    def detrend_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """去趋势处理"""
        try:
            if self.config.detrend_method == "linear":
                # 线性去趋势
                x = np.arange(len(data))
                slope, intercept = np.polyfit(x, data, 1)
                trend = slope * x + intercept
                detrended = data - trend
            elif self.config.detrend_method == "polynomial":
                # 多项式去趋势
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 2)
                trend = np.polyval(coeffs, x)
                detrended = data - trend
            else:
                # 无去趋势
                trend = np.zeros_like(data)
                detrended = data.copy()
            
            return detrended, trend
        
        except Exception as e:
            self.logger.error(f"去趋势处理失败: {e}")
            return data, np.zeros_like(data)
    
    def analyze_seasonality(self, 
                           data: np.ndarray, 
                           times: List[datetime]) -> Dict[str, Any]:
        """季节性分析"""
        try:
            if len(data) < 12:  # 至少需要一年的数据
                return {}
            
            # 提取月份信息
            months = np.array([t.month for t in times])
            
            # 计算月平均值
            monthly_means = {}
            for month in range(1, 13):
                month_data = data[months == month]
                if len(month_data) > 0:
                    monthly_means[month] = np.mean(month_data)
            
            # 计算季节性振幅
            if len(monthly_means) >= 12:
                seasonal_amplitude = np.max(list(monthly_means.values())) - np.min(list(monthly_means.values()))
            else:
                seasonal_amplitude = 0.0
            
            # 傅里叶分析
            if self.config.seasonal_method == "fourier" and len(data) >= 24:
                fft = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data))
                
                # 找到年周期对应的频率
                annual_freq_idx = np.argmin(np.abs(freqs - 1/12))  # 假设数据是月度的
                annual_power = np.abs(fft[annual_freq_idx])
            else:
                annual_power = 0.0
            
            return {
                'monthly_means': monthly_means,
                'seasonal_amplitude': seasonal_amplitude,
                'annual_power': annual_power
            }
        
        except Exception as e:
            self.logger.error(f"季节性分析失败: {e}")
            return {}

class SpatialProcessor:
    """空间处理器"""
    
    def __init__(self, config: GRACEComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def regrid_data(self, 
                   source_data: np.ndarray,
                   source_coords: Tuple[np.ndarray, np.ndarray],
                   target_coords: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """重新网格化数据"""
        try:
            source_lat, source_lon = source_coords
            target_lat, target_lon = target_coords
            
            # 展平源数据和坐标
            source_points = np.column_stack([
                source_lat.flatten(),
                source_lon.flatten()
            ])
            source_values = source_data.flatten()
            
            # 移除无效值
            valid_mask = ~np.isnan(source_values)
            source_points = source_points[valid_mask]
            source_values = source_values[valid_mask]
            
            if len(source_values) == 0:
                return np.full(target_lat.shape, np.nan)
            
            # 目标网格点
            target_points = np.column_stack([
                target_lat.flatten(),
                target_lon.flatten()
            ])
            
            # 插值
            interpolated = griddata(
                source_points,
                source_values,
                target_points,
                method=self.config.interpolation_method,
                fill_value=np.nan
            )
            
            return interpolated.reshape(target_lat.shape)
        
        except Exception as e:
            self.logger.error(f"数据重网格化失败: {e}")
            return np.full(target_coords[0].shape, np.nan)
    
    def compute_spatial_statistics(self, 
                                  model_data: np.ndarray,
                                  grace_data: np.ndarray,
                                  coordinates: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """计算空间统计量"""
        try:
            # 展平数据
            model_flat = model_data.flatten()
            grace_flat = grace_data.flatten()
            
            # 移除无效值
            valid_mask = ~(np.isnan(model_flat) | np.isnan(grace_flat))
            model_valid = model_flat[valid_mask]
            grace_valid = grace_flat[valid_mask]
            
            if len(model_valid) == 0:
                return {}
            
            # 计算统计量
            correlation = np.corrcoef(model_valid, grace_valid)[0, 1]
            rmse = np.sqrt(mean_squared_error(grace_valid, model_valid))
            mae = mean_absolute_error(grace_valid, model_valid)
            bias = np.mean(model_valid - grace_valid)
            r_squared = r2_score(grace_valid, model_valid)
            
            # 空间相关性
            lat, lon = coordinates
            spatial_corr = self._compute_spatial_correlation(model_data, grace_data, lat, lon)
            
            return {
                'correlation': correlation,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'r_squared': r_squared,
                'spatial_correlation': spatial_corr,
                'valid_pixels': len(model_valid),
                'total_pixels': len(model_flat)
            }
        
        except Exception as e:
            self.logger.error(f"空间统计计算失败: {e}")
            return {}
    
    def _compute_spatial_correlation(self, 
                                   model_data: np.ndarray,
                                   grace_data: np.ndarray,
                                   lat: np.ndarray,
                                   lon: np.ndarray) -> float:
        """计算空间相关性"""
        try:
            # 简化的空间相关性计算
            # 这里可以实现更复杂的空间自相关分析
            valid_mask = ~(np.isnan(model_data) | np.isnan(grace_data))
            if np.sum(valid_mask) < 10:
                return np.nan
            
            model_valid = model_data[valid_mask]
            grace_valid = grace_data[valid_mask]
            
            return np.corrcoef(model_valid, grace_valid)[0, 1]
        
        except Exception as e:
            self.logger.warning(f"空间相关性计算失败: {e}")
            return np.nan

class GRACEValidator:
    """GRACE数据验证器"""
    
    def __init__(self, config: GRACEComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_processor = TimeSeriesProcessor(config)
        self.spatial_processor = SpatialProcessor(config)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def validate(self, 
                model: nn.Module,
                grace_observations: List[GRACEObservation]) -> Dict[str, Any]:
        """执行GRACE数据验证"""
        try:
            self.logger.info("开始GRACE数据验证...")
            
            # 过滤观测数据
            filtered_observations = self._filter_observations(grace_observations)
            
            if not filtered_observations:
                self.logger.warning("没有有效的GRACE观测数据")
                return {'error': '没有有效的观测数据'}
            
            # 生成模型预测
            model_predictions = self._generate_model_predictions(model, filtered_observations)
            
            # 执行验证
            validation_results = {}
            
            # 时间序列验证
            if AnalysisType.TIME_SERIES in self.config.analysis_types:
                validation_results['time_series'] = self._validate_time_series(
                    model_predictions, filtered_observations
                )
            
            # 趋势分析
            if AnalysisType.TREND in self.config.analysis_types:
                validation_results['trend_analysis'] = self._analyze_trends(
                    model_predictions, filtered_observations
                )
            
            # 季节性分析
            if AnalysisType.SEASONAL in self.config.analysis_types:
                validation_results['seasonal_analysis'] = self._analyze_seasonality(
                    model_predictions, filtered_observations
                )
            
            # 空间验证
            validation_results['spatial_validation'] = self._validate_spatial(
                model_predictions, filtered_observations
            )
            
            # 不确定性分析
            validation_results['uncertainty_analysis'] = self._analyze_uncertainty(
                model_predictions, filtered_observations
            )
            
            # 绘制结果
            if self.config.enable_plotting:
                self._plot_results(validation_results, filtered_observations)
            
            # 汇总结果
            summary = {
                'total_observations': len(grace_observations),
                'filtered_observations': len(filtered_observations),
                'validation_results': validation_results,
                'config': self.config
            }
            
            self.logger.info("GRACE数据验证完成")
            return summary
        
        except Exception as e:
            self.logger.error(f"GRACE验证失败: {e}")
            return {'error': str(e)}
    
    def _filter_observations(self, observations: List[GRACEObservation]) -> List[GRACEObservation]:
        """过滤观测数据"""
        filtered = []
        
        for obs in observations:
            # 质量过滤
            if self.config.enable_quality_filter:
                quality_levels = {
                    QualityFlag.EXCELLENT: 5,
                    QualityFlag.GOOD: 4,
                    QualityFlag.FAIR: 3,
                    QualityFlag.POOR: 2,
                    QualityFlag.BAD: 1
                }
                
                if quality_levels.get(obs.quality_flag, 0) < quality_levels.get(self.config.min_quality, 0):
                    continue
            
            # 数据覆盖率过滤
            if obs.data_coverage < self.config.min_data_coverage:
                continue
            
            # 异常值检测
            if self.config.outlier_threshold > 0:
                data_flat = obs.data.flatten()
                valid_data = data_flat[~np.isnan(data_flat)]
                
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    
                    if std_val > 0:
                        outlier_mask = np.abs(valid_data - mean_val) > self.config.outlier_threshold * std_val
                        if np.sum(outlier_mask) / len(valid_data) > 0.5:  # 如果超过50%是异常值，跳过
                            continue
            
            filtered.append(obs)
        
        return filtered
    
    def _generate_model_predictions(self, 
                                  model: nn.Module,
                                  observations: List[GRACEObservation]) -> Dict[str, Any]:
        """生成模型预测"""
        predictions = {}
        
        model.eval()
        with torch.no_grad():
            for obs in observations:
                try:
                    # 准备输入数据
                    lat, lon = obs.coordinates
                    
                    # 创建时间特征
                    time_features = self._create_time_features(obs.acquisition_time)
                    
                    # 创建空间网格
                    lat_flat = lat.flatten()
                    lon_flat = lon.flatten()
                    
                    # 组合输入特征
                    inputs = []
                    for i in range(len(lat_flat)):
                        input_vec = [lat_flat[i], lon_flat[i]] + time_features
                        inputs.append(input_vec)
                    
                    inputs_tensor = torch.FloatTensor(inputs)
                    
                    # 模型预测
                    pred_flat = model(inputs_tensor).numpy().flatten()
                    pred_grid = pred_flat.reshape(lat.shape)
                    
                    predictions[obs.observation_id] = {
                        'prediction': pred_grid,
                        'coordinates': obs.coordinates,
                        'time': obs.acquisition_time,
                        'observation': obs
                    }
                
                except Exception as e:
                    self.logger.warning(f"模型预测失败 {obs.observation_id}: {e}")
        
        return predictions
    
    def _create_time_features(self, time: datetime) -> List[float]:
        """创建时间特征"""
        # 年份归一化
        year_norm = (time.year - 2000) / 50.0
        
        # 月份的周期性特征
        month_sin = np.sin(2 * np.pi * time.month / 12)
        month_cos = np.cos(2 * np.pi * time.month / 12)
        
        # 日期的周期性特征
        day_sin = np.sin(2 * np.pi * time.timetuple().tm_yday / 365)
        day_cos = np.cos(2 * np.pi * time.timetuple().tm_yday / 365)
        
        return [year_norm, month_sin, month_cos, day_sin, day_cos]
    
    def _validate_time_series(self, 
                            predictions: Dict[str, Any],
                            observations: List[GRACEObservation]) -> Dict[str, Any]:
        """时间序列验证"""
        try:
            # 收集时间序列数据
            model_series = []
            grace_series = []
            time_series = []
            
            for obs in observations:
                if obs.observation_id in predictions:
                    pred_data = predictions[obs.observation_id]['prediction']
                    grace_data = obs.data
                    
                    # 计算空间平均值
                    pred_mean = np.nanmean(pred_data)
                    grace_mean = np.nanmean(grace_data)
                    
                    if not (np.isnan(pred_mean) or np.isnan(grace_mean)):
                        model_series.append(pred_mean)
                        grace_series.append(grace_mean)
                        time_series.append(obs.acquisition_time)
            
            if len(model_series) < 2:
                return {'error': '时间序列数据不足'}
            
            # 对齐时间序列
            model_aligned, grace_aligned, times_aligned = self.time_processor.align_time_series(
                np.array(model_series), time_series,
                np.array(grace_series), time_series
            )
            
            if len(model_aligned) == 0:
                return {'error': '时间序列对齐失败'}
            
            # 计算时间序列指标
            correlation = np.corrcoef(model_aligned, grace_aligned)[0, 1]
            rmse = np.sqrt(mean_squared_error(grace_aligned, model_aligned))
            mae = mean_absolute_error(grace_aligned, model_aligned)
            bias = np.mean(model_aligned - grace_aligned)
            
            # 趋势分析
            model_detrended, model_trend = self.time_processor.detrend_data(model_aligned)
            grace_detrended, grace_trend = self.time_processor.detrend_data(grace_aligned)
            
            # 计算趋势相关性
            trend_correlation = np.corrcoef(model_trend, grace_trend)[0, 1]
            
            return {
                'correlation': correlation,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'trend_correlation': trend_correlation,
                'data_points': len(model_aligned),
                'time_range': (min(times_aligned), max(times_aligned)),
                'model_series': model_aligned,
                'grace_series': grace_aligned,
                'times': times_aligned
            }
        
        except Exception as e:
            self.logger.error(f"时间序列验证失败: {e}")
            return {'error': str(e)}
    
    def _analyze_trends(self, 
                       predictions: Dict[str, Any],
                       observations: List[GRACEObservation]) -> Dict[str, Any]:
        """趋势分析"""
        try:
            # 收集时间序列数据
            model_values = []
            grace_values = []
            times = []
            
            for obs in sorted(observations, key=lambda x: x.acquisition_time):
                if obs.observation_id in predictions:
                    pred_data = predictions[obs.observation_id]['prediction']
                    grace_data = obs.data
                    
                    # 计算空间平均值
                    pred_mean = np.nanmean(pred_data)
                    grace_mean = np.nanmean(grace_data)
                    
                    if not (np.isnan(pred_mean) or np.isnan(grace_mean)):
                        model_values.append(pred_mean)
                        grace_values.append(grace_mean)
                        times.append(obs.acquisition_time)
            
            if len(model_values) < 3:
                return {'error': '趋势分析数据不足'}
            
            # 转换时间为数值
            time_numeric = [(t - times[0]).total_seconds() / (365.25 * 24 * 3600) for t in times]
            
            # 计算趋势
            model_slope, model_intercept, model_r, _, _ = stats.linregress(time_numeric, model_values)
            grace_slope, grace_intercept, grace_r, _, _ = stats.linregress(time_numeric, grace_values)
            
            # 趋势比较
            slope_ratio = model_slope / grace_slope if grace_slope != 0 else np.inf
            slope_difference = model_slope - grace_slope
            
            return {
                'model_trend': {
                    'slope': model_slope,
                    'intercept': model_intercept,
                    'r_value': model_r
                },
                'grace_trend': {
                    'slope': grace_slope,
                    'intercept': grace_intercept,
                    'r_value': grace_r
                },
                'comparison': {
                    'slope_ratio': slope_ratio,
                    'slope_difference': slope_difference,
                    'trend_correlation': np.corrcoef([model_slope], [grace_slope])[0, 1]
                }
            }
        
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_seasonality(self, 
                           predictions: Dict[str, Any],
                           observations: List[GRACEObservation]) -> Dict[str, Any]:
        """季节性分析"""
        try:
            # 收集时间序列数据
            model_values = []
            grace_values = []
            times = []
            
            for obs in observations:
                if obs.observation_id in predictions:
                    pred_data = predictions[obs.observation_id]['prediction']
                    grace_data = obs.data
                    
                    pred_mean = np.nanmean(pred_data)
                    grace_mean = np.nanmean(grace_data)
                    
                    if not (np.isnan(pred_mean) or np.isnan(grace_mean)):
                        model_values.append(pred_mean)
                        grace_values.append(grace_mean)
                        times.append(obs.acquisition_time)
            
            if len(model_values) < 12:
                return {'error': '季节性分析数据不足'}
            
            # 季节性分析
            model_seasonal = self.time_processor.analyze_seasonality(np.array(model_values), times)
            grace_seasonal = self.time_processor.analyze_seasonality(np.array(grace_values), times)
            
            # 比较季节性特征
            amplitude_ratio = (model_seasonal.get('seasonal_amplitude', 0) / 
                             grace_seasonal.get('seasonal_amplitude', 1))
            
            return {
                'model_seasonality': model_seasonal,
                'grace_seasonality': grace_seasonal,
                'amplitude_ratio': amplitude_ratio
            }
        
        except Exception as e:
            self.logger.error(f"季节性分析失败: {e}")
            return {'error': str(e)}
    
    def _validate_spatial(self, 
                         predictions: Dict[str, Any],
                         observations: List[GRACEObservation]) -> Dict[str, Any]:
        """空间验证"""
        try:
            spatial_results = {}
            
            for scale in self.config.validation_scales:
                scale_results = []
                
                for obs in observations:
                    if obs.observation_id in predictions:
                        pred_data = predictions[obs.observation_id]['prediction']
                        grace_data = obs.data
                        coordinates = obs.coordinates
                        
                        if scale == ValidationScale.PIXEL:
                            # 像素级验证
                            stats = self.spatial_processor.compute_spatial_statistics(
                                pred_data, grace_data, coordinates
                            )
                        elif scale == ValidationScale.REGIONAL:
                            # 区域级验证（空间平均）
                            pred_mean = np.nanmean(pred_data)
                            grace_mean = np.nanmean(grace_data)
                            
                            if not (np.isnan(pred_mean) or np.isnan(grace_mean)):
                                stats = {
                                    'correlation': 1.0,  # 单点相关性
                                    'rmse': abs(pred_mean - grace_mean),
                                    'mae': abs(pred_mean - grace_mean),
                                    'bias': pred_mean - grace_mean,
                                    'r_squared': 1.0
                                }
                            else:
                                stats = {}
                        else:
                            stats = {}
                        
                        if stats:
                            scale_results.append(stats)
                
                # 汇总尺度结果
                if scale_results:
                    spatial_results[scale.value] = self._aggregate_spatial_results(scale_results)
            
            return spatial_results
        
        except Exception as e:
            self.logger.error(f"空间验证失败: {e}")
            return {'error': str(e)}
    
    def _aggregate_spatial_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """汇总空间验证结果"""
        if not results:
            return {}
        
        aggregated = {}
        
        for key in results[0].keys():
            values = [r[key] for r in results if key in r and not np.isnan(r[key])]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def _analyze_uncertainty(self, 
                           predictions: Dict[str, Any],
                           observations: List[GRACEObservation]) -> Dict[str, Any]:
        """不确定性分析"""
        try:
            uncertainty_results = {}
            
            # 收集不确定性数据
            model_uncertainties = []
            grace_uncertainties = []
            prediction_errors = []
            
            for obs in observations:
                if obs.observation_id in predictions and obs.uncertainty is not None:
                    pred_data = predictions[obs.observation_id]['prediction']
                    grace_data = obs.data
                    grace_unc = obs.uncertainty
                    
                    # 计算预测误差
                    error = np.abs(pred_data - grace_data)
                    
                    # 收集统计量
                    model_uncertainties.append(np.nanstd(pred_data))
                    grace_uncertainties.append(np.nanmean(grace_unc))
                    prediction_errors.append(np.nanmean(error))
            
            if model_uncertainties:
                uncertainty_results = {
                    'model_uncertainty_mean': np.mean(model_uncertainties),
                    'grace_uncertainty_mean': np.mean(grace_uncertainties),
                    'prediction_error_mean': np.mean(prediction_errors),
                    'uncertainty_correlation': np.corrcoef(model_uncertainties, grace_uncertainties)[0, 1] if len(model_uncertainties) > 1 else np.nan
                }
            
            return uncertainty_results
        
        except Exception as e:
            self.logger.error(f"不确定性分析失败: {e}")
            return {'error': str(e)}
    
    def _plot_results(self, 
                     results: Dict[str, Any],
                     observations: List[GRACEObservation]):
        """绘制验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('GRACE数据验证结果', fontsize=16, fontweight='bold')
            
            # 时间序列对比
            if 'time_series' in results and 'model_series' in results['time_series']:
                ax = axes[0, 0]
                ts_data = results['time_series']
                
                ax.plot(ts_data['times'], ts_data['model_series'], 'b-', label='模型预测', linewidth=2)
                ax.plot(ts_data['times'], ts_data['grace_series'], 'r-', label='GRACE观测', linewidth=2)
                ax.set_xlabel('时间')
                ax.set_ylabel('等效水高 (mm)')
                ax.set_title(f"时间序列对比 (相关系数: {ts_data.get('correlation', 0):.3f})")
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 散点图
            if 'time_series' in results and 'model_series' in results['time_series']:
                ax = axes[0, 1]
                ts_data = results['time_series']
                
                ax.scatter(ts_data['grace_series'], ts_data['model_series'], alpha=0.6)
                
                # 添加1:1线
                min_val = min(np.min(ts_data['grace_series']), np.min(ts_data['model_series']))
                max_val = max(np.max(ts_data['grace_series']), np.max(ts_data['model_series']))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax.set_xlabel('GRACE观测 (mm)')
                ax.set_ylabel('模型预测 (mm)')
                ax.set_title(f"散点图 (R² = {ts_data.get('correlation', 0)**2:.3f})")
                ax.grid(True, alpha=0.3)
            
            # 趋势分析
            if 'trend_analysis' in results and 'model_trend' in results['trend_analysis']:
                ax = axes[1, 0]
                trend_data = results['trend_analysis']
                
                model_trend = trend_data['model_trend']
                grace_trend = trend_data['grace_trend']
                
                categories = ['模型', 'GRACE']
                slopes = [model_trend['slope'], grace_trend['slope']]
                
                bars = ax.bar(categories, slopes, color=['blue', 'red'], alpha=0.7)
                ax.set_ylabel('趋势斜率 (mm/年)')
                ax.set_title('趋势对比')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, slope in zip(bars, slopes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{slope:.3f}', ha='center', va='bottom')
            
            # 空间验证结果
            if 'spatial_validation' in results:
                ax = axes[1, 1]
                spatial_data = results['spatial_validation']
                
                metrics = []
                values = []
                
                for scale, scale_data in spatial_data.items():
                    if 'correlation_mean' in scale_data:
                        metrics.append(f'{scale}\n相关系数')
                        values.append(scale_data['correlation_mean'])
                
                if metrics:
                    bars = ax.bar(metrics, values, color='green', alpha=0.7)
                    ax.set_ylabel('相关系数')
                    ax.set_title('空间验证结果')
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                    
                    # 添加数值标签
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/grace_validation_results.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       observations: List[GRACEObservation]) -> str:
        """生成验证报告"""
        report_lines = [
            "="*80,
            "GRACE数据验证报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. 数据概览",
            "-"*40,
            f"总观测数量: {results.get('total_observations', 0)}",
            f"过滤后观测数量: {results.get('filtered_observations', 0)}",
        ]
        
        # 时间序列验证结果
        if 'validation_results' in results and 'time_series' in results['validation_results']:
            ts_results = results['validation_results']['time_series']
            
            report_lines.extend([
                "",
                "2. 时间序列验证",
                "-"*40,
                f"相关系数: {ts_results.get('correlation', 'N/A'):.4f}",
                f"RMSE: {ts_results.get('rmse', 'N/A'):.4f} mm",
                f"MAE: {ts_results.get('mae', 'N/A'):.4f} mm",
                f"偏差: {ts_results.get('bias', 'N/A'):.4f} mm",
                f"数据点数: {ts_results.get('data_points', 'N/A')}"
            ])
        
        # 趋势分析结果
        if 'validation_results' in results and 'trend_analysis' in results['validation_results']:
            trend_results = results['validation_results']['trend_analysis']
            
            if 'model_trend' in trend_results:
                model_trend = trend_results['model_trend']
                grace_trend = trend_results['grace_trend']
                
                report_lines.extend([
                    "",
                    "3. 趋势分析",
                    "-"*40,
                    f"模型趋势: {model_trend.get('slope', 'N/A'):.4f} mm/年",
                    f"GRACE趋势: {grace_trend.get('slope', 'N/A'):.4f} mm/年",
                    f"趋势比值: {trend_results.get('comparison', {}).get('slope_ratio', 'N/A'):.4f}"
                ])
        
        # 空间验证结果
        if 'validation_results' in results and 'spatial_validation' in results['validation_results']:
            spatial_results = results['validation_results']['spatial_validation']
            
            report_lines.extend([
                "",
                "4. 空间验证",
                "-"*40
            ])
            
            for scale, scale_data in spatial_results.items():
                if 'correlation_mean' in scale_data:
                    report_lines.extend([
                        f"{scale}级验证:",
                        f"  平均相关系数: {scale_data.get('correlation_mean', 'N/A'):.4f}",
                        f"  平均RMSE: {scale_data.get('rmse_mean', 'N/A'):.4f}",
                        f"  平均偏差: {scale_data.get('bias_mean', 'N/A'):.4f}"
                    ])
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_grace_validator(config: Optional[GRACEComparisonConfig] = None) -> GRACEValidator:
    """创建GRACE验证器"""
    if config is None:
        config = GRACEComparisonConfig()
    
    return GRACEValidator(config)

if __name__ == "__main__":
    # 测试代码
    import torch.nn as nn
    
    # 创建简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(7, 64),  # lat, lon, year, month_sin, month_cos, day_sin, day_cos
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 创建测试数据
    test_observations = [
        GRACEObservation(
            observation_id="grace_001",
            product_type=GRACEProduct.CSR_RL06,
            processing_level=ProcessingLevel.L3,
            data_type=DataType.EQUIVALENT_WATER_HEIGHT,
            acquisition_time=datetime(2023, 6, 15),
            time_span=(datetime(2023, 6, 1), datetime(2023, 6, 30)),
            spatial_extent=(29.0, 31.0, 89.0, 91.0),
            spatial_resolution=1.0,
            data=np.random.normal(100, 20, (3, 3)),
            coordinates=(np.linspace(29.5, 30.5, 3).reshape(3, 1) * np.ones((3, 3)),
                        np.linspace(89.5, 90.5, 3).reshape(1, 3) * np.ones((3, 3))),
            units="mm",
            uncertainty=np.random.uniform(5, 15, (3, 3)),
            quality_flag=QualityFlag.GOOD,
            data_coverage=0.95
        ),
        GRACEObservation(
            observation_id="grace_002",
            product_type=GRACEProduct.JPL_RL06,
            processing_level=ProcessingLevel.L3,
            data_type=DataType.EQUIVALENT_WATER_HEIGHT,
            acquisition_time=datetime(2023, 7, 15),
            time_span=(datetime(2023, 7, 1), datetime(2023, 7, 31)),
            spatial_extent=(29.0, 31.0, 89.0, 91.0),
            spatial_resolution=1.0,
            data=np.random.normal(95, 18, (3, 3)),
            coordinates=(np.linspace(29.5, 30.5, 3).reshape(3, 1) * np.ones((3, 3)),
                        np.linspace(89.5, 90.5, 3).reshape(1, 3) * np.ones((3, 3))),
            units="mm",
            uncertainty=np.random.uniform(4, 12, (3, 3)),
            quality_flag=QualityFlag.EXCELLENT,
            data_coverage=0.98
        )
    ]
    
    # 创建验证器
    config = GRACEComparisonConfig(
        enable_plotting=True,
        save_plots=False,
        verbose=True
    )
    
    validator = create_grace_validator(config)
    
    # 创建测试模型
    model = TestModel()
    
    # 执行验证
    print("开始GRACE数据验证...")
    results = validator.validate(model, test_observations)
    
    # 生成报告
    report = validator.generate_report(results, test_observations)
    print(report)
    
    print("\nGRACE数据验证完成！")