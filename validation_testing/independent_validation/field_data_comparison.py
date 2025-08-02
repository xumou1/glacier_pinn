#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
野外数据对比验证模块

该模块实现了与野外实测数据的对比验证，用于评估模型在真实观测数据上的表现，包括：
- 多种野外数据类型支持
- 时空匹配策略
- 统计显著性检验
- 误差分析和偏差校正
- 数据质量评估
- 不确定性量化

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
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import pandas as pd
from datetime import datetime, timedelta

class FieldDataType(Enum):
    """野外数据类型"""
    TEMPERATURE = "temperature"  # 温度测量
    VELOCITY = "velocity"  # 流速测量
    THICKNESS = "thickness"  # 厚度测量
    MASS_BALANCE = "mass_balance"  # 质量平衡
    SURFACE_ELEVATION = "surface_elevation"  # 表面高程
    METEOROLOGICAL = "meteorological"  # 气象数据
    STRAIN_RATE = "strain_rate"  # 应变率
    STRESS = "stress"  # 应力
    DENSITY = "density"  # 密度
    ALBEDO = "albedo"  # 反照率
    ROUGHNESS = "roughness"  # 表面粗糙度
    UNKNOWN = "unknown"  # 未知类型

class MeasurementMethod(Enum):
    """测量方法"""
    THERMISTOR = "thermistor"  # 温度计
    GPS = "gps"  # GPS测量
    RADAR = "radar"  # 雷达测量
    LIDAR = "lidar"  # 激光雷达
    PHOTOGRAMMETRY = "photogrammetry"  # 摄影测量
    STAKE_MEASUREMENT = "stake_measurement"  # 花杆测量
    WEATHER_STATION = "weather_station"  # 气象站
    MANUAL_SURVEY = "manual_survey"  # 人工测量
    AUTOMATIC_SENSOR = "automatic_sensor"  # 自动传感器
    UNKNOWN = "unknown"  # 未知方法

class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    UNKNOWN = "unknown"  # 未知

class MatchingStrategy(Enum):
    """匹配策略"""
    NEAREST_NEIGHBOR = "nearest_neighbor"  # 最近邻
    BILINEAR_INTERPOLATION = "bilinear_interpolation"  # 双线性插值
    KRIGING = "kriging"  # 克里金插值
    INVERSE_DISTANCE = "inverse_distance"  # 反距离权重
    TEMPORAL_INTERPOLATION = "temporal_interpolation"  # 时间插值
    SPATIOTEMPORAL = "spatiotemporal"  # 时空匹配

class ValidationMetric(Enum):
    """验证指标"""
    BIAS = "bias"  # 偏差
    MAE = "mae"  # 平均绝对误差
    RMSE = "rmse"  # 均方根误差
    CORRELATION = "correlation"  # 相关系数
    R_SQUARED = "r_squared"  # 决定系数
    NASH_SUTCLIFFE = "nash_sutcliffe"  # Nash-Sutcliffe效率系数
    INDEX_OF_AGREEMENT = "index_of_agreement"  # 一致性指数
    PERCENT_BIAS = "percent_bias"  # 百分比偏差
    NORMALIZED_RMSE = "normalized_rmse"  # 标准化RMSE

@dataclass
class FieldMeasurement:
    """野外测量数据"""
    measurement_id: str
    data_type: FieldDataType
    method: MeasurementMethod
    
    # 时空信息
    x: float  # X坐标 (m)
    y: float  # Y坐标 (m)
    z: Optional[float] = None  # Z坐标/高程 (m)
    timestamp: Optional[datetime] = None  # 测量时间
    
    # 测量值
    value: float  # 测量值
    uncertainty: float = 0.0  # 不确定性
    unit: str = ""  # 单位
    
    # 质量信息
    quality: DataQuality = DataQuality.UNKNOWN
    confidence: float = 1.0  # 置信度 (0-1)
    
    # 元数据
    instrument: Optional[str] = None  # 仪器信息
    operator: Optional[str] = None  # 操作员
    weather_conditions: Optional[str] = None  # 天气条件
    notes: Optional[str] = None  # 备注
    
    # 处理标记
    is_outlier: bool = False  # 是否为异常值
    is_validated: bool = False  # 是否已验证
    correction_applied: bool = False  # 是否已校正

@dataclass
class FieldDataComparisonConfig:
    """野外数据对比配置"""
    # 匹配设置
    matching_strategy: MatchingStrategy = MatchingStrategy.NEAREST_NEIGHBOR
    spatial_tolerance: float = 100.0  # 空间容差 (m)
    temporal_tolerance: float = 24.0  # 时间容差 (hours)
    
    # 验证指标
    validation_metrics: List[ValidationMetric] = None
    
    # 质量控制
    min_quality: DataQuality = DataQuality.FAIR
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # 异常值阈值（标准差倍数）
    
    # 数据过滤
    min_confidence: float = 0.5  # 最小置信度
    max_uncertainty: float = 1.0  # 最大不确定性
    exclude_outliers: bool = True  # 排除异常值
    
    # 插值设置
    interpolation_method: str = "linear"  # 插值方法
    extrapolation_allowed: bool = False  # 是否允许外推
    
    # 统计测试
    significance_level: float = 0.05  # 显著性水平
    bootstrap_samples: int = 1000  # 自举样本数
    
    # 偏差校正
    bias_correction: bool = True  # 是否进行偏差校正
    correction_method: str = "linear"  # 校正方法
    
    # 分组分析
    group_by_data_type: bool = True  # 按数据类型分组
    group_by_method: bool = True  # 按测量方法分组
    group_by_quality: bool = True  # 按质量分组
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "field_validation_plots"
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.validation_metrics is None:
            self.validation_metrics = [
                ValidationMetric.BIAS,
                ValidationMetric.MAE,
                ValidationMetric.RMSE,
                ValidationMetric.CORRELATION,
                ValidationMetric.R_SQUARED
            ]

class SpatioTemporalMatcher:
    """时空匹配器"""
    
    def __init__(self, config: FieldDataComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def match_predictions_to_measurements(self, 
                                        model: nn.Module,
                                        measurements: List[FieldMeasurement],
                                        domain_bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """将模型预测与野外测量匹配"""
        matched_data = {
            'measurements': [],
            'predictions': [],
            'coordinates': [],
            'timestamps': [],
            'metadata': []
        }
        
        valid_measurements = self._filter_measurements(measurements)
        
        for measurement in valid_measurements:
            try:
                # 准备输入数据
                input_data = self._prepare_model_input(measurement)
                
                # 获取模型预测
                prediction = self._get_model_prediction(model, input_data)
                
                # 存储匹配的数据
                matched_data['measurements'].append(measurement.value)
                matched_data['predictions'].append(prediction)
                matched_data['coordinates'].append((measurement.x, measurement.y))
                matched_data['timestamps'].append(measurement.timestamp)
                matched_data['metadata'].append({
                    'measurement_id': measurement.measurement_id,
                    'data_type': measurement.data_type,
                    'method': measurement.method,
                    'quality': measurement.quality,
                    'uncertainty': measurement.uncertainty
                })
            
            except Exception as e:
                self.logger.warning(f"匹配测量 {measurement.measurement_id} 失败: {e}")
        
        # 转换为张量
        if matched_data['measurements']:
            matched_data['measurements'] = torch.tensor(matched_data['measurements'], dtype=torch.float32)
            matched_data['predictions'] = torch.tensor(matched_data['predictions'], dtype=torch.float32)
        
        return matched_data
    
    def _filter_measurements(self, measurements: List[FieldMeasurement]) -> List[FieldMeasurement]:
        """过滤测量数据"""
        filtered = []
        
        for measurement in measurements:
            # 质量过滤
            if self._quality_to_score(measurement.quality) < self._quality_to_score(self.config.min_quality):
                continue
            
            # 置信度过滤
            if measurement.confidence < self.config.min_confidence:
                continue
            
            # 不确定性过滤
            if measurement.uncertainty > self.config.max_uncertainty:
                continue
            
            # 异常值过滤
            if self.config.exclude_outliers and measurement.is_outlier:
                continue
            
            filtered.append(measurement)
        
        return filtered
    
    def _quality_to_score(self, quality: DataQuality) -> float:
        """将质量等级转换为数值分数"""
        quality_scores = {
            DataQuality.EXCELLENT: 5.0,
            DataQuality.GOOD: 4.0,
            DataQuality.FAIR: 3.0,
            DataQuality.POOR: 2.0,
            DataQuality.UNKNOWN: 1.0
        }
        return quality_scores.get(quality, 1.0)
    
    def _prepare_model_input(self, measurement: FieldMeasurement) -> torch.Tensor:
        """准备模型输入"""
        # 基本坐标输入
        inputs = [measurement.x, measurement.y]
        
        # 添加时间信息（如果有）
        if measurement.timestamp:
            # 将时间戳转换为相对时间（天）
            reference_time = datetime(2020, 1, 1)  # 参考时间
            time_delta = (measurement.timestamp - reference_time).total_seconds() / (24 * 3600)
            inputs.append(time_delta)
        else:
            inputs.append(0.0)  # 默认时间
        
        # 添加高程信息（如果有）
        if measurement.z is not None:
            inputs.append(measurement.z)
        
        return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    
    def _get_model_prediction(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """获取模型预测"""
        with torch.no_grad():
            model.eval()
            prediction = model(input_data)
            
            if prediction.dim() > 1:
                prediction = prediction[0, 0]  # 取第一个输出的第一个值
            else:
                prediction = prediction[0]
            
            return prediction.item()

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, config: FieldDataComparisonConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_validation_metrics(self, 
                                 predictions: torch.Tensor, 
                                 measurements: torch.Tensor,
                                 metric_types: List[ValidationMetric] = None) -> Dict[str, float]:
        """计算验证指标"""
        if metric_types is None:
            metric_types = self.config.validation_metrics
        
        metrics = {}
        
        # 转换为numpy数组
        pred = predictions.detach().numpy().flatten()
        meas = measurements.detach().numpy().flatten()
        
        # 确保长度一致
        min_len = min(len(pred), len(meas))
        pred = pred[:min_len]
        meas = meas[:min_len]
        
        if len(pred) == 0:
            return metrics
        
        try:
            for metric_type in metric_types:
                if metric_type == ValidationMetric.BIAS:
                    bias = np.mean(pred - meas)
                    metrics['bias'] = float(bias)
                
                elif metric_type == ValidationMetric.MAE:
                    mae = np.mean(np.abs(pred - meas))
                    metrics['mae'] = float(mae)
                
                elif metric_type == ValidationMetric.RMSE:
                    rmse = np.sqrt(np.mean((pred - meas) ** 2))
                    metrics['rmse'] = float(rmse)
                
                elif metric_type == ValidationMetric.CORRELATION:
                    if np.std(pred) > 1e-8 and np.std(meas) > 1e-8:
                        correlation = np.corrcoef(pred, meas)[0, 1]
                        metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                    else:
                        metrics['correlation'] = 0.0
                
                elif metric_type == ValidationMetric.R_SQUARED:
                    ss_res = np.sum((meas - pred) ** 2)
                    ss_tot = np.sum((meas - np.mean(meas)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    metrics['r_squared'] = float(r2)
                
                elif metric_type == ValidationMetric.NASH_SUTCLIFFE:
                    ss_res = np.sum((meas - pred) ** 2)
                    ss_tot = np.sum((meas - np.mean(meas)) ** 2)
                    nse = 1 - (ss_res / (ss_tot + 1e-8))
                    metrics['nash_sutcliffe'] = float(nse)
                
                elif metric_type == ValidationMetric.INDEX_OF_AGREEMENT:
                    mean_meas = np.mean(meas)
                    numerator = np.sum((pred - meas) ** 2)
                    denominator = np.sum((np.abs(pred - mean_meas) + np.abs(meas - mean_meas)) ** 2)
                    ioa = 1 - (numerator / (denominator + 1e-8))
                    metrics['index_of_agreement'] = float(ioa)
                
                elif metric_type == ValidationMetric.PERCENT_BIAS:
                    pbias = 100 * np.sum(pred - meas) / (np.sum(meas) + 1e-8)
                    metrics['percent_bias'] = float(pbias)
                
                elif metric_type == ValidationMetric.NORMALIZED_RMSE:
                    rmse = np.sqrt(np.mean((pred - meas) ** 2))
                    nrmse = rmse / (np.max(meas) - np.min(meas) + 1e-8)
                    metrics['normalized_rmse'] = float(nrmse)
        
        except Exception as e:
            self.logger.warning(f"计算验证指标时出错: {e}")
        
        return metrics
    
    def perform_significance_tests(self, 
                                 predictions: torch.Tensor, 
                                 measurements: torch.Tensor) -> Dict[str, Any]:
        """执行显著性检验"""
        test_results = {}
        
        pred = predictions.detach().numpy().flatten()
        meas = measurements.detach().numpy().flatten()
        
        try:
            # t检验（检验均值差异）
            t_stat, t_pvalue = stats.ttest_rel(pred, meas)
            test_results['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant': t_pvalue < self.config.significance_level
            }
            
            # Wilcoxon符号秩检验（非参数检验）
            w_stat, w_pvalue = stats.wilcoxon(pred, meas)
            test_results['wilcoxon_test'] = {
                'statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant': w_pvalue < self.config.significance_level
            }
            
            # Kolmogorov-Smirnov检验（分布差异）
            ks_stat, ks_pvalue = stats.ks_2samp(pred, meas)
            test_results['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'significant': ks_pvalue < self.config.significance_level
            }
            
            # 相关性显著性检验
            if len(pred) > 2:
                corr_coef, corr_pvalue = stats.pearsonr(pred, meas)
                test_results['correlation_test'] = {
                    'correlation': float(corr_coef),
                    'p_value': float(corr_pvalue),
                    'significant': corr_pvalue < self.config.significance_level
                }
        
        except Exception as e:
            self.logger.warning(f"显著性检验失败: {e}")
        
        return test_results
    
    def detect_outliers(self, data: torch.Tensor, method: str = "zscore") -> torch.Tensor:
        """检测异常值"""
        data_np = data.detach().numpy().flatten()
        
        if method == "zscore":
            z_scores = np.abs(stats.zscore(data_np))
            outlier_mask = z_scores > self.config.outlier_threshold
        elif method == "iqr":
            q1, q3 = np.percentile(data_np, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (data_np < lower_bound) | (data_np > upper_bound)
        else:
            outlier_mask = np.zeros(len(data_np), dtype=bool)
        
        return torch.from_numpy(outlier_mask)
    
    def bootstrap_confidence_intervals(self, 
                                     predictions: torch.Tensor, 
                                     measurements: torch.Tensor,
                                     metric_name: str = "correlation",
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """自举法计算置信区间"""
        pred = predictions.detach().numpy().flatten()
        meas = measurements.detach().numpy().flatten()
        
        n_samples = len(pred)
        bootstrap_metrics = []
        
        for _ in range(self.config.bootstrap_samples):
            # 重采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            pred_boot = pred[indices]
            meas_boot = meas[indices]
            
            # 计算指标
            if metric_name == "correlation":
                if np.std(pred_boot) > 1e-8 and np.std(meas_boot) > 1e-8:
                    metric_value = np.corrcoef(pred_boot, meas_boot)[0, 1]
                    if not np.isnan(metric_value):
                        bootstrap_metrics.append(metric_value)
            elif metric_name == "rmse":
                metric_value = np.sqrt(np.mean((pred_boot - meas_boot) ** 2))
                bootstrap_metrics.append(metric_value)
            elif metric_name == "bias":
                metric_value = np.mean(pred_boot - meas_boot)
                bootstrap_metrics.append(metric_value)
        
        if bootstrap_metrics:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            return {
                'mean': float(np.mean(bootstrap_metrics)),
                'std': float(np.std(bootstrap_metrics)),
                'lower_ci': float(np.percentile(bootstrap_metrics, lower_percentile)),
                'upper_ci': float(np.percentile(bootstrap_metrics, upper_percentile))
            }
        else:
            return {'mean': 0.0, 'std': 0.0, 'lower_ci': 0.0, 'upper_ci': 0.0}

class FieldDataValidator:
    """野外数据验证器"""
    
    def __init__(self, config: FieldDataComparisonConfig):
        self.config = config
        self.matcher = SpatioTemporalMatcher(config)
        self.analyzer = StatisticalAnalyzer(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, 
                model: nn.Module, 
                field_measurements: List[FieldMeasurement]) -> Dict[str, Any]:
        """执行野外数据验证"""
        results = {
            'total_measurements': len(field_measurements),
            'matched_data': {},
            'overall_metrics': {},
            'grouped_analysis': {},
            'statistical_tests': {},
            'outlier_analysis': {},
            'uncertainty_analysis': {},
            'bias_analysis': {}
        }
        
        try:
            # 匹配预测与测量
            matched_data = self.matcher.match_predictions_to_measurements(
                model, field_measurements
            )
            results['matched_data'] = matched_data
            
            if not matched_data['measurements'] or len(matched_data['measurements']) == 0:
                self.logger.warning("没有成功匹配的数据")
                return results
            
            results['valid_measurements'] = len(matched_data['measurements'])
            
            # 总体指标计算
            results['overall_metrics'] = self.analyzer.compute_validation_metrics(
                matched_data['predictions'], matched_data['measurements']
            )
            
            # 统计显著性检验
            results['statistical_tests'] = self.analyzer.perform_significance_tests(
                matched_data['predictions'], matched_data['measurements']
            )
            
            # 异常值分析
            results['outlier_analysis'] = self._analyze_outliers(matched_data)
            
            # 分组分析
            results['grouped_analysis'] = self._perform_grouped_analysis(
                matched_data, field_measurements
            )
            
            # 不确定性分析
            results['uncertainty_analysis'] = self._analyze_uncertainty(matched_data)
            
            # 偏差分析
            results['bias_analysis'] = self._analyze_bias(matched_data)
            
            # 置信区间计算
            results['confidence_intervals'] = self._compute_confidence_intervals(matched_data)
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, matched_data)
        
        except Exception as e:
            self.logger.error(f"野外数据验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_outliers(self, matched_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析异常值"""
        outlier_analysis = {
            'prediction_outliers': {},
            'measurement_outliers': {},
            'residual_outliers': {}
        }
        
        try:
            predictions = matched_data['predictions']
            measurements = matched_data['measurements']
            residuals = predictions - measurements
            
            # 预测值异常值
            pred_outliers = self.analyzer.detect_outliers(predictions)
            outlier_analysis['prediction_outliers'] = {
                'count': int(torch.sum(pred_outliers)),
                'percentage': float(torch.sum(pred_outliers)) / len(predictions) * 100,
                'indices': pred_outliers.nonzero().flatten().tolist()
            }
            
            # 测量值异常值
            meas_outliers = self.analyzer.detect_outliers(measurements)
            outlier_analysis['measurement_outliers'] = {
                'count': int(torch.sum(meas_outliers)),
                'percentage': float(torch.sum(meas_outliers)) / len(measurements) * 100,
                'indices': meas_outliers.nonzero().flatten().tolist()
            }
            
            # 残差异常值
            resid_outliers = self.analyzer.detect_outliers(residuals)
            outlier_analysis['residual_outliers'] = {
                'count': int(torch.sum(resid_outliers)),
                'percentage': float(torch.sum(resid_outliers)) / len(residuals) * 100,
                'indices': resid_outliers.nonzero().flatten().tolist()
            }
        
        except Exception as e:
            self.logger.warning(f"异常值分析失败: {e}")
        
        return outlier_analysis
    
    def _perform_grouped_analysis(self, 
                                matched_data: Dict[str, Any], 
                                field_measurements: List[FieldMeasurement]) -> Dict[str, Any]:
        """执行分组分析"""
        grouped_analysis = {}
        
        try:
            metadata = matched_data['metadata']
            predictions = matched_data['predictions']
            measurements = matched_data['measurements']
            
            # 按数据类型分组
            if self.config.group_by_data_type:
                grouped_analysis['by_data_type'] = self._group_by_attribute(
                    metadata, predictions, measurements, 'data_type'
                )
            
            # 按测量方法分组
            if self.config.group_by_method:
                grouped_analysis['by_method'] = self._group_by_attribute(
                    metadata, predictions, measurements, 'method'
                )
            
            # 按质量分组
            if self.config.group_by_quality:
                grouped_analysis['by_quality'] = self._group_by_attribute(
                    metadata, predictions, measurements, 'quality'
                )
        
        except Exception as e:
            self.logger.warning(f"分组分析失败: {e}")
        
        return grouped_analysis
    
    def _group_by_attribute(self, 
                          metadata: List[Dict], 
                          predictions: torch.Tensor, 
                          measurements: torch.Tensor,
                          attribute: str) -> Dict[str, Any]:
        """按属性分组分析"""
        groups = {}
        
        # 收集分组数据
        for i, meta in enumerate(metadata):
            if attribute in meta:
                group_key = str(meta[attribute])
                if group_key not in groups:
                    groups[group_key] = {'predictions': [], 'measurements': []}
                
                groups[group_key]['predictions'].append(predictions[i].item())
                groups[group_key]['measurements'].append(measurements[i].item())
        
        # 计算每组的指标
        group_results = {}
        for group_key, group_data in groups.items():
            if len(group_data['predictions']) > 0:
                pred_tensor = torch.tensor(group_data['predictions'])
                meas_tensor = torch.tensor(group_data['measurements'])
                
                group_metrics = self.analyzer.compute_validation_metrics(
                    pred_tensor, meas_tensor
                )
                group_metrics['count'] = len(group_data['predictions'])
                group_results[group_key] = group_metrics
        
        return group_results
    
    def _analyze_uncertainty(self, matched_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析不确定性"""
        uncertainty_analysis = {
            'measurement_uncertainty': {},
            'prediction_uncertainty': {},
            'total_uncertainty': {}
        }
        
        try:
            metadata = matched_data['metadata']
            predictions = matched_data['predictions']
            measurements = matched_data['measurements']
            
            # 测量不确定性统计
            uncertainties = [meta.get('uncertainty', 0) for meta in metadata]
            if uncertainties:
                uncertainty_analysis['measurement_uncertainty'] = {
                    'mean': float(np.mean(uncertainties)),
                    'std': float(np.std(uncertainties)),
                    'min': float(np.min(uncertainties)),
                    'max': float(np.max(uncertainties))
                }
            
            # 预测不确定性（基于残差）
            residuals = (predictions - measurements).detach().numpy()
            uncertainty_analysis['prediction_uncertainty'] = {
                'residual_std': float(np.std(residuals)),
                'residual_mean': float(np.mean(np.abs(residuals))),
                'residual_95_percentile': float(np.percentile(np.abs(residuals), 95))
            }
            
            # 总不确定性
            if uncertainties:
                total_uncertainty = np.sqrt(np.array(uncertainties)**2 + np.std(residuals)**2)
                uncertainty_analysis['total_uncertainty'] = {
                    'mean': float(np.mean(total_uncertainty)),
                    'std': float(np.std(total_uncertainty))
                }
        
        except Exception as e:
            self.logger.warning(f"不确定性分析失败: {e}")
        
        return uncertainty_analysis
    
    def _analyze_bias(self, matched_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析偏差"""
        bias_analysis = {
            'overall_bias': {},
            'systematic_bias': {},
            'bias_trends': {}
        }
        
        try:
            predictions = matched_data['predictions'].detach().numpy()
            measurements = matched_data['measurements'].detach().numpy()
            residuals = predictions - measurements
            
            # 总体偏差
            bias_analysis['overall_bias'] = {
                'mean_bias': float(np.mean(residuals)),
                'median_bias': float(np.median(residuals)),
                'bias_std': float(np.std(residuals)),
                'absolute_bias': float(np.mean(np.abs(residuals)))
            }
            
            # 系统性偏差检验
            # 使用游程检验检测系统性偏差
            signs = np.sign(residuals)
            runs, n_runs = self._count_runs(signs)
            expected_runs = 2 * np.sum(signs > 0) * np.sum(signs < 0) / len(signs) + 1
            
            bias_analysis['systematic_bias'] = {
                'runs_count': int(n_runs),
                'expected_runs': float(expected_runs),
                'systematic_bias_detected': n_runs < expected_runs * 0.5
            }
            
            # 偏差趋势（与测量值的关系）
            if len(measurements) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(measurements, residuals)
                bias_analysis['bias_trends'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'correlation': float(r_value),
                    'p_value': float(p_value),
                    'proportional_bias': abs(slope) > 0.1
                }
        
        except Exception as e:
            self.logger.warning(f"偏差分析失败: {e}")
        
        return bias_analysis
    
    def _count_runs(self, sequence: np.ndarray) -> Tuple[List, int]:
        """计算游程"""
        runs = []
        current_run = [sequence[0]]
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run.append(sequence[i])
            else:
                runs.append(current_run)
                current_run = [sequence[i]]
        
        runs.append(current_run)
        return runs, len(runs)
    
    def _compute_confidence_intervals(self, matched_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算置信区间"""
        confidence_intervals = {}
        
        try:
            predictions = matched_data['predictions']
            measurements = matched_data['measurements']
            
            # 主要指标的置信区间
            for metric in ['correlation', 'rmse', 'bias']:
                ci = self.analyzer.bootstrap_confidence_intervals(
                    predictions, measurements, metric
                )
                confidence_intervals[metric] = ci
        
        except Exception as e:
            self.logger.warning(f"置信区间计算失败: {e}")
        
        return confidence_intervals
    
    def _plot_results(self, results: Dict[str, Any], matched_data: Dict[str, Any]):
        """绘制结果"""
        try:
            predictions = matched_data['predictions'].detach().numpy()
            measurements = matched_data['measurements'].detach().numpy()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 散点图
            axes[0, 0].scatter(measurements, predictions, alpha=0.6)
            min_val = min(np.min(measurements), np.min(predictions))
            max_val = max(np.max(measurements), np.max(predictions))
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1线')
            axes[0, 0].set_xlabel('野外测量值')
            axes[0, 0].set_ylabel('模型预测值')
            axes[0, 0].set_title('预测值 vs 测量值')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 残差图
            residuals = predictions - measurements
            axes[0, 1].scatter(measurements, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('野外测量值')
            axes[0, 1].set_ylabel('残差 (预测值 - 测量值)')
            axes[0, 1].set_title('残差分布')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 残差直方图
            axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('残差')
            axes[1, 0].set_ylabel('频数')
            axes[1, 0].set_title('残差直方图')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('残差Q-Q图')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/field_validation_results.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "野外数据对比验证报告",
            "=" * 60,
            f"总测量数据: {results.get('total_measurements', 0)}",
            f"有效匹配数据: {results.get('valid_measurements', 0)}",
            ""
        ]
        
        # 总体指标
        if 'overall_metrics' in results:
            metrics = results['overall_metrics']
            report_lines.append("总体验证指标:")
            for metric_name, value in metrics.items():
                report_lines.append(f"  {metric_name}: {value:.6f}")
            report_lines.append("")
        
        # 统计显著性
        if 'statistical_tests' in results:
            tests = results['statistical_tests']
            report_lines.append("统计显著性检验:")
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    significant = "是" if test_result.get('significant', False) else "否"
                    report_lines.append(f"  {test_name}: p值={test_result.get('p_value', 0):.6f}, 显著={significant}")
            report_lines.append("")
        
        # 异常值分析
        if 'outlier_analysis' in results:
            outliers = results['outlier_analysis']
            report_lines.append("异常值分析:")
            for outlier_type, outlier_info in outliers.items():
                if isinstance(outlier_info, dict):
                    count = outlier_info.get('count', 0)
                    percentage = outlier_info.get('percentage', 0)
                    report_lines.append(f"  {outlier_type}: {count}个 ({percentage:.2f}%)")
            report_lines.append("")
        
        # 偏差分析
        if 'bias_analysis' in results:
            bias = results['bias_analysis']
            report_lines.append("偏差分析:")
            
            if 'overall_bias' in bias:
                ob = bias['overall_bias']
                report_lines.extend([
                    f"  平均偏差: {ob.get('mean_bias', 0):.6f}",
                    f"  绝对偏差: {ob.get('absolute_bias', 0):.6f}",
                    f"  偏差标准差: {ob.get('bias_std', 0):.6f}"
                ])
            
            if 'systematic_bias' in bias:
                sb = bias['systematic_bias']
                systematic = "是" if sb.get('systematic_bias_detected', False) else "否"
                report_lines.append(f"  系统性偏差: {systematic}")
            
            report_lines.append("")
        
        # 分组分析
        if 'grouped_analysis' in results:
            grouped = results['grouped_analysis']
            report_lines.append("分组分析:")
            
            for group_type, group_results in grouped.items():
                if isinstance(group_results, dict):
                    report_lines.append(f"  按{group_type}分组:")
                    for group_name, group_metrics in group_results.items():
                        if isinstance(group_metrics, dict):
                            count = group_metrics.get('count', 0)
                            correlation = group_metrics.get('correlation', 0)
                            rmse = group_metrics.get('rmse', 0)
                            report_lines.append(f"    {group_name}: 数量={count}, 相关性={correlation:.4f}, RMSE={rmse:.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def create_field_validator(config: FieldDataComparisonConfig = None) -> FieldDataValidator:
    """
    创建野外数据验证器
    
    Args:
        config: 配置
        
    Returns:
        FieldDataValidator: 验证器实例
    """
    if config is None:
        config = FieldDataComparisonConfig()
    return FieldDataValidator(config)

if __name__ == "__main__":
    # 测试野外数据对比验证
    
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
    
    # 创建野外测量数据
    field_measurements = []
    
    for i in range(50):
        measurement = FieldMeasurement(
            measurement_id=f"field_{i:03d}",
            data_type=FieldDataType.TEMPERATURE,
            method=MeasurementMethod.THERMISTOR,
            x=np.random.uniform(-1000, 1000),
            y=np.random.uniform(-1000, 1000),
            timestamp=datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)),
            value=np.random.normal(273.15, 10),  # 温度数据
            uncertainty=np.random.uniform(0.1, 0.5),
            unit="K",
            quality=np.random.choice(list(DataQuality)),
            confidence=np.random.uniform(0.7, 1.0)
        )
        field_measurements.append(measurement)
    
    # 创建配置
    config = FieldDataComparisonConfig(
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_field_validator(config)
    
    print("=== 野外数据对比验证测试 ===")
    
    # 执行验证
    results = validator.validate(model, field_measurements)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n野外数据对比验证测试完成！")