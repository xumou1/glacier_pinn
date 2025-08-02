#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多源一致性验证模块

该模块实现了多源数据一致性验证策略，用于评估模型在不同数据源上的一致性表现，包括：
- 数据源分组验证
- 跨源一致性检验
- 源间偏差分析
- 数据质量影响评估
- 融合策略验证
- 不确定性传播分析

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
from scipy.spatial.distance import wasserstein_distance
import pandas as pd

class DataSource(Enum):
    """数据源类型"""
    SATELLITE = "satellite"  # 卫星数据
    FIELD_MEASUREMENT = "field_measurement"  # 野外测量
    REANALYSIS = "reanalysis"  # 再分析数据
    MODEL_OUTPUT = "model_output"  # 模型输出
    REMOTE_SENSING = "remote_sensing"  # 遥感数据
    GRACE = "grace"  # GRACE重力数据
    ICESAT = "icesat"  # ICESat激光测高
    WEATHER_STATION = "weather_station"  # 气象站数据
    UNKNOWN = "unknown"  # 未知来源

class ConsistencyMetric(Enum):
    """一致性指标"""
    CORRELATION = "correlation"  # 相关性
    BIAS = "bias"  # 偏差
    RMSE = "rmse"  # 均方根误差
    MAE = "mae"  # 平均绝对误差
    WASSERSTEIN_DISTANCE = "wasserstein_distance"  # Wasserstein距离
    KL_DIVERGENCE = "kl_divergence"  # KL散度
    CONCORDANCE = "concordance"  # 一致性相关系数
    AGREEMENT = "agreement"  # 一致性指数

class ValidationStrategy(Enum):
    """验证策略"""
    LEAVE_ONE_SOURCE_OUT = "leave_one_source_out"  # 留一源验证
    CROSS_SOURCE = "cross_source"  # 跨源验证
    PAIRWISE_COMPARISON = "pairwise_comparison"  # 成对比较
    ENSEMBLE_VALIDATION = "ensemble_validation"  # 集成验证
    HIERARCHICAL_VALIDATION = "hierarchical_validation"  # 分层验证

class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"  # 高质量
    MEDIUM = "medium"  # 中等质量
    LOW = "low"  # 低质量
    UNKNOWN = "unknown"  # 未知质量

@dataclass
class SourceInfo:
    """数据源信息"""
    source_id: str
    source_type: DataSource
    quality: DataQuality = DataQuality.UNKNOWN
    uncertainty: float = 0.0  # 不确定性
    resolution_spatial: float = 0.0  # 空间分辨率 (m)
    resolution_temporal: float = 0.0  # 时间分辨率 (days)
    coverage_start: Optional[str] = None  # 覆盖开始时间
    coverage_end: Optional[str] = None  # 覆盖结束时间
    reliability_score: float = 1.0  # 可靠性得分 (0-1)
    bias_correction: bool = False  # 是否进行偏差校正
    
@dataclass
class MultisourceConsistencyConfig:
    """多源一致性验证配置"""
    # 基本设置
    validation_strategy: ValidationStrategy = ValidationStrategy.LEAVE_ONE_SOURCE_OUT
    consistency_metrics: List[ConsistencyMetric] = None
    
    # 数据源设置
    primary_source: Optional[str] = None  # 主要数据源
    reference_sources: List[str] = None  # 参考数据源
    
    # 一致性阈值
    correlation_threshold: float = 0.7  # 相关性阈值
    bias_threshold: float = 0.1  # 偏差阈值
    rmse_threshold: float = 1.0  # RMSE阈值
    
    # 质量权重
    quality_weights: Dict[DataQuality, float] = None
    uncertainty_weights: bool = True  # 是否使用不确定性权重
    
    # 融合策略
    fusion_method: str = "weighted_average"  # 融合方法
    outlier_detection: bool = True  # 是否检测异常值
    outlier_threshold: float = 3.0  # 异常值阈值（标准差倍数）
    
    # 分析设置
    analyze_temporal_consistency: bool = True  # 时间一致性分析
    analyze_spatial_consistency: bool = True  # 空间一致性分析
    analyze_uncertainty_propagation: bool = True  # 不确定性传播分析
    
    # 统计测试
    significance_level: float = 0.05  # 显著性水平
    bootstrap_samples: int = 1000  # 自举样本数
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "multisource_validation_plots"
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.consistency_metrics is None:
            self.consistency_metrics = [
                ConsistencyMetric.CORRELATION,
                ConsistencyMetric.BIAS,
                ConsistencyMetric.RMSE,
                ConsistencyMetric.WASSERSTEIN_DISTANCE
            ]
        
        if self.quality_weights is None:
            self.quality_weights = {
                DataQuality.HIGH: 1.0,
                DataQuality.MEDIUM: 0.7,
                DataQuality.LOW: 0.3,
                DataQuality.UNKNOWN: 0.5
            }
        
        if self.reference_sources is None:
            self.reference_sources = []

class ConsistencyAnalyzer:
    """一致性分析器"""
    
    def __init__(self, config: MultisourceConsistencyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_consistency_metrics(self, predictions_1: torch.Tensor, 
                                   predictions_2: torch.Tensor,
                                   metric_types: List[ConsistencyMetric] = None) -> Dict[str, float]:
        """计算一致性指标"""
        if metric_types is None:
            metric_types = self.config.consistency_metrics
        
        metrics = {}
        
        # 转换为numpy数组
        pred1 = predictions_1.detach().numpy().flatten()
        pred2 = predictions_2.detach().numpy().flatten()
        
        # 确保长度一致
        min_len = min(len(pred1), len(pred2))
        pred1 = pred1[:min_len]
        pred2 = pred2[:min_len]
        
        try:
            for metric_type in metric_types:
                if metric_type == ConsistencyMetric.CORRELATION:
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                
                elif metric_type == ConsistencyMetric.BIAS:
                    bias = np.mean(pred1 - pred2)
                    metrics['bias'] = float(bias)
                
                elif metric_type == ConsistencyMetric.RMSE:
                    rmse = np.sqrt(mean_squared_error(pred1, pred2))
                    metrics['rmse'] = float(rmse)
                
                elif metric_type == ConsistencyMetric.MAE:
                    mae = mean_absolute_error(pred1, pred2)
                    metrics['mae'] = float(mae)
                
                elif metric_type == ConsistencyMetric.WASSERSTEIN_DISTANCE:
                    wd = wasserstein_distance(pred1, pred2)
                    metrics['wasserstein_distance'] = float(wd)
                
                elif metric_type == ConsistencyMetric.CONCORDANCE:
                    # 计算一致性相关系数
                    mean1, mean2 = np.mean(pred1), np.mean(pred2)
                    var1, var2 = np.var(pred1), np.var(pred2)
                    covariance = np.mean((pred1 - mean1) * (pred2 - mean2))
                    concordance = 2 * covariance / (var1 + var2 + (mean1 - mean2)**2)
                    metrics['concordance'] = float(concordance)
                
                elif metric_type == ConsistencyMetric.AGREEMENT:
                    # 计算一致性指数
                    diff = pred1 - pred2
                    mean_diff = np.mean(diff)
                    std_diff = np.std(diff)
                    agreement = 1 - np.mean(np.abs(diff - mean_diff)) / (2 * std_diff + 1e-8)
                    metrics['agreement'] = float(agreement)
        
        except Exception as e:
            self.logger.warning(f"计算一致性指标时出错: {e}")
        
        return metrics
    
    def detect_outliers(self, data: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """检测异常值"""
        if threshold is None:
            threshold = self.config.outlier_threshold
        
        data_np = data.detach().numpy().flatten()
        mean_val = np.mean(data_np)
        std_val = np.std(data_np)
        
        # 使用Z-score检测异常值
        z_scores = np.abs((data_np - mean_val) / (std_val + 1e-8))
        outlier_mask = z_scores > threshold
        
        return torch.from_numpy(~outlier_mask)
    
    def compute_weights(self, source_info: Dict[str, SourceInfo], 
                       uncertainties: Dict[str, float] = None) -> Dict[str, float]:
        """计算数据源权重"""
        weights = {}
        
        for source_id, info in source_info.items():
            # 基于质量的权重
            quality_weight = self.config.quality_weights.get(info.quality, 0.5)
            
            # 基于可靠性的权重
            reliability_weight = info.reliability_score
            
            # 基于不确定性的权重
            uncertainty_weight = 1.0
            if self.config.uncertainty_weights and uncertainties and source_id in uncertainties:
                uncertainty = uncertainties[source_id]
                uncertainty_weight = 1.0 / (1.0 + uncertainty)
            
            # 综合权重
            total_weight = quality_weight * reliability_weight * uncertainty_weight
            weights[source_id] = total_weight
        
        # 归一化权重
        total_sum = sum(weights.values())
        if total_sum > 0:
            weights = {k: v / total_sum for k, v in weights.items()}
        
        return weights

class MultisourceValidator:
    """多源验证器"""
    
    def __init__(self, config: MultisourceConsistencyConfig):
        self.config = config
        self.analyzer = ConsistencyAnalyzer(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, model: nn.Module, 
                data: Dict[str, torch.Tensor],
                source_info: Dict[str, SourceInfo],
                source_column: str = 'source_id') -> Dict[str, Any]:
        """执行多源一致性验证"""
        results = {
            'strategy': self.config.validation_strategy.value,
            'source_info': source_info,
            'validation_results': {},
            'consistency_analysis': {},
            'bias_analysis': {},
            'uncertainty_analysis': {},
            'fusion_analysis': {}
        }
        
        try:
            # 执行验证策略
            if self.config.validation_strategy == ValidationStrategy.LEAVE_ONE_SOURCE_OUT:
                results['validation_results'] = self._leave_one_source_out_validation(
                    model, data, source_info, source_column
                )
            elif self.config.validation_strategy == ValidationStrategy.CROSS_SOURCE:
                results['validation_results'] = self._cross_source_validation(
                    model, data, source_info, source_column
                )
            elif self.config.validation_strategy == ValidationStrategy.PAIRWISE_COMPARISON:
                results['validation_results'] = self._pairwise_comparison_validation(
                    model, data, source_info, source_column
                )
            elif self.config.validation_strategy == ValidationStrategy.ENSEMBLE_VALIDATION:
                results['validation_results'] = self._ensemble_validation(
                    model, data, source_info, source_column
                )
            
            # 一致性分析
            results['consistency_analysis'] = self._analyze_consistency(
                results['validation_results'], source_info
            )
            
            # 偏差分析
            results['bias_analysis'] = self._analyze_bias(
                results['validation_results'], source_info
            )
            
            # 不确定性分析
            if self.config.analyze_uncertainty_propagation:
                results['uncertainty_analysis'] = self._analyze_uncertainty(
                    results['validation_results'], source_info
                )
            
            # 融合分析
            results['fusion_analysis'] = self._analyze_fusion(
                results['validation_results'], source_info
            )
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, source_info)
        
        except Exception as e:
            self.logger.error(f"多源验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _leave_one_source_out_validation(self, model: nn.Module, 
                                        data: Dict[str, torch.Tensor],
                                        source_info: Dict[str, SourceInfo],
                                        source_column: str) -> Dict[str, Any]:
        """留一源验证"""
        results = {
            'source_results': {},
            'cross_source_metrics': {},
            'predictions': {},
            'targets': {}
        }
        
        source_ids = list(source_info.keys())
        
        for test_source_id in source_ids:
            try:
                # 分割数据
                test_mask = data[source_column] == test_source_id
                
                if not test_mask.any():
                    self.logger.warning(f"数据源 {test_source_id} 没有数据")
                    continue
                
                # 提取测试数据
                test_data = {key: tensor[test_mask] for key, tensor in data.items()}
                
                # 预测
                predictions = self._predict(model, test_data)
                targets = self._extract_targets(test_data)
                
                # 计算基本指标
                metrics = self._compute_basic_metrics(predictions, targets)
                
                results['source_results'][test_source_id] = metrics
                results['predictions'][test_source_id] = predictions
                results['targets'][test_source_id] = targets
                
                self.logger.info(f"数据源 {test_source_id} 验证完成")
            
            except Exception as e:
                self.logger.warning(f"数据源 {test_source_id} 验证失败: {e}")
        
        # 计算跨源指标
        results['cross_source_metrics'] = self._compute_cross_source_metrics(
            results['predictions'], results['targets']
        )
        
        return results
    
    def _cross_source_validation(self, model: nn.Module, 
                                data: Dict[str, torch.Tensor],
                                source_info: Dict[str, SourceInfo],
                                source_column: str) -> Dict[str, Any]:
        """跨源验证"""
        results = {
            'source_predictions': {},
            'consistency_matrix': {},
            'cross_validation_metrics': {}
        }
        
        source_ids = list(source_info.keys())
        
        # 为每个数据源生成预测
        for source_id in source_ids:
            try:
                source_mask = data[source_column] == source_id
                
                if not source_mask.any():
                    continue
                
                source_data = {key: tensor[source_mask] for key, tensor in data.items()}
                predictions = self._predict(model, source_data)
                
                results['source_predictions'][source_id] = predictions
            
            except Exception as e:
                self.logger.warning(f"数据源 {source_id} 预测失败: {e}")
        
        # 计算一致性矩阵
        results['consistency_matrix'] = self._compute_consistency_matrix(
            results['source_predictions']
        )
        
        return results
    
    def _pairwise_comparison_validation(self, model: nn.Module, 
                                       data: Dict[str, torch.Tensor],
                                       source_info: Dict[str, SourceInfo],
                                       source_column: str) -> Dict[str, Any]:
        """成对比较验证"""
        results = {
            'pairwise_metrics': {},
            'source_rankings': {},
            'agreement_analysis': {}
        }
        
        source_ids = list(source_info.keys())
        
        # 生成所有数据源的预测
        source_predictions = {}
        source_targets = {}
        
        for source_id in source_ids:
            try:
                source_mask = data[source_column] == source_id
                
                if not source_mask.any():
                    continue
                
                source_data = {key: tensor[source_mask] for key, tensor in data.items()}
                predictions = self._predict(model, source_data)
                targets = self._extract_targets(source_data)
                
                source_predictions[source_id] = predictions
                source_targets[source_id] = targets
            
            except Exception as e:
                self.logger.warning(f"数据源 {source_id} 处理失败: {e}")
        
        # 成对比较
        for i, source_1 in enumerate(source_ids):
            for j, source_2 in enumerate(source_ids[i+1:], i+1):
                if source_1 in source_predictions and source_2 in source_predictions:
                    try:
                        # 计算一致性指标
                        consistency_metrics = self.analyzer.compute_consistency_metrics(
                            source_predictions[source_1], source_predictions[source_2]
                        )
                        
                        pair_key = f"{source_1}_vs_{source_2}"
                        results['pairwise_metrics'][pair_key] = consistency_metrics
                    
                    except Exception as e:
                        self.logger.warning(f"成对比较 {source_1} vs {source_2} 失败: {e}")
        
        return results
    
    def _ensemble_validation(self, model: nn.Module, 
                            data: Dict[str, torch.Tensor],
                            source_info: Dict[str, SourceInfo],
                            source_column: str) -> Dict[str, Any]:
        """集成验证"""
        results = {
            'individual_performance': {},
            'ensemble_performance': {},
            'fusion_weights': {},
            'ensemble_predictions': None
        }
        
        source_ids = list(source_info.keys())
        source_predictions = {}
        source_targets = {}
        
        # 获取各数据源的预测
        for source_id in source_ids:
            try:
                source_mask = data[source_column] == source_id
                
                if not source_mask.any():
                    continue
                
                source_data = {key: tensor[source_mask] for key, tensor in data.items()}
                predictions = self._predict(model, source_data)
                targets = self._extract_targets(source_data)
                
                source_predictions[source_id] = predictions
                source_targets[source_id] = targets
                
                # 计算个体性能
                metrics = self._compute_basic_metrics(predictions, targets)
                results['individual_performance'][source_id] = metrics
            
            except Exception as e:
                self.logger.warning(f"数据源 {source_id} 处理失败: {e}")
        
        # 计算融合权重
        uncertainties = {}
        for source_id in source_predictions.keys():
            if source_id in source_info:
                uncertainties[source_id] = source_info[source_id].uncertainty
        
        fusion_weights = self.analyzer.compute_weights(source_info, uncertainties)
        results['fusion_weights'] = fusion_weights
        
        # 集成预测
        if source_predictions:
            ensemble_pred = self._compute_ensemble_prediction(
                source_predictions, fusion_weights
            )
            results['ensemble_predictions'] = ensemble_pred
            
            # 计算集成性能（使用第一个数据源的目标作为参考）
            if source_targets:
                first_target = list(source_targets.values())[0]
                ensemble_metrics = self._compute_basic_metrics(ensemble_pred, first_target)
                results['ensemble_performance'] = ensemble_metrics
        
        return results
    
    def _predict(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """模型预测"""
        with torch.no_grad():
            model.eval()
            
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
                if key not in ['x', 'y', 't', 'source_id']:
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
    
    def _compute_cross_source_metrics(self, predictions: Dict[str, torch.Tensor], 
                                     targets: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """计算跨源指标"""
        cross_metrics = {
            'source_consistency': {},
            'overall_consistency': {},
            'variance_analysis': {}
        }
        
        try:
            source_ids = list(predictions.keys())
            
            # 计算源间一致性
            consistency_scores = []
            for i, source_1 in enumerate(source_ids):
                for j, source_2 in enumerate(source_ids[i+1:], i+1):
                    if source_1 in predictions and source_2 in predictions:
                        consistency = self.analyzer.compute_consistency_metrics(
                            predictions[source_1], predictions[source_2]
                        )
                        consistency_scores.append(consistency.get('correlation', 0))
                        
                        pair_key = f"{source_1}_vs_{source_2}"
                        cross_metrics['source_consistency'][pair_key] = consistency
            
            # 总体一致性
            if consistency_scores:
                cross_metrics['overall_consistency'] = {
                    'mean_correlation': float(np.mean(consistency_scores)),
                    'std_correlation': float(np.std(consistency_scores)),
                    'min_correlation': float(np.min(consistency_scores)),
                    'max_correlation': float(np.max(consistency_scores))
                }
            
            # 方差分析
            if len(predictions) > 1:
                # 计算预测值的方差
                all_predictions = [pred.detach().numpy().flatten() for pred in predictions.values()]
                min_len = min(len(pred) for pred in all_predictions)
                
                aligned_predictions = np.array([pred[:min_len] for pred in all_predictions])
                
                cross_metrics['variance_analysis'] = {
                    'between_source_variance': float(np.var(np.mean(aligned_predictions, axis=1))),
                    'within_source_variance': float(np.mean([np.var(pred) for pred in aligned_predictions])),
                    'total_variance': float(np.var(aligned_predictions.flatten()))
                }
        
        except Exception as e:
            self.logger.warning(f"计算跨源指标失败: {e}")
        
        return cross_metrics
    
    def _compute_consistency_matrix(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """计算一致性矩阵"""
        source_ids = list(predictions.keys())
        n_sources = len(source_ids)
        
        consistency_matrix = np.zeros((n_sources, n_sources))
        
        for i, source_1 in enumerate(source_ids):
            for j, source_2 in enumerate(source_ids):
                if i == j:
                    consistency_matrix[i, j] = 1.0
                else:
                    consistency = self.analyzer.compute_consistency_metrics(
                        predictions[source_1], predictions[source_2]
                    )
                    consistency_matrix[i, j] = consistency.get('correlation', 0)
        
        return {
            'matrix': consistency_matrix.tolist(),
            'source_ids': source_ids,
            'mean_consistency': float(np.mean(consistency_matrix[np.triu_indices(n_sources, k=1)])),
            'min_consistency': float(np.min(consistency_matrix[np.triu_indices(n_sources, k=1)])),
            'max_consistency': float(np.max(consistency_matrix[np.triu_indices(n_sources, k=1)]))
        }
    
    def _compute_ensemble_prediction(self, predictions: Dict[str, torch.Tensor], 
                                   weights: Dict[str, float]) -> torch.Tensor:
        """计算集成预测"""
        if not predictions:
            raise ValueError("没有预测数据")
        
        # 找到最小长度
        min_len = min(len(pred) for pred in predictions.values())
        
        # 加权平均
        ensemble_pred = torch.zeros(min_len)
        total_weight = 0
        
        for source_id, pred in predictions.items():
            weight = weights.get(source_id, 1.0)
            ensemble_pred += weight * pred[:min_len]
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def _analyze_consistency(self, validation_results: Dict[str, Any], 
                           source_info: Dict[str, SourceInfo]) -> Dict[str, Any]:
        """分析一致性"""
        consistency_analysis = {
            'overall_consistency': {},
            'source_rankings': {},
            'consistency_trends': {}
        }
        
        try:
            # 总体一致性评估
            if 'cross_source_metrics' in validation_results:
                csm = validation_results['cross_source_metrics']
                if 'overall_consistency' in csm:
                    consistency_analysis['overall_consistency'] = csm['overall_consistency']
            
            # 数据源排名
            if 'source_results' in validation_results:
                source_results = validation_results['source_results']
                
                # 按相关性排名
                correlations = {}
                for source_id, metrics in source_results.items():
                    correlations[source_id] = metrics.get('correlation', 0)
                
                sorted_sources = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                consistency_analysis['source_rankings'] = {
                    'by_correlation': sorted_sources,
                    'best_source': sorted_sources[0][0] if sorted_sources else None,
                    'worst_source': sorted_sources[-1][0] if sorted_sources else None
                }
        
        except Exception as e:
            self.logger.warning(f"一致性分析失败: {e}")
        
        return consistency_analysis
    
    def _analyze_bias(self, validation_results: Dict[str, Any], 
                     source_info: Dict[str, SourceInfo]) -> Dict[str, Any]:
        """分析偏差"""
        bias_analysis = {
            'source_bias': {},
            'systematic_bias': {},
            'bias_correction_needed': {}
        }
        
        try:
            if 'pairwise_metrics' in validation_results:
                pairwise_metrics = validation_results['pairwise_metrics']
                
                # 分析成对偏差
                biases = []
                for pair_key, metrics in pairwise_metrics.items():
                    bias = metrics.get('bias', 0)
                    biases.append(abs(bias))
                    bias_analysis['source_bias'][pair_key] = bias
                
                # 系统性偏差
                if biases:
                    bias_analysis['systematic_bias'] = {
                        'mean_absolute_bias': float(np.mean(biases)),
                        'max_bias': float(np.max(biases)),
                        'bias_variability': float(np.std(biases))
                    }
                
                # 偏差校正建议
                high_bias_pairs = []
                for pair_key, bias in bias_analysis['source_bias'].items():
                    if abs(bias) > self.config.bias_threshold:
                        high_bias_pairs.append(pair_key)
                
                bias_analysis['bias_correction_needed'] = {
                    'high_bias_pairs': high_bias_pairs,
                    'correction_recommended': len(high_bias_pairs) > 0
                }
        
        except Exception as e:
            self.logger.warning(f"偏差分析失败: {e}")
        
        return bias_analysis
    
    def _analyze_uncertainty(self, validation_results: Dict[str, Any], 
                           source_info: Dict[str, SourceInfo]) -> Dict[str, Any]:
        """分析不确定性"""
        uncertainty_analysis = {
            'source_uncertainties': {},
            'propagated_uncertainty': {},
            'uncertainty_contributions': {}
        }
        
        try:
            # 提取数据源不确定性
            for source_id, info in source_info.items():
                uncertainty_analysis['source_uncertainties'][source_id] = info.uncertainty
            
            # 计算传播不确定性（简化实现）
            if 'fusion_weights' in validation_results:
                weights = validation_results['fusion_weights']
                uncertainties = uncertainty_analysis['source_uncertainties']
                
                # 加权不确定性传播
                total_uncertainty = 0
                for source_id, weight in weights.items():
                    if source_id in uncertainties:
                        total_uncertainty += (weight * uncertainties[source_id]) ** 2
                
                uncertainty_analysis['propagated_uncertainty'] = {
                    'total_uncertainty': float(np.sqrt(total_uncertainty)),
                    'relative_uncertainty': float(np.sqrt(total_uncertainty) / (np.mean(list(uncertainties.values())) + 1e-8))
                }
        
        except Exception as e:
            self.logger.warning(f"不确定性分析失败: {e}")
        
        return uncertainty_analysis
    
    def _analyze_fusion(self, validation_results: Dict[str, Any], 
                       source_info: Dict[str, SourceInfo]) -> Dict[str, Any]:
        """分析融合策略"""
        fusion_analysis = {
            'fusion_performance': {},
            'optimal_weights': {},
            'fusion_recommendations': {}
        }
        
        try:
            # 融合性能评估
            if 'ensemble_performance' in validation_results and 'individual_performance' in validation_results:
                ensemble_perf = validation_results['ensemble_performance']
                individual_perf = validation_results['individual_performance']
                
                # 比较集成与个体性能
                individual_correlations = [perf.get('correlation', 0) for perf in individual_perf.values()]
                ensemble_correlation = ensemble_perf.get('correlation', 0)
                
                fusion_analysis['fusion_performance'] = {
                    'ensemble_correlation': ensemble_correlation,
                    'best_individual_correlation': float(np.max(individual_correlations)) if individual_correlations else 0,
                    'mean_individual_correlation': float(np.mean(individual_correlations)) if individual_correlations else 0,
                    'fusion_improvement': ensemble_correlation - float(np.max(individual_correlations)) if individual_correlations else 0
                }
            
            # 最优权重
            if 'fusion_weights' in validation_results:
                fusion_analysis['optimal_weights'] = validation_results['fusion_weights']
            
            # 融合建议
            fusion_analysis['fusion_recommendations'] = {
                'use_ensemble': fusion_analysis.get('fusion_performance', {}).get('fusion_improvement', 0) > 0,
                'dominant_source': max(fusion_analysis.get('optimal_weights', {}).items(), 
                                     key=lambda x: x[1])[0] if fusion_analysis.get('optimal_weights') else None
            }
        
        except Exception as e:
            self.logger.warning(f"融合分析失败: {e}")
        
        return fusion_analysis
    
    def _plot_results(self, results: Dict[str, Any], source_info: Dict[str, SourceInfo]):
        """绘制结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 数据源性能比较
            if 'validation_results' in results and 'source_results' in results['validation_results']:
                source_results = results['validation_results']['source_results']
                
                source_names = list(source_results.keys())
                correlations = [metrics.get('correlation', 0) for metrics in source_results.values()]
                rmse_values = [metrics.get('rmse', 0) for metrics in source_results.values()]
                
                # 相关性比较
                axes[0, 0].bar(range(len(source_names)), correlations)
                axes[0, 0].set_xlabel('数据源')
                axes[0, 0].set_ylabel('相关性')
                axes[0, 0].set_title('数据源相关性比较')
                axes[0, 0].set_xticks(range(len(source_names)))
                axes[0, 0].set_xticklabels(source_names, rotation=45)
                
                # RMSE比较
                axes[0, 1].bar(range(len(source_names)), rmse_values)
                axes[0, 1].set_xlabel('数据源')
                axes[0, 1].set_ylabel('RMSE')
                axes[0, 1].set_title('数据源RMSE比较')
                axes[0, 1].set_xticks(range(len(source_names)))
                axes[0, 1].set_xticklabels(source_names, rotation=45)
            
            # 一致性矩阵
            if 'validation_results' in results and 'consistency_matrix' in results['validation_results']:
                cm = results['validation_results']['consistency_matrix']
                matrix = np.array(cm['matrix'])
                source_ids = cm['source_ids']
                
                im = axes[1, 0].imshow(matrix, cmap='viridis', vmin=0, vmax=1)
                axes[1, 0].set_xlabel('数据源')
                axes[1, 0].set_ylabel('数据源')
                axes[1, 0].set_title('数据源一致性矩阵')
                axes[1, 0].set_xticks(range(len(source_ids)))
                axes[1, 0].set_yticks(range(len(source_ids)))
                axes[1, 0].set_xticklabels(source_ids, rotation=45)
                axes[1, 0].set_yticklabels(source_ids)
                plt.colorbar(im, ax=axes[1, 0])
            
            # 融合权重
            if 'fusion_analysis' in results and 'optimal_weights' in results['fusion_analysis']:
                weights = results['fusion_analysis']['optimal_weights']
                
                source_names = list(weights.keys())
                weight_values = list(weights.values())
                
                axes[1, 1].pie(weight_values, labels=source_names, autopct='%1.1f%%')
                axes[1, 1].set_title('融合权重分布')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/multisource_validation_results.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "多源一致性验证报告",
            "=" * 60,
            f"验证策略: {results.get('strategy', 'Unknown')}",
            ""
        ]
        
        # 数据源信息
        if 'source_info' in results:
            source_info = results['source_info']
            report_lines.append(f"数据源数量: {len(source_info)}")
            for source_id, info in source_info.items():
                report_lines.append(f"  {source_id}: {info.source_type.value} (质量: {info.quality.value})")
            report_lines.append("")
        
        # 验证结果
        if 'validation_results' in results:
            vr = results['validation_results']
            
            if 'source_results' in vr:
                report_lines.append("数据源性能:")
                for source_id, metrics in vr['source_results'].items():
                    report_lines.extend([
                        f"  {source_id}:",
                        f"    相关性: {metrics.get('correlation', 0):.6f}",
                        f"    RMSE: {metrics.get('rmse', 0):.6f}",
                        f"    MAE: {metrics.get('mae', 0):.6f}"
                    ])
                report_lines.append("")
        
        # 一致性分析
        if 'consistency_analysis' in results:
            ca = results['consistency_analysis']
            report_lines.append("一致性分析:")
            
            if 'overall_consistency' in ca:
                oc = ca['overall_consistency']
                report_lines.extend([
                    f"  平均一致性: {oc.get('mean_correlation', 0):.6f}",
                    f"  一致性范围: [{oc.get('min_correlation', 0):.6f}, {oc.get('max_correlation', 0):.6f}]",
                    f"  一致性标准差: {oc.get('std_correlation', 0):.6f}"
                ])
            
            if 'source_rankings' in ca:
                sr = ca['source_rankings']
                if 'best_source' in sr:
                    report_lines.append(f"  最佳数据源: {sr['best_source']}")
                if 'worst_source' in sr:
                    report_lines.append(f"  最差数据源: {sr['worst_source']}")
            
            report_lines.append("")
        
        # 偏差分析
        if 'bias_analysis' in results:
            ba = results['bias_analysis']
            report_lines.append("偏差分析:")
            
            if 'systematic_bias' in ba:
                sb = ba['systematic_bias']
                report_lines.extend([
                    f"  平均绝对偏差: {sb.get('mean_absolute_bias', 0):.6f}",
                    f"  最大偏差: {sb.get('max_bias', 0):.6f}",
                    f"  偏差变异性: {sb.get('bias_variability', 0):.6f}"
                ])
            
            if 'bias_correction_needed' in ba:
                bcn = ba['bias_correction_needed']
                report_lines.append(f"  需要偏差校正: {'是' if bcn.get('correction_recommended', False) else '否'}")
            
            report_lines.append("")
        
        # 融合分析
        if 'fusion_analysis' in results:
            fa = results['fusion_analysis']
            report_lines.append("融合分析:")
            
            if 'fusion_performance' in fa:
                fp = fa['fusion_performance']
                report_lines.extend([
                    f"  集成相关性: {fp.get('ensemble_correlation', 0):.6f}",
                    f"  最佳个体相关性: {fp.get('best_individual_correlation', 0):.6f}",
                    f"  融合改进: {fp.get('fusion_improvement', 0):.6f}"
                ])
            
            if 'fusion_recommendations' in fa:
                fr = fa['fusion_recommendations']
                report_lines.append(f"  推荐使用集成: {'是' if fr.get('use_ensemble', False) else '否'}")
                if 'dominant_source' in fr:
                    report_lines.append(f"  主导数据源: {fr['dominant_source']}")
            
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def create_multisource_validator(config: MultisourceConsistencyConfig = None) -> MultisourceValidator:
    """
    创建多源验证器
    
    Args:
        config: 配置
        
    Returns:
        MultisourceValidator: 验证器实例
    """
    if config is None:
        config = MultisourceConsistencyConfig()
    return MultisourceValidator(config)

if __name__ == "__main__":
    # 测试多源一致性验证
    
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
    
    # 创建数据源信息
    source_info = {
        'satellite': SourceInfo(
            source_id='satellite',
            source_type=DataSource.SATELLITE,
            quality=DataQuality.HIGH,
            uncertainty=0.1,
            resolution_spatial=30.0,
            reliability_score=0.9
        ),
        'field': SourceInfo(
            source_id='field',
            source_type=DataSource.FIELD_MEASUREMENT,
            quality=DataQuality.HIGH,
            uncertainty=0.05,
            resolution_spatial=1.0,
            reliability_score=0.95
        ),
        'reanalysis': SourceInfo(
            source_id='reanalysis',
            source_type=DataSource.REANALYSIS,
            quality=DataQuality.MEDIUM,
            uncertainty=0.2,
            resolution_spatial=1000.0,
            reliability_score=0.7
        )
    }
    
    # 生成测试数据
    n_samples = 1000
    source_ids = np.random.choice(list(source_info.keys()), n_samples)
    
    data = {
        'x': torch.randn(n_samples) * 1000,  # X坐标 (m)
        'y': torch.randn(n_samples) * 1000,  # Y坐标 (m)
        't': torch.randn(n_samples),  # 时间
        'source_id': source_ids,  # 数据源ID
        'temperature': torch.randn(n_samples) + 273.15  # 温度数据
    }
    
    # 创建配置
    config = MultisourceConsistencyConfig(
        validation_strategy=ValidationStrategy.LEAVE_ONE_SOURCE_OUT,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_multisource_validator(config)
    
    print("=== 多源一致性验证测试 ===")
    
    # 执行验证
    results = validator.validate(model, data, source_info)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n多源一致性验证测试完成！")