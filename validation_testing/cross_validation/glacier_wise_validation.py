#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冰川维度验证模块

该模块实现了基于冰川维度的验证策略，用于评估模型在不同冰川上的泛化能力，包括：
- 冰川分组策略（按大小、类型、海拔、地理位置）
- 留一冰川验证
- 冰川间交叉验证
- 冰川特征相似性分析
- 冰川类型泛化评估
- 区域代表性分析

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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import pandas as pd

class GlacierGroupStrategy(Enum):
    """冰川分组策略"""
    BY_SIZE = "by_size"  # 按冰川大小分组
    BY_TYPE = "by_type"  # 按冰川类型分组
    BY_ELEVATION = "by_elevation"  # 按海拔分组
    BY_REGION = "by_region"  # 按地理区域分组
    BY_CLIMATE = "by_climate"  # 按气候条件分组
    BY_SIMILARITY = "by_similarity"  # 按相似性分组
    RANDOM = "random"  # 随机分组
    LEAVE_ONE_OUT = "leave_one_out"  # 留一验证

class GlacierType(Enum):
    """冰川类型"""
    VALLEY_GLACIER = "valley_glacier"  # 山谷冰川
    CIRQUE_GLACIER = "cirque_glacier"  # 冰斗冰川
    HANGING_GLACIER = "hanging_glacier"  # 悬冰川
    PIEDMONT_GLACIER = "piedmont_glacier"  # 山麓冰川
    TIDEWATER_GLACIER = "tidewater_glacier"  # 潮水冰川
    ICE_CAP = "ice_cap"  # 冰帽
    UNKNOWN = "unknown"  # 未知类型

class GlacierValidationMode(Enum):
    """冰川验证模式"""
    LEAVE_ONE_GLACIER_OUT = "leave_one_glacier_out"  # 留一冰川验证
    CROSS_GLACIER = "cross_glacier"  # 冰川间交叉验证
    GROUPED_VALIDATION = "grouped_validation"  # 分组验证
    SIMILARITY_BASED = "similarity_based"  # 基于相似性的验证
    REGIONAL_HOLDOUT = "regional_holdout"  # 区域留出验证

class GlacierMetric(Enum):
    """冰川验证指标"""
    MSE = "mse"  # 均方误差
    MAE = "mae"  # 平均绝对误差
    RMSE = "rmse"  # 均方根误差
    GLACIER_CORRELATION = "glacier_correlation"  # 冰川相关性
    TYPE_CONSISTENCY = "type_consistency"  # 类型一致性
    SIZE_BIAS = "size_bias"  # 大小偏差
    ELEVATION_BIAS = "elevation_bias"  # 海拔偏差
    REGIONAL_BIAS = "regional_bias"  # 区域偏差

@dataclass
class GlacierInfo:
    """冰川信息"""
    glacier_id: str
    name: str = ""
    glacier_type: GlacierType = GlacierType.UNKNOWN
    area: float = 0.0  # 面积 (km²)
    length: float = 0.0  # 长度 (km)
    elevation_min: float = 0.0  # 最低海拔 (m)
    elevation_max: float = 0.0  # 最高海拔 (m)
    elevation_mean: float = 0.0  # 平均海拔 (m)
    latitude: float = 0.0  # 纬度
    longitude: float = 0.0  # 经度
    region: str = ""  # 地理区域
    climate_zone: str = ""  # 气候带
    
    def __post_init__(self):
        if self.elevation_mean == 0.0 and self.elevation_min > 0 and self.elevation_max > 0:
            self.elevation_mean = (self.elevation_min + self.elevation_max) / 2

@dataclass
class GlacierValidationConfig:
    """冰川验证配置"""
    # 基本设置
    group_strategy: GlacierGroupStrategy = GlacierGroupStrategy.BY_SIZE
    validation_mode: GlacierValidationMode = GlacierValidationMode.LEAVE_ONE_GLACIER_OUT
    
    # 分组参数
    n_groups: int = 5  # 分组数量
    min_glacier_per_group: int = 2  # 每组最少冰川数
    
    # 大小分组参数
    size_bins: List[float] = None  # 大小分箱边界 (km²)
    
    # 海拔分组参数
    elevation_bins: List[float] = None  # 海拔分箱边界 (m)
    
    # 相似性分组参数
    similarity_threshold: float = 0.7  # 相似性阈值
    similarity_features: List[str] = None  # 相似性特征
    
    # 验证指标
    metrics: List[GlacierMetric] = None
    
    # 交叉验证参数
    cv_folds: int = 5  # 交叉验证折数
    random_seed: int = 42
    
    # 分析参数
    analyze_bias: bool = True  # 是否分析偏差
    analyze_similarity: bool = True  # 是否分析相似性
    analyze_transferability: bool = True  # 是否分析可迁移性
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "glacier_validation_plots"
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                GlacierMetric.MSE,
                GlacierMetric.MAE,
                GlacierMetric.GLACIER_CORRELATION,
                GlacierMetric.TYPE_CONSISTENCY
            ]
        
        if self.size_bins is None:
            self.size_bins = [0, 1, 5, 10, 50, float('inf')]  # km²
        
        if self.elevation_bins is None:
            self.elevation_bins = [0, 3000, 4000, 5000, 6000, float('inf')]  # m
        
        if self.similarity_features is None:
            self.similarity_features = ['area', 'elevation_mean', 'latitude', 'longitude']

class GlacierGrouper:
    """冰川分组器"""
    
    def __init__(self, config: GlacierValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def group_glaciers(self, glacier_info: Dict[str, GlacierInfo], 
                      data: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """分组冰川"""
        if self.config.group_strategy == GlacierGroupStrategy.BY_SIZE:
            return self._group_by_size(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.BY_TYPE:
            return self._group_by_type(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.BY_ELEVATION:
            return self._group_by_elevation(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.BY_REGION:
            return self._group_by_region(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.BY_SIMILARITY:
            return self._group_by_similarity(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.RANDOM:
            return self._group_randomly(glacier_info)
        elif self.config.group_strategy == GlacierGroupStrategy.LEAVE_ONE_OUT:
            return self._leave_one_out_groups(glacier_info)
        else:
            raise ValueError(f"不支持的分组策略: {self.config.group_strategy}")
    
    def _group_by_size(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """按大小分组"""
        groups = {}
        
        for glacier_id, info in glacier_info.items():
            # 确定大小分组
            for i, threshold in enumerate(self.config.size_bins[1:]):
                if info.area <= threshold:
                    group_name = f"size_group_{i}"
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].append(glacier_id)
                    break
        
        self.logger.info(f"按大小分组完成: {len(groups)} 个组")
        return groups
    
    def _group_by_type(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """按类型分组"""
        groups = {}
        
        for glacier_id, info in glacier_info.items():
            group_name = f"type_{info.glacier_type.value}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(glacier_id)
        
        self.logger.info(f"按类型分组完成: {len(groups)} 个组")
        return groups
    
    def _group_by_elevation(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """按海拔分组"""
        groups = {}
        
        for glacier_id, info in glacier_info.items():
            # 确定海拔分组
            for i, threshold in enumerate(self.config.elevation_bins[1:]):
                if info.elevation_mean <= threshold:
                    group_name = f"elevation_group_{i}"
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].append(glacier_id)
                    break
        
        self.logger.info(f"按海拔分组完成: {len(groups)} 个组")
        return groups
    
    def _group_by_region(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """按地理区域分组"""
        groups = {}
        
        for glacier_id, info in glacier_info.items():
            group_name = f"region_{info.region}" if info.region else "region_unknown"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(glacier_id)
        
        self.logger.info(f"按区域分组完成: {len(groups)} 个组")
        return groups
    
    def _group_by_similarity(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """按相似性分组"""
        # 提取特征矩阵
        glacier_ids = list(glacier_info.keys())
        features = []
        
        for glacier_id in glacier_ids:
            info = glacier_info[glacier_id]
            feature_vector = []
            
            for feature_name in self.config.similarity_features:
                if hasattr(info, feature_name):
                    value = getattr(info, feature_name)
                    feature_vector.append(float(value) if value is not None else 0.0)
                else:
                    feature_vector.append(0.0)
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # 标准化特征
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # 聚类
        kmeans = KMeans(n_clusters=self.config.n_groups, random_state=self.config.random_seed)
        cluster_labels = kmeans.fit_predict(features)
        
        # 构建分组
        groups = {}
        for i, glacier_id in enumerate(glacier_ids):
            group_name = f"similarity_group_{cluster_labels[i]}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(glacier_id)
        
        self.logger.info(f"按相似性分组完成: {len(groups)} 个组")
        return groups
    
    def _group_randomly(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """随机分组"""
        glacier_ids = list(glacier_info.keys())
        np.random.seed(self.config.random_seed)
        np.random.shuffle(glacier_ids)
        
        groups = {}
        group_size = len(glacier_ids) // self.config.n_groups
        
        for i in range(self.config.n_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < self.config.n_groups - 1 else len(glacier_ids)
            
            group_name = f"random_group_{i}"
            groups[group_name] = glacier_ids[start_idx:end_idx]
        
        self.logger.info(f"随机分组完成: {len(groups)} 个组")
        return groups
    
    def _leave_one_out_groups(self, glacier_info: Dict[str, GlacierInfo]) -> Dict[str, List[str]]:
        """留一验证分组"""
        groups = {}
        glacier_ids = list(glacier_info.keys())
        
        for i, glacier_id in enumerate(glacier_ids):
            group_name = f"fold_{i}"
            groups[group_name] = [glacier_id]
        
        self.logger.info(f"留一验证分组完成: {len(groups)} 个组")
        return groups

class GlacierValidator:
    """冰川验证器"""
    
    def __init__(self, config: GlacierValidationConfig):
        self.config = config
        self.grouper = GlacierGrouper(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, model: nn.Module, data: Dict[str, torch.Tensor], 
                glacier_info: Dict[str, GlacierInfo],
                glacier_id_column: str = 'glacier_id') -> Dict[str, Any]:
        """执行冰川验证"""
        results = {
            'strategy': self.config.group_strategy.value,
            'mode': self.config.validation_mode.value,
            'glacier_groups': {},
            'validation_results': {},
            'glacier_metrics': {},
            'bias_analysis': {},
            'similarity_analysis': {},
            'transferability_analysis': {}
        }
        
        try:
            # 分组冰川
            glacier_groups = self.grouper.group_glaciers(glacier_info, data)
            results['glacier_groups'] = glacier_groups
            
            # 执行验证
            if self.config.validation_mode == GlacierValidationMode.LEAVE_ONE_GLACIER_OUT:
                results['validation_results'] = self._leave_one_glacier_out_validation(
                    model, data, glacier_info, glacier_id_column
                )
            elif self.config.validation_mode == GlacierValidationMode.CROSS_GLACIER:
                results['validation_results'] = self._cross_glacier_validation(
                    model, data, glacier_groups, glacier_id_column
                )
            elif self.config.validation_mode == GlacierValidationMode.GROUPED_VALIDATION:
                results['validation_results'] = self._grouped_validation(
                    model, data, glacier_groups, glacier_id_column
                )
            
            # 计算冰川指标
            results['glacier_metrics'] = self._compute_glacier_metrics(
                results['validation_results'], glacier_info
            )
            
            # 偏差分析
            if self.config.analyze_bias:
                results['bias_analysis'] = self._analyze_bias(
                    results['validation_results'], glacier_info
                )
            
            # 相似性分析
            if self.config.analyze_similarity:
                results['similarity_analysis'] = self._analyze_similarity(
                    results['validation_results'], glacier_info
                )
            
            # 可迁移性分析
            if self.config.analyze_transferability:
                results['transferability_analysis'] = self._analyze_transferability(
                    results['validation_results'], glacier_info
                )
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, glacier_info)
        
        except Exception as e:
            self.logger.error(f"冰川验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _leave_one_glacier_out_validation(self, model: nn.Module, 
                                         data: Dict[str, torch.Tensor],
                                         glacier_info: Dict[str, GlacierInfo],
                                         glacier_id_column: str) -> Dict[str, Any]:
        """留一冰川验证"""
        results = {
            'glacier_results': {},
            'overall_metrics': {},
            'predictions': {},
            'targets': {}
        }
        
        glacier_ids = list(glacier_info.keys())
        
        for test_glacier_id in glacier_ids:
            try:
                # 分割数据
                test_mask = data[glacier_id_column] == test_glacier_id
                train_mask = ~test_mask
                
                # 提取测试数据
                test_data = {key: tensor[test_mask] for key, tensor in data.items()}
                
                # 预测
                predictions = self._predict(model, test_data)
                targets = self._extract_targets(test_data)
                
                # 计算指标
                metrics = self._compute_basic_metrics(predictions, targets)
                
                results['glacier_results'][test_glacier_id] = metrics
                results['predictions'][test_glacier_id] = predictions
                results['targets'][test_glacier_id] = targets
                
                self.logger.info(f"冰川 {test_glacier_id} 验证完成")
            
            except Exception as e:
                self.logger.warning(f"冰川 {test_glacier_id} 验证失败: {e}")
        
        # 计算总体指标
        results['overall_metrics'] = self._compute_overall_metrics(
            results['glacier_results']
        )
        
        return results
    
    def _cross_glacier_validation(self, model: nn.Module, 
                                 data: Dict[str, torch.Tensor],
                                 glacier_groups: Dict[str, List[str]],
                                 glacier_id_column: str) -> Dict[str, Any]:
        """冰川间交叉验证"""
        results = {
            'fold_results': {},
            'overall_metrics': {},
            'group_metrics': {}
        }
        
        group_names = list(glacier_groups.keys())
        
        for i, test_group in enumerate(group_names):
            try:
                test_glacier_ids = glacier_groups[test_group]
                
                # 创建测试掩码
                test_mask = torch.zeros(len(data[glacier_id_column]), dtype=torch.bool)
                for glacier_id in test_glacier_ids:
                    test_mask |= (data[glacier_id_column] == glacier_id)
                
                # 分割数据
                test_data = {key: tensor[test_mask] for key, tensor in data.items()}
                
                # 预测
                predictions = self._predict(model, test_data)
                targets = self._extract_targets(test_data)
                
                # 计算指标
                metrics = self._compute_basic_metrics(predictions, targets)
                
                results['fold_results'][f'fold_{i}'] = {
                    'test_group': test_group,
                    'test_glaciers': test_glacier_ids,
                    'metrics': metrics,
                    'predictions': predictions,
                    'targets': targets
                }
                
                self.logger.info(f"折 {i} (组 {test_group}) 验证完成")
            
            except Exception as e:
                self.logger.warning(f"折 {i} 验证失败: {e}")
        
        # 计算总体指标
        fold_metrics = [fold['metrics'] for fold in results['fold_results'].values()]
        results['overall_metrics'] = self._compute_overall_metrics_from_list(fold_metrics)
        
        return results
    
    def _grouped_validation(self, model: nn.Module, 
                           data: Dict[str, torch.Tensor],
                           glacier_groups: Dict[str, List[str]],
                           glacier_id_column: str) -> Dict[str, Any]:
        """分组验证"""
        results = {
            'group_results': {},
            'overall_metrics': {}
        }
        
        for group_name, glacier_ids in glacier_groups.items():
            try:
                # 创建组掩码
                group_mask = torch.zeros(len(data[glacier_id_column]), dtype=torch.bool)
                for glacier_id in glacier_ids:
                    group_mask |= (data[glacier_id_column] == glacier_id)
                
                # 提取组数据
                group_data = {key: tensor[group_mask] for key, tensor in data.items()}
                
                # 预测
                predictions = self._predict(model, group_data)
                targets = self._extract_targets(group_data)
                
                # 计算指标
                metrics = self._compute_basic_metrics(predictions, targets)
                
                results['group_results'][group_name] = {
                    'glaciers': glacier_ids,
                    'metrics': metrics,
                    'predictions': predictions,
                    'targets': targets
                }
                
                self.logger.info(f"组 {group_name} 验证完成")
            
            except Exception as e:
                self.logger.warning(f"组 {group_name} 验证失败: {e}")
        
        # 计算总体指标
        group_metrics = [group['metrics'] for group in results['group_results'].values()]
        results['overall_metrics'] = self._compute_overall_metrics_from_list(group_metrics)
        
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
                if key not in ['x', 'y', 't', 'glacier_id']:
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
    
    def _compute_overall_metrics(self, glacier_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算总体指标"""
        if not glacier_results:
            return {}
        
        metrics_list = list(glacier_results.values())
        return self._compute_overall_metrics_from_list(metrics_list)
    
    def _compute_overall_metrics_from_list(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """从指标列表计算总体指标"""
        if not metrics_list:
            return {}
        
        overall_metrics = {}
        
        # 计算平均值和标准差
        for metric_name in metrics_list[0].keys():
            values = [metrics[metric_name] for metrics in metrics_list if metric_name in metrics]
            if values:
                overall_metrics[f'{metric_name}_mean'] = np.mean(values)
                overall_metrics[f'{metric_name}_std'] = np.std(values)
                overall_metrics[f'{metric_name}_min'] = np.min(values)
                overall_metrics[f'{metric_name}_max'] = np.max(values)
        
        return overall_metrics
    
    def _compute_glacier_metrics(self, validation_results: Dict[str, Any], 
                                glacier_info: Dict[str, GlacierInfo]) -> Dict[str, Any]:
        """计算冰川特定指标"""
        glacier_metrics = {
            'type_performance': {},
            'size_performance': {},
            'elevation_performance': {},
            'regional_performance': {}
        }
        
        try:
            # 按类型分析性能
            if 'glacier_results' in validation_results:
                glacier_results = validation_results['glacier_results']
                
                # 按类型分组
                type_metrics = {}
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        glacier_type = glacier_info[glacier_id].glacier_type.value
                        if glacier_type not in type_metrics:
                            type_metrics[glacier_type] = []
                        type_metrics[glacier_type].append(metrics)
                
                # 计算类型平均性能
                for glacier_type, metrics_list in type_metrics.items():
                    glacier_metrics['type_performance'][glacier_type] = \
                        self._compute_overall_metrics_from_list(metrics_list)
                
                # 按大小分析性能
                size_metrics = {'small': [], 'medium': [], 'large': []}
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        area = glacier_info[glacier_id].area
                        if area < 1:
                            size_metrics['small'].append(metrics)
                        elif area < 10:
                            size_metrics['medium'].append(metrics)
                        else:
                            size_metrics['large'].append(metrics)
                
                for size_category, metrics_list in size_metrics.items():
                    if metrics_list:
                        glacier_metrics['size_performance'][size_category] = \
                            self._compute_overall_metrics_from_list(metrics_list)
        
        except Exception as e:
            self.logger.warning(f"计算冰川指标时出错: {e}")
        
        return glacier_metrics
    
    def _analyze_bias(self, validation_results: Dict[str, Any], 
                     glacier_info: Dict[str, GlacierInfo]) -> Dict[str, Any]:
        """分析偏差"""
        bias_analysis = {
            'size_bias': {},
            'elevation_bias': {},
            'type_bias': {},
            'regional_bias': {}
        }
        
        try:
            if 'glacier_results' in validation_results:
                glacier_results = validation_results['glacier_results']
                
                # 大小偏差分析
                areas = []
                mse_values = []
                
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        areas.append(glacier_info[glacier_id].area)
                        mse_values.append(metrics.get('mse', 0))
                
                if len(areas) > 1:
                    correlation = np.corrcoef(areas, mse_values)[0, 1]
                    bias_analysis['size_bias'] = {
                        'correlation_with_mse': float(correlation) if not np.isnan(correlation) else 0.0,
                        'bias_direction': 'larger_glaciers_worse' if correlation > 0 else 'smaller_glaciers_worse'
                    }
                
                # 海拔偏差分析
                elevations = []
                mse_values = []
                
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        elevations.append(glacier_info[glacier_id].elevation_mean)
                        mse_values.append(metrics.get('mse', 0))
                
                if len(elevations) > 1:
                    correlation = np.corrcoef(elevations, mse_values)[0, 1]
                    bias_analysis['elevation_bias'] = {
                        'correlation_with_mse': float(correlation) if not np.isnan(correlation) else 0.0,
                        'bias_direction': 'higher_elevation_worse' if correlation > 0 else 'lower_elevation_worse'
                    }
        
        except Exception as e:
            self.logger.warning(f"偏差分析失败: {e}")
        
        return bias_analysis
    
    def _analyze_similarity(self, validation_results: Dict[str, Any], 
                           glacier_info: Dict[str, GlacierInfo]) -> Dict[str, Any]:
        """分析相似性"""
        similarity_analysis = {
            'feature_similarity': {},
            'performance_similarity': {},
            'transferability_matrix': {}
        }
        
        try:
            # 计算冰川特征相似性
            glacier_ids = list(glacier_info.keys())
            features = []
            
            for glacier_id in glacier_ids:
                info = glacier_info[glacier_id]
                feature_vector = [
                    info.area,
                    info.elevation_mean,
                    info.latitude,
                    info.longitude
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # 标准化特征
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            
            # 计算相似性矩阵
            similarity_matrix = cosine_similarity(features)
            
            similarity_analysis['feature_similarity'] = {
                'mean_similarity': float(np.mean(similarity_matrix)),
                'similarity_range': [float(np.min(similarity_matrix)), float(np.max(similarity_matrix))]
            }
        
        except Exception as e:
            self.logger.warning(f"相似性分析失败: {e}")
        
        return similarity_analysis
    
    def _analyze_transferability(self, validation_results: Dict[str, Any], 
                                glacier_info: Dict[str, GlacierInfo]) -> Dict[str, Any]:
        """分析可迁移性"""
        transferability_analysis = {
            'cross_type_transferability': {},
            'cross_size_transferability': {},
            'cross_region_transferability': {},
            'overall_transferability': {}
        }
        
        try:
            if 'glacier_results' in validation_results:
                glacier_results = validation_results['glacier_results']
                
                # 计算总体可迁移性指标
                mse_values = [metrics.get('mse', float('inf')) for metrics in glacier_results.values()]
                correlation_values = [metrics.get('correlation', 0) for metrics in glacier_results.values()]
                
                transferability_analysis['overall_transferability'] = {
                    'mean_mse': float(np.mean(mse_values)),
                    'std_mse': float(np.std(mse_values)),
                    'mean_correlation': float(np.mean(correlation_values)),
                    'transferability_score': float(np.mean(correlation_values)) - float(np.std(mse_values)) / 10
                }
        
        except Exception as e:
            self.logger.warning(f"可迁移性分析失败: {e}")
        
        return transferability_analysis
    
    def _plot_results(self, results: Dict[str, Any], glacier_info: Dict[str, GlacierInfo]):
        """绘制结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制冰川性能分布
            if 'validation_results' in results and 'glacier_results' in results['validation_results']:
                glacier_results = results['validation_results']['glacier_results']
                
                # MSE分布
                mse_values = [metrics.get('mse', 0) for metrics in glacier_results.values()]
                axes[0, 0].hist(mse_values, bins=20, alpha=0.7)
                axes[0, 0].set_xlabel('MSE')
                axes[0, 0].set_ylabel('冰川数量')
                axes[0, 0].set_title('MSE分布')
                
                # 相关性分布
                corr_values = [metrics.get('correlation', 0) for metrics in glacier_results.values()]
                axes[0, 1].hist(corr_values, bins=20, alpha=0.7)
                axes[0, 1].set_xlabel('相关性')
                axes[0, 1].set_ylabel('冰川数量')
                axes[0, 1].set_title('相关性分布')
                
                # 大小vs性能
                areas = []
                mse_vals = []
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        areas.append(glacier_info[glacier_id].area)
                        mse_vals.append(metrics.get('mse', 0))
                
                if areas and mse_vals:
                    axes[1, 0].scatter(areas, mse_vals, alpha=0.6)
                    axes[1, 0].set_xlabel('冰川面积 (km²)')
                    axes[1, 0].set_ylabel('MSE')
                    axes[1, 0].set_title('冰川大小 vs 性能')
                    axes[1, 0].set_xscale('log')
                
                # 海拔vs性能
                elevations = []
                mse_vals = []
                for glacier_id, metrics in glacier_results.items():
                    if glacier_id in glacier_info:
                        elevations.append(glacier_info[glacier_id].elevation_mean)
                        mse_vals.append(metrics.get('mse', 0))
                
                if elevations and mse_vals:
                    axes[1, 1].scatter(elevations, mse_vals, alpha=0.6)
                    axes[1, 1].set_xlabel('平均海拔 (m)')
                    axes[1, 1].set_ylabel('MSE')
                    axes[1, 1].set_title('海拔 vs 性能')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/glacier_validation_results.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "冰川维度验证报告",
            "=" * 60,
            f"验证策略: {results.get('strategy', 'Unknown')}",
            f"验证模式: {results.get('mode', 'Unknown')}",
            ""
        ]
        
        # 分组信息
        if 'glacier_groups' in results:
            groups = results['glacier_groups']
            report_lines.append(f"冰川分组: {len(groups)} 个组")
            for group_name, glacier_ids in groups.items():
                report_lines.append(f"  {group_name}: {len(glacier_ids)} 个冰川")
            report_lines.append("")
        
        # 验证结果
        if 'validation_results' in results:
            vr = results['validation_results']
            
            if 'overall_metrics' in vr:
                om = vr['overall_metrics']
                report_lines.append("总体性能:")
                for metric, value in om.items():
                    report_lines.append(f"  {metric}: {value:.6f}")
                report_lines.append("")
        
        # 冰川指标
        if 'glacier_metrics' in results:
            gm = results['glacier_metrics']
            report_lines.append("冰川特定指标:")
            
            if 'type_performance' in gm:
                report_lines.append("  按类型性能:")
                for glacier_type, metrics in gm['type_performance'].items():
                    report_lines.append(f"    {glacier_type}:")
                    for metric, value in metrics.items():
                        if 'mean' in metric:
                            report_lines.append(f"      {metric}: {value:.6f}")
            
            if 'size_performance' in gm:
                report_lines.append("  按大小性能:")
                for size_category, metrics in gm['size_performance'].items():
                    report_lines.append(f"    {size_category}:")
                    for metric, value in metrics.items():
                        if 'mean' in metric:
                            report_lines.append(f"      {metric}: {value:.6f}")
            
            report_lines.append("")
        
        # 偏差分析
        if 'bias_analysis' in results:
            ba = results['bias_analysis']
            report_lines.append("偏差分析:")
            
            if 'size_bias' in ba:
                sb = ba['size_bias']
                report_lines.extend([
                    f"  大小偏差相关性: {sb.get('correlation_with_mse', 0):.6f}",
                    f"  偏差方向: {sb.get('bias_direction', 'Unknown')}"
                ])
            
            if 'elevation_bias' in ba:
                eb = ba['elevation_bias']
                report_lines.extend([
                    f"  海拔偏差相关性: {eb.get('correlation_with_mse', 0):.6f}",
                    f"  偏差方向: {eb.get('bias_direction', 'Unknown')}"
                ])
            
            report_lines.append("")
        
        # 可迁移性分析
        if 'transferability_analysis' in results:
            ta = results['transferability_analysis']
            if 'overall_transferability' in ta:
                ot = ta['overall_transferability']
                report_lines.append("可迁移性分析:")
                report_lines.extend([
                    f"  平均MSE: {ot.get('mean_mse', 0):.6f}",
                    f"  MSE标准差: {ot.get('std_mse', 0):.6f}",
                    f"  平均相关性: {ot.get('mean_correlation', 0):.6f}",
                    f"  可迁移性得分: {ot.get('transferability_score', 0):.6f}"
                ])
                report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def create_glacier_validator(config: GlacierValidationConfig = None) -> GlacierValidator:
    """
    创建冰川验证器
    
    Args:
        config: 配置
        
    Returns:
        GlacierValidator: 验证器实例
    """
    if config is None:
        config = GlacierValidationConfig()
    return GlacierValidator(config)

if __name__ == "__main__":
    # 测试冰川维度验证
    
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
    n_glaciers = 10
    
    # 创建冰川信息
    glacier_info = {}
    for i in range(n_glaciers):
        glacier_info[f"glacier_{i}"] = GlacierInfo(
            glacier_id=f"glacier_{i}",
            name=f"测试冰川{i}",
            glacier_type=GlacierType.VALLEY_GLACIER if i % 2 == 0 else GlacierType.CIRQUE_GLACIER,
            area=np.random.uniform(0.5, 50.0),  # km²
            elevation_mean=np.random.uniform(3000, 6000),  # m
            latitude=np.random.uniform(28, 35),  # 度
            longitude=np.random.uniform(80, 100),  # 度
            region=f"region_{i // 3}"
        )
    
    # 生成数据
    glacier_ids = np.random.choice(list(glacier_info.keys()), n_samples)
    data = {
        'x': torch.randn(n_samples) * 1000,  # X坐标 (m)
        'y': torch.randn(n_samples) * 1000,  # Y坐标 (m)
        't': torch.randn(n_samples),  # 时间
        'glacier_id': glacier_ids,  # 冰川ID
        'temperature': torch.randn(n_samples) + 273.15  # 温度数据
    }
    
    # 创建配置
    config = GlacierValidationConfig(
        group_strategy=GlacierGroupStrategy.BY_SIZE,
        validation_mode=GlacierValidationMode.LEAVE_ONE_GLACIER_OUT,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_glacier_validator(config)
    
    print("=== 冰川维度验证测试 ===")
    
    # 执行验证
    results = validator.validate(model, data, glacier_info)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n冰川维度验证测试完成！")