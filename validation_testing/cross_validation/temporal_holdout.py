#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间维度留出验证模块

该模块实现了基于时间维度的留出验证策略，用于评估模型的时间泛化能力，包括：
- 时间序列分割策略
- 前向验证（Forward Validation）
- 滑动窗口验证
- 时间间隔验证
- 季节性验证
- 长期趋势验证

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
from datetime import datetime, timedelta
import pandas as pd

class TemporalSplitStrategy(Enum):
    """时间分割策略"""
    FORWARD_CHAINING = "forward_chaining"  # 前向链式验证
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    BLOCKED_TIME_SERIES = "blocked_time_series"  # 时间块分割
    SEASONAL_SPLIT = "seasonal_split"  # 季节性分割
    YEAR_WISE_SPLIT = "year_wise_split"  # 年度分割
    CUSTOM_INTERVALS = "custom_intervals"  # 自定义间隔

class ValidationMetric(Enum):
    """验证指标"""
    MSE = "mse"  # 均方误差
    MAE = "mae"  # 平均绝对误差
    RMSE = "rmse"  # 均方根误差
    MAPE = "mape"  # 平均绝对百分比误差
    R2 = "r2"  # 决定系数
    CORRELATION = "correlation"  # 相关系数
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # 时间一致性

class TemporalValidationMode(Enum):
    """时间验证模式"""
    SINGLE_STEP = "single_step"  # 单步预测
    MULTI_STEP = "multi_step"  # 多步预测
    RECURSIVE = "recursive"  # 递归预测
    DIRECT = "direct"  # 直接预测
    ENSEMBLE = "ensemble"  # 集成预测

@dataclass
class TemporalHoldoutConfig:
    """时间留出验证配置"""
    # 基本设置
    split_strategy: TemporalSplitStrategy = TemporalSplitStrategy.FORWARD_CHAINING
    validation_mode: TemporalValidationMode = TemporalValidationMode.SINGLE_STEP
    
    # 时间分割参数
    train_ratio: float = 0.7  # 训练集比例
    validation_ratio: float = 0.15  # 验证集比例
    test_ratio: float = 0.15  # 测试集比例
    
    # 滑动窗口参数
    window_size: int = 365  # 窗口大小（天）
    step_size: int = 30  # 步长（天）
    min_train_size: int = 730  # 最小训练集大小（天）
    
    # 预测参数
    prediction_horizon: int = 30  # 预测时间范围（天）
    max_prediction_steps: int = 10  # 最大预测步数
    
    # 季节性参数
    seasonal_periods: List[int] = None  # 季节周期（天）
    consider_leap_years: bool = True
    
    # 验证指标
    metrics: List[ValidationMetric] = None
    
    # 数据处理
    normalize_data: bool = True
    handle_missing_data: bool = True
    interpolation_method: str = "linear"
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "temporal_validation_plots"
    
    # 并行处理
    n_jobs: int = 1
    
    # 日志设置
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [365, 30, 7]  # 年、月、周
        
        if self.metrics is None:
            self.metrics = [
                ValidationMetric.MSE,
                ValidationMetric.MAE,
                ValidationMetric.R2,
                ValidationMetric.CORRELATION
            ]

class TemporalDataSplitter:
    """时间数据分割器"""
    
    def __init__(self, config: TemporalHoldoutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, data: Dict[str, torch.Tensor], 
                   time_column: str = 't') -> Dict[str, Dict[str, torch.Tensor]]:
        """分割数据"""
        if self.config.split_strategy == TemporalSplitStrategy.FORWARD_CHAINING:
            return self._forward_chaining_split(data, time_column)
        elif self.config.split_strategy == TemporalSplitStrategy.SLIDING_WINDOW:
            return self._sliding_window_split(data, time_column)
        elif self.config.split_strategy == TemporalSplitStrategy.BLOCKED_TIME_SERIES:
            return self._blocked_time_series_split(data, time_column)
        elif self.config.split_strategy == TemporalSplitStrategy.SEASONAL_SPLIT:
            return self._seasonal_split(data, time_column)
        elif self.config.split_strategy == TemporalSplitStrategy.YEAR_WISE_SPLIT:
            return self._year_wise_split(data, time_column)
        else:
            raise ValueError(f"不支持的分割策略: {self.config.split_strategy}")
    
    def _forward_chaining_split(self, data: Dict[str, torch.Tensor], 
                               time_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """前向链式分割"""
        time_data = data[time_column]
        n_samples = len(time_data)
        
        # 计算分割点
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 分割所有数据
        for key, tensor in data.items():
            splits['train'][key] = tensor[:train_end]
            splits['validation'][key] = tensor[train_end:val_end]
            splits['test'][key] = tensor[val_end:]
        
        self.logger.info(
            f"前向链式分割完成: 训练集 {train_end}, 验证集 {val_end - train_end}, "
            f"测试集 {n_samples - val_end}"
        )
        
        return splits
    
    def _sliding_window_split(self, data: Dict[str, torch.Tensor], 
                             time_column: str) -> List[Dict[str, Dict[str, torch.Tensor]]]:
        """滑动窗口分割"""
        time_data = data[time_column]
        n_samples = len(time_data)
        
        splits = []
        start_idx = 0
        
        while start_idx + self.config.min_train_size + self.config.prediction_horizon <= n_samples:
            train_end = start_idx + self.config.window_size
            test_start = train_end
            test_end = min(test_start + self.config.prediction_horizon, n_samples)
            
            if train_end > n_samples:
                break
            
            split = {
                'train': {},
                'test': {}
            }
            
            # 分割数据
            for key, tensor in data.items():
                split['train'][key] = tensor[start_idx:train_end]
                split['test'][key] = tensor[test_start:test_end]
            
            splits.append(split)
            start_idx += self.config.step_size
        
        self.logger.info(f"滑动窗口分割完成: 生成 {len(splits)} 个窗口")
        return splits
    
    def _blocked_time_series_split(self, data: Dict[str, torch.Tensor], 
                                  time_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """时间块分割"""
        time_data = data[time_column]
        n_samples = len(time_data)
        
        # 计算块大小
        block_size = n_samples // 5  # 分成5块
        
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # 使用前3块作为训练，第4块作为验证，第5块作为测试
        train_indices = list(range(0, 3 * block_size))
        val_indices = list(range(3 * block_size, 4 * block_size))
        test_indices = list(range(4 * block_size, n_samples))
        
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_indices]
            splits['validation'][key] = tensor[val_indices]
            splits['test'][key] = tensor[test_indices]
        
        self.logger.info(
            f"时间块分割完成: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, "
            f"测试集 {len(test_indices)}"
        )
        
        return splits
    
    def _seasonal_split(self, data: Dict[str, torch.Tensor], 
                       time_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """季节性分割"""
        time_data = data[time_column]
        
        # 假设时间数据是以天为单位的时间戳
        # 将时间转换为日期
        if isinstance(time_data, torch.Tensor):
            time_array = time_data.numpy()
        else:
            time_array = time_data
        
        # 计算季节（简化为按月份）
        seasons = (time_array % (365 * 24 * 3600)) // (30 * 24 * 3600)  # 月份
        
        # 按季节分割
        spring_summer = (seasons >= 2) & (seasons <= 7)  # 3-8月
        autumn_winter = ~spring_summer
        
        splits = {
            'train': {},  # 春夏数据
            'test': {}    # 秋冬数据
        }
        
        for key, tensor in data.items():
            splits['train'][key] = tensor[spring_summer]
            splits['test'][key] = tensor[autumn_winter]
        
        self.logger.info(
            f"季节性分割完成: 春夏 {spring_summer.sum()}, 秋冬 {autumn_winter.sum()}"
        )
        
        return splits
    
    def _year_wise_split(self, data: Dict[str, torch.Tensor], 
                        time_column: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """年度分割"""
        time_data = data[time_column]
        
        # 假设时间数据是以天为单位的时间戳
        if isinstance(time_data, torch.Tensor):
            time_array = time_data.numpy()
        else:
            time_array = time_data
        
        # 计算年份
        years = time_array // (365 * 24 * 3600)
        unique_years = np.unique(years)
        
        # 使用前70%年份作为训练，后30%作为测试
        n_train_years = int(len(unique_years) * 0.7)
        train_years = unique_years[:n_train_years]
        test_years = unique_years[n_train_years:]
        
        train_mask = np.isin(years, train_years)
        test_mask = np.isin(years, test_years)
        
        splits = {
            'train': {},
            'test': {}
        }
        
        for key, tensor in data.items():
            splits['train'][key] = tensor[train_mask]
            splits['test'][key] = tensor[test_mask]
        
        self.logger.info(
            f"年度分割完成: 训练年份 {len(train_years)}, 测试年份 {len(test_years)}"
        )
        
        return splits

class TemporalValidator:
    """时间验证器"""
    
    def __init__(self, config: TemporalHoldoutConfig):
        self.config = config
        self.splitter = TemporalDataSplitter(config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate(self, model: nn.Module, data: Dict[str, torch.Tensor], 
                time_column: str = 't') -> Dict[str, Any]:
        """执行时间验证"""
        results = {
            'strategy': self.config.split_strategy.value,
            'mode': self.config.validation_mode.value,
            'splits_results': [],
            'overall_metrics': {},
            'temporal_analysis': {}
        }
        
        try:
            # 分割数据
            splits = self.splitter.split_data(data, time_column)
            
            if isinstance(splits, list):  # 滑动窗口返回列表
                for i, split in enumerate(splits):
                    split_result = self._validate_split(model, split, f"window_{i}")
                    results['splits_results'].append(split_result)
            else:  # 其他策略返回字典
                split_result = self._validate_split(model, splits, "single_split")
                results['splits_results'].append(split_result)
            
            # 计算总体指标
            results['overall_metrics'] = self._compute_overall_metrics(results['splits_results'])
            
            # 时间分析
            results['temporal_analysis'] = self._analyze_temporal_performance(
                results['splits_results'], data, time_column
            )
            
            # 可视化
            if self.config.enable_plotting:
                self._plot_results(results, data, time_column)
            
        except Exception as e:
            self.logger.error(f"时间验证失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _validate_split(self, model: nn.Module, split: Dict[str, Dict[str, torch.Tensor]], 
                       split_name: str) -> Dict[str, Any]:
        """验证单个分割"""
        result = {
            'split_name': split_name,
            'metrics': {},
            'predictions': {},
            'targets': {}
        }
        
        try:
            # 获取训练和测试数据
            if 'train' in split and 'test' in split:
                train_data = split['train']
                test_data = split['test']
            elif 'train' in split and 'validation' in split:
                train_data = split['train']
                test_data = split['validation']
            else:
                raise ValueError("分割数据格式不正确")
            
            # 预测
            if self.config.validation_mode == TemporalValidationMode.SINGLE_STEP:
                predictions = self._single_step_prediction(model, test_data)
            elif self.config.validation_mode == TemporalValidationMode.MULTI_STEP:
                predictions = self._multi_step_prediction(model, train_data, test_data)
            elif self.config.validation_mode == TemporalValidationMode.RECURSIVE:
                predictions = self._recursive_prediction(model, train_data, test_data)
            else:
                predictions = self._single_step_prediction(model, test_data)
            
            # 获取目标值（假设模型输出的第一个维度是目标）
            targets = self._extract_targets(test_data)
            
            # 计算指标
            result['metrics'] = self._compute_metrics(predictions, targets)
            result['predictions'] = predictions
            result['targets'] = targets
            
        except Exception as e:
            self.logger.error(f"验证分割 {split_name} 失败: {e}")
            result['error'] = str(e)
        
        return result
    
    def _single_step_prediction(self, model: nn.Module, 
                               test_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步预测"""
        model.eval()
        with torch.no_grad():
            # 构造输入
            inputs = self._prepare_inputs(test_data)
            predictions = model(inputs)
            
            if predictions.dim() > 1:
                predictions = predictions[:, 0]  # 取第一个输出
        
        return predictions
    
    def _multi_step_prediction(self, model: nn.Module, 
                              train_data: Dict[str, torch.Tensor],
                              test_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """多步预测"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for step in range(min(self.config.max_prediction_steps, len(test_data['t']))):
                # 构造当前步的输入
                current_input = {}
                for key, tensor in test_data.items():
                    if step < len(tensor):
                        current_input[key] = tensor[step:step+1]
                    else:
                        break
                
                if not current_input:
                    break
                
                inputs = self._prepare_inputs(current_input)
                pred = model(inputs)
                
                if pred.dim() > 1:
                    pred = pred[:, 0]
                
                predictions.append(pred)
        
        return torch.cat(predictions) if predictions else torch.tensor([])
    
    def _recursive_prediction(self, model: nn.Module, 
                             train_data: Dict[str, torch.Tensor],
                             test_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """递归预测"""
        # 简化实现，实际应该使用前一步的预测作为下一步的输入
        return self._multi_step_prediction(model, train_data, test_data)
    
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
        # 假设目标值在 'target' 或 'y' 键中
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
    
    def _compute_metrics(self, predictions: torch.Tensor, 
                        targets: torch.Tensor) -> Dict[str, float]:
        """计算验证指标"""
        metrics = {}
        
        # 确保张量形状一致
        min_len = min(len(predictions), len(targets))
        pred = predictions[:min_len]
        targ = targets[:min_len]
        
        try:
            for metric in self.config.metrics:
                if metric == ValidationMetric.MSE:
                    metrics['mse'] = torch.mean((pred - targ) ** 2).item()
                elif metric == ValidationMetric.MAE:
                    metrics['mae'] = torch.mean(torch.abs(pred - targ)).item()
                elif metric == ValidationMetric.RMSE:
                    metrics['rmse'] = torch.sqrt(torch.mean((pred - targ) ** 2)).item()
                elif metric == ValidationMetric.MAPE:
                    metrics['mape'] = torch.mean(torch.abs((pred - targ) / (targ + 1e-8))).item() * 100
                elif metric == ValidationMetric.R2:
                    ss_res = torch.sum((targ - pred) ** 2)
                    ss_tot = torch.sum((targ - torch.mean(targ)) ** 2)
                    metrics['r2'] = (1 - ss_res / (ss_tot + 1e-8)).item()
                elif metric == ValidationMetric.CORRELATION:
                    pred_centered = pred - torch.mean(pred)
                    targ_centered = targ - torch.mean(targ)
                    correlation = torch.sum(pred_centered * targ_centered) / (
                        torch.sqrt(torch.sum(pred_centered ** 2)) * 
                        torch.sqrt(torch.sum(targ_centered ** 2)) + 1e-8
                    )
                    metrics['correlation'] = correlation.item()
        
        except Exception as e:
            self.logger.warning(f"计算指标时出错: {e}")
        
        return metrics
    
    def _compute_overall_metrics(self, splits_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算总体指标"""
        overall_metrics = {}
        
        # 收集所有分割的指标
        all_metrics = {}
        for result in splits_results:
            if 'metrics' in result:
                for metric_name, value in result['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # 计算平均值和标准差
        for metric_name, values in all_metrics.items():
            if values:
                overall_metrics[f'{metric_name}_mean'] = np.mean(values)
                overall_metrics[f'{metric_name}_std'] = np.std(values)
                overall_metrics[f'{metric_name}_min'] = np.min(values)
                overall_metrics[f'{metric_name}_max'] = np.max(values)
        
        return overall_metrics
    
    def _analyze_temporal_performance(self, splits_results: List[Dict[str, Any]], 
                                    data: Dict[str, torch.Tensor], 
                                    time_column: str) -> Dict[str, Any]:
        """分析时间性能"""
        analysis = {
            'trend_analysis': {},
            'seasonal_analysis': {},
            'stability_analysis': {}
        }
        
        try:
            # 趋势分析
            if len(splits_results) > 1:
                mse_values = [r['metrics'].get('mse', float('inf')) for r in splits_results if 'metrics' in r]
                if len(mse_values) > 1:
                    # 计算性能趋势
                    trend_slope = np.polyfit(range(len(mse_values)), mse_values, 1)[0]
                    analysis['trend_analysis'] = {
                        'performance_trend': 'improving' if trend_slope < 0 else 'degrading',
                        'trend_slope': trend_slope,
                        'performance_stability': np.std(mse_values)
                    }
            
            # 季节性分析（简化）
            analysis['seasonal_analysis'] = {
                'seasonal_variation': 'low',  # 简化实现
                'peak_performance_season': 'summer'
            }
            
            # 稳定性分析
            all_correlations = [r['metrics'].get('correlation', 0) for r in splits_results if 'metrics' in r]
            if all_correlations:
                analysis['stability_analysis'] = {
                    'mean_correlation': np.mean(all_correlations),
                    'correlation_stability': np.std(all_correlations),
                    'min_correlation': np.min(all_correlations)
                }
        
        except Exception as e:
            self.logger.warning(f"时间分析失败: {e}")
        
        return analysis
    
    def _plot_results(self, results: Dict[str, Any], 
                     data: Dict[str, torch.Tensor], time_column: str):
        """绘制结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制预测vs真实值
            if results['splits_results']:
                split_result = results['splits_results'][0]
                if 'predictions' in split_result and 'targets' in split_result:
                    pred = split_result['predictions']
                    targ = split_result['targets']
                    
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().numpy()
                    if isinstance(targ, torch.Tensor):
                        targ = targ.detach().numpy()
                    
                    axes[0, 0].scatter(targ, pred, alpha=0.6)
                    axes[0, 0].plot([targ.min(), targ.max()], [targ.min(), targ.max()], 'r--')
                    axes[0, 0].set_xlabel('真实值')
                    axes[0, 0].set_ylabel('预测值')
                    axes[0, 0].set_title('预测 vs 真实值')
            
            # 绘制时间序列
            if results['splits_results']:
                split_result = results['splits_results'][0]
                if 'predictions' in split_result and 'targets' in split_result:
                    pred = split_result['predictions']
                    targ = split_result['targets']
                    
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().numpy()
                    if isinstance(targ, torch.Tensor):
                        targ = targ.detach().numpy()
                    
                    time_steps = range(len(pred))
                    axes[0, 1].plot(time_steps, targ, label='真实值', alpha=0.7)
                    axes[0, 1].plot(time_steps, pred, label='预测值', alpha=0.7)
                    axes[0, 1].set_xlabel('时间步')
                    axes[0, 1].set_ylabel('值')
                    axes[0, 1].set_title('时间序列预测')
                    axes[0, 1].legend()
            
            # 绘制指标变化
            if len(results['splits_results']) > 1:
                mse_values = [r['metrics'].get('mse', 0) for r in results['splits_results'] if 'metrics' in r]
                mae_values = [r['metrics'].get('mae', 0) for r in results['splits_results'] if 'metrics' in r]
                
                axes[1, 0].plot(mse_values, label='MSE', marker='o')
                axes[1, 0].plot(mae_values, label='MAE', marker='s')
                axes[1, 0].set_xlabel('分割索引')
                axes[1, 0].set_ylabel('误差')
                axes[1, 0].set_title('验证指标变化')
                axes[1, 0].legend()
            
            # 绘制相关性分析
            if len(results['splits_results']) > 1:
                corr_values = [r['metrics'].get('correlation', 0) for r in results['splits_results'] if 'metrics' in r]
                axes[1, 1].plot(corr_values, marker='o', color='green')
                axes[1, 1].set_xlabel('分割索引')
                axes[1, 1].set_ylabel('相关系数')
                axes[1, 1].set_title('预测相关性变化')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/temporal_validation_results.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "时间维度留出验证报告",
            "=" * 60,
            f"验证策略: {results.get('strategy', 'Unknown')}",
            f"验证模式: {results.get('mode', 'Unknown')}",
            f"分割数量: {len(results.get('splits_results', []))}",
            ""
        ]
        
        # 总体指标
        if 'overall_metrics' in results:
            report_lines.append("总体性能指标:")
            for metric, value in results['overall_metrics'].items():
                report_lines.append(f"  {metric}: {value:.6f}")
            report_lines.append("")
        
        # 时间分析
        if 'temporal_analysis' in results:
            ta = results['temporal_analysis']
            report_lines.append("时间分析:")
            
            if 'trend_analysis' in ta:
                trend = ta['trend_analysis']
                report_lines.extend([
                    f"  性能趋势: {trend.get('performance_trend', 'Unknown')}",
                    f"  趋势斜率: {trend.get('trend_slope', 0):.6f}",
                    f"  性能稳定性: {trend.get('performance_stability', 0):.6f}"
                ])
            
            if 'stability_analysis' in ta:
                stability = ta['stability_analysis']
                report_lines.extend([
                    f"  平均相关性: {stability.get('mean_correlation', 0):.6f}",
                    f"  相关性稳定性: {stability.get('correlation_stability', 0):.6f}",
                    f"  最小相关性: {stability.get('min_correlation', 0):.6f}"
                ])
            
            report_lines.append("")
        
        # 分割详情
        if 'splits_results' in results:
            report_lines.append("分割详细结果:")
            for i, split_result in enumerate(results['splits_results']):
                if 'metrics' in split_result:
                    report_lines.append(f"  分割 {i+1}:")
                    for metric, value in split_result['metrics'].items():
                        report_lines.append(f"    {metric}: {value:.6f}")
                    report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def create_temporal_validator(config: TemporalHoldoutConfig = None) -> TemporalValidator:
    """
    创建时间验证器
    
    Args:
        config: 配置
        
    Returns:
        TemporalValidator: 验证器实例
    """
    if config is None:
        config = TemporalHoldoutConfig()
    return TemporalValidator(config)

if __name__ == "__main__":
    # 测试时间维度留出验证
    
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
        'x': torch.randn(n_samples),
        'y': torch.randn(n_samples),
        't': torch.linspace(0, 365*24*3600, n_samples),  # 一年的时间
        'temperature': torch.randn(n_samples) + 273.15  # 温度数据
    }
    
    # 创建配置
    config = TemporalHoldoutConfig(
        split_strategy=TemporalSplitStrategy.FORWARD_CHAINING,
        validation_mode=TemporalValidationMode.SINGLE_STEP,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_temporal_validator(config)
    
    print("=== 时间维度留出验证测试 ===")
    
    # 执行验证
    results = validator.validate(model, data)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n时间维度留出验证测试完成！")