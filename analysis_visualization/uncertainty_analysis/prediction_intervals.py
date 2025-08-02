#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测区间分析模块

本模块提供冰川PINNs模型预测区间的计算和分析功能，包括：
- 贝叶斯预测区间
- Bootstrap预测区间
- 分位数回归预测区间
- 集成模型预测区间
- 时间序列预测区间
- 空间预测区间

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionIntervalCalculator:
    """
    预测区间计算器
    
    提供多种预测区间计算方法，包括参数化和非参数化方法。
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化预测区间计算器
        
        参数:
            confidence_level: 置信水平 (0, 1)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("置信水平必须在0和1之间")
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.lower_quantile = self.alpha / 2
        self.upper_quantile = 1 - self.alpha / 2
        
    def bootstrap_prediction_interval(self,
                                    model: Callable,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_test: np.ndarray,
                                    n_bootstrap: int = 1000,
                                    random_state: Optional[int] = None) -> Dict:
        """
        Bootstrap预测区间
        
        参数:
            model: 预测模型（可调用对象）
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            n_bootstrap: Bootstrap采样次数
            random_state: 随机种子
            
        返回:
            预测区间结果
        """
        try:
            if random_state is not None:
                np.random.seed(random_state)
            
            n_train = len(X_train)
            n_test = len(X_test)
            
            # 存储Bootstrap预测结果
            bootstrap_predictions = np.zeros((n_bootstrap, n_test))
            
            logger.info(f"开始Bootstrap采样，共{n_bootstrap}次...")
            
            for i in range(n_bootstrap):
                # Bootstrap采样
                bootstrap_indices = np.random.choice(n_train, size=n_train, replace=True)
                X_bootstrap = X_train[bootstrap_indices]
                y_bootstrap = y_train[bootstrap_indices]
                
                # 训练模型并预测
                try:
                    # 如果模型有fit方法（sklearn风格）
                    if hasattr(model, 'fit'):
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_bootstrap, y_bootstrap)
                        predictions = model_copy.predict(X_test)
                    else:
                        # 自定义模型
                        predictions = model(X_bootstrap, y_bootstrap, X_test)
                    
                    bootstrap_predictions[i] = predictions
                    
                except Exception as e:
                    logger.warning(f"Bootstrap第{i+1}次采样失败: {e}")
                    bootstrap_predictions[i] = np.nan
                
                if (i + 1) % 100 == 0:
                    logger.info(f"完成Bootstrap采样: {i+1}/{n_bootstrap}")
            
            # 计算预测区间
            valid_predictions = bootstrap_predictions[~np.isnan(bootstrap_predictions).any(axis=1)]
            
            if len(valid_predictions) < 10:
                raise ValueError("有效的Bootstrap预测太少")
            
            # 计算分位数
            lower_bound = np.percentile(valid_predictions, 
                                      self.lower_quantile * 100, axis=0)
            upper_bound = np.percentile(valid_predictions, 
                                      self.upper_quantile * 100, axis=0)
            mean_prediction = np.mean(valid_predictions, axis=0)
            std_prediction = np.std(valid_predictions, axis=0)
            
            # 计算区间宽度
            interval_width = upper_bound - lower_bound
            
            return {
                'method': 'bootstrap',
                'confidence_level': self.confidence_level,
                'n_bootstrap': len(valid_predictions),
                'mean_prediction': mean_prediction,
                'std_prediction': std_prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'all_predictions': valid_predictions
            }
            
        except Exception as e:
            logger.error(f"Bootstrap预测区间计算失败: {e}")
            raise
    
    def quantile_regression_interval(self,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   solver: str = 'highs') -> Dict:
        """
        分位数回归预测区间
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            solver: 求解器
            
        返回:
            预测区间结果
        """
        try:
            # 训练分位数回归模型
            quantiles = [self.lower_quantile, 0.5, self.upper_quantile]
            models = {}
            predictions = {}
            
            for q in quantiles:
                logger.info(f"训练{q:.3f}分位数回归模型...")
                
                model = QuantileRegressor(quantile=q, solver=solver, alpha=0.01)
                model.fit(X_train, y_train)
                
                pred = model.predict(X_test)
                
                models[q] = model
                predictions[q] = pred
            
            lower_bound = predictions[self.lower_quantile]
            upper_bound = predictions[self.upper_quantile]
            median_prediction = predictions[0.5]
            interval_width = upper_bound - lower_bound
            
            return {
                'method': 'quantile_regression',
                'confidence_level': self.confidence_level,
                'models': models,
                'median_prediction': median_prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'quantiles': quantiles
            }
            
        except Exception as e:
            logger.error(f"分位数回归预测区间计算失败: {e}")
            raise
    
    def ensemble_prediction_interval(self,
                                   models: List[Callable],
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   method: str = 'simple') -> Dict:
        """
        集成模型预测区间
        
        参数:
            models: 模型列表
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            method: 集成方法 ('simple', 'weighted', 'stacking')
            
        返回:
            预测区间结果
        """
        try:
            n_models = len(models)
            n_test = len(X_test)
            
            # 存储所有模型的预测结果
            all_predictions = np.zeros((n_models, n_test))
            model_weights = np.ones(n_models) / n_models
            
            logger.info(f"训练{n_models}个集成模型...")
            
            for i, model in enumerate(models):
                try:
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    else:
                        predictions = model(X_train, y_train, X_test)
                    
                    all_predictions[i] = predictions
                    
                    logger.info(f"完成模型{i+1}/{n_models}")
                    
                except Exception as e:
                    logger.warning(f"模型{i+1}训练失败: {e}")
                    all_predictions[i] = np.nan
            
            # 移除失败的模型
            valid_mask = ~np.isnan(all_predictions).any(axis=1)
            valid_predictions = all_predictions[valid_mask]
            valid_weights = model_weights[valid_mask]
            
            if len(valid_predictions) == 0:
                raise ValueError("所有模型都训练失败")
            
            # 重新归一化权重
            valid_weights = valid_weights / np.sum(valid_weights)
            
            if method == 'simple':
                # 简单平均
                ensemble_mean = np.mean(valid_predictions, axis=0)
                ensemble_std = np.std(valid_predictions, axis=0)
                
            elif method == 'weighted':
                # 加权平均（基于交叉验证性能）
                weights = self._calculate_model_weights(
                    [models[i] for i in range(len(models)) if valid_mask[i]],
                    X_train, y_train
                )
                ensemble_mean = np.average(valid_predictions, axis=0, weights=weights)
                ensemble_std = np.sqrt(np.average(
                    (valid_predictions - ensemble_mean)**2, axis=0, weights=weights
                ))
                
            else:
                # 简单方法作为默认
                ensemble_mean = np.mean(valid_predictions, axis=0)
                ensemble_std = np.std(valid_predictions, axis=0)
            
            # 计算预测区间（假设正态分布）
            z_score = stats.norm.ppf(self.upper_quantile)
            lower_bound = ensemble_mean - z_score * ensemble_std
            upper_bound = ensemble_mean + z_score * ensemble_std
            
            # 也可以使用经验分位数
            empirical_lower = np.percentile(valid_predictions, 
                                          self.lower_quantile * 100, axis=0)
            empirical_upper = np.percentile(valid_predictions, 
                                          self.upper_quantile * 100, axis=0)
            
            return {
                'method': 'ensemble',
                'ensemble_method': method,
                'confidence_level': self.confidence_level,
                'n_models': len(valid_predictions),
                'ensemble_mean': ensemble_mean,
                'ensemble_std': ensemble_std,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'empirical_lower': empirical_lower,
                'empirical_upper': empirical_upper,
                'interval_width': upper_bound - lower_bound,
                'all_predictions': valid_predictions
            }
            
        except Exception as e:
            logger.error(f"集成预测区间计算失败: {e}")
            raise
    
    def bayesian_prediction_interval(self,
                                   model_class: type,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   n_samples: int = 1000,
                                   **model_kwargs) -> Dict:
        """
        贝叶斯预测区间（使用变分推断近似）
        
        参数:
            model_class: 贝叶斯模型类
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            n_samples: 采样次数
            **model_kwargs: 模型参数
            
        返回:
            预测区间结果
        """
        try:
            # 这里实现一个简化的贝叶斯线性回归
            return self._bayesian_linear_regression(
                X_train, y_train, X_test, n_samples
            )
            
        except Exception as e:
            logger.error(f"贝叶斯预测区间计算失败: {e}")
            raise
    
    def temporal_prediction_interval(self,
                                   time_series: pd.Series,
                                   forecast_horizon: int,
                                   method: str = 'arima') -> Dict:
        """
        时间序列预测区间
        
        参数:
            time_series: 时间序列数据
            forecast_horizon: 预测步长
            method: 预测方法 ('arima', 'exponential_smoothing', 'bootstrap')
            
        返回:
            预测区间结果
        """
        try:
            if method == 'bootstrap':
                return self._bootstrap_time_series_interval(
                    time_series, forecast_horizon
                )
            elif method == 'residual_bootstrap':
                return self._residual_bootstrap_interval(
                    time_series, forecast_horizon
                )
            else:
                # 简单的移动窗口方法
                return self._moving_window_interval(
                    time_series, forecast_horizon
                )
                
        except Exception as e:
            logger.error(f"时间序列预测区间计算失败: {e}")
            raise
    
    def spatial_prediction_interval(self,
                                  spatial_data: np.ndarray,
                                  coordinates: np.ndarray,
                                  prediction_locations: np.ndarray,
                                  method: str = 'kriging') -> Dict:
        """
        空间预测区间
        
        参数:
            spatial_data: 空间观测数据
            coordinates: 观测点坐标
            prediction_locations: 预测点坐标
            method: 空间插值方法
            
        返回:
            预测区间结果
        """
        try:
            if method == 'kriging':
                return self._kriging_prediction_interval(
                    spatial_data, coordinates, prediction_locations
                )
            else:
                return self._spatial_bootstrap_interval(
                    spatial_data, coordinates, prediction_locations
                )
                
        except Exception as e:
            logger.error(f"空间预测区间计算失败: {e}")
            raise
    
    def _calculate_model_weights(self,
                               models: List[Callable],
                               X: np.ndarray,
                               y: np.ndarray,
                               cv_folds: int = 5) -> np.ndarray:
        """
        基于交叉验证计算模型权重
        """
        from sklearn.model_selection import cross_val_score
        
        scores = []
        for model in models:
            try:
                if hasattr(model, 'fit'):
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                              scoring='neg_mean_squared_error')
                    scores.append(-np.mean(cv_scores))
                else:
                    # 对于自定义模型，使用简单的holdout验证
                    split_idx = int(0.8 * len(X))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    pred = model(X_train, y_train, X_val)
                    mse = mean_squared_error(y_val, pred)
                    scores.append(mse)
            except:
                scores.append(np.inf)
        
        # 转换为权重（分数越低权重越高）
        scores = np.array(scores)
        weights = 1 / (scores + 1e-8)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _bayesian_linear_regression(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  n_samples: int) -> Dict:
        """
        简化的贝叶斯线性回归
        """
        # 添加偏置项
        X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_bias = np.column_stack([np.ones(len(X_test)), X_test])
        
        # 先验参数
        alpha = 1.0  # 噪声精度的先验
        beta = 1.0   # 权重精度的先验
        
        # 后验参数
        S_inv = beta * np.eye(X_train_bias.shape[1]) + alpha * X_train_bias.T @ X_train_bias
        S = np.linalg.inv(S_inv)
        m = alpha * S @ X_train_bias.T @ y_train
        
        # 预测分布参数
        y_mean = X_test_bias @ m
        y_var = 1/alpha + np.sum((X_test_bias @ S) * X_test_bias, axis=1)
        y_std = np.sqrt(y_var)
        
        # 计算预测区间
        t_value = stats.t.ppf(self.upper_quantile, df=len(y_train)-X_train_bias.shape[1])
        lower_bound = y_mean - t_value * y_std
        upper_bound = y_mean + t_value * y_std
        
        return {
            'method': 'bayesian',
            'confidence_level': self.confidence_level,
            'mean_prediction': y_mean,
            'std_prediction': y_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _bootstrap_time_series_interval(self,
                                      time_series: pd.Series,
                                      forecast_horizon: int,
                                      n_bootstrap: int = 1000) -> Dict:
        """
        时间序列Bootstrap预测区间
        """
        # 简单的移动平均预测
        window_size = min(12, len(time_series) // 4)
        
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Bootstrap采样
            bootstrap_series = time_series.sample(n=len(time_series), replace=True)
            
            # 简单预测（移动平均）
            last_values = bootstrap_series.tail(window_size).values
            forecast = [np.mean(last_values)] * forecast_horizon
            
            bootstrap_forecasts.append(forecast)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # 计算预测区间
        lower_bound = np.percentile(bootstrap_forecasts, 
                                  self.lower_quantile * 100, axis=0)
        upper_bound = np.percentile(bootstrap_forecasts, 
                                  self.upper_quantile * 100, axis=0)
        mean_forecast = np.mean(bootstrap_forecasts, axis=0)
        
        return {
            'method': 'time_series_bootstrap',
            'confidence_level': self.confidence_level,
            'forecast_horizon': forecast_horizon,
            'mean_forecast': mean_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _residual_bootstrap_interval(self,
                                   time_series: pd.Series,
                                   forecast_horizon: int,
                                   n_bootstrap: int = 1000) -> Dict:
        """
        残差Bootstrap预测区间
        """
        # 拟合简单趋势模型
        X = np.arange(len(time_series)).reshape(-1, 1)
        y = time_series.values
        
        # 线性趋势
        from sklearn.linear_model import LinearRegression
        trend_model = LinearRegression()
        trend_model.fit(X, y)
        
        # 计算残差
        fitted = trend_model.predict(X)
        residuals = y - fitted
        
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Bootstrap残差
            bootstrap_residuals = np.random.choice(residuals, 
                                                  size=forecast_horizon, 
                                                  replace=True)
            
            # 预测趋势
            future_X = np.arange(len(time_series), 
                               len(time_series) + forecast_horizon).reshape(-1, 1)
            trend_forecast = trend_model.predict(future_X)
            
            # 添加Bootstrap残差
            forecast = trend_forecast + bootstrap_residuals
            bootstrap_forecasts.append(forecast)
        
        bootstrap_forecasts = np.array(bootstrap_forecasts)
        
        # 计算预测区间
        lower_bound = np.percentile(bootstrap_forecasts, 
                                  self.lower_quantile * 100, axis=0)
        upper_bound = np.percentile(bootstrap_forecasts, 
                                  self.upper_quantile * 100, axis=0)
        mean_forecast = np.mean(bootstrap_forecasts, axis=0)
        
        return {
            'method': 'residual_bootstrap',
            'confidence_level': self.confidence_level,
            'forecast_horizon': forecast_horizon,
            'mean_forecast': mean_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _moving_window_interval(self,
                              time_series: pd.Series,
                              forecast_horizon: int,
                              window_size: Optional[int] = None) -> Dict:
        """
        移动窗口预测区间
        """
        if window_size is None:
            window_size = min(24, len(time_series) // 3)
        
        # 计算移动窗口统计
        rolling_mean = time_series.rolling(window=window_size).mean()
        rolling_std = time_series.rolling(window=window_size).std()
        
        # 最后的统计值作为预测
        last_mean = rolling_mean.iloc[-1]
        last_std = rolling_std.iloc[-1]
        
        # 预测区间
        z_score = stats.norm.ppf(self.upper_quantile)
        
        mean_forecast = np.full(forecast_horizon, last_mean)
        lower_bound = mean_forecast - z_score * last_std
        upper_bound = mean_forecast + z_score * last_std
        
        return {
            'method': 'moving_window',
            'confidence_level': self.confidence_level,
            'window_size': window_size,
            'forecast_horizon': forecast_horizon,
            'mean_forecast': mean_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _kriging_prediction_interval(self,
                                   spatial_data: np.ndarray,
                                   coordinates: np.ndarray,
                                   prediction_locations: np.ndarray) -> Dict:
        """
        Kriging预测区间（简化实现）
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        
        # 使用高斯过程作为Kriging的近似
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        
        gp.fit(coordinates, spatial_data)
        
        # 预测
        mean_pred, std_pred = gp.predict(prediction_locations, return_std=True)
        
        # 计算预测区间
        z_score = stats.norm.ppf(self.upper_quantile)
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return {
            'method': 'kriging',
            'confidence_level': self.confidence_level,
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def _spatial_bootstrap_interval(self,
                                  spatial_data: np.ndarray,
                                  coordinates: np.ndarray,
                                  prediction_locations: np.ndarray,
                                  n_bootstrap: int = 500) -> Dict:
        """
        空间Bootstrap预测区间
        """
        from sklearn.neighbors import KNeighborsRegressor
        
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(len(spatial_data), 
                                     size=len(spatial_data), 
                                     replace=True)
            
            bootstrap_coords = coordinates[indices]
            bootstrap_data = spatial_data[indices]
            
            # KNN插值
            knn = KNeighborsRegressor(n_neighbors=min(5, len(bootstrap_data)))
            knn.fit(bootstrap_coords, bootstrap_data)
            
            predictions = knn.predict(prediction_locations)
            bootstrap_predictions.append(predictions)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # 计算预测区间
        lower_bound = np.percentile(bootstrap_predictions, 
                                  self.lower_quantile * 100, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, 
                                  self.upper_quantile * 100, axis=0)
        mean_prediction = np.mean(bootstrap_predictions, axis=0)
        
        return {
            'method': 'spatial_bootstrap',
            'confidence_level': self.confidence_level,
            'mean_prediction': mean_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }

class PredictionIntervalEvaluator:
    """
    预测区间评估器
    
    提供预测区间质量评估的各种指标。
    """
    
    def __init__(self):
        """
        初始化评估器
        """
        pass
    
    def evaluate_coverage(self,
                        y_true: np.ndarray,
                        lower_bound: np.ndarray,
                        upper_bound: np.ndarray,
                        confidence_level: float) -> Dict:
        """
        评估预测区间覆盖率
        
        参数:
            y_true: 真实值
            lower_bound: 下界
            upper_bound: 上界
            confidence_level: 置信水平
            
        返回:
            覆盖率评估结果
        """
        try:
            # 计算覆盖率
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            
            # 计算区间宽度
            interval_width = upper_bound - lower_bound
            mean_width = np.mean(interval_width)
            
            # 计算覆盖率偏差
            coverage_deviation = abs(coverage - confidence_level)
            
            # 计算条件覆盖率（分段）
            n_bins = 5
            bin_edges = np.linspace(0, len(y_true), n_bins + 1, dtype=int)
            conditional_coverage = []
            
            for i in range(n_bins):
                start, end = bin_edges[i], bin_edges[i + 1]
                if end > start:
                    bin_coverage = np.mean(
                        (y_true[start:end] >= lower_bound[start:end]) & 
                        (y_true[start:end] <= upper_bound[start:end])
                    )
                    conditional_coverage.append(bin_coverage)
            
            # 计算覆盖率的置信区间
            n = len(y_true)
            coverage_se = np.sqrt(coverage * (1 - coverage) / n)
            coverage_ci_lower = coverage - 1.96 * coverage_se
            coverage_ci_upper = coverage + 1.96 * coverage_se
            
            return {
                'coverage': coverage,
                'target_coverage': confidence_level,
                'coverage_deviation': coverage_deviation,
                'mean_interval_width': mean_width,
                'conditional_coverage': conditional_coverage,
                'coverage_confidence_interval': (coverage_ci_lower, coverage_ci_upper),
                'n_observations': n
            }
            
        except Exception as e:
            logger.error(f"覆盖率评估失败: {e}")
            raise
    
    def evaluate_sharpness(self,
                         lower_bound: np.ndarray,
                         upper_bound: np.ndarray) -> Dict:
        """
        评估预测区间锐度（窄度）
        
        参数:
            lower_bound: 下界
            upper_bound: 上界
            
        返回:
            锐度评估结果
        """
        try:
            interval_width = upper_bound - lower_bound
            
            return {
                'mean_width': np.mean(interval_width),
                'median_width': np.median(interval_width),
                'std_width': np.std(interval_width),
                'min_width': np.min(interval_width),
                'max_width': np.max(interval_width),
                'width_percentiles': {
                    '25%': np.percentile(interval_width, 25),
                    '75%': np.percentile(interval_width, 75),
                    '90%': np.percentile(interval_width, 90),
                    '95%': np.percentile(interval_width, 95)
                }
            }
            
        except Exception as e:
            logger.error(f"锐度评估失败: {e}")
            raise
    
    def evaluate_calibration(self,
                           y_true: np.ndarray,
                           lower_bound: np.ndarray,
                           upper_bound: np.ndarray,
                           n_bins: int = 10) -> Dict:
        """
        评估预测区间校准性
        
        参数:
            y_true: 真实值
            lower_bound: 下界
            upper_bound: 上界
            n_bins: 分箱数量
            
        返回:
            校准性评估结果
        """
        try:
            # 计算预测区间的中心和宽度
            center = (lower_bound + upper_bound) / 2
            width = upper_bound - lower_bound
            
            # 计算标准化残差
            residuals = (y_true - center) / width
            
            # 分箱分析
            bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            observed_frequencies = []
            expected_frequencies = []
            
            for i in range(n_bins):
                # 观测频率
                in_bin = (residuals >= bin_edges[i]) & (residuals < bin_edges[i + 1])
                observed_freq = np.mean(in_bin)
                observed_frequencies.append(observed_freq)
                
                # 期望频率（均匀分布）
                expected_freq = 1 / n_bins
                expected_frequencies.append(expected_freq)
            
            # 计算卡方统计量
            chi_square = np.sum(
                (np.array(observed_frequencies) - np.array(expected_frequencies))**2 / 
                np.array(expected_frequencies)
            )
            
            # 计算p值
            p_value = 1 - stats.chi2.cdf(chi_square, df=n_bins - 1)
            
            return {
                'bin_centers': bin_centers,
                'observed_frequencies': observed_frequencies,
                'expected_frequencies': expected_frequencies,
                'chi_square_statistic': chi_square,
                'p_value': p_value,
                'is_well_calibrated': p_value > 0.05
            }
            
        except Exception as e:
            logger.error(f"校准性评估失败: {e}")
            raise
    
    def comprehensive_evaluation(self,
                               y_true: np.ndarray,
                               predictions: Dict,
                               confidence_level: float) -> Dict:
        """
        综合评估预测区间
        
        参数:
            y_true: 真实值
            predictions: 预测结果字典
            confidence_level: 置信水平
            
        返回:
            综合评估结果
        """
        try:
            results = {
                'method': predictions.get('method', 'unknown'),
                'confidence_level': confidence_level
            }
            
            # 覆盖率评估
            coverage_results = self.evaluate_coverage(
                y_true, predictions['lower_bound'], 
                predictions['upper_bound'], confidence_level
            )
            results['coverage'] = coverage_results
            
            # 锐度评估
            sharpness_results = self.evaluate_sharpness(
                predictions['lower_bound'], predictions['upper_bound']
            )
            results['sharpness'] = sharpness_results
            
            # 校准性评估
            calibration_results = self.evaluate_calibration(
                y_true, predictions['lower_bound'], predictions['upper_bound']
            )
            results['calibration'] = calibration_results
            
            # 计算综合得分
            coverage_score = 1 - coverage_results['coverage_deviation']
            sharpness_score = 1 / (1 + sharpness_results['mean_width'])
            calibration_score = 1 if calibration_results['is_well_calibrated'] else 0.5
            
            overall_score = (coverage_score + sharpness_score + calibration_score) / 3
            results['overall_score'] = overall_score
            
            return results
            
        except Exception as e:
            logger.error(f"综合评估失败: {e}")
            raise

class PredictionIntervalVisualizer:
    """
    预测区间可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_prediction_intervals(self,
                                X_test: np.ndarray,
                                y_true: Optional[np.ndarray],
                                predictions: Dict,
                                title: str = "预测区间",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制预测区间
        
        参数:
            X_test: 测试特征（用于x轴）
            y_true: 真实值（可选）
            predictions: 预测结果
            title: 图标题
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 如果X_test是多维的，使用第一维或索引
        if X_test.ndim > 1:
            x_axis = np.arange(len(X_test))
            xlabel = "样本索引"
        else:
            x_axis = X_test
            xlabel = "X值"
        
        # 绘制预测区间
        ax.fill_between(x_axis, 
                       predictions['lower_bound'], 
                       predictions['upper_bound'],
                       alpha=0.3, color='blue', 
                       label=f"{predictions.get('confidence_level', 0.95)*100:.0f}%预测区间")
        
        # 绘制预测均值
        if 'mean_prediction' in predictions:
            ax.plot(x_axis, predictions['mean_prediction'], 
                   'b-', linewidth=2, label='预测均值')
        elif 'median_prediction' in predictions:
            ax.plot(x_axis, predictions['median_prediction'], 
                   'b-', linewidth=2, label='预测中位数')
        
        # 绘制真实值
        if y_true is not None:
            ax.scatter(x_axis, y_true, color='red', s=30, 
                      alpha=0.7, label='真实值', zorder=5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('值')
        ax.set_title(f"{title} - {predictions.get('method', '未知方法')}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        if y_true is not None:
            coverage = np.mean((y_true >= predictions['lower_bound']) & 
                             (y_true <= predictions['upper_bound']))
            mean_width = np.mean(predictions['upper_bound'] - predictions['lower_bound'])
            
            info_text = f"覆盖率: {coverage:.3f}\n平均宽度: {mean_width:.3f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interval_comparison(self,
                               X_test: np.ndarray,
                               y_true: np.ndarray,
                               predictions_list: List[Dict],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        比较多种预测区间方法
        
        参数:
            X_test: 测试特征
            y_true: 真实值
            predictions_list: 预测结果列表
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        n_methods = len(predictions_list)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4*n_methods))
        
        if n_methods == 1:
            axes = [axes]
        
        # 如果X_test是多维的，使用索引
        if X_test.ndim > 1:
            x_axis = np.arange(len(X_test))
            xlabel = "样本索引"
        else:
            x_axis = X_test
            xlabel = "X值"
        
        for i, predictions in enumerate(predictions_list):
            ax = axes[i]
            
            # 绘制预测区间
            ax.fill_between(x_axis, 
                           predictions['lower_bound'], 
                           predictions['upper_bound'],
                           alpha=0.3, color=self.colors[i % len(self.colors)])
            
            # 绘制预测均值
            if 'mean_prediction' in predictions:
                ax.plot(x_axis, predictions['mean_prediction'], 
                       color=self.colors[i % len(self.colors)], linewidth=2)
            
            # 绘制真实值
            ax.scatter(x_axis, y_true, color='red', s=20, alpha=0.7)
            
            # 计算覆盖率
            coverage = np.mean((y_true >= predictions['lower_bound']) & 
                             (y_true <= predictions['upper_bound']))
            mean_width = np.mean(predictions['upper_bound'] - predictions['lower_bound'])
            
            method_name = predictions.get('method', f'方法{i+1}')
            ax.set_title(f"{method_name} - 覆盖率: {coverage:.3f}, 平均宽度: {mean_width:.3f}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel('值')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_coverage_analysis(self,
                             evaluation_results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制覆盖率分析
        
        参数:
            evaluation_results: 评估结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('预测区间覆盖率分析', fontsize=16, fontweight='bold')
        
        # 覆盖率对比
        ax1 = axes[0, 0]
        coverage_data = evaluation_results['coverage']
        
        categories = ['实际覆盖率', '目标覆盖率']
        values = [coverage_data['coverage'], coverage_data['target_coverage']]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(categories, values, color=colors)
        ax1.set_ylabel('覆盖率')
        ax1.set_title('覆盖率对比')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 条件覆盖率
        ax2 = axes[0, 1]
        if 'conditional_coverage' in coverage_data:
            cond_coverage = coverage_data['conditional_coverage']
            bins = range(1, len(cond_coverage) + 1)
            
            ax2.plot(bins, cond_coverage, 'o-', linewidth=2, markersize=8)
            ax2.axhline(y=coverage_data['target_coverage'], color='red', 
                       linestyle='--', label='目标覆盖率')
            ax2.set_xlabel('时间段')
            ax2.set_ylabel('覆盖率')
            ax2.set_title('条件覆盖率')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 区间宽度分布
        ax3 = axes[1, 0]
        sharpness_data = evaluation_results['sharpness']
        
        width_stats = [
            sharpness_data['min_width'],
            sharpness_data['width_percentiles']['25%'],
            sharpness_data['median_width'],
            sharpness_data['width_percentiles']['75%'],
            sharpness_data['max_width']
        ]
        
        labels = ['最小值', '25%', '中位数', '75%', '最大值']
        ax3.bar(labels, width_stats, color='lightgreen')
        ax3.set_ylabel('区间宽度')
        ax3.set_title('区间宽度分布')
        ax3.tick_params(axis='x', rotation=45)
        
        # 校准性分析
        ax4 = axes[1, 1]
        if 'calibration' in evaluation_results:
            calib_data = evaluation_results['calibration']
            
            ax4.bar(range(len(calib_data['observed_frequencies'])), 
                   calib_data['observed_frequencies'], 
                   alpha=0.7, label='观测频率')
            ax4.bar(range(len(calib_data['expected_frequencies'])), 
                   calib_data['expected_frequencies'], 
                   alpha=0.7, label='期望频率')
            
            ax4.set_xlabel('分箱')
            ax4.set_ylabel('频率')
            ax4.set_title(f"校准性分析 (p={calib_data['p_value']:.3f})")
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """
    主函数 - 演示预测区间分析功能
    """
    # 创建示例数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成非线性数据
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    true_function = lambda x: 2 * np.sin(x) + 0.5 * x
    noise_std = 0.3
    
    y = true_function(X.ravel()) + np.random.normal(0, noise_std, n_samples)
    
    # 分割数据
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    
    # 初始化预测区间计算器
    pi_calculator = PredictionIntervalCalculator(confidence_level=0.95)
    evaluator = PredictionIntervalEvaluator()
    visualizer = PredictionIntervalVisualizer()
    
    print("\n开始计算预测区间...")
    
    # 1. Bootstrap预测区间
    print("\n1. Bootstrap预测区间")
    from sklearn.ensemble import RandomForestRegressor
    
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    bootstrap_results = pi_calculator.bootstrap_prediction_interval(
        rf_model, X_train, y_train, X_test, n_bootstrap=100
    )
    print(f"Bootstrap方法 - 平均区间宽度: {np.mean(bootstrap_results['interval_width']):.3f}")
    
    # 2. 分位数回归预测区间
    print("\n2. 分位数回归预测区间")
    quantile_results = pi_calculator.quantile_regression_interval(
        X_train, y_train, X_test
    )
    print(f"分位数回归 - 平均区间宽度: {np.mean(quantile_results['interval_width']):.3f}")
    
    # 3. 集成模型预测区间
    print("\n3. 集成模型预测区间")
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    
    models = [
        RandomForestRegressor(n_estimators=50, random_state=42),
        GradientBoostingRegressor(n_estimators=50, random_state=42),
        LinearRegression()
    ]
    
    ensemble_results = pi_calculator.ensemble_prediction_interval(
        models, X_train, y_train, X_test, method='simple'
    )
    print(f"集成方法 - 平均区间宽度: {np.mean(ensemble_results['interval_width']):.3f}")
    
    # 4. 贝叶斯预测区间
    print("\n4. 贝叶斯预测区间")
    bayesian_results = pi_calculator.bayesian_prediction_interval(
        None, X_train, y_train, X_test
    )
    print(f"贝叶斯方法 - 平均区间宽度: {np.mean(bayesian_results['interval_width']):.3f}")
    
    # 评估所有方法
    print("\n\n=== 预测区间评估 ===")
    
    all_results = [
        bootstrap_results,
        quantile_results,
        ensemble_results,
        bayesian_results
    ]
    
    for i, results in enumerate(all_results):
        method_name = results['method']
        
        # 综合评估
        evaluation = evaluator.comprehensive_evaluation(
            y_test, results, pi_calculator.confidence_level
        )
        
        print(f"\n{method_name.upper()}方法:")
        print(f"  覆盖率: {evaluation['coverage']['coverage']:.3f}")
        print(f"  平均宽度: {evaluation['sharpness']['mean_width']:.3f}")
        print(f"  校准性: {'良好' if evaluation['calibration']['is_well_calibrated'] else '需改进'}")
        print(f"  综合得分: {evaluation['overall_score']:.3f}")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 单个方法可视化
    fig1 = visualizer.plot_prediction_intervals(
        X_test, y_test, bootstrap_results, "Bootstrap预测区间"
    )
    
    # 方法比较
    fig2 = visualizer.plot_interval_comparison(
        X_test, y_test, all_results
    )
    
    # 覆盖率分析
    bootstrap_evaluation = evaluator.comprehensive_evaluation(
        y_test, bootstrap_results, pi_calculator.confidence_level
    )
    fig3 = visualizer.plot_coverage_analysis(bootstrap_evaluation)
    
    plt.show()
    
    print("\n预测区间分析完成！")

if __name__ == "__main__":
    main()