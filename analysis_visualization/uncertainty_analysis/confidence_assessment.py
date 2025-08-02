#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
置信度评估模块

本模块提供冰川PINNs模型置信度评估功能，包括：
- 预测置信度计算
- 模型不确定性量化
- 置信区间估计
- 可靠性评估
- 校准分析
- 置信度可视化

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceAssessor:
    """
    置信度评估器
    
    提供多种置信度评估方法，用于量化模型预测的可靠性。
    """
    
    def __init__(self, model: Optional[Callable] = None):
        """
        初始化置信度评估器
        
        参数:
            model: 待评估的模型（可调用对象）
        """
        self.model = model
        self.scaler = StandardScaler()
        self.calibration_model = None
        
    def prediction_confidence(self,
                            X: np.ndarray,
                            method: str = 'ensemble',
                            n_bootstrap: int = 100,
                            confidence_level: float = 0.95) -> Dict:
        """
        预测置信度计算
        
        参数:
            X: 输入数据
            method: 置信度计算方法 ('ensemble', 'bootstrap', 'bayesian', 'dropout')
            n_bootstrap: Bootstrap采样次数
            confidence_level: 置信水平
            
        返回:
            预测置信度结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行置信度评估")
            
            if method == 'ensemble':
                return self._ensemble_confidence(X, confidence_level)
            elif method == 'bootstrap':
                return self._bootstrap_confidence(X, n_bootstrap, confidence_level)
            elif method == 'bayesian':
                return self._bayesian_confidence(X, confidence_level)
            elif method == 'dropout':
                return self._dropout_confidence(X, confidence_level)
            else:
                raise ValueError(f"不支持的置信度计算方法: {method}")
                
        except Exception as e:
            logger.error(f"预测置信度计算失败: {e}")
            raise
    
    def model_uncertainty_quantification(self,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_test: np.ndarray,
                                       uncertainty_types: List[str] = ['aleatoric', 'epistemic']) -> Dict:
        """
        模型不确定性量化
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            uncertainty_types: 不确定性类型列表
            
        返回:
            不确定性量化结果
        """
        try:
            results = {
                'uncertainty_types': uncertainty_types,
                'X_test_shape': X_test.shape
            }
            
            # 偶然不确定性（数据噪声）
            if 'aleatoric' in uncertainty_types:
                logger.info("计算偶然不确定性...")
                aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(
                    X_train, y_train, X_test
                )
                results['aleatoric'] = aleatoric_uncertainty
            
            # 认知不确定性（模型不确定性）
            if 'epistemic' in uncertainty_types:
                logger.info("计算认知不确定性...")
                epistemic_uncertainty = self._calculate_epistemic_uncertainty(
                    X_train, y_train, X_test
                )
                results['epistemic'] = epistemic_uncertainty
            
            # 总不确定性
            if 'aleatoric' in results and 'epistemic' in results:
                total_uncertainty = np.sqrt(
                    results['aleatoric']['variance'] + 
                    results['epistemic']['variance']
                )
                results['total'] = {
                    'uncertainty': total_uncertainty,
                    'variance': results['aleatoric']['variance'] + results['epistemic']['variance']
                }
            
            return results
            
        except Exception as e:
            logger.error(f"模型不确定性量化失败: {e}")
            raise
    
    def confidence_interval_estimation(self,
                                     X: np.ndarray,
                                     y_true: Optional[np.ndarray] = None,
                                     method: str = 'bootstrap',
                                     confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> Dict:
        """
        置信区间估计
        
        参数:
            X: 输入数据
            y_true: 真实值（可选，用于评估）
            method: 估计方法 ('bootstrap', 'analytical', 'quantile')
            confidence_levels: 置信水平列表
            
        返回:
            置信区间估计结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行置信区间估计")
            
            results = {
                'method': method,
                'confidence_levels': confidence_levels,
                'n_samples': len(X)
            }
            
            # 获取预测值
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X)
            else:
                y_pred = np.array([self.model(x) for x in X])
            
            results['predictions'] = y_pred
            
            # 计算置信区间
            confidence_intervals = {}
            
            for conf_level in confidence_levels:
                if method == 'bootstrap':
                    ci = self._bootstrap_confidence_interval(X, conf_level)
                elif method == 'analytical':
                    ci = self._analytical_confidence_interval(X, conf_level)
                elif method == 'quantile':
                    ci = self._quantile_confidence_interval(X, conf_level)
                else:
                    raise ValueError(f"不支持的置信区间估计方法: {method}")
                
                confidence_intervals[conf_level] = ci
            
            results['confidence_intervals'] = confidence_intervals
            
            # 如果提供了真实值，计算覆盖率
            if y_true is not None:
                coverage_rates = {}
                for conf_level, ci in confidence_intervals.items():
                    coverage = np.mean(
                        (y_true >= ci['lower']) & (y_true <= ci['upper'])
                    )
                    coverage_rates[conf_level] = coverage
                
                results['coverage_rates'] = coverage_rates
                results['y_true'] = y_true
            
            return results
            
        except Exception as e:
            logger.error(f"置信区间估计失败: {e}")
            raise
    
    def reliability_assessment(self,
                             X: np.ndarray,
                             y_true: np.ndarray,
                             confidence_scores: Optional[np.ndarray] = None) -> Dict:
        """
        可靠性评估
        
        参数:
            X: 输入数据
            y_true: 真实值
            confidence_scores: 置信度分数（可选）
            
        返回:
            可靠性评估结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行可靠性评估")
            
            # 获取预测值
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X)
            else:
                y_pred = np.array([self.model(x) for x in X])
            
            # 计算预测误差
            errors = np.abs(y_pred - y_true)
            squared_errors = (y_pred - y_true) ** 2
            
            # 基本可靠性指标
            reliability_metrics = {
                'mae': np.mean(errors),
                'rmse': np.sqrt(np.mean(squared_errors)),
                'mape': np.mean(np.abs(errors / (y_true + 1e-8))) * 100,
                'r2': 1 - np.sum(squared_errors) / np.sum((y_true - np.mean(y_true)) ** 2)
            }
            
            # 如果没有提供置信度分数，计算默认置信度
            if confidence_scores is None:
                confidence_scores = self._calculate_default_confidence(X)
            
            # 置信度-准确性关系
            conf_accuracy_relation = self._analyze_confidence_accuracy_relation(
                confidence_scores, errors
            )
            
            # 可靠性分层分析
            reliability_stratified = self._stratified_reliability_analysis(
                confidence_scores, errors, n_bins=10
            )
            
            # 异常值检测
            outliers = self._detect_prediction_outliers(y_pred, y_true, errors)
            
            return {
                'basic_metrics': reliability_metrics,
                'confidence_accuracy_relation': conf_accuracy_relation,
                'stratified_analysis': reliability_stratified,
                'outliers': outliers,
                'predictions': y_pred,
                'true_values': y_true,
                'errors': errors,
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            logger.error(f"可靠性评估失败: {e}")
            raise
    
    def calibration_analysis(self,
                           X: np.ndarray,
                           y_true: np.ndarray,
                           confidence_scores: Optional[np.ndarray] = None,
                           n_bins: int = 10) -> Dict:
        """
        校准分析
        
        参数:
            X: 输入数据
            y_true: 真实值
            confidence_scores: 置信度分数
            n_bins: 分箱数量
            
        返回:
            校准分析结果
        """
        try:
            if self.model is None:
                raise ValueError("需要提供模型进行校准分析")
            
            # 获取预测值
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X)
            else:
                y_pred = np.array([self.model(x) for x in X])
            
            # 如果没有提供置信度分数，计算默认置信度
            if confidence_scores is None:
                confidence_scores = self._calculate_default_confidence(X)
            
            # 校准曲线
            calibration_curve_data = self._compute_calibration_curve(
                confidence_scores, y_pred, y_true, n_bins
            )
            
            # 校准指标
            calibration_metrics = self._compute_calibration_metrics(
                confidence_scores, y_pred, y_true
            )
            
            # 可靠性图
            reliability_diagram = self._create_reliability_diagram(
                confidence_scores, y_pred, y_true, n_bins
            )
            
            # 校准后的置信度
            calibrated_confidence = self._calibrate_confidence(
                confidence_scores, y_pred, y_true
            )
            
            return {
                'calibration_curve': calibration_curve_data,
                'calibration_metrics': calibration_metrics,
                'reliability_diagram': reliability_diagram,
                'calibrated_confidence': calibrated_confidence,
                'original_confidence': confidence_scores,
                'predictions': y_pred,
                'true_values': y_true
            }
            
        except Exception as e:
            logger.error(f"校准分析失败: {e}")
            raise
    
    def comprehensive_confidence_assessment(self,
                                          X_train: np.ndarray,
                                          y_train: np.ndarray,
                                          X_test: np.ndarray,
                                          y_test: np.ndarray) -> Dict:
        """
        综合置信度评估
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            
        返回:
            综合置信度评估结果
        """
        try:
            logger.info("开始综合置信度评估...")
            
            results = {
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            # 1. 预测置信度
            logger.info("计算预测置信度...")
            pred_confidence = self.prediction_confidence(
                X_test, method='bootstrap', n_bootstrap=50
            )
            results['prediction_confidence'] = pred_confidence
            
            # 2. 不确定性量化
            logger.info("量化模型不确定性...")
            uncertainty_quantification = self.model_uncertainty_quantification(
                X_train, y_train, X_test
            )
            results['uncertainty_quantification'] = uncertainty_quantification
            
            # 3. 置信区间估计
            logger.info("估计置信区间...")
            confidence_intervals = self.confidence_interval_estimation(
                X_test, y_test, method='bootstrap'
            )
            results['confidence_intervals'] = confidence_intervals
            
            # 4. 可靠性评估
            logger.info("评估模型可靠性...")
            reliability = self.reliability_assessment(
                X_test, y_test, pred_confidence.get('confidence_scores')
            )
            results['reliability'] = reliability
            
            # 5. 校准分析
            logger.info("进行校准分析...")
            calibration = self.calibration_analysis(
                X_test, y_test, pred_confidence.get('confidence_scores')
            )
            results['calibration'] = calibration
            
            # 6. 综合评分
            logger.info("计算综合评分...")
            overall_score = self._calculate_overall_confidence_score(results)
            results['overall_score'] = overall_score
            
            return results
            
        except Exception as e:
            logger.error(f"综合置信度评估失败: {e}")
            raise
    
    def _ensemble_confidence(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        集成方法置信度计算
        """
        # 简化实现：使用多个随机初始化的模型
        n_models = 10
        predictions = []
        
        for i in range(n_models):
            # 这里应该是不同的模型实例
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(X)
            else:
                pred = np.array([self.model(x) for x in X])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        # 置信度分数（基于预测方差）
        confidence_scores = 1 / (1 + std_pred)
        
        return {
            'method': 'ensemble',
            'predictions': mean_pred,
            'std': std_pred,
            'confidence_scores': confidence_scores,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    
    def _bootstrap_confidence(self, X: np.ndarray, n_bootstrap: int, confidence_level: float) -> Dict:
        """
        Bootstrap方法置信度计算
        """
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap采样（这里简化为添加噪声）
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(X)
            else:
                pred = np.array([self.model(x) for x in X])
            
            # 添加小量噪声模拟Bootstrap效果
            noise = np.random.normal(0, 0.01 * np.std(pred), len(pred))
            predictions.append(pred + noise)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        # 置信度分数
        confidence_scores = 1 / (1 + std_pred)
        
        return {
            'method': 'bootstrap',
            'predictions': mean_pred,
            'std': std_pred,
            'confidence_scores': confidence_scores,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
    
    def _bayesian_confidence(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        贝叶斯方法置信度计算
        """
        # 简化的贝叶斯实现
        if hasattr(self.model, 'predict'):
            mean_pred = self.model.predict(X)
        else:
            mean_pred = np.array([self.model(x) for x in X])
        
        # 假设先验不确定性
        prior_std = 0.1 * np.std(mean_pred)
        
        # 后验不确定性（简化）
        posterior_std = np.full_like(mean_pred, prior_std)
        
        # 置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_pred - z_score * posterior_std
        upper_bound = mean_pred + z_score * posterior_std
        
        # 置信度分数
        confidence_scores = 1 / (1 + posterior_std)
        
        return {
            'method': 'bayesian',
            'predictions': mean_pred,
            'std': posterior_std,
            'confidence_scores': confidence_scores,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    
    def _dropout_confidence(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        Dropout方法置信度计算
        """
        # 简化实现：模拟dropout效果
        n_samples = 20
        predictions = []
        
        for i in range(n_samples):
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(X)
            else:
                pred = np.array([self.model(x) for x in X])
            
            # 模拟dropout效果（随机置零部分预测）
            dropout_mask = np.random.random(len(pred)) > 0.1
            pred = pred * dropout_mask
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        # 置信度分数
        confidence_scores = 1 / (1 + std_pred)
        
        return {
            'method': 'dropout',
            'predictions': mean_pred,
            'std': std_pred,
            'confidence_scores': confidence_scores,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'n_samples': n_samples
        }
    
    def _calculate_aleatoric_uncertainty(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """
        计算偶然不确定性（数据噪声）
        """
        # 使用残差估计数据噪声
        if hasattr(self.model, 'predict'):
            y_train_pred = self.model.predict(X_train)
        else:
            y_train_pred = np.array([self.model(x) for x in X_train])
        
        residuals = y_train - y_train_pred
        noise_variance = np.var(residuals)
        
        # 对测试数据，假设相同的噪声水平
        aleatoric_variance = np.full(len(X_test), noise_variance)
        aleatoric_uncertainty = np.sqrt(aleatoric_variance)
        
        return {
            'uncertainty': aleatoric_uncertainty,
            'variance': aleatoric_variance,
            'noise_variance': noise_variance,
            'type': 'aleatoric'
        }
    
    def _calculate_epistemic_uncertainty(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """
        计算认知不确定性（模型不确定性）
        """
        # 使用交叉验证估计模型不确定性
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        predictions = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练模型（这里简化为使用原模型）
            if hasattr(self.model, 'predict'):
                fold_pred = self.model.predict(X_test)
            else:
                fold_pred = np.array([self.model(x) for x in X_test])
            
            predictions.append(fold_pred)
        
        predictions = np.array(predictions)
        
        # 计算模型间的方差
        epistemic_variance = np.var(predictions, axis=0)
        epistemic_uncertainty = np.sqrt(epistemic_variance)
        
        return {
            'uncertainty': epistemic_uncertainty,
            'variance': epistemic_variance,
            'type': 'epistemic'
        }
    
    def _bootstrap_confidence_interval(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        Bootstrap置信区间
        """
        n_bootstrap = 100
        predictions = []
        
        for i in range(n_bootstrap):
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(X)
            else:
                pred = np.array([self.model(x) for x in X])
            
            # 添加Bootstrap噪声
            noise = np.random.normal(0, 0.01 * np.std(pred), len(pred))
            predictions.append(pred + noise)
        
        predictions = np.array(predictions)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': upper_bound - lower_bound
        }
    
    def _analytical_confidence_interval(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        解析置信区间
        """
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
        else:
            predictions = np.array([self.model(x) for x in X])
        
        # 假设预测误差的标准差
        pred_std = 0.1 * np.std(predictions)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin_of_error = z_score * pred_std
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': 2 * margin_of_error
        }
    
    def _quantile_confidence_interval(self, X: np.ndarray, confidence_level: float) -> Dict:
        """
        分位数置信区间
        """
        # 使用分位数回归的简化实现
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
        else:
            predictions = np.array([self.model(x) for x in X])
        
        # 假设预测分布
        alpha = 1 - confidence_level
        
        # 使用正态分布近似
        pred_std = 0.1 * np.std(predictions)
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bound = predictions + stats.norm.ppf(lower_quantile) * pred_std
        upper_bound = predictions + stats.norm.ppf(upper_quantile) * pred_std
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'width': upper_bound - lower_bound
        }
    
    def _calculate_default_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        计算默认置信度分数
        """
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X)
        else:
            predictions = np.array([self.model(x) for x in X])
        
        # 基于预测值的变异性计算置信度
        pred_std = np.std(predictions)
        confidence_scores = 1 / (1 + np.abs(predictions - np.mean(predictions)) / pred_std)
        
        return confidence_scores
    
    def _analyze_confidence_accuracy_relation(self, confidence_scores: np.ndarray, errors: np.ndarray) -> Dict:
        """
        分析置信度与准确性的关系
        """
        # 计算相关性
        correlation, p_value = stats.pearsonr(confidence_scores, -errors)  # 负号因为高置信度应对应低误差
        
        # 分箱分析
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidence_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_confidence = []
        bin_accuracy = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence.append(np.mean(confidence_scores[mask]))
                bin_accuracy.append(np.mean(errors[mask]))
            else:
                bin_confidence.append(np.nan)
                bin_accuracy.append(np.nan)
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'bin_confidence': np.array(bin_confidence),
            'bin_accuracy': np.array(bin_accuracy),
            'bin_edges': bin_edges
        }
    
    def _stratified_reliability_analysis(self, confidence_scores: np.ndarray, errors: np.ndarray, n_bins: int = 10) -> Dict:
        """
        分层可靠性分析
        """
        # 按置信度分层
        confidence_percentiles = np.percentile(confidence_scores, np.linspace(0, 100, n_bins + 1))
        
        strata_results = []
        
        for i in range(n_bins):
            if i == 0:
                mask = confidence_scores <= confidence_percentiles[i + 1]
            elif i == n_bins - 1:
                mask = confidence_scores > confidence_percentiles[i]
            else:
                mask = (confidence_scores > confidence_percentiles[i]) & (confidence_scores <= confidence_percentiles[i + 1])
            
            if np.sum(mask) > 0:
                stratum_errors = errors[mask]
                stratum_confidence = confidence_scores[mask]
                
                strata_results.append({
                    'stratum': i,
                    'confidence_range': (confidence_percentiles[i], confidence_percentiles[i + 1]),
                    'n_samples': np.sum(mask),
                    'mean_confidence': np.mean(stratum_confidence),
                    'mean_error': np.mean(stratum_errors),
                    'std_error': np.std(stratum_errors),
                    'median_error': np.median(stratum_errors)
                })
        
        return {
            'strata': strata_results,
            'n_bins': n_bins
        }
    
    def _detect_prediction_outliers(self, y_pred: np.ndarray, y_true: np.ndarray, errors: np.ndarray) -> Dict:
        """
        检测预测异常值
        """
        # 使用IQR方法检测异常值
        q1 = np.percentile(errors, 25)
        q3 = np.percentile(errors, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (errors < lower_bound) | (errors > upper_bound)
        
        return {
            'outlier_indices': np.where(outlier_mask)[0],
            'outlier_errors': errors[outlier_mask],
            'outlier_predictions': y_pred[outlier_mask],
            'outlier_true_values': y_true[outlier_mask],
            'n_outliers': np.sum(outlier_mask),
            'outlier_percentage': np.mean(outlier_mask) * 100,
            'error_bounds': (lower_bound, upper_bound)
        }
    
    def _compute_calibration_curve(self, confidence_scores: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, n_bins: int) -> Dict:
        """
        计算校准曲线
        """
        # 将置信度转换为准确性指标
        accuracy_scores = 1 - np.abs(y_pred - y_true) / (np.std(y_true) + 1e-8)
        accuracy_scores = np.clip(accuracy_scores, 0, 1)
        
        # 计算校准曲线
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy_scores[in_bin].mean()
                confidence_in_bin = confidence_scores[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0
                confidence_in_bin = (bin_lower + bin_upper) / 2
                count_in_bin = 0
            
            bin_confidences.append(confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(count_in_bin)
        
        return {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }
    
    def _compute_calibration_metrics(self, confidence_scores: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        计算校准指标
        """
        # 将置信度转换为准确性指标
        accuracy_scores = 1 - np.abs(y_pred - y_true) / (np.std(y_true) + 1e-8)
        accuracy_scores = np.clip(accuracy_scores, 0, 1)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy_scores[in_bin].mean()
                confidence_in_bin = confidence_scores[in_bin].mean()
                ece += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracy_scores[in_bin].mean()
                confidence_in_bin = confidence_scores[in_bin].mean()
                mce = max(mce, np.abs(confidence_in_bin - accuracy_in_bin))
        
        # Brier Score
        brier_score = np.mean((confidence_scores - accuracy_scores) ** 2)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score
        }
    
    def _create_reliability_diagram(self, confidence_scores: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, n_bins: int) -> Dict:
        """
        创建可靠性图数据
        """
        calibration_data = self._compute_calibration_curve(confidence_scores, y_pred, y_true, n_bins)
        
        return {
            'bin_confidences': calibration_data['bin_confidences'],
            'bin_accuracies': calibration_data['bin_accuracies'],
            'bin_counts': calibration_data['bin_counts'],
            'perfect_calibration': np.linspace(0, 1, n_bins)
        }
    
    def _calibrate_confidence(self, confidence_scores: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        校准置信度分数
        """
        # 将置信度转换为准确性指标
        accuracy_scores = 1 - np.abs(y_pred - y_true) / (np.std(y_true) + 1e-8)
        accuracy_scores = np.clip(accuracy_scores, 0, 1)
        
        # 使用等渗回归进行校准
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        calibrated_scores = iso_reg.fit_transform(confidence_scores, accuracy_scores)
        
        return calibrated_scores
    
    def _calculate_overall_confidence_score(self, results: Dict) -> Dict:
        """
        计算综合置信度评分
        """
        scores = {}
        
        # 可靠性评分
        if 'reliability' in results:
            reliability_metrics = results['reliability']['basic_metrics']
            # 基于R²计算可靠性评分
            reliability_score = max(0, reliability_metrics.get('r2', 0))
            scores['reliability'] = reliability_score
        
        # 校准评分
        if 'calibration' in results:
            calibration_metrics = results['calibration']['calibration_metrics']
            # 基于ECE计算校准评分（ECE越小越好）
            ece = calibration_metrics.get('ece', 1)
            calibration_score = max(0, 1 - ece)
            scores['calibration'] = calibration_score
        
        # 置信区间覆盖率评分
        if 'confidence_intervals' in results and 'coverage_rates' in results['confidence_intervals']:
            coverage_rates = results['confidence_intervals']['coverage_rates']
            # 使用95%置信区间的覆盖率
            coverage_score = coverage_rates.get(0.95, 0)
            scores['coverage'] = coverage_score
        
        # 不确定性量化评分
        if 'uncertainty_quantification' in results:
            # 基于不确定性的合理性评分（简化）
            uncertainty_score = 0.8  # 默认评分
            scores['uncertainty'] = uncertainty_score
        
        # 计算综合评分
        if scores:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = 0.5  # 默认中等评分
        
        return {
            'overall': overall_score,
            'component_scores': scores,
            'grade': self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """
        将评分转换为等级
        """
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'

class ConfidenceVisualizer:
    """
    置信度评估可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_prediction_confidence(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制预测置信度
        
        参数:
            results: 预测置信度结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'预测置信度分析 ({results["method"]})', fontsize=16, fontweight='bold')
        
        predictions = results['predictions']
        confidence_scores = results['confidence_scores']
        lower_bound = results['lower_bound']
        upper_bound = results['upper_bound']
        
        # 预测值与置信区间
        ax1 = axes[0, 0]
        x_indices = np.arange(len(predictions))
        
        ax1.plot(x_indices, predictions, 'b-', label='预测值', alpha=0.8)
        ax1.fill_between(x_indices, lower_bound, upper_bound, alpha=0.3, label='置信区间')
        
        ax1.set_xlabel('样本索引')
        ax1.set_ylabel('预测值')
        ax1.set_title('预测值与置信区间')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 置信度分数分布
        ax2 = axes[0, 1]
        ax2.hist(confidence_scores, bins=30, alpha=0.7, color='green')
        ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(confidence_scores):.3f}')
        
        ax2.set_xlabel('置信度分数')
        ax2.set_ylabel('频次')
        ax2.set_title('置信度分数分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 置信区间宽度
        ax3 = axes[1, 0]
        interval_width = upper_bound - lower_bound
        
        ax3.plot(x_indices, interval_width, 'r-', alpha=0.8)
        ax3.axhline(np.mean(interval_width), color='blue', linestyle='--',
                   label=f'平均宽度: {np.mean(interval_width):.3f}')
        
        ax3.set_xlabel('样本索引')
        ax3.set_ylabel('置信区间宽度')
        ax3.set_title('置信区间宽度变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 置信度 vs 不确定性
        ax4 = axes[1, 1]
        uncertainty = results['std']
        
        scatter = ax4.scatter(confidence_scores, uncertainty, alpha=0.6, c=predictions, cmap='viridis')
        ax4.set_xlabel('置信度分数')
        ax4.set_ylabel('预测不确定性')
        ax4.set_title('置信度 vs 不确定性')
        plt.colorbar(scatter, ax=ax4, label='预测值')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_quantification(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制不确定性量化结果
        
        参数:
            results: 不确定性量化结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('不确定性量化分析', fontsize=16, fontweight='bold')
        
        # 不确定性类型比较
        ax1 = axes[0, 0]
        
        uncertainty_types = []
        uncertainty_values = []
        
        if 'aleatoric' in results:
            uncertainty_types.append('偶然不确定性')
            uncertainty_values.append(np.mean(results['aleatoric']['uncertainty']))
        
        if 'epistemic' in results:
            uncertainty_types.append('认知不确定性')
            uncertainty_values.append(np.mean(results['epistemic']['uncertainty']))
        
        if 'total' in results:
            uncertainty_types.append('总不确定性')
            uncertainty_values.append(np.mean(results['total']['uncertainty']))
        
        bars = ax1.bar(uncertainty_types, uncertainty_values, alpha=0.7, 
                      color=['blue', 'orange', 'green'][:len(uncertainty_types)])
        ax1.set_ylabel('平均不确定性')
        ax1.set_title('不确定性类型比较')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, uncertainty_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 不确定性分布
        ax2 = axes[0, 1]
        
        if 'aleatoric' in results and 'epistemic' in results:
            aleatoric_unc = results['aleatoric']['uncertainty']
            epistemic_unc = results['epistemic']['uncertainty']
            
            ax2.hist(aleatoric_unc, bins=30, alpha=0.5, label='偶然不确定性', color='blue')
            ax2.hist(epistemic_unc, bins=30, alpha=0.5, label='认知不确定性', color='orange')
            
            ax2.set_xlabel('不确定性值')
            ax2.set_ylabel('频次')
            ax2.set_title('不确定性分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 不确定性空间分布
        ax3 = axes[1, 0]
        
        if 'total' in results:
            total_uncertainty = results['total']['uncertainty']
            x_indices = np.arange(len(total_uncertainty))
            
            ax3.plot(x_indices, total_uncertainty, 'g-', alpha=0.8, label='总不确定性')
            
            if 'aleatoric' in results:
                ax3.plot(x_indices, results['aleatoric']['uncertainty'], 
                        'b--', alpha=0.6, label='偶然不确定性')
            
            if 'epistemic' in results:
                ax3.plot(x_indices, results['epistemic']['uncertainty'], 
                        'r--', alpha=0.6, label='认知不确定性')
            
            ax3.set_xlabel('样本索引')
            ax3.set_ylabel('不确定性')
            ax3.set_title('不确定性空间变化')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 不确定性相关性
        ax4 = axes[1, 1]
        
        if 'aleatoric' in results and 'epistemic' in results:
            aleatoric_unc = results['aleatoric']['uncertainty']
            epistemic_unc = results['epistemic']['uncertainty']
            
            ax4.scatter(aleatoric_unc, epistemic_unc, alpha=0.6)
            
            # 计算相关性
            correlation, p_value = stats.pearsonr(aleatoric_unc, epistemic_unc)
            ax4.text(0.05, 0.95, f'相关系数: {correlation:.3f}\np值: {p_value:.3f}',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax4.set_xlabel('偶然不确定性')
            ax4.set_ylabel('认知不确定性')
            ax4.set_title('不确定性相关性')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_reliability_assessment(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制可靠性评估结果
        
        参数:
            results: 可靠性评估结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('可靠性评估', fontsize=16, fontweight='bold')
        
        predictions = results['predictions']
        true_values = results['true_values']
        errors = results['errors']
        confidence_scores = results['confidence_scores']
        
        # 预测 vs 真实值
        ax1 = axes[0, 0]
        ax1.scatter(true_values, predictions, alpha=0.6)
        
        # 添加完美预测线
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完美预测')
        
        # 计算R²
        r2 = results['basic_metrics']['r2']
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('预测准确性')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 误差分布
        ax2 = axes[0, 1]
        ax2.hist(errors, bins=30, alpha=0.7, color='orange')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'平均误差: {np.mean(errors):.3f}')
        ax2.axvline(np.median(errors), color='blue', linestyle='--', 
                   label=f'中位误差: {np.median(errors):.3f}')
        
        ax2.set_xlabel('预测误差')
        ax2.set_ylabel('频次')
        ax2.set_title('误差分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 置信度 vs 误差
        ax3 = axes[1, 0]
        
        if 'confidence_accuracy_relation' in results:
            relation = results['confidence_accuracy_relation']
            
            ax3.scatter(confidence_scores, errors, alpha=0.6)
            
            # 添加趋势线
            z = np.polyfit(confidence_scores, errors, 1)
            p = np.poly1d(z)
            ax3.plot(confidence_scores, p(confidence_scores), "r--", alpha=0.8)
            
            # 显示相关性
            correlation = relation['correlation']
            ax3.text(0.05, 0.95, f'相关系数: {correlation:.3f}',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel('置信度分数')
        ax3.set_ylabel('预测误差')
        ax3.set_title('置信度 vs 误差')
        ax3.grid(True, alpha=0.3)
        
        # 分层可靠性
        ax4 = axes[1, 1]
        
        if 'stratified_analysis' in results:
            strata = results['stratified_analysis']['strata']
            
            stratum_indices = [s['stratum'] for s in strata]
            mean_errors = [s['mean_error'] for s in strata]
            mean_confidences = [s['mean_confidence'] for s in strata]
            
            bars = ax4.bar(stratum_indices, mean_errors, alpha=0.7, color='purple')
            
            # 添加置信度标签
            for i, (bar, conf) in enumerate(zip(bars, mean_confidences)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax4.set_xlabel('置信度分层')
            ax4.set_ylabel('平均误差')
            ax4.set_title('分层可靠性分析')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_analysis(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制校准分析结果
        
        参数:
            results: 校准分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('校准分析', fontsize=16, fontweight='bold')
        
        # 可靠性图
        ax1 = axes[0, 0]
        
        if 'reliability_diagram' in results:
            reliability = results['reliability_diagram']
            
            bin_confidences = reliability['bin_confidences']
            bin_accuracies = reliability['bin_accuracies']
            bin_counts = reliability['bin_counts']
            
            # 绘制校准曲线
            ax1.plot(bin_confidences, bin_accuracies, 'bo-', label='校准曲线', markersize=8)
            
            # 绘