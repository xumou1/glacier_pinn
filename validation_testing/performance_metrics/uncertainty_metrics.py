#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不确定性指标模块

该模块实现了用于评估模型预测不确定性的各种指标，包括认知不确定性、
偶然不确定性、预测区间、置信度评估和不确定性校准等。

主要功能:
- 不确定性量化
- 预测区间评估
- 置信度校准
- 不确定性分解
- 集成不确定性
- 贝叶斯不确定性

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats
from scipy.special import erfinv
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UncertaintyType(Enum):
    """不确定性类型"""
    ALEATORIC = "aleatoric"        # 偶然不确定性(数据噪声)
    EPISTEMIC = "epistemic"        # 认知不确定性(模型不确定性)
    TOTAL = "total"                # 总不确定性
    PREDICTIVE = "predictive"      # 预测不确定性

class UncertaintyMetric(Enum):
    """不确定性指标类型"""
    # 基础不确定性指标
    PREDICTION_INTERVAL_COVERAGE = "pic"           # 预测区间覆盖率
    PREDICTION_INTERVAL_WIDTH = "piw"              # 预测区间宽度
    MEAN_PREDICTION_INTERVAL_WIDTH = "mpiw"        # 平均预测区间宽度
    
    # 校准指标
    CALIBRATION_ERROR = "calibration_error"        # 校准误差
    RELIABILITY_DIAGRAM = "reliability_diagram"    # 可靠性图
    BRIER_SCORE = "brier_score"                   # Brier分数
    
    # 信息论指标
    ENTROPY = "entropy"                           # 熵
    MUTUAL_INFORMATION = "mutual_information"     # 互信息
    EXPECTED_CALIBRATION_ERROR = "ece"            # 期望校准误差
    
    # 分布指标
    VARIANCE = "variance"                         # 方差
    STANDARD_DEVIATION = "std"                    # 标准差
    COEFFICIENT_OF_VARIATION = "cv"               # 变异系数
    
    # 集成指标
    ENSEMBLE_VARIANCE = "ensemble_variance"       # 集成方差
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"  # 集成分歧
    
    # 贝叶斯指标
    EPISTEMIC_UNCERTAINTY = "epistemic_uncertainty"  # 认知不确定性
    ALEATORIC_UNCERTAINTY = "aleatoric_uncertainty"  # 偶然不确定性
    
    # 质量指标
    UNCERTAINTY_QUALITY = "uncertainty_quality"   # 不确定性质量
    SHARPNESS = "sharpness"                       # 锐度
    RESOLUTION = "resolution"                     # 分辨率

class CalibrationMethod(Enum):
    """校准方法"""
    PLATT_SCALING = "platt"        # Platt缩放
    ISOTONIC_REGRESSION = "isotonic"  # 等渗回归
    TEMPERATURE_SCALING = "temperature"  # 温度缩放
    HISTOGRAM_BINNING = "histogram"   # 直方图分箱

class EnsembleMethod(Enum):
    """集成方法"""
    BOOTSTRAP = "bootstrap"        # Bootstrap
    BAGGING = "bagging"           # Bagging
    DROPOUT = "dropout"           # Dropout
    DEEP_ENSEMBLE = "deep_ensemble"  # 深度集成
    BAYESIAN = "bayesian"         # 贝叶斯方法

@dataclass
class UncertaintyConfig:
    """不确定性评估配置"""
    # 基本设置
    metrics: List[UncertaintyMetric] = field(default_factory=lambda: [
        UncertaintyMetric.PREDICTION_INTERVAL_COVERAGE,
        UncertaintyMetric.MEAN_PREDICTION_INTERVAL_WIDTH,
        UncertaintyMetric.CALIBRATION_ERROR,
        UncertaintyMetric.ENTROPY
    ])
    
    # 预测区间设置
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    prediction_interval_method: str = "gaussian"  # gaussian, quantile, bootstrap
    
    # 校准设置
    calibration_method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION
    calibration_bins: int = 10
    
    # 集成设置
    ensemble_method: EnsembleMethod = EnsembleMethod.BOOTSTRAP
    ensemble_size: int = 100
    dropout_rate: float = 0.1
    
    # 贝叶斯设置
    enable_bayesian: bool = False
    mcmc_samples: int = 1000
    burn_in: int = 200
    
    # 分解设置
    enable_uncertainty_decomposition: bool = True
    decomposition_method: str = "variance"  # variance, information
    
    # 质量评估
    enable_quality_assessment: bool = True
    quality_threshold: float = 0.1
    
    # 统计设置
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./uncertainty_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

@dataclass
class UncertaintyResult:
    """不确定性评估结果"""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    uncertainty_type: Optional[UncertaintyType] = None
    confidence_level: Optional[float] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PredictionInterval:
    """预测区间计算器"""
    
    @staticmethod
    def gaussian_interval(predictions: np.ndarray, 
                         uncertainties: np.ndarray, 
                         confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """基于高斯分布的预测区间"""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        
        return lower, upper
    
    @staticmethod
    def quantile_interval(ensemble_predictions: np.ndarray, 
                         confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """基于分位数的预测区间"""
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower = np.quantile(ensemble_predictions, lower_quantile, axis=0)
        upper = np.quantile(ensemble_predictions, upper_quantile, axis=0)
        
        return lower, upper
    
    @staticmethod
    def bootstrap_interval(y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          confidence_level: float = 0.95,
                          n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """基于Bootstrap的预测区间"""
        residuals = y_true - y_pred
        n_samples = len(residuals)
        
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # 重采样残差
            bootstrap_residuals = np.random.choice(residuals, n_samples, replace=True)
            bootstrap_pred = y_pred + bootstrap_residuals
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        alpha = 1 - confidence_level
        lower = np.quantile(bootstrap_predictions, alpha / 2, axis=0)
        upper = np.quantile(bootstrap_predictions, 1 - alpha / 2, axis=0)
        
        return lower, upper

class CalibrationAnalyzer:
    """校准分析器"""
    
    def __init__(self, method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION):
        self.method = method
        self.calibrator = None
    
    def fit_calibration(self, probabilities: np.ndarray, y_true: np.ndarray):
        """拟合校准模型"""
        if self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probabilities, y_true)
        elif self.method == CalibrationMethod.PLATT_SCALING:
            # 简化的Platt缩放实现
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(probabilities.reshape(-1, 1), y_true)
    
    def calibrate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率"""
        if self.calibrator is None:
            raise ValueError("校准器未拟合")
        
        if self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            return self.calibrator.predict(probabilities)
        elif self.method == CalibrationMethod.PLATT_SCALING:
            return self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        
        return probabilities
    
    def expected_calibration_error(self, probabilities: np.ndarray, 
                                  y_true: np.ndarray, 
                                  n_bins: int = 10) -> float:
        """期望校准误差"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(probabilities)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在当前bin中的样本
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def reliability_diagram_data(self, probabilities: np.ndarray, 
                               y_true: np.ndarray, 
                               n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """可靠性图数据"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_true[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)

class EnsembleUncertainty:
    """集成不确定性分析器"""
    
    def __init__(self, method: EnsembleMethod = EnsembleMethod.BOOTSTRAP):
        self.method = method
    
    def compute_ensemble_uncertainty(self, 
                                   ensemble_predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """计算集成不确定性"""
        # ensemble_predictions shape: (n_models, n_samples)
        
        # 平均预测
        mean_prediction = np.mean(ensemble_predictions, axis=0)
        
        # 总方差(认知不确定性)
        epistemic_uncertainty = np.var(ensemble_predictions, axis=0)
        
        # 平均方差(偶然不确定性) - 如果有的话
        # 这里简化处理，假设偶然不确定性为0
        aleatoric_uncertainty = np.zeros_like(mean_prediction)
        
        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'mean_prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'ensemble_std': np.std(ensemble_predictions, axis=0)
        }
    
    def ensemble_disagreement(self, ensemble_predictions: np.ndarray) -> float:
        """集成分歧度"""
        # 计算所有模型对之间的平均分歧
        n_models = ensemble_predictions.shape[0]
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean((ensemble_predictions[i] - ensemble_predictions[j]) ** 2)
                disagreements.append(disagreement)
        
        return np.mean(disagreements)
    
    def prediction_diversity(self, ensemble_predictions: np.ndarray) -> float:
        """预测多样性"""
        mean_prediction = np.mean(ensemble_predictions, axis=0)
        diversity = np.mean(np.var(ensemble_predictions - mean_prediction[None, :], axis=0))
        return diversity

class BayesianUncertainty:
    """贝叶斯不确定性分析器"""
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
    
    def monte_carlo_dropout_uncertainty(self, 
                                      model: nn.Module, 
                                      x: torch.Tensor, 
                                      n_samples: int = 100,
                                      dropout_rate: float = 0.1) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout不确定性"""
        model.train()  # 启用dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # 前向传播时保持dropout激活
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # 计算统计量
        mean_pred = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        return {
            'mean_prediction': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictions': predictions
        }
    
    def variational_uncertainty(self, 
                              mean: torch.Tensor, 
                              log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """变分不确定性"""
        # 从变分分布采样
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        samples = mean + eps * std
        
        return {
            'mean_prediction': mean,
            'aleatoric_uncertainty': torch.exp(log_var),
            'samples': samples
        }

class UncertaintyDecomposer:
    """不确定性分解器"""
    
    def variance_decomposition(self, 
                             ensemble_predictions: np.ndarray,
                             individual_uncertainties: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """方差分解"""
        # 认知不确定性(模型间方差)
        epistemic = np.var(ensemble_predictions, axis=0)
        
        # 偶然不确定性(模型内方差的平均)
        if individual_uncertainties is not None:
            aleatoric = np.mean(individual_uncertainties, axis=0)
        else:
            # 如果没有个体不确定性，假设为0
            aleatoric = np.zeros_like(epistemic)
        
        # 总不确定性
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }
    
    def information_decomposition(self, 
                                ensemble_predictions: np.ndarray) -> Dict[str, float]:
        """信息论分解"""
        # 计算熵
        def entropy(predictions):
            # 简化的熵计算
            hist, _ = np.histogram(predictions, bins=50, density=True)
            hist = hist[hist > 0]  # 移除零值
            return -np.sum(hist * np.log(hist))
        
        # 总熵
        all_predictions = ensemble_predictions.flatten()
        total_entropy = entropy(all_predictions)
        
        # 平均个体熵
        individual_entropies = []
        for i in range(ensemble_predictions.shape[0]):
            individual_entropies.append(entropy(ensemble_predictions[i]))
        
        mean_individual_entropy = np.mean(individual_entropies)
        
        # 互信息(认知不确定性的信息论度量)
        mutual_information = total_entropy - mean_individual_entropy
        
        return {
            'total_entropy': total_entropy,
            'mean_individual_entropy': mean_individual_entropy,
            'mutual_information': mutual_information
        }

class UncertaintyAnalyzer:
    """不确定性分析器"""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.calibration_analyzer = CalibrationAnalyzer(config.calibration_method)
        self.ensemble_analyzer = EnsembleUncertainty(config.ensemble_method)
        self.bayesian_analyzer = BayesianUncertainty(config.mcmc_samples)
        self.decomposer = UncertaintyDecomposer()
        
        # 设置随机种子
        np.random.seed(config.random_seed)
    
    def analyze_uncertainty(self, 
                          y_true: np.ndarray,
                          predictions: Union[np.ndarray, Dict[str, np.ndarray]],
                          uncertainties: Optional[np.ndarray] = None,
                          ensemble_predictions: Optional[np.ndarray] = None,
                          probabilities: Optional[np.ndarray] = None) -> Dict[str, UncertaintyResult]:
        """分析不确定性"""
        try:
            results = {}
            
            # 处理输入
            if isinstance(predictions, dict):
                y_pred = predictions.get('mean', predictions.get('prediction'))
            else:
                y_pred = predictions
            
            # 基础不确定性指标
            if uncertainties is not None:
                basic_results = self._compute_basic_uncertainty_metrics(
                    y_true, y_pred, uncertainties
                )
                results.update(basic_results)
            
            # 预测区间分析
            if uncertainties is not None or ensemble_predictions is not None:
                interval_results = self._compute_prediction_intervals(
                    y_true, y_pred, uncertainties, ensemble_predictions
                )
                results.update(interval_results)
            
            # 校准分析
            if probabilities is not None:
                calibration_results = self._compute_calibration_metrics(
                    y_true, probabilities
                )
                results.update(calibration_results)
            
            # 集成不确定性分析
            if ensemble_predictions is not None:
                ensemble_results = self._compute_ensemble_metrics(
                    y_true, ensemble_predictions
                )
                results.update(ensemble_results)
            
            # 不确定性分解
            if self.config.enable_uncertainty_decomposition and ensemble_predictions is not None:
                decomposition_results = self._compute_uncertainty_decomposition(
                    ensemble_predictions, uncertainties
                )
                results.update(decomposition_results)
            
            # 质量评估
            if self.config.enable_quality_assessment:
                quality_results = self._compute_quality_metrics(
                    y_true, y_pred, uncertainties, ensemble_predictions
                )
                results.update(quality_results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"不确定性分析失败: {e}")
            return {}
    
    def _compute_basic_uncertainty_metrics(self, 
                                         y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         uncertainties: np.ndarray) -> Dict[str, UncertaintyResult]:
        """计算基础不确定性指标"""
        results = {}
        
        try:
            # 方差和标准差
            if UncertaintyMetric.VARIANCE in self.config.metrics:
                variance = np.mean(uncertainties ** 2)
                results['variance'] = UncertaintyResult(
                    metric_name='variance',
                    value=variance,
                    uncertainty_type=UncertaintyType.TOTAL,
                    sample_size=len(uncertainties)
                )
            
            if UncertaintyMetric.STANDARD_DEVIATION in self.config.metrics:
                std = np.mean(uncertainties)
                results['std'] = UncertaintyResult(
                    metric_name='std',
                    value=std,
                    uncertainty_type=UncertaintyType.TOTAL,
                    sample_size=len(uncertainties)
                )
            
            # 变异系数
            if UncertaintyMetric.COEFFICIENT_OF_VARIATION in self.config.metrics:
                cv = np.std(uncertainties) / np.mean(np.abs(y_pred)) if np.mean(np.abs(y_pred)) > 0 else np.inf
                results['cv'] = UncertaintyResult(
                    metric_name='cv',
                    value=cv,
                    uncertainty_type=UncertaintyType.TOTAL,
                    sample_size=len(uncertainties)
                )
            
            # 熵
            if UncertaintyMetric.ENTROPY in self.config.metrics:
                # 假设高斯分布计算熵
                entropy = 0.5 * np.log(2 * np.pi * np.e * np.mean(uncertainties ** 2))
                results['entropy'] = UncertaintyResult(
                    metric_name='entropy',
                    value=entropy,
                    uncertainty_type=UncertaintyType.TOTAL,
                    sample_size=len(uncertainties)
                )
        
        except Exception as e:
            self.logger.warning(f"基础不确定性指标计算失败: {e}")
        
        return results
    
    def _compute_prediction_intervals(self, 
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    uncertainties: Optional[np.ndarray] = None,
                                    ensemble_predictions: Optional[np.ndarray] = None) -> Dict[str, UncertaintyResult]:
        """计算预测区间指标"""
        results = {}
        
        try:
            for confidence_level in self.config.confidence_levels:
                # 计算预测区间
                if uncertainties is not None:
                    lower, upper = PredictionInterval.gaussian_interval(
                        y_pred, uncertainties, confidence_level
                    )
                elif ensemble_predictions is not None:
                    lower, upper = PredictionInterval.quantile_interval(
                        ensemble_predictions, confidence_level
                    )
                else:
                    continue
                
                # 覆盖率
                if UncertaintyMetric.PREDICTION_INTERVAL_COVERAGE in self.config.metrics:
                    coverage = np.mean((y_true >= lower) & (y_true <= upper))
                    results[f'pic_{confidence_level}'] = UncertaintyResult(
                        metric_name=f'pic_{confidence_level}',
                        value=coverage,
                        confidence_level=confidence_level,
                        sample_size=len(y_true),
                        metadata={'expected_coverage': confidence_level}
                    )
                
                # 区间宽度
                if UncertaintyMetric.PREDICTION_INTERVAL_WIDTH in self.config.metrics:
                    width = upper - lower
                    mean_width = np.mean(width)
                    results[f'piw_{confidence_level}'] = UncertaintyResult(
                        metric_name=f'piw_{confidence_level}',
                        value=mean_width,
                        confidence_level=confidence_level,
                        sample_size=len(width),
                        metadata={'width_std': np.std(width)}
                    )
        
        except Exception as e:
            self.logger.warning(f"预测区间指标计算失败: {e}")
        
        return results
    
    def _compute_calibration_metrics(self, 
                                   y_true: np.ndarray,
                                   probabilities: np.ndarray) -> Dict[str, UncertaintyResult]:
        """计算校准指标"""
        results = {}
        
        try:
            # 期望校准误差
            if UncertaintyMetric.EXPECTED_CALIBRATION_ERROR in self.config.metrics:
                ece = self.calibration_analyzer.expected_calibration_error(
                    probabilities, y_true, self.config.calibration_bins
                )
                results['ece'] = UncertaintyResult(
                    metric_name='ece',
                    value=ece,
                    sample_size=len(probabilities),
                    metadata={'n_bins': self.config.calibration_bins}
                )
            
            # Brier分数
            if UncertaintyMetric.BRIER_SCORE in self.config.metrics:
                brier_score = np.mean((probabilities - y_true) ** 2)
                results['brier_score'] = UncertaintyResult(
                    metric_name='brier_score',
                    value=brier_score,
                    sample_size=len(probabilities)
                )
        
        except Exception as e:
            self.logger.warning(f"校准指标计算失败: {e}")
        
        return results
    
    def _compute_ensemble_metrics(self, 
                                y_true: np.ndarray,
                                ensemble_predictions: np.ndarray) -> Dict[str, UncertaintyResult]:
        """计算集成指标"""
        results = {}
        
        try:
            # 集成方差
            if UncertaintyMetric.ENSEMBLE_VARIANCE in self.config.metrics:
                ensemble_var = np.mean(np.var(ensemble_predictions, axis=0))
                results['ensemble_variance'] = UncertaintyResult(
                    metric_name='ensemble_variance',
                    value=ensemble_var,
                    uncertainty_type=UncertaintyType.EPISTEMIC,
                    sample_size=ensemble_predictions.shape[1]
                )
            
            # 集成分歧
            if UncertaintyMetric.ENSEMBLE_DISAGREEMENT in self.config.metrics:
                disagreement = self.ensemble_analyzer.ensemble_disagreement(ensemble_predictions)
                results['ensemble_disagreement'] = UncertaintyResult(
                    metric_name='ensemble_disagreement',
                    value=disagreement,
                    uncertainty_type=UncertaintyType.EPISTEMIC,
                    sample_size=ensemble_predictions.shape[1]
                )
        
        except Exception as e:
            self.logger.warning(f"集成指标计算失败: {e}")
        
        return results
    
    def _compute_uncertainty_decomposition(self, 
                                         ensemble_predictions: np.ndarray,
                                         individual_uncertainties: Optional[np.ndarray] = None) -> Dict[str, UncertaintyResult]:
        """计算不确定性分解"""
        results = {}
        
        try:
            if self.config.decomposition_method == "variance":
                decomposition = self.decomposer.variance_decomposition(
                    ensemble_predictions, individual_uncertainties
                )
                
                # 认知不确定性
                if UncertaintyMetric.EPISTEMIC_UNCERTAINTY in self.config.metrics:
                    epistemic = np.mean(decomposition['epistemic'])
                    results['epistemic_uncertainty'] = UncertaintyResult(
                        metric_name='epistemic_uncertainty',
                        value=epistemic,
                        uncertainty_type=UncertaintyType.EPISTEMIC,
                        sample_size=ensemble_predictions.shape[1]
                    )
                
                # 偶然不确定性
                if UncertaintyMetric.ALEATORIC_UNCERTAINTY in self.config.metrics:
                    aleatoric = np.mean(decomposition['aleatoric'])
                    results['aleatoric_uncertainty'] = UncertaintyResult(
                        metric_name='aleatoric_uncertainty',
                        value=aleatoric,
                        uncertainty_type=UncertaintyType.ALEATORIC,
                        sample_size=ensemble_predictions.shape[1]
                    )
            
            elif self.config.decomposition_method == "information":
                decomposition = self.decomposer.information_decomposition(ensemble_predictions)
                
                # 互信息
                if UncertaintyMetric.MUTUAL_INFORMATION in self.config.metrics:
                    mi = decomposition['mutual_information']
                    results['mutual_information'] = UncertaintyResult(
                        metric_name='mutual_information',
                        value=mi,
                        uncertainty_type=UncertaintyType.EPISTEMIC,
                        sample_size=ensemble_predictions.shape[1]
                    )
        
        except Exception as e:
            self.logger.warning(f"不确定性分解计算失败: {e}")
        
        return results
    
    def _compute_quality_metrics(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               uncertainties: Optional[np.ndarray] = None,
                               ensemble_predictions: Optional[np.ndarray] = None) -> Dict[str, UncertaintyResult]:
        """计算质量指标"""
        results = {}
        
        try:
            # 锐度(不确定性的平均值)
            if UncertaintyMetric.SHARPNESS in self.config.metrics and uncertainties is not None:
                sharpness = np.mean(uncertainties)
                results['sharpness'] = UncertaintyResult(
                    metric_name='sharpness',
                    value=sharpness,
                    sample_size=len(uncertainties)
                )
            
            # 分辨率(不确定性与误差的相关性)
            if UncertaintyMetric.RESOLUTION in self.config.metrics and uncertainties is not None:
                errors = np.abs(y_true - y_pred)
                if np.std(uncertainties) > 0 and np.std(errors) > 0:
                    resolution = np.corrcoef(uncertainties, errors)[0, 1]
                else:
                    resolution = 0.0
                
                results['resolution'] = UncertaintyResult(
                    metric_name='resolution',
                    value=resolution,
                    sample_size=len(uncertainties)
                )
            
            # 不确定性质量(综合指标)
            if UncertaintyMetric.UNCERTAINTY_QUALITY in self.config.metrics:
                if uncertainties is not None:
                    errors = np.abs(y_true - y_pred)
                    # 简单的质量指标:不确定性高的地方误差也应该高
                    quality = np.corrcoef(uncertainties, errors)[0, 1] if np.std(uncertainties) > 0 and np.std(errors) > 0 else 0.0
                else:
                    quality = 0.0
                
                results['uncertainty_quality'] = UncertaintyResult(
                    metric_name='uncertainty_quality',
                    value=quality,
                    sample_size=len(y_true)
                )
        
        except Exception as e:
            self.logger.warning(f"质量指标计算失败: {e}")
        
        return results
    
    def plot_results(self, 
                    results: Dict[str, UncertaintyResult],
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    uncertainties: Optional[np.ndarray] = None,
                    ensemble_predictions: Optional[np.ndarray] = None,
                    probabilities: Optional[np.ndarray] = None):
        """绘制不确定性分析结果"""
        try:
            n_plots = 2 + (1 if uncertainties is not None else 0) + (1 if probabilities is not None else 0)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('不确定性分析结果', fontsize=16, fontweight='bold')
            
            axes = axes.flatten()
            plot_idx = 0
            
            # 预测 vs 真实值(带不确定性)
            ax = axes[plot_idx]
            plot_idx += 1
            
            if uncertainties is not None:
                # 带误差棒的散点图
                ax.errorbar(y_true, y_pred, yerr=uncertainties, fmt='o', alpha=0.6, capsize=2)
            else:
                ax.scatter(y_true, y_pred, alpha=0.6)
            
            # 1:1线
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title('预测 vs 真实值')
            ax.grid(True, alpha=0.3)
            
            # 不确定性 vs 误差
            if uncertainties is not None:
                ax = axes[plot_idx]
                plot_idx += 1
                
                errors = np.abs(y_true - y_pred)
                ax.scatter(uncertainties, errors, alpha=0.6)
                
                # 添加趋势线
                z = np.polyfit(uncertainties, errors, 1)
                p = np.poly1d(z)
                ax.plot(uncertainties, p(uncertainties), "r--", alpha=0.8)
                
                ax.set_xlabel('预测不确定性')
                ax.set_ylabel('绝对误差')
                ax.set_title('不确定性 vs 误差')
                ax.grid(True, alpha=0.3)
            
            # 校准图
            if probabilities is not None:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # 计算校准曲线
                bin_centers, bin_accuracies, bin_counts = self.calibration_analyzer.reliability_diagram_data(
                    probabilities, y_true, self.config.calibration_bins
                )
                
                # 绘制校准图
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='完美校准')
                ax.plot(bin_centers, bin_accuracies, 'o-', label='实际校准')
                
                ax.set_xlabel('预测概率')
                ax.set_ylabel('实际频率')
                ax.set_title('校准图')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 不确定性分布
            if uncertainties is not None:
                ax = axes[plot_idx]
                plot_idx += 1
                
                ax.hist(uncertainties, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(uncertainties), color='red', linestyle='--', 
                          label=f'平均值: {np.mean(uncertainties):.3f}')
                ax.set_xlabel('不确定性')
                ax.set_ylabel('频次')
                ax.set_title('不确定性分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/uncertainty_analysis.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, UncertaintyResult],
                       y_true: np.ndarray,
                       y_pred: np.ndarray) -> str:
        """生成不确定性分析报告"""
        from datetime import datetime
        
        report_lines = [
            "="*80,
            "不确定性分析报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"样本数量: {len(y_true)}",
            "",
            "1. 基础不确定性指标",
            "-"*40
        ]
        
        # 基础指标
        basic_metrics = ['variance', 'std', 'cv', 'entropy']
        for metric in basic_metrics:
            if metric in results:
                result = results[metric]
                report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 预测区间指标
        interval_metrics = [key for key in results.keys() if 'pic_' in key or 'piw_' in key]
        if interval_metrics:
            report_lines.extend([
                "",
                "2. 预测区间指标",
                "-"*40
            ])
            
            for confidence_level in self.config.confidence_levels:
                pic_key = f'pic_{confidence_level}'
                piw_key = f'piw_{confidence_level}'
                
                if pic_key in results and piw_key in results:
                    pic_result = results[pic_key]
                    piw_result = results[piw_key]
                    report_lines.append(f"置信水平 {confidence_level*100:.0f}%:")
                    report_lines.append(f"  覆盖率: {pic_result.value:.4f} (期望: {confidence_level:.4f})")
                    report_lines.append(f"  平均宽度: {piw_result.value:.4f}")
        
        # 校准指标
        calibration_metrics = ['ece', 'brier_score']
        if any(metric in results for metric in calibration_metrics):
            report_lines.extend([
                "",
                "3. 校准指标",
                "-"*40
            ])
            
            for metric in calibration_metrics:
                if metric in results:
                    result = results[metric]
                    report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 集成指标
        ensemble_metrics = ['ensemble_variance', 'ensemble_disagreement']
        if any(metric in results for metric in ensemble_metrics):
            report_lines.extend([
                "",
                "4. 集成不确定性指标",
                "-"*40
            ])
            
            for metric in ensemble_metrics:
                if metric in results:
                    result = results[metric]
                    report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 不确定性分解
        decomposition_metrics = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'mutual_information']
        if any(metric in results for metric in decomposition_metrics):
            report_lines.extend([
                "",
                "5. 不确定性分解",
                "-"*40
            ])
            
            for metric in decomposition_metrics:
                if metric in results:
                    result = results[metric]
                    report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        # 质量指标
        quality_metrics = ['sharpness', 'resolution', 'uncertainty_quality']
        if any(metric in results for metric in quality_metrics):
            report_lines.extend([
                "",
                "6. 不确定性质量指标",
                "-"*40
            ])
            
            for metric in quality_metrics:
                if metric in results:
                    result = results[metric]
                    report_lines.append(f"{metric.upper()}: {result.value:.6f}")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_uncertainty_analyzer(config: Optional[UncertaintyConfig] = None) -> UncertaintyAnalyzer:
    """创建不确定性分析器"""
    if config is None:
        config = UncertaintyConfig()
    
    return UncertaintyAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    print("开始不确定性分析测试...")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_ensemble = 10
    
    # 真实值
    y_true = np.random.normal(100, 20, n_samples)
    
    # 预测值
    y_pred = y_true + np.random.normal(0, 5, n_samples)
    
    # 不确定性
    uncertainties = np.abs(np.random.normal(5, 2, n_samples))
    
    # 集成预测
    ensemble_predictions = np.array([
        y_true + np.random.normal(0, 5, n_samples) for _ in range(n_ensemble)
    ])
    
    # 概率(用于校准分析)
    probabilities = np.random.uniform(0, 1, n_samples)
    y_binary = (y_true > np.median(y_true)).astype(int)
    
    # 创建配置
    config = UncertaintyConfig(
        metrics=[
            UncertaintyMetric.PREDICTION_INTERVAL_COVERAGE,
            UncertaintyMetric.MEAN_PREDICTION_INTERVAL_WIDTH,
            UncertaintyMetric.EXPECTED_CALIBRATION_ERROR,
            UncertaintyMetric.VARIANCE,
            UncertaintyMetric.ENTROPY,
            UncertaintyMetric.ENSEMBLE_VARIANCE,
            UncertaintyMetric.EPISTEMIC_UNCERTAINTY,
            UncertaintyMetric.UNCERTAINTY_QUALITY
        ],
        confidence_levels=[0.68, 0.95],
        enable_uncertainty_decomposition=True,
        enable_quality_assessment=True,
        enable_plotting=True,
        verbose=True
    )
    
    # 创建分析器
    analyzer = create_uncertainty_analyzer(config)
    
    # 分析不确定性
    results = analyzer.analyze_uncertainty(
        y_true=y_true,
        predictions=y_pred,
        uncertainties=uncertainties,
        ensemble_predictions=ensemble_predictions,
        probabilities=probabilities
    )
    
    # 打印结果
    print("\n不确定性分析完成！")
    print(f"计算了 {len(results)} 个指标")
    
    # 生成报告
    report = analyzer.generate_report(results, y_true, y_pred)
    print("\n" + report)
    
    # 绘制结果
    analyzer.plot_results(results, y_true, y_pred, uncertainties, ensemble_predictions, probabilities)
    
    print("\n不确定性分析测试完成！")