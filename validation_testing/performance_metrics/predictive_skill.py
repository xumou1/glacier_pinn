#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测技能评估模块

该模块实现了用于评估模型预测技能的各种指标，包括预测准确性、可靠性、
分辨率、技能得分、概率预测评估和时空预测能力等。

主要功能:
- 确定性预测技能
- 概率预测技能
- 时间序列预测技能
- 空间预测技能
- 极端事件预测技能
- 多尺度预测技能

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
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss, log_loss, roc_auc_score, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictiveSkillMetric(Enum):
    """预测技能指标类型"""
    # 确定性预测指标
    SKILL_SCORE = "skill_score"                    # 技能得分
    NASH_SUTCLIFFE = "nash_sutcliffe"              # Nash-Sutcliffe效率系数
    KLING_GUPTA = "kling_gupta"                    # Kling-Gupta效率
    INDEX_OF_AGREEMENT = "index_of_agreement"      # 一致性指数
    
    # 概率预测指标
    BRIER_SCORE = "brier_score"                    # Brier得分
    RELIABILITY = "reliability"                    # 可靠性
    RESOLUTION = "resolution"                      # 分辨率
    SHARPNESS = "sharpness"                        # 锐度
    CALIBRATION = "calibration"                    # 校准度
    
    # 时间序列预测指标
    PERSISTENCE_SKILL = "persistence_skill"        # 持续性技能
    TREND_SKILL = "trend_skill"                    # 趋势技能
    SEASONAL_SKILL = "seasonal_skill"              # 季节性技能
    FORECAST_HORIZON_SKILL = "forecast_horizon_skill"  # 预报时效技能
    
    # 空间预测指标
    SPATIAL_CORRELATION = "spatial_correlation"    # 空间相关性
    PATTERN_CORRELATION = "pattern_correlation"    # 模式相关性
    SPATIAL_SKILL_SCORE = "spatial_skill_score"    # 空间技能得分
    
    # 极端事件预测指标
    EXTREME_EVENT_SKILL = "extreme_event_skill"    # 极端事件技能
    HIT_RATE = "hit_rate"                          # 命中率
    FALSE_ALARM_RATE = "false_alarm_rate"          # 虚警率
    CRITICAL_SUCCESS_INDEX = "critical_success_index"  # 临界成功指数
    
    # 综合指标
    OVERALL_SKILL = "overall_skill"                # 总体技能
    PREDICTIVE_POWER = "predictive_power"          # 预测能力

class PredictionType(Enum):
    """预测类型"""
    DETERMINISTIC = "deterministic"                # 确定性预测
    PROBABILISTIC = "probabilistic"                # 概率预测
    ENSEMBLE = "ensemble"                          # 集合预测
    CATEGORICAL = "categorical"                    # 分类预测

class SkillScoreType(Enum):
    """技能得分类型"""
    MURPHY = "murphy"                              # Murphy技能得分
    PEIRCE = "peirce"                              # Peirce技能得分
    HEIDKE = "heidke"                              # Heidke技能得分
    EQUITABLE_THREAT = "equitable_threat"          # 公平威胁得分

class TimeScale(Enum):
    """时间尺度"""
    DAILY = "daily"                                # 日尺度
    WEEKLY = "weekly"                              # 周尺度
    MONTHLY = "monthly"                            # 月尺度
    SEASONAL = "seasonal"                          # 季节尺度
    ANNUAL = "annual"                              # 年尺度

@dataclass
class PredictiveSkillConfig:
    """预测技能评估配置"""
    # 基本设置
    metrics: List[PredictiveSkillMetric] = field(default_factory=lambda: [
        PredictiveSkillMetric.SKILL_SCORE,
        PredictiveSkillMetric.NASH_SUTCLIFFE,
        PredictiveSkillMetric.RELIABILITY,
        PredictiveSkillMetric.SPATIAL_CORRELATION
    ])
    
    prediction_type: PredictionType = PredictionType.DETERMINISTIC
    
    # 技能得分设置
    skill_score_types: List[SkillScoreType] = field(default_factory=lambda: [
        SkillScoreType.MURPHY
    ])
    reference_forecast: str = "climatology"  # 参考预报：climatology, persistence
    
    # 概率预测设置
    probability_thresholds: List[float] = field(default_factory=lambda: [
        0.1, 0.25, 0.5, 0.75, 0.9
    ])
    calibration_bins: int = 10
    
    # 时间序列设置
    time_scales: List[TimeScale] = field(default_factory=lambda: [
        TimeScale.DAILY, TimeScale.MONTHLY
    ])
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 7, 30])  # 天数
    enable_persistence_baseline: bool = True
    enable_climatology_baseline: bool = True
    
    # 空间预测设置
    spatial_scales: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])  # km
    enable_pattern_analysis: bool = True
    
    # 极端事件设置
    extreme_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 0.9,    # 90th percentile
        'low': 0.1,     # 10th percentile
        'very_high': 0.95,  # 95th percentile
        'very_low': 0.05    # 5th percentile
    })
    
    # 集合预测设置
    ensemble_size: int = 50
    ensemble_spread_threshold: float = 0.1
    
    # 统计设置
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    significance_test: bool = True
    
    # 交叉验证设置
    enable_cross_validation: bool = True
    cv_folds: int = 5
    temporal_cv: bool = True  # 时间序列交叉验证
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./predictive_skill_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

@dataclass
class PredictiveSkillResult:
    """预测技能评估结果"""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    baseline_value: Optional[float] = None
    skill_improvement: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DeterministicSkillCalculator:
    """确定性预测技能计算器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_skill_scores(self, 
                             predictions: np.ndarray,
                             observations: np.ndarray,
                             reference: Optional[np.ndarray] = None) -> Dict[str, PredictiveSkillResult]:
        """计算技能得分"""
        results = {}
        
        try:
            # 生成参考预报
            if reference is None:
                reference = self._generate_reference_forecast(observations)
            
            # Nash-Sutcliffe效率系数
            if PredictiveSkillMetric.NASH_SUTCLIFFE in self.config.metrics:
                nse = self._calculate_nash_sutcliffe(predictions, observations)
                results['nash_sutcliffe'] = PredictiveSkillResult(
                    metric_name='nash_sutcliffe',
                    value=nse,
                    metadata={'description': 'Nash-Sutcliffe效率系数'}
                )
            
            # Kling-Gupta效率
            if PredictiveSkillMetric.KLING_GUPTA in self.config.metrics:
                kge = self._calculate_kling_gupta(predictions, observations)
                results['kling_gupta'] = PredictiveSkillResult(
                    metric_name='kling_gupta',
                    value=kge,
                    metadata={'description': 'Kling-Gupta效率'}
                )
            
            # 一致性指数
            if PredictiveSkillMetric.INDEX_OF_AGREEMENT in self.config.metrics:
                ioa = self._calculate_index_of_agreement(predictions, observations)
                results['index_of_agreement'] = PredictiveSkillResult(
                    metric_name='index_of_agreement',
                    value=ioa,
                    metadata={'description': '一致性指数'}
                )
            
            # 技能得分
            if PredictiveSkillMetric.SKILL_SCORE in self.config.metrics:
                for score_type in self.config.skill_score_types:
                    skill_score = self._calculate_skill_score(
                        predictions, observations, reference, score_type
                    )
                    results[f'skill_score_{score_type.value}'] = PredictiveSkillResult(
                        metric_name=f'skill_score_{score_type.value}',
                        value=skill_score,
                        metadata={'score_type': score_type.value}
                    )
            
            return results
        
        except Exception as e:
            self.logger.error(f"确定性技能计算失败: {e}")
            return {}
    
    def _generate_reference_forecast(self, observations: np.ndarray) -> np.ndarray:
        """生成参考预报"""
        if self.config.reference_forecast == "climatology":
            # 气候态：使用历史平均
            return np.full_like(observations, np.mean(observations))
        elif self.config.reference_forecast == "persistence":
            # 持续性：使用前一时刻的值
            reference = np.zeros_like(observations)
            reference[1:] = observations[:-1]
            reference[0] = observations[0]  # 第一个值保持不变
            return reference
        else:
            return np.zeros_like(observations)
    
    def _calculate_nash_sutcliffe(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算Nash-Sutcliffe效率系数"""
        try:
            numerator = np.sum((observations - predictions) ** 2)
            denominator = np.sum((observations - np.mean(observations)) ** 2)
            
            if denominator == 0:
                return 0.0
            
            nse = 1 - (numerator / denominator)
            return nse
        
        except Exception as e:
            self.logger.warning(f"Nash-Sutcliffe计算失败: {e}")
            return 0.0
    
    def _calculate_kling_gupta(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算Kling-Gupta效率"""
        try:
            # 相关系数
            r = np.corrcoef(predictions, observations)[0, 1]
            if np.isnan(r):
                r = 0.0
            
            # 偏差比
            alpha = np.std(predictions) / np.std(observations) if np.std(observations) > 0 else 1.0
            
            # 均值比
            beta = np.mean(predictions) / np.mean(observations) if np.mean(observations) != 0 else 1.0
            
            # KGE计算
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            return kge
        
        except Exception as e:
            self.logger.warning(f"Kling-Gupta计算失败: {e}")
            return 0.0
    
    def _calculate_index_of_agreement(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算一致性指数"""
        try:
            obs_mean = np.mean(observations)
            numerator = np.sum((predictions - observations) ** 2)
            denominator = np.sum((np.abs(predictions - obs_mean) + np.abs(observations - obs_mean)) ** 2)
            
            if denominator == 0:
                return 1.0
            
            ioa = 1 - (numerator / denominator)
            return ioa
        
        except Exception as e:
            self.logger.warning(f"一致性指数计算失败: {e}")
            return 0.0
    
    def _calculate_skill_score(self, 
                             predictions: np.ndarray,
                             observations: np.ndarray,
                             reference: np.ndarray,
                             score_type: SkillScoreType) -> float:
        """计算技能得分"""
        try:
            if score_type == SkillScoreType.MURPHY:
                # Murphy技能得分
                mse_pred = mean_squared_error(observations, predictions)
                mse_ref = mean_squared_error(observations, reference)
                
                if mse_ref == 0:
                    return 1.0 if mse_pred == 0 else 0.0
                
                skill_score = 1 - (mse_pred / mse_ref)
                return skill_score
            
            else:
                # 其他技能得分的简化实现
                return self._calculate_nash_sutcliffe(predictions, observations)
        
        except Exception as e:
            self.logger.warning(f"技能得分计算失败: {e}")
            return 0.0

class ProbabilisticSkillCalculator:
    """概率预测技能计算器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_probabilistic_skills(self,
                                     probability_predictions: np.ndarray,
                                     binary_observations: np.ndarray) -> Dict[str, PredictiveSkillResult]:
        """计算概率预测技能"""
        results = {}
        
        try:
            # Brier得分
            if PredictiveSkillMetric.BRIER_SCORE in self.config.metrics:
                brier = self._calculate_brier_score(probability_predictions, binary_observations)
                results['brier_score'] = PredictiveSkillResult(
                    metric_name='brier_score',
                    value=brier,
                    metadata={'description': 'Brier得分 (越小越好)'}
                )
            
            # 可靠性和分辨率
            if (PredictiveSkillMetric.RELIABILITY in self.config.metrics or 
                PredictiveSkillMetric.RESOLUTION in self.config.metrics):
                reliability, resolution = self._calculate_reliability_resolution(
                    probability_predictions, binary_observations
                )
                
                if PredictiveSkillMetric.RELIABILITY in self.config.metrics:
                    results['reliability'] = PredictiveSkillResult(
                        metric_name='reliability',
                        value=reliability,
                        metadata={'description': '可靠性 (越小越好)'}
                    )
                
                if PredictiveSkillMetric.RESOLUTION in self.config.metrics:
                    results['resolution'] = PredictiveSkillResult(
                        metric_name='resolution',
                        value=resolution,
                        metadata={'description': '分辨率 (越大越好)'}
                    )
            
            # 锐度
            if PredictiveSkillMetric.SHARPNESS in self.config.metrics:
                sharpness = self._calculate_sharpness(probability_predictions)
                results['sharpness'] = PredictiveSkillResult(
                    metric_name='sharpness',
                    value=sharpness,
                    metadata={'description': '锐度'}
                )
            
            # 校准度
            if PredictiveSkillMetric.CALIBRATION in self.config.metrics:
                calibration_error = self._calculate_calibration_error(
                    probability_predictions, binary_observations
                )
                results['calibration'] = PredictiveSkillResult(
                    metric_name='calibration',
                    value=calibration_error,
                    metadata={'description': '校准误差 (越小越好)'}
                )
            
            return results
        
        except Exception as e:
            self.logger.error(f"概率技能计算失败: {e}")
            return {}
    
    def _calculate_brier_score(self, probabilities: np.ndarray, observations: np.ndarray) -> float:
        """计算Brier得分"""
        try:
            return brier_score_loss(observations, probabilities)
        except Exception as e:
            self.logger.warning(f"Brier得分计算失败: {e}")
            return 1.0
    
    def _calculate_reliability_resolution(self, probabilities: np.ndarray, 
                                        observations: np.ndarray) -> Tuple[float, float]:
        """计算可靠性和分辨率"""
        try:
            # 将概率分箱
            bins = np.linspace(0, 1, self.config.calibration_bins + 1)
            bin_indices = np.digitize(probabilities, bins) - 1
            bin_indices = np.clip(bin_indices, 0, self.config.calibration_bins - 1)
            
            reliability = 0.0
            resolution = 0.0
            base_rate = np.mean(observations)
            
            for i in range(self.config.calibration_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_prob = np.mean(probabilities[mask])
                    bin_freq = np.mean(observations[mask])
                    bin_count = np.sum(mask)
                    
                    # 可靠性：预测概率与观测频率的差异
                    reliability += bin_count * (bin_prob - bin_freq) ** 2
                    
                    # 分辨率：各箱观测频率与基础率的差异
                    resolution += bin_count * (bin_freq - base_rate) ** 2
            
            reliability /= len(probabilities)
            resolution /= len(probabilities)
            
            return reliability, resolution
        
        except Exception as e:
            self.logger.warning(f"可靠性和分辨率计算失败: {e}")
            return 1.0, 0.0
    
    def _calculate_sharpness(self, probabilities: np.ndarray) -> float:
        """计算锐度"""
        try:
            # 锐度：概率分布的方差
            return np.var(probabilities)
        except Exception as e:
            self.logger.warning(f"锐度计算失败: {e}")
            return 0.0
    
    def _calculate_calibration_error(self, probabilities: np.ndarray, observations: np.ndarray) -> float:
        """计算校准误差"""
        try:
            # 使用sklearn的校准曲线
            fraction_of_positives, mean_predicted_value = calibration_curve(
                observations, probabilities, n_bins=self.config.calibration_bins
            )
            
            # 计算校准误差（期望校准误差）
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            return calibration_error
        
        except Exception as e:
            self.logger.warning(f"校准误差计算失败: {e}")
            return 1.0

class TemporalSkillCalculator:
    """时间序列预测技能计算器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_temporal_skills(self,
                                predictions: np.ndarray,
                                observations: np.ndarray,
                                time_steps: Optional[np.ndarray] = None) -> Dict[str, PredictiveSkillResult]:
        """计算时间序列预测技能"""
        results = {}
        
        try:
            # 持续性技能
            if PredictiveSkillMetric.PERSISTENCE_SKILL in self.config.metrics:
                persistence_skill = self._calculate_persistence_skill(predictions, observations)
                results['persistence_skill'] = PredictiveSkillResult(
                    metric_name='persistence_skill',
                    value=persistence_skill,
                    metadata={'description': '相对于持续性预报的技能'}
                )
            
            # 趋势技能
            if PredictiveSkillMetric.TREND_SKILL in self.config.metrics:
                trend_skill = self._calculate_trend_skill(predictions, observations)
                results['trend_skill'] = PredictiveSkillResult(
                    metric_name='trend_skill',
                    value=trend_skill,
                    metadata={'description': '趋势预测技能'}
                )
            
            # 季节性技能
            if PredictiveSkillMetric.SEASONAL_SKILL in self.config.metrics and time_steps is not None:
                seasonal_skill = self._calculate_seasonal_skill(predictions, observations, time_steps)
                results['seasonal_skill'] = PredictiveSkillResult(
                    metric_name='seasonal_skill',
                    value=seasonal_skill,
                    metadata={'description': '季节性预测技能'}
                )
            
            # 预报时效技能
            if PredictiveSkillMetric.FORECAST_HORIZON_SKILL in self.config.metrics:
                horizon_skills = self._calculate_forecast_horizon_skill(predictions, observations)
                for horizon, skill in horizon_skills.items():
                    results[f'forecast_horizon_skill_{horizon}'] = PredictiveSkillResult(
                        metric_name=f'forecast_horizon_skill_{horizon}',
                        value=skill,
                        metadata={'horizon': horizon, 'description': f'{horizon}步预报技能'}
                    )
            
            return results
        
        except Exception as e:
            self.logger.error(f"时间序列技能计算失败: {e}")
            return {}
    
    def _calculate_persistence_skill(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算持续性技能"""
        try:
            # 生成持续性预报
            persistence_forecast = np.zeros_like(observations)
            persistence_forecast[1:] = observations[:-1]
            persistence_forecast[0] = observations[0]
            
            # 计算技能得分
            mse_pred = mean_squared_error(observations, predictions)
            mse_pers = mean_squared_error(observations, persistence_forecast)
            
            if mse_pers == 0:
                return 1.0 if mse_pred == 0 else 0.0
            
            skill = 1 - (mse_pred / mse_pers)
            return skill
        
        except Exception as e:
            self.logger.warning(f"持续性技能计算失败: {e}")
            return 0.0
    
    def _calculate_trend_skill(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算趋势技能"""
        try:
            # 计算趋势（一阶差分）
            pred_trend = np.diff(predictions)
            obs_trend = np.diff(observations)
            
            # 趋势方向一致性
            trend_agreement = np.mean(np.sign(pred_trend) == np.sign(obs_trend))
            
            return trend_agreement
        
        except Exception as e:
            self.logger.warning(f"趋势技能计算失败: {e}")
            return 0.0
    
    def _calculate_seasonal_skill(self, predictions: np.ndarray, 
                                observations: np.ndarray,
                                time_steps: np.ndarray) -> float:
        """计算季节性技能"""
        try:
            # 简化实现：假设时间步长为天，计算年内日期的季节性
            if len(time_steps) < 365:
                return 0.0
            
            # 计算季节性成分（简化为年内变化）
            day_of_year = time_steps % 365
            
            # 按日期分组计算平均值
            seasonal_pred = np.zeros_like(predictions)
            seasonal_obs = np.zeros_like(observations)
            
            for day in range(365):
                mask = day_of_year == day
                if np.sum(mask) > 0:
                    seasonal_pred[mask] = np.mean(predictions[mask])
                    seasonal_obs[mask] = np.mean(observations[mask])
            
            # 计算季节性相关性
            correlation = np.corrcoef(seasonal_pred, seasonal_obs)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            self.logger.warning(f"季节性技能计算失败: {e}")
            return 0.0
    
    def _calculate_forecast_horizon_skill(self, predictions: np.ndarray, 
                                        observations: np.ndarray) -> Dict[int, float]:
        """计算不同预报时效的技能"""
        horizon_skills = {}
        
        try:
            for horizon in self.config.forecast_horizons:
                if len(predictions) > horizon:
                    # 计算指定时效的预报技能
                    pred_horizon = predictions[:-horizon]
                    obs_horizon = observations[horizon:]
                    
                    # 使用相关系数作为技能指标
                    correlation = np.corrcoef(pred_horizon, obs_horizon)[0, 1]
                    skill = correlation if not np.isnan(correlation) else 0.0
                    
                    horizon_skills[horizon] = skill
        
        except Exception as e:
            self.logger.warning(f"预报时效技能计算失败: {e}")
        
        return horizon_skills

class SpatialSkillCalculator:
    """空间预测技能计算器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_spatial_skills(self,
                               predictions: np.ndarray,
                               observations: np.ndarray,
                               coordinates: Optional[np.ndarray] = None) -> Dict[str, PredictiveSkillResult]:
        """计算空间预测技能"""
        results = {}
        
        try:
            # 空间相关性
            if PredictiveSkillMetric.SPATIAL_CORRELATION in self.config.metrics:
                spatial_corr = self._calculate_spatial_correlation(predictions, observations, coordinates)
                results['spatial_correlation'] = PredictiveSkillResult(
                    metric_name='spatial_correlation',
                    value=spatial_corr,
                    metadata={'description': '空间相关性'}
                )
            
            # 模式相关性
            if PredictiveSkillMetric.PATTERN_CORRELATION in self.config.metrics:
                pattern_corr = self._calculate_pattern_correlation(predictions, observations)
                results['pattern_correlation'] = PredictiveSkillResult(
                    metric_name='pattern_correlation',
                    value=pattern_corr,
                    metadata={'description': '空间模式相关性'}
                )
            
            # 空间技能得分
            if PredictiveSkillMetric.SPATIAL_SKILL_SCORE in self.config.metrics:
                spatial_skill = self._calculate_spatial_skill_score(predictions, observations, coordinates)
                results['spatial_skill_score'] = PredictiveSkillResult(
                    metric_name='spatial_skill_score',
                    value=spatial_skill,
                    metadata={'description': '空间技能得分'}
                )
            
            return results
        
        except Exception as e:
            self.logger.error(f"空间技能计算失败: {e}")
            return {}
    
    def _calculate_spatial_correlation(self, predictions: np.ndarray, 
                                     observations: np.ndarray,
                                     coordinates: Optional[np.ndarray] = None) -> float:
        """计算空间相关性"""
        try:
            # 简单的空间相关性：整体相关系数
            correlation = np.corrcoef(predictions.flatten(), observations.flatten())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            self.logger.warning(f"空间相关性计算失败: {e}")
            return 0.0
    
    def _calculate_pattern_correlation(self, predictions: np.ndarray, observations: np.ndarray) -> float:
        """计算模式相关性"""
        try:
            # 去除空间平均后的相关性
            pred_anomaly = predictions - np.mean(predictions)
            obs_anomaly = observations - np.mean(observations)
            
            correlation = np.corrcoef(pred_anomaly.flatten(), obs_anomaly.flatten())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            self.logger.warning(f"模式相关性计算失败: {e}")
            return 0.0
    
    def _calculate_spatial_skill_score(self, predictions: np.ndarray, 
                                     observations: np.ndarray,
                                     coordinates: Optional[np.ndarray] = None) -> float:
        """计算空间技能得分"""
        try:
            # 使用空间平均作为参考
            spatial_mean = np.mean(observations)
            reference = np.full_like(observations, spatial_mean)
            
            # 计算技能得分
            mse_pred = mean_squared_error(observations, predictions)
            mse_ref = mean_squared_error(observations, reference)
            
            if mse_ref == 0:
                return 1.0 if mse_pred == 0 else 0.0
            
            skill = 1 - (mse_pred / mse_ref)
            return skill
        
        except Exception as e:
            self.logger.warning(f"空间技能得分计算失败: {e}")
            return 0.0

class ExtremeEventSkillCalculator:
    """极端事件预测技能计算器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_extreme_event_skills(self,
                                     predictions: np.ndarray,
                                     observations: np.ndarray) -> Dict[str, PredictiveSkillResult]:
        """计算极端事件预测技能"""
        results = {}
        
        try:
            for threshold_name, threshold_value in self.config.extreme_thresholds.items():
                # 定义极端事件
                obs_threshold = np.percentile(observations, threshold_value * 100)
                
                # 二值化
                obs_binary = (observations > obs_threshold).astype(int)
                pred_binary = (predictions > obs_threshold).astype(int)
                
                # 计算混淆矩阵
                tp = np.sum((pred_binary == 1) & (obs_binary == 1))
                fp = np.sum((pred_binary == 1) & (obs_binary == 0))
                tn = np.sum((pred_binary == 0) & (obs_binary == 0))
                fn = np.sum((pred_binary == 0) & (obs_binary == 1))
                
                # 命中率
                hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # 虚警率
                false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                # 临界成功指数
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                
                # 存储结果
                results[f'hit_rate_{threshold_name}'] = PredictiveSkillResult(
                    metric_name=f'hit_rate_{threshold_name}',
                    value=hit_rate,
                    metadata={'threshold': threshold_name, 'threshold_value': threshold_value}
                )
                
                results[f'false_alarm_rate_{threshold_name}'] = PredictiveSkillResult(
                    metric_name=f'false_alarm_rate_{threshold_name}',
                    value=false_alarm_rate,
                    metadata={'threshold': threshold_name, 'threshold_value': threshold_value}
                )
                
                results[f'critical_success_index_{threshold_name}'] = PredictiveSkillResult(
                    metric_name=f'critical_success_index_{threshold_name}',
                    value=csi,
                    metadata={'threshold': threshold_name, 'threshold_value': threshold_value}
                )
            
            return results
        
        except Exception as e:
            self.logger.error(f"极端事件技能计算失败: {e}")
            return {}

class PredictiveSkillAnalyzer:
    """预测技能分析器"""
    
    def __init__(self, config: PredictiveSkillConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化计算器
        self.deterministic_calculator = DeterministicSkillCalculator(config)
        self.probabilistic_calculator = ProbabilisticSkillCalculator(config)
        self.temporal_calculator = TemporalSkillCalculator(config)
        self.spatial_calculator = SpatialSkillCalculator(config)
        self.extreme_calculator = ExtremeEventSkillCalculator(config)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
    
    def analyze_predictive_skill(self,
                               predictions: np.ndarray,
                               observations: np.ndarray,
                               coordinates: Optional[np.ndarray] = None,
                               time_steps: Optional[np.ndarray] = None,
                               probability_predictions: Optional[np.ndarray] = None) -> Dict[str, PredictiveSkillResult]:
        """分析预测技能"""
        try:
            results = {}
            
            # 确定性预测技能
            if self.config.prediction_type in [PredictionType.DETERMINISTIC, PredictionType.ENSEMBLE]:
                deterministic_results = self.deterministic_calculator.calculate_skill_scores(
                    predictions, observations
                )
                results.update(deterministic_results)
            
            # 概率预测技能
            if (self.config.prediction_type == PredictionType.PROBABILISTIC and 
                probability_predictions is not None):
                # 需要二值化观测数据
                threshold = np.median(observations)
                binary_obs = (observations > threshold).astype(int)
                
                probabilistic_results = self.probabilistic_calculator.calculate_probabilistic_skills(
                    probability_predictions, binary_obs
                )
                results.update(probabilistic_results)
            
            # 时间序列技能
            temporal_results = self.temporal_calculator.calculate_temporal_skills(
                predictions, observations, time_steps
            )
            results.update(temporal_results)
            
            # 空间技能
            spatial_results = self.spatial_calculator.calculate_spatial_skills(
                predictions, observations, coordinates
            )
            results.update(spatial_results)
            
            # 极端事件技能
            extreme_results = self.extreme_calculator.calculate_extreme_event_skills(
                predictions, observations
            )
            results.update(extreme_results)
            
            # 计算综合技能
            overall_results = self._calculate_overall_skill(results)
            results.update(overall_results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"预测技能分析失败: {e}")
            return {}
    
    def _calculate_overall_skill(self, results: Dict[str, PredictiveSkillResult]) -> Dict[str, PredictiveSkillResult]:
        """计算综合技能"""
        overall_results = {}
        
        try:
            if not results:
                return overall_results
            
            # 总体技能
            if PredictiveSkillMetric.OVERALL_SKILL in self.config.metrics:
                skill_values = []
                weights = []
                
                # 收集各种技能指标
                for name, result in results.items():
                    if 'skill' in name.lower() or name in ['nash_sutcliffe', 'kling_gupta']:
                        skill_values.append(result.value)
                        weights.append(1.0)  # 简化：等权重
                
                if skill_values:
                    overall_skill = np.average(skill_values, weights=weights)
                    
                    overall_results['overall_skill'] = PredictiveSkillResult(
                        metric_name='overall_skill',
                        value=overall_skill,
                        metadata={
                            'n_metrics': len(skill_values),
                            'component_skills': skill_values
                        }
                    )
            
            # 预测能力
            if PredictiveSkillMetric.PREDICTIVE_POWER in self.config.metrics:
                # 基于相关性和技能得分的综合指标
                correlations = []
                skills = []
                
                for name, result in results.items():
                    if 'correlation' in name:
                        correlations.append(result.value)
                    elif 'skill' in name:
                        skills.append(result.value)
                
                if correlations or skills:
                    all_values = correlations + skills
                    predictive_power = np.mean(all_values)
                    
                    overall_results['predictive_power'] = PredictiveSkillResult(
                        metric_name='predictive_power',
                        value=predictive_power,
                        metadata={
                            'n_correlations': len(correlations),
                            'n_skills': len(skills)
                        }
                    )
        
        except Exception as e:
            self.logger.warning(f"综合技能计算失败: {e}")
        
        return overall_results
    
    def plot_results(self, 
                    results: Dict[str, PredictiveSkillResult],
                    predictions: np.ndarray,
                    observations: np.ndarray):
        """绘制预测技能分析结果"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('预测技能分析结果', fontsize=16, fontweight='bold')
            
            # 散点图：预测 vs 观测
            ax = axes[0, 0]
            ax.scatter(observations, predictions, alpha=0.6)
            min_val = min(np.min(observations), np.min(predictions))
            max_val = max(np.max(observations), np.max(predictions))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1线')
            ax.set_xlabel('观测值')
            ax.set_ylabel('预测值')
            ax.set_title('预测 vs 观测')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 技能得分雷达图
            ax = axes[0, 1]
            skill_metrics = {k: v for k, v in results.items() 
                           if 'skill' in k.lower() or k in ['nash_sutcliffe', 'kling_gupta']}
            
            if skill_metrics:
                names = list(skill_metrics.keys())
                values = [r.value for r in skill_metrics.values()]
                
                # 简化的条形图代替雷达图
                bars = ax.bar(range(len(names)), values)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('技能得分')
                ax.set_title('各项技能指标')
                ax.grid(True, alpha=0.3)
                
                # 颜色编码
                for i, bar in enumerate(bars):
                    if values[i] > 0.7:
                        bar.set_color('green')
                    elif values[i] > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            # 残差分析
            ax = axes[0, 2]
            residuals = predictions - observations
            ax.scatter(observations, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('观测值')
            ax.set_ylabel('残差')
            ax.set_title('残差分析')
            ax.grid(True, alpha=0.3)
            
            # 时间序列比较（如果数据足够长）
            ax = axes[1, 0]
            if len(predictions) > 10:
                time_indices = np.arange(len(predictions))
                ax.plot(time_indices, observations, 'b-', label='观测', alpha=0.7)
                ax.plot(time_indices, predictions, 'r-', label='预测', alpha=0.7)
                ax.set_xlabel('时间步')
                ax.set_ylabel('值')
                ax.set_title('时间序列比较')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 误差分布
            ax = axes[1, 1]
            errors = np.abs(predictions - observations)
            ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('绝对误差')
            ax.set_ylabel('频数')
            ax.set_title('误差分布')
            ax.grid(True, alpha=0.3)
            
            # 综合评分
            ax = axes[1, 2]
            if 'overall_skill' in results:
                score = results['overall_skill'].value
                
                # 创建仪表盘
                theta = np.linspace(0, np.pi, 100)
                r = np.ones_like(theta)
                
                ax.plot(theta, r, 'k-', linewidth=2)
                
                # 得分指针
                score_angle = np.pi * (1 - (score + 1) / 2)  # 将[-1,1]映射到[π,0]
                ax.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
                
                ax.set_xlim(0, np.pi)
                ax.set_ylim(0, 1.2)
                ax.set_title(f'总体技能得分: {score:.3f}')
                ax.set_xticks([0, np.pi/2, np.pi])
                ax.set_xticklabels(['-1', '0', '1'])
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/predictive_skill_analysis.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, PredictiveSkillResult],
                       predictions: np.ndarray,
                       observations: np.ndarray) -> str:
        """生成预测技能分析报告"""
        from datetime import datetime
        
        report_lines = [
            "="*80,
            "预测技能分析报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"预测类型: {self.config.prediction_type.value}",
            f"样本数量: {len(predictions)}",
            f"评估指标数: {len(results)}",
            "",
            "1. 基本统计",
            "-"*40,
            f"观测值范围: [{np.min(observations):.4f}, {np.max(observations):.4f}]",
            f"预测值范围: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]",
            f"观测值均值: {np.mean(observations):.4f}",
            f"预测值均值: {np.mean(predictions):.4f}",
            f"观测值标准差: {np.std(observations):.4f}",
            f"预测值标准差: {np.std(predictions):.4f}",
            ""
        ]
        
        # 确定性技能指标
        deterministic_metrics = {
            k: v for k, v in results.items() 
            if k in ['nash_sutcliffe', 'kling_gupta', 'index_of_agreement'] or 'skill_score' in k
        }
        
        if deterministic_metrics:
            report_lines.extend([
                "2. 确定性预测技能",
                "-"*40
            ])
            
            for name, result in deterministic_metrics.items():
                if result.value >= 0.7:
                    assessment = "优秀"
                elif result.value >= 0.5:
                    assessment = "良好"
                elif result.value >= 0.3:
                    assessment = "一般"
                else:
                    assessment = "较差"
                
                report_lines.append(f"{name}: {result.value:.4f} ({assessment})")
        
        # 时间序列技能
        temporal_metrics = {k: v for k, v in results.items() if 'temporal' in k or 'persistence' in k or 'trend' in k}
        if temporal_metrics:
            report_lines.extend([
                "",
                "3. 时间序列预测技能",
                "-"*40
            ])
            
            for name, result in temporal_metrics.items():
                report_lines.append(f"{name}: {result.value:.4f}")
        
        # 空间预测技能
        spatial_metrics = {k: v for k, v in results.items() if 'spatial' in k or 'pattern' in k}
        if spatial_metrics:
            report_lines.extend([
                "",
                "4. 空间预测技能",
                "-"*40
            ])
            
            for name, result in spatial_metrics.items():
                report_lines.append(f"{name}: {result.value:.4f}")
        
        # 极端事件技能
        extreme_metrics = {k: v for k, v in results.items() if 'extreme' in k or 'hit_rate' in k or 'critical_success' in k}
        if extreme_metrics:
            report_lines.extend([
                "",
                "5. 极端事件预测技能",
                "-"*40
            ])
            
            for name, result in extreme_metrics.items():
                report_lines.append(f"{name}: {result.value:.4f}")
        
        # 综合评估
        if 'overall_skill' in results:
            overall_skill = results['overall_skill']
            report_lines.extend([
                "",
                "6. 综合评估",
                "-"*40,
                f"总体技能得分: {overall_skill.value:.4f}"
            ])
            
            if overall_skill.value >= 0.7:
                assessment = "优秀 - 模型具有很强的预测技能"
            elif overall_skill.value >= 0.5:
                assessment = "良好 - 模型具有较好的预测技能"
            elif overall_skill.value >= 0.3:
                assessment = "一般 - 模型具有一定的预测技能"
            else:
                assessment = "较差 - 模型预测技能有限"
            
            report_lines.append(f"评估结果: {assessment}")
        
        if 'predictive_power' in results:
            power = results['predictive_power']
            report_lines.append(f"预测能力: {power.value:.4f}")
        
        # 建议
        report_lines.extend([
            "",
            "7. 改进建议",
            "-"*40
        ])
        
        # 基于结果给出建议
        if 'nash_sutcliffe' in results and results['nash_sutcliffe'].value < 0.5:
            report_lines.append("• Nash-Sutcliffe效率较低，建议检查模型结构和参数")
        
        if 'spatial_correlation' in results and results['spatial_correlation'].value < 0.6:
            report_lines.append("• 空间相关性较低，建议改进空间表示能力")
        
        if 'persistence_skill' in results and results['persistence_skill'].value < 0.2:
            report_lines.append("• 相对于持续性预报的改进有限，建议增强时间动态建模")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_predictive_skill_analyzer(config: Optional[PredictiveSkillConfig] = None) -> PredictiveSkillAnalyzer:
    """创建预测技能分析器"""
    if config is None:
        config = PredictiveSkillConfig()
    
    return PredictiveSkillAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    print("开始预测技能分析测试...")
    
    # 生成测试数据
    np.random.seed(42)
    n_points = 200
    
    # 创建具有一定技能的预测数据
    true_signal = np.sin(np.linspace(0, 4*np.pi, n_points)) + 0.5 * np.sin(np.linspace(0, 8*np.pi, n_points))
    observations = true_signal + np.random.normal(0, 0.2, n_points)
    
    # 模拟预测：添加一些偏差和噪声
    predictions = 0.8 * true_signal + 0.1 * np.random.normal(0, 0.3, n_points) + 0.1
    
    # 坐标和时间
    coordinates = np.random.uniform(-1, 1, (n_points, 2))
    time_steps = np.arange(n_points)
    
    # 概率预测（简化）
    threshold = np.median(observations)
    probability_predictions = 1 / (1 + np.exp(-(predictions - threshold)))
    
    # 创建配置
    config = PredictiveSkillConfig(
        metrics=[
            PredictiveSkillMetric.SKILL_SCORE,
            PredictiveSkillMetric.NASH_SUTCLIFFE,
            PredictiveSkillMetric.KLING_GUPTA,
            PredictiveSkillMetric.PERSISTENCE_SKILL,
            PredictiveSkillMetric.SPATIAL_CORRELATION,
            PredictiveSkillMetric.OVERALL_SKILL
        ],
        prediction_type=PredictionType.DETERMINISTIC,
        enable_plotting=True,
        verbose=True
    )
    
    # 创建分析器
    analyzer = create_predictive_skill_analyzer(config)
    
    # 进行分析
    results = analyzer.analyze_predictive_skill(
        predictions=predictions,
        observations=observations,
        coordinates=coordinates,
        time_steps=time_steps,
        probability_predictions=probability_predictions
    )
    
    # 打印结果
    print("\n预测技能分析完成！")
    print(f"评估了 {len(results)} 个指标")
    
    # 生成报告
    report = analyzer.generate_report(results, predictions, observations)
    print("\n" + report)
    
    # 绘制结果
    analyzer.plot_results(results, predictions, observations)
    
    print("\n预测技能分析测试完成！")