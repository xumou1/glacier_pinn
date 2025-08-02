#!/usr/bin/env python3
"""
季节模式分析模块

该模块提供冰川时间序列数据的季节模式分析功能，包括：
- 季节性检测和量化
- 季节周期分析
- 季节异常检测
- 多年季节对比
- 季节趋势分析

作者：冰川PINNs项目组
日期：2024年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import calendar

warnings.filterwarnings('ignore')


class SeasonalMethod(Enum):
    """季节分析方法枚举"""
    CLASSICAL_DECOMPOSITION = "classical"
    STL_DECOMPOSITION = "stl"
    X13_ARIMA = "x13"
    FOURIER_ANALYSIS = "fourier"
    WAVELET_ANALYSIS = "wavelet"
    EMPIRICAL_MODE = "emd"


class SeasonalPeriod(Enum):
    """季节周期枚举"""
    MONTHLY = 12
    QUARTERLY = 4
    WEEKLY = 52
    DAILY = 365
    CUSTOM = 0


class AnomalyMethod(Enum):
    """异常检测方法枚举"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER = "lof"
    SEASONAL_HYBRID = "seasonal_hybrid"


class AggregationMethod(Enum):
    """聚合方法枚举"""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    PERCENTILE = "percentile"


@dataclass
class SeasonalConfig:
    """季节分析配置"""
    method: SeasonalMethod = SeasonalMethod.STL_DECOMPOSITION
    period: SeasonalPeriod = SeasonalPeriod.MONTHLY
    custom_period: Optional[int] = None
    anomaly_method: AnomalyMethod = AnomalyMethod.SEASONAL_HYBRID
    anomaly_threshold: float = 2.0
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    min_periods_per_season: int = 3
    confidence_level: float = 0.95
    detrend: bool = True
    normalize: bool = False
    smooth_seasonal: bool = True
    smooth_window: int = 3
    detect_multiple_periods: bool = True
    max_periods: int = 3
    seasonal_strength_threshold: float = 0.1


@dataclass
class SeasonalData:
    """季节分析数据结构"""
    time_series: pd.DataFrame
    variable_names: List[str]
    time_column: str = "time"
    frequency: str = "M"  # 数据频率：D(日), M(月), Q(季), Y(年)
    metadata: Dict[str, Any] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    coordinates: Optional[Dict[str, float]] = None
    glacier_id: Optional[str] = None


@dataclass
class SeasonalResult:
    """季节分析结果"""
    variable: str
    seasonal_component: np.ndarray
    trend_component: np.ndarray
    residual_component: np.ndarray
    seasonal_strength: float
    dominant_periods: List[int]
    seasonal_peaks: Dict[str, float]  # 季节峰值时间和强度
    seasonal_troughs: Dict[str, float]  # 季节谷值时间和强度
    seasonal_amplitude: float
    seasonal_phase: float
    seasonal_anomalies: List[int]  # 异常点索引
    yearly_patterns: Dict[int, np.ndarray]  # 每年的季节模式
    seasonal_statistics: Dict[str, float]
    decomposition_quality: float  # 分解质量指标


class SeasonalDetector:
    """季节性检测器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
    
    def detect_seasonality(self, data: np.ndarray, time_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """检测时间序列的季节性"""
        results = {
            "has_seasonality": False,
            "dominant_periods": [],
            "seasonal_strength": 0.0,
            "seasonal_peaks": {},
            "seasonal_troughs": {},
            "fourier_analysis": {}
        }
        
        if len(data) < 24:  # 需要至少2年的月度数据
            return results
        
        # 傅里叶分析检测周期性
        fourier_results = self._fourier_analysis(data)
        results["fourier_analysis"] = fourier_results
        
        # 自相关分析
        autocorr_results = self._autocorrelation_analysis(data)
        
        # 组合分析结果
        dominant_periods = []
        if fourier_results["dominant_frequencies"]:
            for freq in fourier_results["dominant_frequencies"]:
                if freq > 0:
                    period = int(1 / freq)
                    if 3 <= period <= len(data) // 3:
                        dominant_periods.append(period)
        
        # 检查常见的季节周期
        common_periods = [12, 6, 4, 3]  # 年、半年、季、三月周期
        for period in common_periods:
            if period <= len(data) // 3:
                strength = self._calculate_seasonal_strength(data, period)
                if strength > self.config.seasonal_strength_threshold:
                    if period not in dominant_periods:
                        dominant_periods.append(period)
        
        # 计算总体季节强度
        if dominant_periods:
            seasonal_strengths = [self._calculate_seasonal_strength(data, p) for p in dominant_periods]
            results["seasonal_strength"] = max(seasonal_strengths)
            results["has_seasonality"] = results["seasonal_strength"] > self.config.seasonal_strength_threshold
            results["dominant_periods"] = sorted(dominant_periods, key=lambda p: self._calculate_seasonal_strength(data, p), reverse=True)
        
        # 检测季节峰值和谷值
        if results["has_seasonality"] and dominant_periods:
            main_period = dominant_periods[0]
            peaks, troughs = self._find_seasonal_extremes(data, main_period)
            results["seasonal_peaks"] = peaks
            results["seasonal_troughs"] = troughs
        
        return results
    
    def _fourier_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """傅里叶分析检测周期性"""
        # 去除趋势
        detrended = signal.detrend(data)
        
        # FFT分析
        fft_values = fft(detrended)
        frequencies = fftfreq(len(data))
        
        # 计算功率谱
        power_spectrum = np.abs(fft_values) ** 2
        
        # 只考虑正频率
        positive_freq_mask = frequencies > 0
        pos_frequencies = frequencies[positive_freq_mask]
        pos_power = power_spectrum[positive_freq_mask]
        
        # 找到主要频率
        if len(pos_power) > 0:
            # 找到功率最大的几个频率
            top_indices = np.argsort(pos_power)[-self.config.max_periods:]
            dominant_frequencies = pos_frequencies[top_indices]
            dominant_powers = pos_power[top_indices]
            
            # 过滤掉功率太小的频率
            power_threshold = np.max(pos_power) * 0.1
            significant_mask = dominant_powers > power_threshold
            dominant_frequencies = dominant_frequencies[significant_mask]
            dominant_powers = dominant_powers[significant_mask]
        else:
            dominant_frequencies = np.array([])
            dominant_powers = np.array([])
        
        return {
            "frequencies": pos_frequencies,
            "power_spectrum": pos_power,
            "dominant_frequencies": dominant_frequencies,
            "dominant_powers": dominant_powers
        }
    
    def _autocorrelation_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """自相关分析"""
        # 计算自相关函数
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # 标准化
        
        # 找到自相关峰值
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=3)
        peaks += 1  # 调整索引
        
        return {
            "autocorrelation": autocorr,
            "peaks": peaks,
            "peak_values": autocorr[peaks] if len(peaks) > 0 else np.array([])
        }
    
    def _calculate_seasonal_strength(self, data: np.ndarray, period: int) -> float:
        """计算指定周期的季节强度"""
        if period >= len(data) or period < 2:
            return 0.0
        
        try:
            # 使用STL分解
            ts = pd.Series(data)
            stl = STL(ts, seasonal=period, robust=True)
            decomposition = stl.fit()
            
            # 计算季节强度：季节分量方差 / (季节分量方差 + 残差方差)
            seasonal_var = np.var(decomposition.seasonal)
            residual_var = np.var(decomposition.resid)
            
            if seasonal_var + residual_var > 0:
                strength = seasonal_var / (seasonal_var + residual_var)
            else:
                strength = 0.0
            
            return strength
        except Exception:
            return 0.0
    
    def _find_seasonal_extremes(self, data: np.ndarray, period: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """找到季节性极值"""
        if period >= len(data):
            return {}, {}
        
        # 重塑数据为季节周期
        n_complete_cycles = len(data) // period
        if n_complete_cycles < 2:
            return {}, {}
        
        reshaped_data = data[:n_complete_cycles * period].reshape(n_complete_cycles, period)
        
        # 计算每个季节位置的平均值
        seasonal_mean = np.mean(reshaped_data, axis=0)
        
        # 找到峰值和谷值
        peak_idx = np.argmax(seasonal_mean)
        trough_idx = np.argmin(seasonal_mean)
        
        peaks = {
            "position": peak_idx,
            "value": seasonal_mean[peak_idx],
            "relative_position": peak_idx / period
        }
        
        troughs = {
            "position": trough_idx,
            "value": seasonal_mean[trough_idx],
            "relative_position": trough_idx / period
        }
        
        return peaks, troughs


class SeasonalDecomposer:
    """季节分解器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
    
    def decompose(self, data: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """分解时间序列"""
        if self.config.method == SeasonalMethod.STL_DECOMPOSITION:
            return self._stl_decompose(data, period)
        elif self.config.method == SeasonalMethod.CLASSICAL_DECOMPOSITION:
            return self._classical_decompose(data, period)
        else:
            return self._stl_decompose(data, period)
    
    def _stl_decompose(self, data: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """STL分解"""
        try:
            ts = pd.Series(data)
            stl = STL(ts, seasonal=period, robust=True)
            decomposition = stl.fit()
            
            return {
                "trend": decomposition.trend.values,
                "seasonal": decomposition.seasonal.values,
                "residual": decomposition.resid.values
            }
        except Exception:
            return self._fallback_decompose(data, period)
    
    def _classical_decompose(self, data: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """经典分解"""
        try:
            ts = pd.Series(data)
            decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
            
            return {
                "trend": decomposition.trend.values,
                "seasonal": decomposition.seasonal.values,
                "residual": decomposition.resid.values
            }
        except Exception:
            return self._fallback_decompose(data, period)
    
    def _fallback_decompose(self, data: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        """备用分解方法"""
        # 简单的移动平均趋势
        if len(data) > period:
            trend = pd.Series(data).rolling(window=period, center=True).mean().values
        else:
            trend = np.full_like(data, np.mean(data))
        
        # 去趋势
        detrended = data - np.nan_to_num(trend, nan=np.mean(data))
        
        # 简单的季节分量（周期平均）
        seasonal = np.zeros_like(data)
        if period < len(data):
            for i in range(period):
                indices = np.arange(i, len(data), period)
                if len(indices) > 0:
                    seasonal_value = np.mean(detrended[indices])
                    seasonal[indices] = seasonal_value
        
        # 残差
        residual = data - np.nan_to_num(trend, nan=np.mean(data)) - seasonal
        
        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual
        }


class SeasonalAnomalyDetector:
    """季节异常检测器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
    
    def detect_anomalies(self, 
                        data: np.ndarray, 
                        seasonal_component: np.ndarray,
                        residual_component: np.ndarray) -> List[int]:
        """检测季节异常"""
        if self.config.anomaly_method == AnomalyMethod.SEASONAL_HYBRID:
            return self._seasonal_hybrid_detection(data, seasonal_component, residual_component)
        elif self.config.anomaly_method == AnomalyMethod.Z_SCORE:
            return self._z_score_detection(residual_component)
        elif self.config.anomaly_method == AnomalyMethod.IQR:
            return self._iqr_detection(residual_component)
        else:
            return self._seasonal_hybrid_detection(data, seasonal_component, residual_component)
    
    def _seasonal_hybrid_detection(self, 
                                  data: np.ndarray, 
                                  seasonal: np.ndarray, 
                                  residual: np.ndarray) -> List[int]:
        """季节混合异常检测"""
        anomalies = []
        
        # 基于残差的异常检测
        residual_anomalies = self._z_score_detection(residual)
        anomalies.extend(residual_anomalies)
        
        # 基于季节偏离的异常检测
        seasonal_anomalies = self._seasonal_deviation_detection(data, seasonal)
        anomalies.extend(seasonal_anomalies)
        
        # 去重并排序
        anomalies = sorted(list(set(anomalies)))
        
        return anomalies
    
    def _z_score_detection(self, data: np.ndarray) -> List[int]:
        """Z分数异常检测"""
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        anomalies = np.where(z_scores > self.config.anomaly_threshold)[0].tolist()
        return anomalies
    
    def _iqr_detection(self, data: np.ndarray) -> List[int]:
        """IQR异常检测"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
        return anomalies
    
    def _seasonal_deviation_detection(self, data: np.ndarray, seasonal: np.ndarray) -> List[int]:
        """季节偏离异常检测"""
        # 计算实际值与季节期望的偏离
        expected = seasonal
        deviation = np.abs(data - expected)
        
        # 使用偏离的Z分数
        z_scores = np.abs(stats.zscore(deviation, nan_policy='omit'))
        anomalies = np.where(z_scores > self.config.anomaly_threshold)[0].tolist()
        
        return anomalies


class SeasonalPatternAnalyzer:
    """季节模式分析器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
    
    def analyze_yearly_patterns(self, 
                               data: np.ndarray, 
                               time_index: pd.DatetimeIndex,
                               period: int = 12) -> Dict[int, np.ndarray]:
        """分析每年的季节模式"""
        yearly_patterns = {}
        
        # 按年分组
        df = pd.DataFrame({'value': data, 'time': time_index})
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            if len(year_data) >= self.config.min_periods_per_season:
                # 按月聚合（如果是月度数据）
                if period == 12:
                    monthly_data = year_data.groupby('month')['value'].agg(self.config.aggregation_method.value)
                    # 确保有12个月的数据
                    full_year = pd.Series(index=range(1, 13), dtype=float)
                    full_year.update(monthly_data)
                    yearly_patterns[year] = full_year.values
                else:
                    # 其他周期的处理
                    yearly_patterns[year] = year_data['value'].values
        
        return yearly_patterns
    
    def calculate_seasonal_statistics(self, 
                                    seasonal_component: np.ndarray,
                                    period: int) -> Dict[str, float]:
        """计算季节统计量"""
        stats_dict = {
            "amplitude": np.max(seasonal_component) - np.min(seasonal_component),
            "mean": np.mean(seasonal_component),
            "std": np.std(seasonal_component),
            "range": np.ptp(seasonal_component),
            "coefficient_of_variation": np.std(seasonal_component) / np.abs(np.mean(seasonal_component)) if np.mean(seasonal_component) != 0 else 0
        }
        
        # 计算季节强度
        if period < len(seasonal_component):
            # 重塑为周期矩阵
            n_complete_cycles = len(seasonal_component) // period
            if n_complete_cycles > 1:
                reshaped = seasonal_component[:n_complete_cycles * period].reshape(n_complete_cycles, period)
                
                # 计算周期内方差和周期间方差
                within_period_var = np.mean(np.var(reshaped, axis=1))
                between_period_var = np.var(np.mean(reshaped, axis=0))
                
                if within_period_var + between_period_var > 0:
                    stats_dict["seasonal_strength"] = between_period_var / (within_period_var + between_period_var)
                else:
                    stats_dict["seasonal_strength"] = 0.0
            else:
                stats_dict["seasonal_strength"] = 0.0
        else:
            stats_dict["seasonal_strength"] = 0.0
        
        return stats_dict
    
    def compare_yearly_patterns(self, yearly_patterns: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """比较不同年份的季节模式"""
        if len(yearly_patterns) < 2:
            return {"comparison_possible": False}
        
        years = sorted(yearly_patterns.keys())
        patterns_matrix = np.array([yearly_patterns[year] for year in years])
        
        # 计算年际相关性
        correlations = np.corrcoef(patterns_matrix)
        
        # 计算平均模式
        mean_pattern = np.nanmean(patterns_matrix, axis=0)
        std_pattern = np.nanstd(patterns_matrix, axis=0)
        
        # 计算每年与平均模式的偏离
        deviations = {}
        for i, year in enumerate(years):
            deviation = np.sqrt(np.mean((patterns_matrix[i] - mean_pattern) ** 2))
            deviations[year] = deviation
        
        # 趋势分析（如果有足够的年份）
        trend_analysis = {}
        if len(years) >= 5:
            for month_idx in range(patterns_matrix.shape[1]):
                month_values = patterns_matrix[:, month_idx]
                if not np.all(np.isnan(month_values)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years, month_values)
                    trend_analysis[f"month_{month_idx + 1}"] = {
                        "slope": slope,
                        "p_value": p_value,
                        "r_squared": r_value ** 2
                    }
        
        return {
            "comparison_possible": True,
            "years": years,
            "correlations": correlations,
            "mean_pattern": mean_pattern,
            "std_pattern": std_pattern,
            "yearly_deviations": deviations,
            "trend_analysis": trend_analysis,
            "most_stable_year": min(deviations.keys(), key=deviations.get) if deviations else None,
            "most_variable_year": max(deviations.keys(), key=deviations.get) if deviations else None
        }


class SeasonalVisualizer:
    """季节可视化器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_seasonal_decomposition(self, 
                                   time_index: pd.DatetimeIndex,
                                   original: np.ndarray,
                                   decomposition: Dict[str, np.ndarray],
                                   title: str = "季节分解") -> plt.Figure:
        """绘制季节分解图"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 原始序列
        axes[0].plot(time_index, original, 'b-', linewidth=1.5)
        axes[0].set_title('原始序列')
        axes[0].set_ylabel('数值')
        axes[0].grid(True, alpha=0.3)
        
        # 趋势分量
        if 'trend' in decomposition:
            axes[1].plot(time_index, decomposition['trend'], 'g-', linewidth=1.5)
            axes[1].set_title('趋势分量')
            axes[1].set_ylabel('趋势')
            axes[1].grid(True, alpha=0.3)
        
        # 季节分量
        if 'seasonal' in decomposition:
            axes[2].plot(time_index, decomposition['seasonal'], 'orange', linewidth=1.5)
            axes[2].set_title('季节分量')
            axes[2].set_ylabel('季节性')
            axes[2].grid(True, alpha=0.3)
        
        # 残差分量
        if 'residual' in decomposition:
            axes[3].plot(time_index, decomposition['residual'], 'r-', linewidth=1.5)
            axes[3].set_title('残差分量')
            axes[3].set_xlabel('时间')
            axes[3].set_ylabel('残差')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_seasonal_pattern(self, 
                             seasonal_component: np.ndarray,
                             period: int = 12,
                             title: str = "季节模式") -> plt.Figure:
        """绘制季节模式图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 季节分量时间序列
        ax1.plot(seasonal_component, 'b-', linewidth=2)
        ax1.set_title('季节分量时间序列')
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('季节分量')
        ax1.grid(True, alpha=0.3)
        
        # 平均季节模式
        if period <= len(seasonal_component):
            n_complete_cycles = len(seasonal_component) // period
            if n_complete_cycles > 1:
                reshaped = seasonal_component[:n_complete_cycles * period].reshape(n_complete_cycles, period)
                mean_pattern = np.mean(reshaped, axis=0)
                std_pattern = np.std(reshaped, axis=0)
                
                x = np.arange(1, period + 1)
                ax2.plot(x, mean_pattern, 'ro-', linewidth=2, markersize=6, label='平均模式')
                ax2.fill_between(x, mean_pattern - std_pattern, mean_pattern + std_pattern, 
                               alpha=0.3, label='标准差范围')
                
                if period == 12:
                    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(month_names)
                    ax2.set_xlabel('月份')
                else:
                    ax2.set_xlabel(f'周期位置 (周期={period})')
                
                ax2.set_title('平均季节模式')
                ax2.set_ylabel('季节分量')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_yearly_comparison(self, 
                              yearly_patterns: Dict[int, np.ndarray],
                              title: str = "年际季节对比") -> plt.Figure:
        """绘制年际季节对比图"""
        if len(yearly_patterns) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '数据不足，无法进行年际对比', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        years = sorted(yearly_patterns.keys())
        period = len(list(yearly_patterns.values())[0])
        
        # 所有年份的季节模式
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        x = np.arange(1, period + 1)
        
        for i, year in enumerate(years):
            pattern = yearly_patterns[year]
            ax1.plot(x, pattern, 'o-', color=colors[i], label=str(year), 
                    linewidth=1.5, markersize=4, alpha=0.8)
        
        if period == 12:
            month_names = [calendar.month_abbr[i] for i in range(1, 13)]
            ax1.set_xticks(x)
            ax1.set_xticklabels(month_names)
            ax1.set_xlabel('月份')
        else:
            ax1.set_xlabel(f'周期位置 (周期={period})')
        
        ax1.set_title('各年季节模式对比')
        ax1.set_ylabel('数值')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 热力图显示年际变化
        patterns_matrix = np.array([yearly_patterns[year] for year in years])
        im = ax2.imshow(patterns_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        
        ax2.set_yticks(range(len(years)))
        ax2.set_yticklabels(years)
        ax2.set_ylabel('年份')
        
        if period == 12:
            ax2.set_xticks(range(12))
            ax2.set_xticklabels(month_names)
            ax2.set_xlabel('月份')
        else:
            ax2.set_xlabel(f'周期位置 (周期={period})')
        
        ax2.set_title('年际季节变化热力图')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('数值')
        
        plt.tight_layout()
        return fig
    
    def plot_anomalies(self, 
                      time_index: pd.DatetimeIndex,
                      data: np.ndarray,
                      anomalies: List[int],
                      title: str = "季节异常检测") -> plt.Figure:
        """绘制异常检测图"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # 绘制原始数据
        ax.plot(time_index, data, 'b-', linewidth=1.5, label='原始数据', alpha=0.7)
        
        # 标记异常点
        if anomalies:
            anomaly_times = time_index[anomalies]
            anomaly_values = data[anomalies]
            ax.scatter(anomaly_times, anomaly_values, color='red', s=50, 
                      label=f'异常点 ({len(anomalies)}个)', zorder=5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间')
        ax.set_ylabel('数值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_seasonal_plot(self, 
                                        time_index: pd.DatetimeIndex,
                                        original: np.ndarray,
                                        decomposition: Dict[str, np.ndarray],
                                        title: str = "交互式季节分析") -> go.Figure:
        """创建交互式季节分析图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('原始序列', '趋势分量', '季节分量', '残差分量'),
            vertical_spacing=0.1
        )
        
        # 原始序列
        fig.add_trace(
            go.Scatter(x=time_index, y=original, mode='lines', name='原始序列',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 趋势分量
        if 'trend' in decomposition:
            fig.add_trace(
                go.Scatter(x=time_index, y=decomposition['trend'], mode='lines', 
                          name='趋势分量', line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        # 季节分量
        if 'seasonal' in decomposition:
            fig.add_trace(
                go.Scatter(x=time_index, y=decomposition['seasonal'], mode='lines', 
                          name='季节分量', line=dict(color='orange', width=2)),
                row=2, col=1
            )
        
        # 残差分量
        if 'residual' in decomposition:
            fig.add_trace(
                go.Scatter(x=time_index, y=decomposition['residual'], mode='lines', 
                          name='残差分量', line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig


class SeasonalAnalyzer:
    """主要的季节分析器"""
    
    def __init__(self, config: SeasonalConfig):
        self.config = config
        self.detector = SeasonalDetector(config)
        self.decomposer = SeasonalDecomposer(config)
        self.anomaly_detector = SeasonalAnomalyDetector(config)
        self.pattern_analyzer = SeasonalPatternAnalyzer(config)
        self.visualizer = SeasonalVisualizer(config)
    
    def analyze(self, data: SeasonalData) -> Dict[str, SeasonalResult]:
        """执行完整的季节分析"""
        results = {}
        
        for variable in data.variable_names:
            if variable not in data.time_series.columns:
                continue
            
            # 提取数据
            time_index = pd.to_datetime(data.time_series[data.time_column])
            var_data = data.time_series[variable].values
            
            # 移除NaN值
            mask = ~np.isnan(var_data)
            clean_time = time_index[mask]
            clean_data = var_data[mask]
            
            if len(clean_data) < 24:  # 需要至少2年的数据
                continue
            
            # 季节性检测
            seasonality_info = self.detector.detect_seasonality(clean_data, clean_time)
            
            if not seasonality_info["has_seasonality"]:
                # 如果没有检测到季节性，使用默认周期
                period = self.config.period.value if self.config.period != SeasonalPeriod.CUSTOM else 12
                dominant_periods = [period]
            else:
                dominant_periods = seasonality_info["dominant_periods"]
                period = dominant_periods[0] if dominant_periods else 12
            
            # 季节分解
            decomposition = self.decomposer.decompose(clean_data, period)
            
            # 异常检测
            anomalies = self.anomaly_detector.detect_anomalies(
                clean_data, 
                decomposition['seasonal'], 
                decomposition['residual']
            )
            
            # 年际模式分析
            yearly_patterns = self.pattern_analyzer.analyze_yearly_patterns(
                clean_data, clean_time, period
            )
            
            # 季节统计
            seasonal_stats = self.pattern_analyzer.calculate_seasonal_statistics(
                decomposition['seasonal'], period
            )
            
            # 计算季节振幅和相位
            seasonal_amplitude = np.max(decomposition['seasonal']) - np.min(decomposition['seasonal'])
            
            # 简单的相位计算（峰值位置）
            if period <= len(decomposition['seasonal']):
                n_cycles = len(decomposition['seasonal']) // period
                if n_cycles > 0:
                    reshaped = decomposition['seasonal'][:n_cycles * period].reshape(n_cycles, period)
                    mean_cycle = np.mean(reshaped, axis=0)
                    peak_position = np.argmax(mean_cycle)
                    seasonal_phase = peak_position / period * 2 * np.pi
                else:
                    seasonal_phase = 0.0
            else:
                seasonal_phase = 0.0
            
            # 分解质量评估
            reconstructed = decomposition['trend'] + decomposition['seasonal']
            decomposition_quality = 1 - np.var(clean_data - np.nan_to_num(reconstructed)) / np.var(clean_data)
            
            # 创建结果对象
            result = SeasonalResult(
                variable=variable,
                seasonal_component=decomposition['seasonal'],
                trend_component=decomposition['trend'],
                residual_component=decomposition['residual'],
                seasonal_strength=seasonality_info["seasonal_strength"],
                dominant_periods=dominant_periods,
                seasonal_peaks=seasonality_info["seasonal_peaks"],
                seasonal_troughs=seasonality_info["seasonal_troughs"],
                seasonal_amplitude=seasonal_amplitude,
                seasonal_phase=seasonal_phase,
                seasonal_anomalies=anomalies,
                yearly_patterns=yearly_patterns,
                seasonal_statistics=seasonal_stats,
                decomposition_quality=decomposition_quality
            )
            
            results[variable] = result
        
        return results
    
    def generate_summary(self, results: Dict[str, SeasonalResult]) -> Dict[str, Any]:
        """生成分析摘要"""
        summary = {
            "total_variables": len(results),
            "seasonal_variables": 0,
            "strong_seasonal": 0,
            "moderate_seasonal": 0,
            "weak_seasonal": 0,
            "variables_with_anomalies": 0,
            "average_seasonal_strength": 0.0,
            "common_periods": {},
            "seasonal_details": {}
        }
        
        if not results:
            return summary
        
        seasonal_strengths = []
        all_periods = []
        
        for variable, result in results.items():
            # 统计季节强度
            strength = result.seasonal_strength
            seasonal_strengths.append(strength)
            
            if strength > 0.1:
                summary["seasonal_variables"] += 1
                if strength > 0.7:
                    summary["strong_seasonal"] += 1
                elif strength > 0.3:
                    summary["moderate_seasonal"] += 1
                else:
                    summary["weak_seasonal"] += 1
            
            # 统计异常
            if result.seasonal_anomalies:
                summary["variables_with_anomalies"] += 1
            
            # 收集周期信息
            all_periods.extend(result.dominant_periods)
            
            # 详细信息
            summary["seasonal_details"][variable] = {
                "seasonal_strength": strength,
                "dominant_periods": result.dominant_periods,
                "seasonal_amplitude": result.seasonal_amplitude,
                "anomalies_count": len(result.seasonal_anomalies),
                "decomposition_quality": result.decomposition_quality
            }
        
        # 计算平均季节强度
        if seasonal_strengths:
            summary["average_seasonal_strength"] = np.mean(seasonal_strengths)
        
        # 统计常见周期
        if all_periods:
            from collections import Counter
            period_counts = Counter(all_periods)
            summary["common_periods"] = dict(period_counts.most_common(5))
        
        return summary
    
    def create_visualizations(self, 
                            data: SeasonalData, 
                            results: Dict[str, SeasonalResult]) -> Dict[str, plt.Figure]:
        """创建可视化图表"""
        figures = {}
        
        for variable, result in results.items():
            if variable not in data.time_series.columns:
                continue
            
            # 提取数据
            time_index = pd.to_datetime(data.time_series[data.time_column])
            var_data = data.time_series[variable].values
            
            # 移除NaN值
            mask = ~np.isnan(var_data)
            clean_time = time_index[mask]
            clean_data = var_data[mask]
            
            # 分解图
            decomposition = {
                'trend': result.trend_component,
                'seasonal': result.seasonal_component,
                'residual': result.residual_component
            }
            
            fig_decomp = self.visualizer.plot_seasonal_decomposition(
                clean_time, clean_data, decomposition,
                title=f"{variable} 季节分解"
            )
            figures[f"{variable}_decomposition"] = fig_decomp
            
            # 季节模式图
            period = result.dominant_periods[0] if result.dominant_periods else 12
            fig_pattern = self.visualizer.plot_seasonal_pattern(
                result.seasonal_component, period,
                title=f"{variable} 季节模式"
            )
            figures[f"{variable}_pattern"] = fig_pattern
            
            # 年际对比图
            if len(result.yearly_patterns) > 1:
                fig_yearly = self.visualizer.plot_yearly_comparison(
                    result.yearly_patterns,
                    title=f"{variable} 年际季节对比"
                )
                figures[f"{variable}_yearly"] = fig_yearly
            
            # 异常检测图
            if result.seasonal_anomalies:
                fig_anomalies = self.visualizer.plot_anomalies(
                    clean_time, clean_data, result.seasonal_anomalies,
                    title=f"{variable} 季节异常检测"
                )
                figures[f"{variable}_anomalies"] = fig_anomalies
        
        return figures
    
    def save_results(self, 
                    results: Dict[str, SeasonalResult], 
                    summary: Dict[str, Any],
                    output_dir: str = "seasonal_analysis_results"):
        """保存分析结果"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存摘要
        with open(os.path.join(output_dir, "seasonal_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细结果
        results_data = {}
        for variable, result in results.items():
            results_data[variable] = {
                "seasonal_strength": result.seasonal_strength,
                "dominant_periods": result.dominant_periods,
                "seasonal_amplitude": result.seasonal_amplitude,
                "seasonal_phase": result.seasonal_phase,
                "anomalies_count": len(result.seasonal_anomalies),
                "decomposition_quality": result.decomposition_quality,
                "seasonal_statistics": result.seasonal_statistics
            }
        
        with open(os.path.join(output_dir, "seasonal_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"季节分析结果已保存到: {output_dir}")


def create_seasonal_analyzer(config: Optional[SeasonalConfig] = None) -> SeasonalAnalyzer:
    """创建季节分析器的工厂函数"""
    if config is None:
        config = SeasonalConfig()
    return SeasonalAnalyzer(config)


if __name__ == "__main__":
    # 示例用法
    import datetime
    
    # 创建测试数据
    np.random.seed(42)
    n_years = 5
    n_months = n_years * 12
    
    # 创建月度时间序列
    dates = pd.date_range(start='2019-01-01', periods=n_months, freq='M')
    
    # 模拟冰川数据（带明显季节性）
    time_numeric = np.arange(n_months)
    
    # 长期趋势
    trend = -0.1 * time_numeric
    
    # 年周期（夏季融化，冬季积累）
    seasonal_annual = 5 * np.sin(2 * np.pi * time_numeric / 12 - np.pi/2)  # 夏季最小
    
    # 半年周期（较弱）
    seasonal_semi = 1 * np.sin(2 * np.pi * time_numeric / 6)
    
    # 噪声
    noise = np.random.normal(0, 1, n_months)
    
    # 组合数据
    glacier_mass_balance = 100 + trend + seasonal_annual + seasonal_semi + noise
    glacier_velocity = 200 + 0.5 * trend + 3 * seasonal_annual + 0.5 * noise
    glacier_temperature = 0 + 0.02 * time_numeric + 8 * np.sin(2 * np.pi * time_numeric / 12) + 0.5 * noise
    
    # 添加一些异常值
    anomaly_indices = [15, 28, 45]
    glacier_mass_balance[anomaly_indices] += np.random.normal(0, 5, len(anomaly_indices))
    
    # 创建数据框
    df = pd.DataFrame({
        'time': dates,
        'mass_balance': glacier_mass_balance,
        'velocity': glacier_velocity,
        'temperature': glacier_temperature
    })
    
    # 创建SeasonalData对象
    seasonal_data = SeasonalData(
        time_series=df,
        variable_names=['mass_balance', 'velocity', 'temperature'],
        time_column='time',
        frequency='M',
        metadata={'glacier_id': 'test_glacier', 'region': 'test_region'},
        units={'mass_balance': 'mm w.e.', 'velocity': 'm/year', 'temperature': '°C'}
    )
    
    # 创建配置
    config = SeasonalConfig(
        method=SeasonalMethod.STL_DECOMPOSITION,
        period=SeasonalPeriod.MONTHLY,
        anomaly_method=AnomalyMethod.SEASONAL_HYBRID,
        anomaly_threshold=2.0,
        detect_multiple_periods=True
    )
    
    # 创建分析器
    analyzer = create_seasonal_analyzer(config)
    
    # 执行分析
    print("执行季节分析...")
    results = analyzer.analyze(seasonal_data)
    
    # 生成摘要
    summary = analyzer.generate_summary(results)
    
    # 打印结果
    print("\n=== 季节分析摘要 ===")
    print(f"总变量数: {summary['total_variables']}")
    print(f"季节性变量数: {summary['seasonal_variables']}")
    print(f"强季节性: {summary['strong_seasonal']}")
    print(f"中等季节性: {summary['moderate_seasonal']}")
    print(f"弱季节性: {summary['weak_seasonal']}")
    print(f"平均季节强度: {summary['average_seasonal_strength']:.3f}")
    print(f"常见周期: {summary['common_periods']}")
    
    print("\n=== 详细结果 ===")
    for variable, result in results.items():
        print(f"\n{variable}:")
        print(f"  季节强度: {result.seasonal_strength:.3f}")
        print(f"  主要周期: {result.dominant_periods}")
        print(f"  季节振幅: {result.seasonal_amplitude:.3f}")
        print(f"  异常点数: {len(result.seasonal_anomalies)}")
        print(f"  分解质量: {result.decomposition_quality:.3f}")
        print(f"  年际模式数: {len(result.yearly_patterns)}")
    
    # 创建可视化
    print("\n创建可视化图表...")
    figures = analyzer.create_visualizations(seasonal_data, results)
    
    # 保存结果
    analyzer.save_results(results, summary)
    
    # 显示图表
    plt.show()
    
    print("\n季节分析完成！")