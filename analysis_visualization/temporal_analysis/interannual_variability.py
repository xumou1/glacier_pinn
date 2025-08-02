#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年际变异分析模块

本模块提供冰川质量平衡和厚度变化的年际变异性分析功能，包括：
- 年际变异性指标计算
- 周期性分析
- 气候指数相关性分析
- 异常年份识别
- 变异性空间分布分析

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterannualVariabilityAnalyzer:
    """
    年际变异性分析器
    
    提供多种年际变异性分析方法，包括变异性指标计算、
    周期性分析、气候指数相关性分析等功能。
    """
    
    def __init__(self, reference_period: Optional[Tuple[str, str]] = None):
        """
        初始化年际变异性分析器
        
        参数:
            reference_period: 参考时期 (start_year, end_year)
        """
        self.reference_period = reference_period
        self.climate_indices = {}
        
    def calculate_variability_metrics(self, 
                                    time_series: pd.Series,
                                    annual_aggregation: str = 'mean') -> Dict:
        """
        计算年际变异性指标
        
        参数:
            time_series: 时间序列数据
            annual_aggregation: 年度聚合方法 ('mean', 'sum', 'min', 'max')
            
        返回:
            变异性指标字典
        """
        try:
            # 转换为年度数据
            annual_data = self._aggregate_to_annual(time_series, annual_aggregation)
            
            if len(annual_data) < 3:
                raise ValueError("年度数据点太少，无法计算变异性指标")
            
            # 基本统计量
            mean_val = annual_data.mean()
            std_val = annual_data.std()
            cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
            
            # 变异性指标
            metrics = {
                'mean': mean_val,
                'std': std_val,
                'coefficient_of_variation': cv,
                'range': annual_data.max() - annual_data.min(),
                'iqr': annual_data.quantile(0.75) - annual_data.quantile(0.25),
                'skewness': stats.skew(annual_data),
                'kurtosis': stats.kurtosis(annual_data)
            }
            
            # 相对变异性（相对于参考期）
            if self.reference_period:
                ref_data = self._get_reference_data(annual_data)
                if len(ref_data) > 0:
                    ref_mean = ref_data.mean()
                    ref_std = ref_data.std()
                    
                    metrics.update({
                        'reference_mean': ref_mean,
                        'reference_std': ref_std,
                        'relative_change': (mean_val - ref_mean) / ref_mean * 100,
                        'variability_change': (std_val - ref_std) / ref_std * 100
                    })
            
            # 年际变化率
            annual_changes = annual_data.diff().dropna()
            metrics.update({
                'mean_annual_change': annual_changes.mean(),
                'std_annual_change': annual_changes.std(),
                'max_annual_increase': annual_changes.max(),
                'max_annual_decrease': annual_changes.min()
            })
            
            # 持续性指标
            metrics.update(self._calculate_persistence_metrics(annual_data))
            
            return metrics
            
        except Exception as e:
            logger.error(f"变异性指标计算失败: {e}")
            raise
    
    def spectral_analysis(self, 
                         time_series: pd.Series,
                         method: str = 'fft') -> Dict:
        """
        频谱分析
        
        参数:
            time_series: 时间序列数据
            method: 分析方法 ('fft', 'welch', 'periodogram')
            
        返回:
            频谱分析结果
        """
        try:
            # 预处理数据
            data = time_series.dropna()
            if len(data) < 10:
                raise ValueError("数据点太少，无法进行频谱分析")
            
            # 去趋势
            detrended = signal.detrend(data.values)
            
            if method == 'fft':
                # 快速傅里叶变换
                n = len(detrended)
                fft_vals = fft(detrended)
                freqs = fftfreq(n, d=1.0)  # 假设采样间隔为1
                
                # 只取正频率部分
                pos_mask = freqs > 0
                freqs = freqs[pos_mask]
                power = np.abs(fft_vals[pos_mask])**2
                
            elif method == 'welch':
                # Welch方法
                freqs, power = signal.welch(detrended, nperseg=min(len(detrended)//4, 256))
                
            elif method == 'periodogram':
                # 周期图
                freqs, power = signal.periodogram(detrended)
            
            # 转换为周期
            periods = 1 / freqs[freqs > 0]
            power_periods = power[freqs > 0]
            
            # 找到主要周期
            dominant_indices = np.argsort(power_periods)[-5:]  # 前5个主要周期
            dominant_periods = periods[dominant_indices]
            dominant_powers = power_periods[dominant_indices]
            
            return {
                'method': method,
                'frequencies': freqs,
                'power_spectrum': power,
                'periods': periods,
                'power_periods': power_periods,
                'dominant_periods': dominant_periods,
                'dominant_powers': dominant_powers,
                'total_power': np.sum(power)
            }
            
        except Exception as e:
            logger.error(f"频谱分析失败: {e}")
            raise
    
    def climate_correlation_analysis(self,
                                   glacier_data: pd.Series,
                                   climate_indices: Dict[str, pd.Series]) -> Dict:
        """
        气候指数相关性分析
        
        参数:
            glacier_data: 冰川数据
            climate_indices: 气候指数字典 (如ENSO, NAO, PDO等)
            
        返回:
            相关性分析结果
        """
        try:
            # 转换为年度数据
            annual_glacier = self._aggregate_to_annual(glacier_data, 'mean')
            
            correlations = {}
            
            for index_name, index_data in climate_indices.items():
                # 转换气候指数为年度数据
                annual_index = self._aggregate_to_annual(index_data, 'mean')
                
                # 找到共同时间段
                common_years = annual_glacier.index.intersection(annual_index.index)
                
                if len(common_years) < 5:
                    logger.warning(f"与{index_name}的共同时间段太短")
                    continue
                
                glacier_common = annual_glacier.loc[common_years]
                index_common = annual_index.loc[common_years]
                
                # 计算相关性
                pearson_r, pearson_p = stats.pearsonr(glacier_common, index_common)
                spearman_r, spearman_p = stats.spearmanr(glacier_common, index_common)
                
                # 滞后相关性分析
                lag_correlations = self._calculate_lag_correlations(
                    glacier_common, index_common, max_lag=3
                )
                
                correlations[index_name] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'lag_correlations': lag_correlations,
                    'n_years': len(common_years)
                }
            
            return correlations
            
        except Exception as e:
            logger.error(f"气候相关性分析失败: {e}")
            raise
    
    def anomaly_detection(self,
                         time_series: pd.Series,
                         threshold: float = 2.0,
                         method: str = 'zscore') -> Dict:
        """
        异常年份检测
        
        参数:
            time_series: 时间序列数据
            threshold: 异常阈值
            method: 检测方法 ('zscore', 'iqr', 'isolation_forest')
            
        返回:
            异常检测结果
        """
        try:
            annual_data = self._aggregate_to_annual(time_series, 'mean')
            
            if method == 'zscore':
                # Z-score方法
                z_scores = np.abs(stats.zscore(annual_data))
                anomalies = annual_data[z_scores > threshold]
                
            elif method == 'iqr':
                # 四分位距方法
                Q1 = annual_data.quantile(0.25)
                Q3 = annual_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                anomalies = annual_data[
                    (annual_data < lower_bound) | (annual_data > upper_bound)
                ]
                
            elif method == 'isolation_forest':
                # 孤立森林方法
                from sklearn.ensemble import IsolationForest
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(annual_data.values.reshape(-1, 1))
                anomalies = annual_data[outliers == -1]
            
            # 分类异常类型
            mean_val = annual_data.mean()
            extreme_high = anomalies[anomalies > mean_val]
            extreme_low = anomalies[anomalies < mean_val]
            
            return {
                'method': method,
                'threshold': threshold,
                'anomalies': anomalies,
                'extreme_high_years': extreme_high.index.tolist(),
                'extreme_low_years': extreme_low.index.tolist(),
                'n_anomalies': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(annual_data) * 100
            }
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            raise
    
    def regime_shift_analysis(self,
                            time_series: pd.Series,
                            min_regime_length: int = 5) -> Dict:
        """
        状态转换分析
        
        参数:
            time_series: 时间序列数据
            min_regime_length: 最小状态长度
            
        返回:
            状态转换分析结果
        """
        try:
            annual_data = self._aggregate_to_annual(time_series, 'mean')
            
            # 使用K-means聚类识别不同状态
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(annual_data.values.reshape(-1, 1))
            
            # 尝试不同的聚类数
            best_k = 2
            best_score = -np.inf
            
            for k in range(2, min(6, len(annual_data)//3)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                score = kmeans.inertia_
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # 最终聚类
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            regime_labels = kmeans.fit_predict(scaled_data)
            
            # 识别状态转换点
            regime_changes = []
            current_regime = regime_labels[0]
            
            for i, regime in enumerate(regime_labels[1:], 1):
                if regime != current_regime:
                    regime_changes.append(annual_data.index[i])
                    current_regime = regime
            
            # 计算每个状态的特征
            regime_stats = {}
            for regime_id in range(best_k):
                regime_mask = regime_labels == regime_id
                regime_data = annual_data[regime_mask]
                
                regime_stats[f'regime_{regime_id}'] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'years': regime_data.index.tolist(),
                    'duration': len(regime_data)
                }
            
            return {
                'n_regimes': best_k,
                'regime_labels': regime_labels,
                'regime_changes': regime_changes,
                'regime_statistics': regime_stats,
                'regime_series': pd.Series(regime_labels, index=annual_data.index)
            }
            
        except Exception as e:
            logger.error(f"状态转换分析失败: {e}")
            raise
    
    def spatial_variability_analysis(self,
                                   spatial_data: np.ndarray,
                                   coordinates: np.ndarray) -> Dict:
        """
        空间变异性分析
        
        参数:
            spatial_data: 空间数据 (time, lat, lon)
            coordinates: 坐标信息
            
        返回:
            空间变异性分析结果
        """
        try:
            nt, nlat, nlon = spatial_data.shape
            
            # 计算每个网格点的年际变异性
            cv_map = np.full((nlat, nlon), np.nan)
            std_map = np.full((nlat, nlon), np.nan)
            trend_map = np.full((nlat, nlon), np.nan)
            
            for i in range(nlat):
                for j in range(nlon):
                    ts = spatial_data[:, i, j]
                    
                    if np.isnan(ts).all():
                        continue
                    
                    valid_data = ts[~np.isnan(ts)]
                    if len(valid_data) < 5:
                        continue
                    
                    # 变异系数
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    cv_map[i, j] = std_val / abs(mean_val) if mean_val != 0 else np.nan
                    std_map[i, j] = std_val
                    
                    # 趋势
                    time_idx = np.arange(len(valid_data))
                    slope, _, _, _, _ = stats.linregress(time_idx, valid_data)
                    trend_map[i, j] = slope
            
            # 空间相关性分析
            spatial_correlation = self._calculate_spatial_correlation(spatial_data)
            
            # 主成分分析
            pca_results = self._spatial_pca_analysis(spatial_data)
            
            return {
                'cv_map': cv_map,
                'std_map': std_map,
                'trend_map': trend_map,
                'spatial_correlation': spatial_correlation,
                'pca_results': pca_results,
                'coordinates': coordinates
            }
            
        except Exception as e:
            logger.error(f"空间变异性分析失败: {e}")
            raise
    
    def _aggregate_to_annual(self, 
                           time_series: pd.Series, 
                           method: str) -> pd.Series:
        """
        将时间序列聚合为年度数据
        """
        if method == 'mean':
            return time_series.groupby(time_series.index.year).mean()
        elif method == 'sum':
            return time_series.groupby(time_series.index.year).sum()
        elif method == 'min':
            return time_series.groupby(time_series.index.year).min()
        elif method == 'max':
            return time_series.groupby(time_series.index.year).max()
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
    
    def _get_reference_data(self, annual_data: pd.Series) -> pd.Series:
        """
        获取参考期数据
        """
        if not self.reference_period:
            return pd.Series()
        
        start_year, end_year = self.reference_period
        return annual_data[
            (annual_data.index >= int(start_year)) & 
            (annual_data.index <= int(end_year))
        ]
    
    def _calculate_persistence_metrics(self, annual_data: pd.Series) -> Dict:
        """
        计算持续性指标
        """
        # 自相关
        autocorr_1 = annual_data.autocorr(lag=1)
        autocorr_2 = annual_data.autocorr(lag=2)
        
        # Hurst指数（简化计算）
        def hurst_exponent(ts):
            lags = range(2, min(20, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        try:
            hurst = hurst_exponent(annual_data.values)
        except:
            hurst = np.nan
        
        return {
            'autocorr_lag1': autocorr_1,
            'autocorr_lag2': autocorr_2,
            'hurst_exponent': hurst
        }
    
    def _calculate_lag_correlations(self,
                                  series1: pd.Series,
                                  series2: pd.Series,
                                  max_lag: int) -> Dict:
        """
        计算滞后相关性
        """
        lag_corrs = {}
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr, p_val = stats.pearsonr(series1, series2)
            elif lag > 0:
                # series2滞后
                s1 = series1.iloc[:-lag]
                s2 = series2.iloc[lag:]
                if len(s1) > 3:
                    corr, p_val = stats.pearsonr(s1, s2)
                else:
                    corr, p_val = np.nan, np.nan
            else:
                # series1滞后
                s1 = series1.iloc[-lag:]
                s2 = series2.iloc[:lag]
                if len(s1) > 3:
                    corr, p_val = stats.pearsonr(s1, s2)
                else:
                    corr, p_val = np.nan, np.nan
            
            lag_corrs[f'lag_{lag}'] = {'correlation': corr, 'p_value': p_val}
        
        return lag_corrs
    
    def _calculate_spatial_correlation(self, spatial_data: np.ndarray) -> Dict:
        """
        计算空间相关性
        """
        nt, nlat, nlon = spatial_data.shape
        
        # 重塑数据
        reshaped_data = spatial_data.reshape(nt, -1)
        
        # 移除全NaN列
        valid_cols = ~np.isnan(reshaped_data).all(axis=0)
        valid_data = reshaped_data[:, valid_cols]
        
        if valid_data.shape[1] < 2:
            return {'mean_correlation': np.nan, 'correlation_matrix': None}
        
        # 计算相关矩阵
        corr_matrix = np.corrcoef(valid_data.T)
        
        # 移除对角线元素
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix[mask]
        
        return {
            'mean_correlation': np.nanmean(correlations),
            'correlation_matrix': corr_matrix,
            'correlation_std': np.nanstd(correlations)
        }
    
    def _spatial_pca_analysis(self, spatial_data: np.ndarray) -> Dict:
        """
        空间主成分分析
        """
        nt, nlat, nlon = spatial_data.shape
        
        # 重塑数据
        reshaped_data = spatial_data.reshape(nt, -1)
        
        # 移除全NaN列
        valid_cols = ~np.isnan(reshaped_data).all(axis=0)
        valid_data = reshaped_data[:, valid_cols]
        
        if valid_data.shape[1] < 2:
            return {'explained_variance_ratio': None, 'components': None}
        
        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(valid_data.T).T
        
        # PCA
        pca = PCA(n_components=min(5, valid_data.shape[0], valid_data.shape[1]))
        pca.fit(scaled_data)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'n_components': pca.n_components_
        }

class VariabilityVisualizer:
    """
    年际变异性可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_variability_analysis(self,
                                time_series: pd.Series,
                                variability_results: Dict,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制变异性分析结果
        
        参数:
            time_series: 原始时间序列
            variability_results: 变异性分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('年际变异性分析结果', fontsize=16, fontweight='bold')
        
        # 年度时间序列
        analyzer = InterannualVariabilityAnalyzer()
        annual_data = analyzer._aggregate_to_annual(time_series, 'mean')
        
        ax1 = axes[0, 0]
        ax1.plot(annual_data.index, annual_data.values, 'o-', linewidth=2)
        ax1.axhline(y=annual_data.mean(), color='r', linestyle='--', 
                   label=f'平均值: {annual_data.mean():.2f}')
        ax1.fill_between(annual_data.index, 
                        annual_data.mean() - annual_data.std(),
                        annual_data.mean() + annual_data.std(),
                        alpha=0.3, label=f'±1σ: {annual_data.std():.2f}')
        ax1.set_title('年度时间序列')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 变异性指标
        ax2 = axes[0, 1]
        metrics = ['std', 'coefficient_of_variation', 'range', 'iqr']
        values = [variability_results.get(m, 0) for m in metrics]
        labels = ['标准差', '变异系数', '极差', '四分位距']
        
        bars = ax2.bar(labels, values, color=self.colors[:len(metrics)])
        ax2.set_title('变异性指标')
        ax2.set_ylabel('值')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 分布直方图
        ax3 = axes[0, 2]
        ax3.hist(annual_data.values, bins=15, alpha=0.7, edgecolor='black')
        ax3.axvline(annual_data.mean(), color='r', linestyle='--', 
                   label='平均值')
        ax3.axvline(annual_data.median(), color='g', linestyle='--', 
                   label='中位数')
        ax3.set_title('数据分布')
        ax3.set_xlabel('值')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 年际变化
        ax4 = axes[1, 0]
        annual_changes = annual_data.diff().dropna()
        ax4.plot(annual_changes.index, annual_changes.values, 'o-')
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('年际变化')
        ax4.set_xlabel('年份')
        ax4.set_ylabel('变化量')
        ax4.grid(True, alpha=0.3)
        
        # 自相关
        ax5 = axes[1, 1]
        lags = range(1, min(11, len(annual_data)//2))
        autocorrs = [annual_data.autocorr(lag=lag) for lag in lags]
        
        ax5.bar(lags, autocorrs, alpha=0.7)
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_title('自相关函数')
        ax5.set_xlabel('滞后期')
        ax5.set_ylabel('自相关系数')
        ax5.grid(True, alpha=0.3)
        
        # 统计信息
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
        变异性统计信息:
        
        平均值: {variability_results.get('mean', 'N/A'):.3f}
        标准差: {variability_results.get('std', 'N/A'):.3f}
        变异系数: {variability_results.get('coefficient_of_variation', 'N/A'):.3f}
        偏度: {variability_results.get('skewness', 'N/A'):.3f}
        峰度: {variability_results.get('kurtosis', 'N/A'):.3f}
        
        年际变化:
        平均变化: {variability_results.get('mean_annual_change', 'N/A'):.3f}
        最大增加: {variability_results.get('max_annual_increase', 'N/A'):.3f}
        最大减少: {variability_results.get('max_annual_decrease', 'N/A'):.3f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spectral_analysis(self,
                             spectral_results: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制频谱分析结果
        
        参数:
            spectral_results: 频谱分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('频谱分析结果', fontsize=16, fontweight='bold')
        
        # 功率谱
        ax1 = axes[0]
        freqs = spectral_results['frequencies']
        power = spectral_results['power_spectrum']
        
        ax1.loglog(freqs, power)
        ax1.set_title('功率谱密度')
        ax1.set_xlabel('频率')
        ax1.set_ylabel('功率')
        ax1.grid(True, alpha=0.3)
        
        # 周期图
        ax2 = axes[1]
        periods = spectral_results['periods']
        power_periods = spectral_results['power_periods']
        
        # 只显示合理的周期范围
        valid_mask = (periods >= 2) & (periods <= 50)
        
        ax2.semilogx(periods[valid_mask], power_periods[valid_mask])
        
        # 标记主要周期
        dominant_periods = spectral_results['dominant_periods']
        dominant_powers = spectral_results['dominant_powers']
        
        for period, power in zip(dominant_periods, dominant_powers):
            if 2 <= period <= 50:
                ax2.axvline(x=period, color='r', linestyle='--', alpha=0.7)
                ax2.text(period, power, f'{period:.1f}', 
                        rotation=90, verticalalignment='bottom')
        
        ax2.set_title('周期分析')
        ax2.set_xlabel('周期')
        ax2.set_ylabel('功率')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """
    主函数 - 演示年际变异性分析功能
    """
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2020-12-31', freq='M')
    
    # 模拟冰川质量平衡数据
    trend = -0.3 * np.arange(len(dates)) / 12
    seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    interannual = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / (3.5 * 12))  # 3.5年周期
    noise = np.random.normal(0, 0.8, len(dates))
    
    glacier_mb = trend + seasonal + interannual + noise
    ts = pd.Series(glacier_mb, index=dates)
    
    # 初始化分析器
    analyzer = InterannualVariabilityAnalyzer(reference_period=('2000', '2010'))
    visualizer = VariabilityVisualizer()
    
    print("开始年际变异性分析...")
    
    # 变异性指标计算
    variability_metrics = analyzer.calculate_variability_metrics(ts)
    print(f"\n变异性指标:")
    print(f"变异系数: {variability_metrics['coefficient_of_variation']:.3f}")
    print(f"标准差: {variability_metrics['std']:.3f}")
    print(f"偏度: {variability_metrics['skewness']:.3f}")
    
    # 频谱分析
    spectral_results = analyzer.spectral_analysis(ts)
    print(f"\n主要周期: {spectral_results['dominant_periods']}")
    
    # 异常检测
    anomaly_results = analyzer.anomaly_detection(ts)
    print(f"\n异常年份检测:")
    print(f"检测到 {anomaly_results['n_anomalies']} 个异常年份")
    print(f"极端高值年份: {anomaly_results['extreme_high_years']}")
    print(f"极端低值年份: {anomaly_results['extreme_low_years']}")
    
    # 状态转换分析
    regime_results = analyzer.regime_shift_analysis(ts)
    print(f"\n状态转换分析:")
    print(f"识别出 {regime_results['n_regimes']} 个状态")
    print(f"状态转换点: {regime_results['regime_changes']}")
    
    # 可视化
    fig1 = visualizer.plot_variability_analysis(ts, variability_metrics)
    fig2 = visualizer.plot_spectral_analysis(spectral_results)
    
    plt.show()
    
    print("\n年际变异性分析完成！")

if __name__ == "__main__":
    main()