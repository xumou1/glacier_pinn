#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极端事件分析模块

本模块提供冰川极端事件的识别、分析和预测功能，包括：
- 极端质量损失事件检测
- 极端天气事件影响分析
- 复合极端事件识别
- 极端事件频率和强度分析
- 极端值理论应用

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtremeEventAnalyzer:
    """
    极端事件分析器
    
    提供多种极端事件分析方法，包括阈值法、百分位数法、
    极值理论、复合事件分析等功能。
    """
    
    def __init__(self, 
                 threshold_method: str = 'percentile',
                 threshold_value: float = 95.0):
        """
        初始化极端事件分析器
        
        参数:
            threshold_method: 阈值确定方法 ('percentile', 'std', 'absolute')
            threshold_value: 阈值数值
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        
    def detect_extreme_events(self, 
                            time_series: pd.Series,
                            event_type: str = 'both') -> Dict:
        """
        检测极端事件
        
        参数:
            time_series: 时间序列数据
            event_type: 事件类型 ('high', 'low', 'both')
            
        返回:
            极端事件检测结果
        """
        try:
            # 计算阈值
            thresholds = self._calculate_thresholds(time_series)
            
            # 检测极端事件
            extreme_events = {
                'high_threshold': thresholds['high'],
                'low_threshold': thresholds['low'],
                'method': self.threshold_method
            }
            
            if event_type in ['high', 'both']:
                high_events = time_series[time_series > thresholds['high']]
                extreme_events['high_events'] = {
                    'events': high_events,
                    'dates': high_events.index.tolist(),
                    'values': high_events.values.tolist(),
                    'count': len(high_events),
                    'frequency': len(high_events) / len(time_series) * 100
                }
            
            if event_type in ['low', 'both']:
                low_events = time_series[time_series < thresholds['low']]
                extreme_events['low_events'] = {
                    'events': low_events,
                    'dates': low_events.index.tolist(),
                    'values': low_events.values.tolist(),
                    'count': len(low_events),
                    'frequency': len(low_events) / len(time_series) * 100
                }
            
            # 事件聚类分析
            if event_type == 'both':
                extreme_events['clustering'] = self._cluster_extreme_events(
                    time_series, thresholds
                )
            
            return extreme_events
            
        except Exception as e:
            logger.error(f"极端事件检测失败: {e}")
            raise
    
    def extreme_value_analysis(self, 
                             time_series: pd.Series,
                             method: str = 'gev',
                             block_size: str = 'annual') -> Dict:
        """
        极值理论分析
        
        参数:
            time_series: 时间序列数据
            method: 分析方法 ('gev', 'gpd', 'gumbel')
            block_size: 块大小 ('annual', 'seasonal', 'monthly')
            
        返回:
            极值分析结果
        """
        try:
            # 提取块最大值
            if block_size == 'annual':
                block_maxima = time_series.groupby(time_series.index.year).max()
            elif block_size == 'seasonal':
                block_maxima = time_series.groupby([time_series.index.year, 
                                                  time_series.index.quarter]).max()
            elif block_size == 'monthly':
                block_maxima = time_series.groupby([time_series.index.year, 
                                                  time_series.index.month]).max()
            else:
                raise ValueError(f"不支持的块大小: {block_size}")
            
            block_maxima = block_maxima.dropna()
            
            if len(block_maxima) < 10:
                raise ValueError("块最大值数据点太少，无法进行极值分析")
            
            results = {
                'method': method,
                'block_size': block_size,
                'block_maxima': block_maxima,
                'n_blocks': len(block_maxima)
            }
            
            if method == 'gev':
                # 广义极值分布拟合
                gev_params = self._fit_gev_distribution(block_maxima.values)
                results.update(gev_params)
                
            elif method == 'gpd':
                # 广义帕累托分布拟合（超阈值方法）
                threshold = np.percentile(time_series.dropna(), 95)
                exceedances = time_series[time_series > threshold] - threshold
                
                if len(exceedances) < 10:
                    raise ValueError("超阈值数据点太少")
                
                gpd_params = self._fit_gpd_distribution(exceedances.values)
                results.update(gpd_params)
                results['threshold'] = threshold
                
            elif method == 'gumbel':
                # Gumbel分布拟合
                gumbel_params = self._fit_gumbel_distribution(block_maxima.values)
                results.update(gumbel_params)
            
            # 计算重现期
            return_periods = [2, 5, 10, 20, 50, 100]
            return_levels = self._calculate_return_levels(
                results, return_periods, method
            )
            results['return_analysis'] = {
                'return_periods': return_periods,
                'return_levels': return_levels
            }
            
            return results
            
        except Exception as e:
            logger.error(f"极值分析失败: {e}")
            raise
    
    def compound_event_analysis(self,
                              primary_series: pd.Series,
                              secondary_series: pd.Series,
                              correlation_threshold: float = 0.3) -> Dict:
        """
        复合极端事件分析
        
        参数:
            primary_series: 主要变量时间序列
            secondary_series: 次要变量时间序列
            correlation_threshold: 相关性阈值
            
        返回:
            复合事件分析结果
        """
        try:
            # 对齐时间序列
            common_index = primary_series.index.intersection(secondary_series.index)
            primary_aligned = primary_series.loc[common_index]
            secondary_aligned = secondary_series.loc[common_index]
            
            if len(common_index) < 10:
                raise ValueError("共同时间段数据点太少")
            
            # 计算阈值
            primary_threshold = np.percentile(primary_aligned.dropna(), 95)
            secondary_threshold = np.percentile(secondary_aligned.dropna(), 95)
            
            # 识别单独极端事件
            primary_extremes = primary_aligned > primary_threshold
            secondary_extremes = secondary_aligned > secondary_threshold
            
            # 识别复合极端事件
            compound_events = primary_extremes & secondary_extremes
            
            # 计算条件概率
            p_primary = np.sum(primary_extremes) / len(primary_extremes)
            p_secondary = np.sum(secondary_extremes) / len(secondary_extremes)
            p_compound = np.sum(compound_events) / len(compound_events)
            
            # 独立性检验
            p_independent = p_primary * p_secondary
            dependence_ratio = p_compound / p_independent if p_independent > 0 else np.inf
            
            # 相关性分析
            correlation, p_value = stats.pearsonr(
                primary_aligned.dropna(), secondary_aligned.dropna()
            )
            
            # 滞后相关性
            lag_correlations = self._calculate_lag_correlations(
                primary_aligned, secondary_aligned, max_lag=5
            )
            
            return {
                'primary_threshold': primary_threshold,
                'secondary_threshold': secondary_threshold,
                'primary_extreme_count': np.sum(primary_extremes),
                'secondary_extreme_count': np.sum(secondary_extremes),
                'compound_event_count': np.sum(compound_events),
                'compound_event_dates': common_index[compound_events].tolist(),
                'probabilities': {
                    'primary': p_primary,
                    'secondary': p_secondary,
                    'compound': p_compound,
                    'independent': p_independent
                },
                'dependence_ratio': dependence_ratio,
                'correlation': correlation,
                'correlation_p_value': p_value,
                'lag_correlations': lag_correlations,
                'is_dependent': dependence_ratio > 1.5 and correlation > correlation_threshold
            }
            
        except Exception as e:
            logger.error(f"复合事件分析失败: {e}")
            raise
    
    def event_duration_analysis(self, 
                              time_series: pd.Series,
                              threshold: Optional[float] = None) -> Dict:
        """
        事件持续时间分析
        
        参数:
            time_series: 时间序列数据
            threshold: 阈值，如果为None则自动计算
            
        返回:
            持续时间分析结果
        """
        try:
            if threshold is None:
                threshold = np.percentile(time_series.dropna(), 95)
            
            # 识别超阈值事件
            above_threshold = time_series > threshold
            
            # 计算事件持续时间
            durations = []
            intensities = []
            start_dates = []
            end_dates = []
            
            in_event = False
            event_start = None
            event_values = []
            
            for date, value in time_series.items():
                if above_threshold[date] and not in_event:
                    # 事件开始
                    in_event = True
                    event_start = date
                    event_values = [value]
                elif above_threshold[date] and in_event:
                    # 事件继续
                    event_values.append(value)
                elif not above_threshold[date] and in_event:
                    # 事件结束
                    in_event = False
                    duration = len(event_values)
                    intensity = np.mean(event_values) - threshold
                    
                    durations.append(duration)
                    intensities.append(intensity)
                    start_dates.append(event_start)
                    end_dates.append(date)
                    
                    event_values = []
            
            # 处理最后一个事件（如果序列结束时仍在事件中）
            if in_event and event_values:
                duration = len(event_values)
                intensity = np.mean(event_values) - threshold
                durations.append(duration)
                intensities.append(intensity)
                start_dates.append(event_start)
                end_dates.append(time_series.index[-1])
            
            if not durations:
                return {
                    'threshold': threshold,
                    'n_events': 0,
                    'durations': [],
                    'intensities': []
                }
            
            return {
                'threshold': threshold,
                'n_events': len(durations),
                'durations': durations,
                'intensities': intensities,
                'start_dates': start_dates,
                'end_dates': end_dates,
                'duration_stats': {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'max': np.max(durations),
                    'min': np.min(durations)
                },
                'intensity_stats': {
                    'mean': np.mean(intensities),
                    'std': np.std(intensities),
                    'max': np.max(intensities),
                    'min': np.min(intensities)
                }
            }
            
        except Exception as e:
            logger.error(f"事件持续时间分析失败: {e}")
            raise
    
    def spatial_extreme_analysis(self,
                               spatial_data: np.ndarray,
                               coordinates: np.ndarray,
                               return_period: int = 100) -> Dict:
        """
        空间极端事件分析
        
        参数:
            spatial_data: 空间数据 (time, lat, lon)
            coordinates: 坐标信息
            return_period: 重现期（年）
            
        返回:
            空间极端分析结果
        """
        try:
            nt, nlat, nlon = spatial_data.shape
            
            # 初始化结果数组
            return_level_map = np.full((nlat, nlon), np.nan)
            extreme_frequency_map = np.full((nlat, nlon), np.nan)
            
            for i in range(nlat):
                for j in range(nlon):
                    ts = spatial_data[:, i, j]
                    
                    if np.isnan(ts).all():
                        continue
                    
                    valid_data = ts[~np.isnan(ts)]
                    if len(valid_data) < 10:
                        continue
                    
                    try:
                        # 年最大值
                        # 假设数据是月度数据，每12个点为一年
                        n_years = len(valid_data) // 12
                        if n_years < 3:
                            continue
                        
                        annual_maxima = []
                        for year in range(n_years):
                            year_data = valid_data[year*12:(year+1)*12]
                            if len(year_data) > 0:
                                annual_maxima.append(np.max(year_data))
                        
                        if len(annual_maxima) < 3:
                            continue
                        
                        # GEV拟合
                        gev_params = self._fit_gev_distribution(np.array(annual_maxima))
                        
                        if gev_params['success']:
                            # 计算重现水平
                            return_level = self._gev_return_level(
                                gev_params['shape'], gev_params['location'], 
                                gev_params['scale'], return_period
                            )
                            return_level_map[i, j] = return_level
                        
                        # 极端事件频率
                        threshold = np.percentile(valid_data, 95)
                        extreme_count = np.sum(valid_data > threshold)
                        extreme_frequency_map[i, j] = extreme_count / len(valid_data) * 100
                        
                    except:
                        continue
            
            return {
                'return_level_map': return_level_map,
                'extreme_frequency_map': extreme_frequency_map,
                'return_period': return_period,
                'coordinates': coordinates
            }
            
        except Exception as e:
            logger.error(f"空间极端分析失败: {e}")
            raise
    
    def _calculate_thresholds(self, time_series: pd.Series) -> Dict:
        """
        计算极端事件阈值
        """
        data = time_series.dropna()
        
        if self.threshold_method == 'percentile':
            high_threshold = np.percentile(data, self.threshold_value)
            low_threshold = np.percentile(data, 100 - self.threshold_value)
        elif self.threshold_method == 'std':
            mean_val = data.mean()
            std_val = data.std()
            high_threshold = mean_val + self.threshold_value * std_val
            low_threshold = mean_val - self.threshold_value * std_val
        elif self.threshold_method == 'absolute':
            high_threshold = self.threshold_value
            low_threshold = -self.threshold_value
        else:
            raise ValueError(f"不支持的阈值方法: {self.threshold_method}")
        
        return {'high': high_threshold, 'low': low_threshold}
    
    def _cluster_extreme_events(self, 
                              time_series: pd.Series, 
                              thresholds: Dict) -> Dict:
        """
        对极端事件进行聚类分析
        """
        # 提取极端事件
        high_events = time_series[time_series > thresholds['high']]
        low_events = time_series[time_series < thresholds['low']]
        
        all_extremes = pd.concat([high_events, low_events])
        
        if len(all_extremes) < 3:
            return {'n_clusters': 0, 'labels': []}
        
        # 准备聚类数据（时间和数值）
        time_numeric = pd.to_numeric(all_extremes.index)
        features = np.column_stack([
            (time_numeric - time_numeric.min()) / (time_numeric.max() - time_numeric.min()),
            (all_extremes.values - all_extremes.min()) / (all_extremes.max() - all_extremes.min())
        ])
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.3, min_samples=2)
        labels = clustering.fit_predict(features)
        
        return {
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'labels': labels,
            'events': all_extremes
        }
    
    def _fit_gev_distribution(self, data: np.ndarray) -> Dict:
        """
        拟合广义极值分布
        """
        try:
            # 使用scipy的genextreme分布
            params = stats.genextreme.fit(data)
            shape, location, scale = params
            
            # 计算拟合优度
            ks_stat, ks_p = stats.kstest(data, lambda x: stats.genextreme.cdf(x, *params))
            
            return {
                'shape': shape,
                'location': location,
                'scale': scale,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'success': True
            }
        except:
            return {'success': False}
    
    def _fit_gpd_distribution(self, data: np.ndarray) -> Dict:
        """
        拟合广义帕累托分布
        """
        try:
            # 使用scipy的genpareto分布
            params = stats.genpareto.fit(data)
            shape, location, scale = params
            
            # 计算拟合优度
            ks_stat, ks_p = stats.kstest(data, lambda x: stats.genpareto.cdf(x, *params))
            
            return {
                'shape': shape,
                'location': location,
                'scale': scale,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'success': True
            }
        except:
            return {'success': False}
    
    def _fit_gumbel_distribution(self, data: np.ndarray) -> Dict:
        """
        拟合Gumbel分布
        """
        try:
            # 使用scipy的gumbel_r分布
            params = stats.gumbel_r.fit(data)
            location, scale = params
            
            # 计算拟合优度
            ks_stat, ks_p = stats.kstest(data, lambda x: stats.gumbel_r.cdf(x, *params))
            
            return {
                'location': location,
                'scale': scale,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'success': True
            }
        except:
            return {'success': False}
    
    def _calculate_return_levels(self, 
                               fit_results: Dict, 
                               return_periods: List[int],
                               method: str) -> List[float]:
        """
        计算重现水平
        """
        return_levels = []
        
        for T in return_periods:
            if method == 'gev' and fit_results.get('success', False):
                level = self._gev_return_level(
                    fit_results['shape'], fit_results['location'], 
                    fit_results['scale'], T
                )
            elif method == 'gumbel' and fit_results.get('success', False):
                level = self._gumbel_return_level(
                    fit_results['location'], fit_results['scale'], T
                )
            else:
                level = np.nan
            
            return_levels.append(level)
        
        return return_levels
    
    def _gev_return_level(self, shape: float, location: float, 
                         scale: float, return_period: int) -> float:
        """
        计算GEV分布的重现水平
        """
        p = 1 - 1/return_period
        
        if abs(shape) < 1e-6:  # shape ≈ 0, Gumbel分布
            return location - scale * np.log(-np.log(p))
        else:
            return location + scale/shape * ((-np.log(p))**(-shape) - 1)
    
    def _gumbel_return_level(self, location: float, scale: float, 
                           return_period: int) -> float:
        """
        计算Gumbel分布的重现水平
        """
        p = 1 - 1/return_period
        return location - scale * np.log(-np.log(p))
    
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
                s1 = series1.iloc[:-lag]
                s2 = series2.iloc[lag:]
                if len(s1) > 3:
                    corr, p_val = stats.pearsonr(s1, s2)
                else:
                    corr, p_val = np.nan, np.nan
            else:
                s1 = series1.iloc[-lag:]
                s2 = series2.iloc[:lag]
                if len(s1) > 3:
                    corr, p_val = stats.pearsonr(s1, s2)
                else:
                    corr, p_val = np.nan, np.nan
            
            lag_corrs[f'lag_{lag}'] = {'correlation': corr, 'p_value': p_val}
        
        return lag_corrs

class ExtremeEventVisualizer:
    """
    极端事件可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_extreme_events(self,
                          time_series: pd.Series,
                          extreme_results: Dict,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制极端事件检测结果
        
        参数:
            time_series: 原始时间序列
            extreme_results: 极端事件检测结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('极端事件分析结果', fontsize=16, fontweight='bold')
        
        # 时间序列和极端事件
        ax1 = axes[0, 0]
        ax1.plot(time_series.index, time_series.values, 'b-', alpha=0.7, 
                label='原始数据')
        
        # 绘制阈值线
        if 'high_threshold' in extreme_results:
            ax1.axhline(y=extreme_results['high_threshold'], color='r', 
                       linestyle='--', label=f"高阈值: {extreme_results['high_threshold']:.2f}")
        
        if 'low_threshold' in extreme_results:
            ax1.axhline(y=extreme_results['low_threshold'], color='g', 
                       linestyle='--', label=f"低阈值: {extreme_results['low_threshold']:.2f}")
        
        # 标记极端事件
        if 'high_events' in extreme_results:
            high_events = extreme_results['high_events']['events']
            ax1.scatter(high_events.index, high_events.values, 
                       color='red', s=50, label='高极值事件', zorder=5)
        
        if 'low_events' in extreme_results:
            low_events = extreme_results['low_events']['events']
            ax1.scatter(low_events.index, low_events.values, 
                       color='green', s=50, label='低极值事件', zorder=5)
        
        ax1.set_title('极端事件检测')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 极端事件频率
        ax2 = axes[0, 1]
        categories = []
        frequencies = []
        
        if 'high_events' in extreme_results:
            categories.append('高极值')
            frequencies.append(extreme_results['high_events']['frequency'])
        
        if 'low_events' in extreme_results:
            categories.append('低极值')
            frequencies.append(extreme_results['low_events']['frequency'])
        
        if categories:
            bars = ax2.bar(categories, frequencies, 
                          color=['red', 'green'][:len(categories)])
            ax2.set_title('极端事件频率')
            ax2.set_ylabel('频率 (%)')
            
            # 添加数值标签
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{freq:.1f}%', ha='center', va='bottom')
        
        # 极端值分布
        ax3 = axes[1, 0]
        all_extremes = []
        labels = []
        
        if 'high_events' in extreme_results:
            all_extremes.extend(extreme_results['high_events']['values'])
            labels.extend(['高极值'] * len(extreme_results['high_events']['values']))
        
        if 'low_events' in extreme_results:
            all_extremes.extend(extreme_results['low_events']['values'])
            labels.extend(['低极值'] * len(extreme_results['low_events']['values']))
        
        if all_extremes:
            df_extremes = pd.DataFrame({'值': all_extremes, '类型': labels})
            
            for i, event_type in enumerate(df_extremes['类型'].unique()):
                data = df_extremes[df_extremes['类型'] == event_type]['值']
                ax3.hist(data, bins=10, alpha=0.7, 
                        label=event_type, color=['red', 'green'][i])
            
            ax3.set_title('极端值分布')
            ax3.set_xlabel('值')
            ax3.set_ylabel('频次')
            ax3.legend()
        
        # 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        极端事件统计信息:
        
        检测方法: {extreme_results.get('method', 'N/A')}
        """
        
        if 'high_events' in extreme_results:
            stats_text += f"""
        
        高极值事件:
        - 阈值: {extreme_results['high_threshold']:.3f}
        - 事件数: {extreme_results['high_events']['count']}
        - 频率: {extreme_results['high_events']['frequency']:.1f}%
        """
        
        if 'low_events' in extreme_results:
            stats_text += f"""
        
        低极值事件:
        - 阈值: {extreme_results['low_threshold']:.3f}
        - 事件数: {extreme_results['low_events']['count']}
        - 频率: {extreme_results['low_events']['frequency']:.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_return_level_analysis(self,
                                 extreme_value_results: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制重现水平分析结果
        
        参数:
            extreme_value_results: 极值分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('重现水平分析', fontsize=16, fontweight='bold')
        
        # 块最大值时间序列
        ax1 = axes[0, 0]
        block_maxima = extreme_value_results['block_maxima']
        ax1.plot(block_maxima.index, block_maxima.values, 'o-')
        ax1.set_title(f'块最大值 ({extreme_value_results["block_size"]})')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('最大值')
        ax1.grid(True, alpha=0.3)
        
        # 重现水平图
        ax2 = axes[0, 1]
        if 'return_analysis' in extreme_value_results:
            return_periods = extreme_value_results['return_analysis']['return_periods']
            return_levels = extreme_value_results['return_analysis']['return_levels']
            
            ax2.semilogx(return_periods, return_levels, 'o-', linewidth=2)
            ax2.set_title('重现水平')
            ax2.set_xlabel('重现期 (年)')
            ax2.set_ylabel('重现水平')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for period, level in zip(return_periods, return_levels):
                if not np.isnan(level):
                    ax2.annotate(f'{level:.2f}', (period, level),
                               xytext=(5, 5), textcoords='offset points')
        
        # 分布拟合
        ax3 = axes[1, 0]
        if extreme_value_results.get('success', False):
            # 绘制经验分布和理论分布
            data = block_maxima.values
            sorted_data = np.sort(data)
            n = len(data)
            empirical_prob = np.arange(1, n+1) / (n+1)
            
            ax3.plot(sorted_data, empirical_prob, 'o', label='经验分布')
            
            # 理论分布
            if extreme_value_results['method'] == 'gev':
                shape = extreme_value_results['shape']
                location = extreme_value_results['location']
                scale = extreme_value_results['scale']
                theoretical_prob = stats.genextreme.cdf(sorted_data, shape, 
                                                       loc=location, scale=scale)
            elif extreme_value_results['method'] == 'gumbel':
                location = extreme_value_results['location']
                scale = extreme_value_results['scale']
                theoretical_prob = stats.gumbel_r.cdf(sorted_data, 
                                                     loc=location, scale=scale)
            
            ax3.plot(sorted_data, theoretical_prob, '-', label='理论分布')
            ax3.set_title('分布拟合检验')
            ax3.set_xlabel('值')
            ax3.set_ylabel('累积概率')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 参数信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if extreme_value_results.get('success', False):
            method = extreme_value_results['method'].upper()
            stats_text = f"""
            {method}分布拟合结果:
            
            方法: {extreme_value_results['method']}
            块大小: {extreme_value_results['block_size']}
            块数量: {extreme_value_results['n_blocks']}
            """
            
            if 'shape' in extreme_value_results:
                stats_text += f"""
            
            参数估计:
            - 形状参数: {extreme_value_results['shape']:.4f}
            - 位置参数: {extreme_value_results['location']:.4f}
            - 尺度参数: {extreme_value_results['scale']:.4f}
            """
            
            if 'ks_p_value' in extreme_value_results:
                stats_text += f"""
            
            拟合检验:
            - KS统计量: {extreme_value_results['ks_statistic']:.4f}
            - p值: {extreme_value_results['ks_p_value']:.4f}
            """
        else:
            stats_text = "分布拟合失败"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """
    主函数 - 演示极端事件分析功能
    """
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2020-12-31', freq='M')
    
    # 模拟冰川质量平衡数据（包含极端事件）
    base_trend = -0.3 * np.arange(len(dates)) / 12
    seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 1, len(dates))
    
    # 添加一些极端事件
    extreme_events = np.zeros(len(dates))
    extreme_indices = np.random.choice(len(dates), size=10, replace=False)
    extreme_events[extreme_indices] = np.random.normal(0, 3, 10)
    
    glacier_mb = base_trend + seasonal + noise + extreme_events
    ts = pd.Series(glacier_mb, index=dates)
    
    # 初始化分析器
    analyzer = ExtremeEventAnalyzer(threshold_method='percentile', threshold_value=95)
    visualizer = ExtremeEventVisualizer()
    
    print("开始极端事件分析...")
    
    # 极端事件检测
    extreme_results = analyzer.detect_extreme_events(ts, event_type='both')
    print(f"\n极端事件检测结果:")
    if 'high_events' in extreme_results:
        print(f"高极值事件: {extreme_results['high_events']['count']} 个")
        print(f"高极值频率: {extreme_results['high_events']['frequency']:.1f}%")
    if 'low_events' in extreme_results:
        print(f"低极值事件: {extreme_results['low_events']['count']} 个")
        print(f"低极值频率: {extreme_results['low_events']['frequency']:.1f}%")
    
    # 极值理论分析
    evt_results = analyzer.extreme_value_analysis(ts, method='gev', block_size='annual')
    print(f"\n极值理论分析:")
    if evt_results.get('success', False):
        print(f"GEV参数 - 形状: {evt_results['shape']:.4f}, "
              f"位置: {evt_results['location']:.4f}, "
              f"尺度: {evt_results['scale']:.4f}")
        
        return_levels = evt_results['return_analysis']['return_levels']
        return_periods = evt_results['return_analysis']['return_periods']
        print(f"重现水平:")
        for period, level in zip(return_periods, return_levels):
            print(f"  {period}年重现期: {level:.2f}")
    
    # 事件持续时间分析
    duration_results = analyzer.event_duration_analysis(ts)
    print(f"\n事件持续时间分析:")
    print(f"检测到 {duration_results['n_events']} 个持续事件")
    if duration_results['n_events'] > 0:
        print(f"平均持续时间: {duration_results['duration_stats']['mean']:.1f}")
        print(f"最长持续时间: {duration_results['duration_stats']['max']}")
    
    # 可视化
    fig1 = visualizer.plot_extreme_events(ts, extreme_results)
    
    if evt_results.get('success', False):
        fig2 = visualizer.plot_return_level_analysis(evt_results)
    
    plt.show()
    
    print("\n极端事件分析完成！")

if __name__ == "__main__":
    main()