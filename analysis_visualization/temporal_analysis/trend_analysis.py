#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势分析模块

本模块提供冰川质量平衡和厚度变化的长期趋势分析功能，包括：
- 线性和非线性趋势检测
- 趋势显著性检验
- 变化点检测
- 趋势空间分布分析
- 多时间尺度趋势分解

作者: 冰川PINNs项目组
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """
    冰川趋势分析器
    
    提供多种趋势分析方法，包括线性趋势、非线性趋势、
    变化点检测和趋势显著性检验等功能。
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化趋势分析器
        
        参数:
            confidence_level: 置信水平，默认0.95
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def linear_trend_analysis(self, 
                            time_series: pd.Series,
                            method: str = 'ols') -> Dict:
        """
        线性趋势分析
        
        参数:
            time_series: 时间序列数据
            method: 回归方法 ('ols', 'theil_sen', 'mann_kendall')
            
        返回:
            包含趋势参数的字典
        """
        try:
            # 准备数据
            valid_data = time_series.dropna()
            if len(valid_data) < 3:
                raise ValueError("有效数据点太少，无法进行趋势分析")
                
            x = np.arange(len(valid_data)).reshape(-1, 1)
            y = valid_data.values
            
            results = {
                'method': method,
                'n_points': len(valid_data),
                'time_span': valid_data.index[-1] - valid_data.index[0]
            }
            
            if method == 'ols':
                # 普通最小二乘法
                model = LinearRegression()
                model.fit(x, y)
                
                slope = model.coef_[0]
                intercept = model.intercept_
                y_pred = model.predict(x)
                
                # 计算统计量
                r2 = r2_score(y, y_pred)
                residuals = y - y_pred
                mse = np.mean(residuals**2)
                
                # t检验
                n = len(y)
                df = n - 2
                se_slope = np.sqrt(mse / np.sum((x.flatten() - np.mean(x))**2))
                t_stat = slope / se_slope
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                results.update({
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r2,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'confidence_interval': self._calculate_slope_ci(slope, se_slope, df)
                })
                
            elif method == 'theil_sen':
                # Theil-Sen回归（对异常值鲁棒）
                model = TheilSenRegressor()
                model.fit(x, y)
                
                slope = model.coef_[0]
                intercept = model.intercept_
                
                results.update({
                    'slope': slope,
                    'intercept': intercept,
                    'robust_method': True
                })
                
            elif method == 'mann_kendall':
                # Mann-Kendall趋势检验
                mk_result = self._mann_kendall_test(y)
                results.update(mk_result)
                
            return results
            
        except Exception as e:
            logger.error(f"线性趋势分析失败: {e}")
            raise
    
    def nonlinear_trend_analysis(self,
                               time_series: pd.Series,
                               degree: int = 2) -> Dict:
        """
        非线性趋势分析
        
        参数:
            time_series: 时间序列数据
            degree: 多项式阶数
            
        返回:
            非线性趋势分析结果
        """
        try:
            valid_data = time_series.dropna()
            x = np.arange(len(valid_data)).reshape(-1, 1)
            y = valid_data.values
            
            # 多项式特征
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x)
            
            # 拟合模型
            model = LinearRegression()
            model.fit(x_poly, y)
            
            y_pred = model.predict(x_poly)
            r2 = r2_score(y, y_pred)
            
            # 计算AIC和BIC
            n = len(y)
            k = degree + 1  # 参数个数
            mse = np.mean((y - y_pred)**2)
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            
            return {
                'degree': degree,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r_squared': r2,
                'aic': aic,
                'bic': bic,
                'fitted_values': y_pred
            }
            
        except Exception as e:
            logger.error(f"非线性趋势分析失败: {e}")
            raise
    
    def changepoint_detection(self,
                            time_series: pd.Series,
                            method: str = 'pelt') -> Dict:
        """
        变化点检测
        
        参数:
            time_series: 时间序列数据
            method: 检测方法 ('pelt', 'binary_segmentation', 'window')
            
        返回:
            变化点检测结果
        """
        try:
            valid_data = time_series.dropna()
            
            if method == 'pelt':
                # 使用PELT算法（需要ruptures库）
                try:
                    import ruptures as rpt
                    model = "rbf"
                    algo = rpt.Pelt(model=model).fit(valid_data.values)
                    changepoints = algo.predict(pen=10)
                    
                    return {
                        'method': 'PELT',
                        'changepoints': changepoints[:-1],  # 移除最后一个点
                        'n_changepoints': len(changepoints) - 1
                    }
                except ImportError:
                    logger.warning("ruptures库未安装，使用简单方法")
                    return self._simple_changepoint_detection(valid_data)
                    
            elif method == 'window':
                return self._window_based_changepoint(valid_data)
            else:
                return self._simple_changepoint_detection(valid_data)
                
        except Exception as e:
            logger.error(f"变化点检测失败: {e}")
            raise
    
    def trend_decomposition(self,
                          time_series: pd.Series,
                          period: Optional[int] = None) -> Dict:
        """
        趋势分解
        
        参数:
            time_series: 时间序列数据
            period: 周期长度
            
        返回:
            分解结果
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            valid_data = time_series.dropna()
            
            if period is None:
                # 自动检测周期
                period = min(len(valid_data) // 2, 12)
            
            decomposition = seasonal_decompose(
                valid_data, 
                model='additive', 
                period=period
            )
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"趋势分解失败: {e}")
            raise
    
    def spatial_trend_analysis(self,
                             spatial_data: np.ndarray,
                             coordinates: np.ndarray) -> Dict:
        """
        空间趋势分析
        
        参数:
            spatial_data: 空间数据 (time, lat, lon)
            coordinates: 坐标信息 (lat, lon)
            
        返回:
            空间趋势分析结果
        """
        try:
            nt, nlat, nlon = spatial_data.shape
            
            # 初始化结果数组
            slope_map = np.full((nlat, nlon), np.nan)
            pvalue_map = np.full((nlat, nlon), np.nan)
            r2_map = np.full((nlat, nlon), np.nan)
            
            # 对每个网格点进行趋势分析
            for i in range(nlat):
                for j in range(nlon):
                    ts = spatial_data[:, i, j]
                    
                    # 跳过无效数据
                    if np.isnan(ts).all():
                        continue
                    
                    # 创建时间序列
                    valid_mask = ~np.isnan(ts)
                    if np.sum(valid_mask) < 3:
                        continue
                    
                    time_idx = np.arange(nt)[valid_mask]
                    values = ts[valid_mask]
                    
                    # 线性回归
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_idx, values
                    )
                    
                    slope_map[i, j] = slope
                    pvalue_map[i, j] = p_value
                    r2_map[i, j] = r_value**2
            
            return {
                'slope_map': slope_map,
                'pvalue_map': pvalue_map,
                'r2_map': r2_map,
                'significant_mask': pvalue_map < self.alpha,
                'coordinates': coordinates
            }
            
        except Exception as e:
            logger.error(f"空间趋势分析失败: {e}")
            raise
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict:
        """
        Mann-Kendall趋势检验
        """
        n = len(data)
        
        # 计算S统计量
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # 计算方差
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # 标准化
        if S > 0:
            z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            z = (S + 1) / np.sqrt(var_S)
        else:
            z = 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Theil-Sen斜率估计
        slopes = []
        for i in range(n-1):
            for j in range(i+1, n):
                if j != i:
                    slopes.append((data[j] - data[i]) / (j - i))
        
        slope = np.median(slopes) if slopes else 0
        
        return {
            'S_statistic': S,
            'z_statistic': z,
            'p_value': p_value,
            'slope': slope,
            'trend': 'increasing' if S > 0 else 'decreasing' if S < 0 else 'no trend',
            'significant': p_value < self.alpha
        }
    
    def _calculate_slope_ci(self, slope: float, se_slope: float, df: int) -> Tuple[float, float]:
        """
        计算斜率的置信区间
        """
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin = t_critical * se_slope
        return (slope - margin, slope + margin)
    
    def _simple_changepoint_detection(self, data: pd.Series) -> Dict:
        """
        简单的变化点检测方法
        """
        # 使用滑动窗口检测均值变化
        window_size = max(5, len(data) // 10)
        changepoints = []
        
        for i in range(window_size, len(data) - window_size):
            before = data.iloc[i-window_size:i]
            after = data.iloc[i:i+window_size]
            
            # t检验
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # 严格的阈值
                changepoints.append(i)
        
        return {
            'method': 'simple_window',
            'changepoints': changepoints,
            'n_changepoints': len(changepoints)
        }
    
    def _window_based_changepoint(self, data: pd.Series) -> Dict:
        """
        基于窗口的变化点检测
        """
        # 计算滑动方差
        window_size = max(5, len(data) // 20)
        rolling_var = data.rolling(window=window_size).var()
        
        # 检测方差突变点
        var_changes = np.abs(np.diff(rolling_var.dropna()))
        threshold = np.percentile(var_changes, 95)
        
        changepoints = np.where(var_changes > threshold)[0] + window_size
        
        return {
            'method': 'variance_based',
            'changepoints': changepoints.tolist(),
            'n_changepoints': len(changepoints)
        }

class TrendVisualizer:
    """
    趋势分析可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
        """
        plt.style.use(style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_trend_analysis(self,
                          time_series: pd.Series,
                          trend_results: Dict,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制趋势分析结果
        
        参数:
            time_series: 原始时间序列
            trend_results: 趋势分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('冰川趋势分析结果', fontsize=16, fontweight='bold')
        
        # 原始数据和趋势线
        ax1 = axes[0, 0]
        ax1.plot(time_series.index, time_series.values, 'o-', 
                alpha=0.7, label='观测数据')
        
        if 'slope' in trend_results:
            x_trend = np.arange(len(time_series))
            y_trend = (trend_results['slope'] * x_trend + 
                      trend_results['intercept'])
            ax1.plot(time_series.index, y_trend, 'r-', 
                    linewidth=2, label=f"趋势线 (斜率: {trend_results['slope']:.4f})")
        
        ax1.set_title('时间序列和趋势')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 残差分析
        if 'fitted_values' in trend_results:
            ax2 = axes[0, 1]
            residuals = time_series.values - trend_results['fitted_values']
            ax2.plot(time_series.index, residuals, 'o-', alpha=0.7)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('残差分析')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('残差')
            ax2.grid(True, alpha=0.3)
        
        # 统计信息
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        stats_text = f"""
        趋势分析统计信息:
        
        方法: {trend_results.get('method', 'N/A')}
        数据点数: {trend_results.get('n_points', 'N/A')}
        R²: {trend_results.get('r_squared', 'N/A'):.4f}
        p值: {trend_results.get('p_value', 'N/A'):.4f}
        显著性: {'是' if trend_results.get('significant', False) else '否'}
        """
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 分布直方图
        ax4 = axes[1, 1]
        ax4.hist(time_series.dropna().values, bins=20, alpha=0.7, 
                edgecolor='black')
        ax4.set_title('数据分布')
        ax4.set_xlabel('值')
        ax4.set_ylabel('频次')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_trends(self,
                          spatial_results: Dict,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制空间趋势分析结果
        
        参数:
            spatial_results: 空间趋势分析结果
            save_path: 保存路径
            
        返回:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('空间趋势分析结果', fontsize=16, fontweight='bold')
        
        # 趋势斜率图
        im1 = axes[0, 0].imshow(spatial_results['slope_map'], 
                               cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('趋势斜率')
        plt.colorbar(im1, ax=axes[0, 0], label='斜率')
        
        # 显著性图
        im2 = axes[0, 1].imshow(spatial_results['significant_mask'], 
                               cmap='RdYlBu', aspect='auto')
        axes[0, 1].set_title('趋势显著性')
        plt.colorbar(im2, ax=axes[0, 1], label='显著 (1) / 不显著 (0)')
        
        # R²图
        im3 = axes[1, 0].imshow(spatial_results['r2_map'], 
                               cmap='viridis', aspect='auto')
        axes[1, 0].set_title('拟合优度 (R²)')
        plt.colorbar(im3, ax=axes[1, 0], label='R²')
        
        # p值图
        im4 = axes[1, 1].imshow(spatial_results['pvalue_map'], 
                               cmap='viridis_r', aspect='auto')
        axes[1, 1].set_title('p值分布')
        plt.colorbar(im4, ax=axes[1, 1], label='p值')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """
    主函数 - 演示趋势分析功能
    """
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2020-12-31', freq='M')
    
    # 模拟冰川质量平衡数据（带趋势和噪声）
    trend = -0.5 * np.arange(len(dates)) / 12  # 年下降趋势
    seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # 季节变化
    noise = np.random.normal(0, 1, len(dates))
    
    glacier_mb = trend + seasonal + noise
    ts = pd.Series(glacier_mb, index=dates)
    
    # 初始化分析器
    analyzer = TrendAnalyzer()
    visualizer = TrendVisualizer()
    
    print("开始冰川趋势分析...")
    
    # 线性趋势分析
    linear_results = analyzer.linear_trend_analysis(ts, method='ols')
    print(f"\n线性趋势分析结果:")
    print(f"斜率: {linear_results['slope']:.4f}")
    print(f"R²: {linear_results['r_squared']:.4f}")
    print(f"p值: {linear_results['p_value']:.4f}")
    print(f"显著性: {linear_results['significant']}")
    
    # Mann-Kendall检验
    mk_results = analyzer.linear_trend_analysis(ts, method='mann_kendall')
    print(f"\nMann-Kendall检验结果:")
    print(f"趋势: {mk_results['trend']}")
    print(f"p值: {mk_results['p_value']:.4f}")
    print(f"显著性: {mk_results['significant']}")
    
    # 变化点检测
    cp_results = analyzer.changepoint_detection(ts)
    print(f"\n变化点检测结果:")
    print(f"检测到 {cp_results['n_changepoints']} 个变化点")
    
    # 趋势分解
    decomp_results = analyzer.trend_decomposition(ts)
    print(f"\n趋势分解完成")
    
    # 可视化
    fig1 = visualizer.plot_trend_analysis(ts, linear_results)
    plt.show()
    
    print("\n趋势分析完成！")

if __name__ == "__main__":
    main()