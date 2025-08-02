#!/usr/bin/env python3
"""
阶段1：长期趋势训练

实现长期趋势学习的训练阶段，包括：
- 长期数据预处理
- 趋势特征提取
- 长期模式学习
- 趋势验证

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TrendType(Enum):
    """趋势类型枚举"""
    LINEAR = "linear"  # 线性趋势
    EXPONENTIAL = "exponential"  # 指数趋势
    POLYNOMIAL = "polynomial"  # 多项式趋势
    SEASONAL = "seasonal"  # 季节性趋势
    CYCLIC = "cyclic"  # 周期性趋势
    NONLINEAR = "nonlinear"  # 非线性趋势

@dataclass
class LongTermConfig:
    """长期趋势配置"""
    time_horizon: int = 365  # 时间跨度（天）
    trend_window: int = 30  # 趋势窗口大小
    smoothing_factor: float = 0.1  # 平滑因子
    min_trend_strength: float = 0.3  # 最小趋势强度
    detrend_method: str = "linear"  # 去趋势方法
    seasonal_periods: List[int] = None  # 季节性周期
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [365, 30, 7]  # 年、月、周周期

class TrendExtractor:
    """
    趋势提取器
    
    从时间序列数据中提取长期趋势
    """
    
    def __init__(self, config: LongTermConfig):
        """
        初始化趋势提取器
        
        Args:
            config: 长期趋势配置
        """
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.trend_models = {}
        self.logger = logging.getLogger(__name__)
    
    def extract_linear_trend(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        提取线性趋势
        
        Args:
            data: 时间序列数据 [time_points, features]
            time_points: 时间点
            
        Returns:
            Dict: 线性趋势信息
        """
        trends = {}
        
        for i in range(data.shape[1]):
            # 线性回归拟合
            coeffs = np.polyfit(time_points, data[:, i], 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # 计算趋势强度
            fitted_values = slope * time_points + intercept
            residuals = data[:, i] - fitted_values
            trend_strength = 1 - np.var(residuals) / np.var(data[:, i])
            
            trends[f'feature_{i}'] = {
                'slope': slope,
                'intercept': intercept,
                'trend_strength': trend_strength,
                'fitted_values': fitted_values,
                'residuals': residuals
            }
        
        return trends
    
    def extract_seasonal_trend(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        提取季节性趋势
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 季节性趋势信息
        """
        seasonal_trends = {}
        
        for period in self.config.seasonal_periods:
            if len(time_points) < 2 * period:
                continue
            
            # 计算季节性分量
            seasonal_component = np.zeros_like(data)
            
            for i in range(data.shape[1]):
                # 简化的季节性分解
                signal = data[:, i]
                
                # 移动平均去趋势
                if len(signal) >= period:
                    trend = np.convolve(signal, np.ones(period)/period, mode='same')
                    detrended = signal - trend
                    
                    # 计算季节性模式
                    seasonal_pattern = np.zeros(period)
                    for j in range(period):
                        indices = np.arange(j, len(detrended), period)
                        if len(indices) > 0:
                            seasonal_pattern[j] = np.mean(detrended[indices])
                    
                    # 重复季节性模式
                    seasonal_full = np.tile(seasonal_pattern, len(signal) // period + 1)[:len(signal)]
                    seasonal_component[:, i] = seasonal_full
            
            seasonal_trends[f'period_{period}'] = {
                'seasonal_component': seasonal_component,
                'period': period
            }
        
        return seasonal_trends
    
    def extract_nonlinear_trend(self, data: np.ndarray, time_points: np.ndarray, 
                              degree: int = 3) -> Dict[str, Any]:
        """
        提取非线性趋势
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            degree: 多项式度数
            
        Returns:
            Dict: 非线性趋势信息
        """
        nonlinear_trends = {}
        
        for i in range(data.shape[1]):
            # 多项式拟合
            coeffs = np.polyfit(time_points, data[:, i], degree)
            poly_func = np.poly1d(coeffs)
            fitted_values = poly_func(time_points)
            
            # 计算拟合质量
            residuals = data[:, i] - fitted_values
            r_squared = 1 - np.sum(residuals**2) / np.sum((data[:, i] - np.mean(data[:, i]))**2)
            
            nonlinear_trends[f'feature_{i}'] = {
                'coefficients': coeffs,
                'fitted_values': fitted_values,
                'residuals': residuals,
                'r_squared': r_squared,
                'degree': degree
            }
        
        return nonlinear_trends
    
    def decompose_trends(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        综合趋势分解
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 趋势分解结果
        """
        decomposition = {
            'original_data': data,
            'time_points': time_points
        }
        
        # 提取线性趋势
        linear_trends = self.extract_linear_trend(data, time_points)
        decomposition['linear_trends'] = linear_trends
        
        # 提取季节性趋势
        seasonal_trends = self.extract_seasonal_trend(data, time_points)
        decomposition['seasonal_trends'] = seasonal_trends
        
        # 提取非线性趋势
        nonlinear_trends = self.extract_nonlinear_trend(data, time_points)
        decomposition['nonlinear_trends'] = nonlinear_trends
        
        # 计算残差
        total_fitted = np.zeros_like(data)
        for i in range(data.shape[1]):
            total_fitted[:, i] += linear_trends[f'feature_{i}']['fitted_values']
            
            # 添加主要季节性分量
            if seasonal_trends:
                main_period = max(seasonal_trends.keys(), 
                                key=lambda x: seasonal_trends[x]['period'])
                total_fitted[:, i] += seasonal_trends[main_period]['seasonal_component'][:, i]
        
        decomposition['residuals'] = data - total_fitted
        decomposition['fitted_values'] = total_fitted
        
        return decomposition

class LongTermTrendLearner(nn.Module):
    """
    长期趋势学习器
    
    使用神经网络学习长期趋势模式
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, trend_types: List[TrendType]):
        """
        初始化长期趋势学习器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            trend_types: 趋势类型列表
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trend_types = trend_types
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # 趋势特定的输出层
        self.trend_heads = nn.ModuleDict()
        for trend_type in trend_types:
            self.trend_heads[trend_type.value] = nn.Linear(output_dim, output_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
        
        # 时间编码
        self.time_encoding = nn.Linear(1, output_dim)
    
    def encode_time(self, time_points: torch.Tensor) -> torch.Tensor:
        """
        时间编码
        
        Args:
            time_points: 时间点 [batch_size, seq_len, 1]
            
        Returns:
            Tensor: 时间编码 [batch_size, seq_len, output_dim]
        """
        # 正弦余弦位置编码
        time_encoded = self.time_encoding(time_points)
        
        # 添加周期性编码
        seq_len = time_points.shape[1]
        position = torch.arange(seq_len, device=time_points.device).float().unsqueeze(0).unsqueeze(-1)
        
        # 不同频率的正弦余弦编码
        div_term = torch.exp(torch.arange(0, self.output_dim, 2, device=time_points.device).float() * 
                           -(np.log(10000.0) / self.output_dim))
        
        pos_encoding = torch.zeros(1, seq_len, self.output_dim, device=time_points.device)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return time_encoded + pos_encoding
    
    def forward(self, x: torch.Tensor, time_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            time_points: 时间点 [batch_size, seq_len, 1]
            
        Returns:
            Dict: 趋势预测结果
        """
        batch_size, seq_len, _ = x.shape
        
        # 时间编码
        time_encoded = self.encode_time(time_points)
        
        # 特征提取
        features = self.network(x)  # [batch_size, seq_len, output_dim]
        
        # 结合时间编码
        combined_features = features + time_encoded
        
        # 自注意力
        attended_features, attention_weights = self.attention(
            combined_features, combined_features, combined_features
        )
        
        # 趋势特定预测
        trend_predictions = {}
        for trend_type in self.trend_types:
            trend_output = self.trend_heads[trend_type.value](attended_features)
            trend_predictions[trend_type.value] = trend_output
        
        return {
            'features': features,
            'attended_features': attended_features,
            'attention_weights': attention_weights,
            'trend_predictions': trend_predictions,
            'time_encoding': time_encoded
        }

class LongTermTrendTrainer:
    """
    长期趋势训练器
    
    管理长期趋势学习的训练过程
    """
    
    def __init__(self, model: LongTermTrendLearner, config: LongTermConfig):
        """
        初始化长期趋势训练器
        
        Args:
            model: 趋势学习模型
            config: 长期趋势配置
        """
        self.model = model
        self.config = config
        self.trend_extractor = TrendExtractor(config)
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizer(self, learning_rate: float = 1e-3, 
                       weight_decay: float = 1e-4) -> None:
        """
        设置优化器
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
    
    def compute_trend_loss(self, predictions: Dict[str, torch.Tensor], 
                          targets: torch.Tensor, 
                          trend_weights: Dict[str, float] = None) -> torch.Tensor:
        """
        计算趋势损失
        
        Args:
            predictions: 趋势预测结果
            targets: 目标值 [batch_size, seq_len, output_dim]
            trend_weights: 趋势权重
            
        Returns:
            Tensor: 总损失
        """
        if trend_weights is None:
            trend_weights = {trend_type.value: 1.0 for trend_type in self.model.trend_types}
        
        total_loss = torch.tensor(0.0, device=targets.device)
        
        # 趋势特定损失
        for trend_type, weight in trend_weights.items():
            if trend_type in predictions['trend_predictions']:
                trend_pred = predictions['trend_predictions'][trend_type]
                trend_loss = nn.MSELoss()(trend_pred, targets)
                total_loss += weight * trend_loss
        
        # 注意力正则化
        attention_weights = predictions['attention_weights']
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        attention_reg = -torch.mean(attention_entropy)  # 鼓励多样化注意力
        total_loss += 0.01 * attention_reg
        
        # 时间一致性损失
        features = predictions['features']
        if features.shape[1] > 1:
            time_consistency_loss = torch.mean(
                torch.abs(features[:, 1:] - features[:, :-1])
            )
            total_loss += 0.1 * time_consistency_loss
        
        return total_loss
    
    def train_epoch(self, data_loader, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            data_loader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, time_points, targets) in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(inputs, time_points)
            
            # 计算损失
            loss = self.compute_trend_loss(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}"
                )
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self, data_loader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            data_loader: 验证数据加载器
            
        Returns:
            Dict: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, time_points, targets in data_loader:
                predictions = self.model(inputs, time_points)
                
                # 计算损失
                loss = self.compute_trend_loss(predictions, targets)
                total_loss += loss.item()
                
                # 计算MAE（使用主要趋势预测）
                main_trend = list(predictions['trend_predictions'].values())[0]
                mae = torch.mean(torch.abs(main_trend - targets))
                total_mae += mae.item()
                
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def analyze_trends(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        分析数据中的趋势
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 趋势分析结果
        """
        # 使用趋势提取器分析
        decomposition = self.trend_extractor.decompose_trends(data, time_points)
        
        # 使用模型预测趋势
        self.model.eval()
        with torch.no_grad():
            # 转换为tensor
            data_tensor = torch.FloatTensor(data).unsqueeze(0)  # [1, seq_len, features]
            time_tensor = torch.FloatTensor(time_points).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            
            model_predictions = self.model(data_tensor, time_tensor)
        
        analysis_results = {
            'statistical_decomposition': decomposition,
            'model_predictions': {
                k: v.squeeze(0).numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in model_predictions.items() 
                if isinstance(v, torch.Tensor)
            },
            'trend_summary': self._summarize_trends(decomposition)
        }
        
        return analysis_results
    
    def _summarize_trends(self, decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """
        总结趋势信息
        
        Args:
            decomposition: 趋势分解结果
            
        Returns:
            Dict: 趋势总结
        """
        summary = {
            'dominant_trends': [],
            'trend_strengths': {},
            'seasonal_components': [],
            'overall_direction': 'stable'
        }
        
        # 分析线性趋势
        linear_trends = decomposition['linear_trends']
        for feature, trend_info in linear_trends.items():
            strength = trend_info['trend_strength']
            slope = trend_info['slope']
            
            summary['trend_strengths'][feature] = strength
            
            if strength > self.config.min_trend_strength:
                direction = 'increasing' if slope > 0 else 'decreasing'
                summary['dominant_trends'].append({
                    'feature': feature,
                    'type': 'linear',
                    'direction': direction,
                    'strength': strength,
                    'slope': slope
                })
        
        # 分析季节性趋势
        seasonal_trends = decomposition['seasonal_trends']
        for period_key, seasonal_info in seasonal_trends.items():
            period = seasonal_info['period']
            component = seasonal_info['seasonal_component']
            
            # 计算季节性强度
            seasonal_strength = np.std(component) / (np.std(decomposition['original_data']) + 1e-8)
            
            if seasonal_strength > 0.1:  # 阈值
                summary['seasonal_components'].append({
                    'period': period,
                    'strength': seasonal_strength
                })
        
        # 确定总体方向
        avg_slopes = [info['slope'] for info in linear_trends.values()]
        if avg_slopes:
            avg_slope = np.mean(avg_slopes)
            if avg_slope > 0.01:
                summary['overall_direction'] = 'increasing'
            elif avg_slope < -0.01:
                summary['overall_direction'] = 'decreasing'
        
        return summary

def create_longterm_trend_trainer(
    input_dim: int,
    hidden_dims: List[int] = None,
    output_dim: int = None,
    trend_types: List[TrendType] = None,
    config: LongTermConfig = None
) -> LongTermTrendTrainer:
    """
    创建长期趋势训练器
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度
        output_dim: 输出维度
        trend_types: 趋势类型
        config: 配置
        
    Returns:
        LongTermTrendTrainer: 训练器实例
    """
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    
    if output_dim is None:
        output_dim = input_dim
    
    if trend_types is None:
        trend_types = [TrendType.LINEAR, TrendType.SEASONAL, TrendType.NONLINEAR]
    
    if config is None:
        config = LongTermConfig()
    
    # 创建模型
    model = LongTermTrendLearner(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        trend_types=trend_types
    )
    
    # 创建训练器
    trainer = LongTermTrendTrainer(model, config)
    
    return trainer

if __name__ == "__main__":
    # 测试长期趋势学习
    
    # 创建配置
    config = LongTermConfig(
        time_horizon=365,
        trend_window=30,
        smoothing_factor=0.1,
        seasonal_periods=[365, 30, 7]
    )
    
    # 创建训练器
    trainer = create_longterm_trend_trainer(
        input_dim=5,
        hidden_dims=[64, 32],
        output_dim=5,
        trend_types=[TrendType.LINEAR, TrendType.SEASONAL],
        config=config
    )
    
    # 生成测试数据
    time_points = np.linspace(0, 365, 1000)
    
    # 模拟长期趋势数据
    data = np.zeros((1000, 5))
    for i in range(5):
        # 线性趋势
        linear_trend = 0.01 * i * time_points
        
        # 季节性分量
        seasonal = 0.5 * np.sin(2 * np.pi * time_points / 365) + \
                  0.2 * np.sin(2 * np.pi * time_points / 30)
        
        # 噪声
        noise = 0.1 * np.random.randn(1000)
        
        data[:, i] = linear_trend + seasonal + noise
    
    # 分析趋势
    analysis_results = trainer.analyze_trends(data, time_points)
    
    print("=== 长期趋势分析结果 ===")
    trend_summary = analysis_results['trend_summary']
    
    print(f"总体方向: {trend_summary['overall_direction']}")
    print(f"主导趋势数量: {len(trend_summary['dominant_trends'])}")
    print(f"季节性分量数量: {len(trend_summary['seasonal_components'])}")
    
    for trend in trend_summary['dominant_trends']:
        print(f"  {trend['feature']}: {trend['type']} {trend['direction']} (强度: {trend['strength']:.3f})")
    
    for seasonal in trend_summary['seasonal_components']:
        print(f"  季节性周期 {seasonal['period']} 天 (强度: {seasonal['strength']:.3f})")
    
    print("\n长期趋势学习测试完成！")