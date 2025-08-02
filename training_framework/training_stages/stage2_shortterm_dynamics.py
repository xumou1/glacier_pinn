#!/usr/bin/env python3
"""
阶段2：短期动态训练

实现短期动态学习的训练阶段，包括：
- 短期变化检测
- 动态特征提取
- 瞬态响应学习
- 快速适应机制

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
from scipy import signal
from scipy.fft import fft, fftfreq

class DynamicsType(Enum):
    """动态类型枚举"""
    TRANSIENT = "transient"  # 瞬态响应
    OSCILLATORY = "oscillatory"  # 振荡动态
    IMPULSE = "impulse"  # 脉冲响应
    STEP = "step"  # 阶跃响应
    CHAOTIC = "chaotic"  # 混沌动态
    PERIODIC = "periodic"  # 周期性动态

@dataclass
class ShortTermConfig:
    """短期动态配置"""
    time_window: int = 24  # 时间窗口（小时）
    sampling_rate: float = 1.0  # 采样率（小时）
    detection_threshold: float = 0.1  # 变化检测阈值
    response_time: float = 6.0  # 响应时间（小时）
    adaptation_rate: float = 0.01  # 适应率
    frequency_bands: List[Tuple[float, float]] = None  # 频率带
    
    def __post_init__(self):
        if self.frequency_bands is None:
            # 定义不同的频率带（单位：1/小时）
            self.frequency_bands = [
                (0.0, 0.1),    # 低频（>10小时周期）
                (0.1, 0.5),    # 中频（2-10小时周期）
                (0.5, 2.0),    # 高频（0.5-2小时周期）
            ]

class DynamicsDetector:
    """
    动态检测器
    
    检测和分析短期动态变化
    """
    
    def __init__(self, config: ShortTermConfig):
        """
        初始化动态检测器
        
        Args:
            config: 短期动态配置
        """
        self.config = config
        self.scaler = StandardScaler()
        self.change_points = []
        self.logger = logging.getLogger(__name__)
    
    def detect_change_points(self, data: np.ndarray, time_points: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测变化点
        
        Args:
            data: 时间序列数据 [time_points, features]
            time_points: 时间点
            
        Returns:
            List: 变化点信息列表
        """
        change_points = []
        
        for i in range(data.shape[1]):
            signal_data = data[:, i]
            
            # 计算移动方差
            window_size = min(int(self.config.time_window / 2), len(signal_data) // 4)
            if window_size < 2:
                continue
            
            moving_var = np.array([
                np.var(signal_data[max(0, j-window_size):j+window_size+1])
                for j in range(len(signal_data))
            ])
            
            # 检测方差突变点
            var_diff = np.abs(np.diff(moving_var))
            threshold = self.config.detection_threshold * np.std(var_diff)
            
            change_indices = np.where(var_diff > threshold)[0]
            
            for idx in change_indices:
                if idx < len(time_points) - 1:
                    change_points.append({
                        'feature_index': i,
                        'time_index': idx,
                        'time_point': time_points[idx],
                        'magnitude': var_diff[idx],
                        'type': 'variance_change'
                    })
        
        # 检测均值突变点
        for i in range(data.shape[1]):
            signal_data = data[:, i]
            
            # 计算移动平均
            window_size = min(int(self.config.time_window / 4), len(signal_data) // 8)
            if window_size < 2:
                continue
            
            moving_mean = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
            
            # 检测均值突变
            mean_diff = np.abs(np.diff(moving_mean))
            threshold = self.config.detection_threshold * np.std(mean_diff)
            
            change_indices = np.where(mean_diff > threshold)[0]
            
            for idx in change_indices:
                if idx < len(time_points) - 1:
                    change_points.append({
                        'feature_index': i,
                        'time_index': idx,
                        'time_point': time_points[idx],
                        'magnitude': mean_diff[idx],
                        'type': 'mean_change'
                    })
        
        # 按时间排序
        change_points.sort(key=lambda x: x['time_point'])
        
        return change_points
    
    def analyze_frequency_content(self, data: np.ndarray, 
                                time_points: np.ndarray) -> Dict[str, Any]:
        """
        分析频率内容
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 频率分析结果
        """
        frequency_analysis = {}
        
        # 计算采样频率
        if len(time_points) > 1:
            dt = np.mean(np.diff(time_points))
            fs = 1.0 / dt
        else:
            fs = 1.0
        
        for i in range(data.shape[1]):
            signal_data = data[:, i]
            
            # FFT分析
            fft_values = fft(signal_data)
            frequencies = fftfreq(len(signal_data), 1/fs)
            
            # 功率谱密度
            power_spectrum = np.abs(fft_values) ** 2
            
            # 分析不同频率带的能量
            band_energies = {}
            for j, (low_freq, high_freq) in enumerate(self.config.frequency_bands):
                band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                band_energy = np.sum(power_spectrum[band_mask])
                band_energies[f'band_{j}'] = {
                    'frequency_range': (low_freq, high_freq),
                    'energy': band_energy,
                    'relative_energy': band_energy / (np.sum(power_spectrum) + 1e-8)
                }
            
            # 主导频率
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_frequency = frequencies[dominant_freq_idx]
            
            frequency_analysis[f'feature_{i}'] = {
                'frequencies': frequencies[:len(frequencies)//2],
                'power_spectrum': power_spectrum[:len(power_spectrum)//2],
                'band_energies': band_energies,
                'dominant_frequency': dominant_frequency,
                'spectral_centroid': np.sum(frequencies[:len(frequencies)//2] * 
                                          power_spectrum[:len(power_spectrum)//2]) / 
                                   (np.sum(power_spectrum[:len(power_spectrum)//2]) + 1e-8)
            }
        
        return frequency_analysis
    
    def detect_oscillations(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        检测振荡模式
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 振荡检测结果
        """
        oscillation_results = {}
        
        for i in range(data.shape[1]):
            signal_data = data[:, i]
            
            # 寻找峰值和谷值
            peaks, peak_properties = signal.find_peaks(signal_data, height=np.mean(signal_data))
            troughs, trough_properties = signal.find_peaks(-signal_data, height=-np.mean(signal_data))
            
            # 计算振荡特征
            if len(peaks) > 1 and len(troughs) > 1:
                # 周期估计
                peak_intervals = np.diff(time_points[peaks])
                avg_period = np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
                
                # 振幅估计
                peak_values = signal_data[peaks]
                trough_values = signal_data[troughs]
                avg_amplitude = (np.mean(peak_values) - np.mean(trough_values)) / 2
                
                # 规律性评估
                period_std = np.std(peak_intervals) if len(peak_intervals) > 0 else float('inf')
                regularity = 1.0 / (1.0 + period_std / (avg_period + 1e-8))
                
                oscillation_results[f'feature_{i}'] = {
                    'has_oscillation': True,
                    'period': avg_period,
                    'amplitude': avg_amplitude,
                    'regularity': regularity,
                    'num_peaks': len(peaks),
                    'num_troughs': len(troughs),
                    'peak_times': time_points[peaks],
                    'trough_times': time_points[troughs]
                }
            else:
                oscillation_results[f'feature_{i}'] = {
                    'has_oscillation': False,
                    'period': 0,
                    'amplitude': 0,
                    'regularity': 0,
                    'num_peaks': len(peaks),
                    'num_troughs': len(troughs)
                }
        
        return oscillation_results

class ShortTermDynamicsLearner(nn.Module):
    """
    短期动态学习器
    
    使用神经网络学习短期动态模式
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dynamics_types: List[DynamicsType]):
        """
        初始化短期动态学习器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dynamics_types: 动态类型列表
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dynamics_types = dynamics_types
        
        # LSTM用于捕获时序依赖
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 卷积层用于捕获局部模式
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dims[0],
            kernel_size=3,
            padding=1
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=4,
            batch_first=True
        )
        
        # 特征融合网络
        fusion_layers = []
        prev_dim = hidden_dims[0] * 2  # LSTM + Conv1D
        
        for hidden_dim in hidden_dims[1:]:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # 动态类型特定的输出头
        self.dynamics_heads = nn.ModuleDict()
        for dynamics_type in dynamics_types:
            self.dynamics_heads[dynamics_type.value] = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], output_dim)
            )
        
        # 变化检测头
        self.change_detector = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        # 频率分析头
        self.frequency_analyzer = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], len(dynamics_types))
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            Dict: 动态预测结果
        """
        batch_size, seq_len, input_dim = x.shape
        
        # LSTM特征提取
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # 卷积特征提取
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        conv_out = self.conv1d(x_conv)  # [batch_size, hidden_dim, seq_len]
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # 注意力机制
        attended_lstm, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # 特征融合
        fused_features = torch.cat([attended_lstm, conv_out], dim=-1)
        fusion_out = self.fusion_network(fused_features)
        
        # 取最后一个时间步的特征用于预测
        final_features = fusion_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 动态类型特定预测
        dynamics_predictions = {}
        for dynamics_type in self.dynamics_types:
            dynamics_output = self.dynamics_heads[dynamics_type.value](final_features)
            dynamics_predictions[dynamics_type.value] = dynamics_output
        
        # 变化检测
        change_probability = self.change_detector(final_features)
        
        # 频率分析
        frequency_features = self.frequency_analyzer(final_features)
        
        return {
            'lstm_features': lstm_out,
            'conv_features': conv_out,
            'fused_features': fusion_out,
            'attention_weights': attention_weights,
            'dynamics_predictions': dynamics_predictions,
            'change_probability': change_probability,
            'frequency_features': frequency_features
        }

class AdaptiveLearningRate:
    """
    自适应学习率调度器
    
    根据短期动态特征调整学习率
    """
    
    def __init__(self, base_lr: float = 1e-3, adaptation_factor: float = 0.1):
        """
        初始化自适应学习率调度器
        
        Args:
            base_lr: 基础学习率
            adaptation_factor: 适应因子
        """
        self.base_lr = base_lr
        self.adaptation_factor = adaptation_factor
        self.change_history = []
        self.performance_history = []
    
    def update(self, change_detected: bool, performance_metric: float) -> float:
        """
        更新学习率
        
        Args:
            change_detected: 是否检测到变化
            performance_metric: 性能指标
            
        Returns:
            float: 调整后的学习率
        """
        self.change_history.append(change_detected)
        self.performance_history.append(performance_metric)
        
        # 保持历史记录长度
        if len(self.change_history) > 10:
            self.change_history.pop(0)
            self.performance_history.pop(0)
        
        # 计算变化频率
        change_frequency = sum(self.change_history) / len(self.change_history)
        
        # 计算性能趋势
        if len(self.performance_history) > 1:
            performance_trend = (self.performance_history[-1] - 
                               self.performance_history[0]) / len(self.performance_history)
        else:
            performance_trend = 0
        
        # 调整学习率
        if change_detected:
            # 检测到变化时增加学习率
            lr_multiplier = 1.0 + self.adaptation_factor
        elif change_frequency > 0.5:
            # 变化频繁时适度增加学习率
            lr_multiplier = 1.0 + self.adaptation_factor * 0.5
        elif performance_trend < 0:
            # 性能下降时减少学习率
            lr_multiplier = 1.0 - self.adaptation_factor * 0.5
        else:
            # 稳定状态
            lr_multiplier = 1.0
        
        adjusted_lr = self.base_lr * lr_multiplier
        
        return max(adjusted_lr, self.base_lr * 0.1)  # 设置最小学习率

class ShortTermDynamicsTrainer:
    """
    短期动态训练器
    
    管理短期动态学习的训练过程
    """
    
    def __init__(self, model: ShortTermDynamicsLearner, config: ShortTermConfig):
        """
        初始化短期动态训练器
        
        Args:
            model: 动态学习模型
            config: 短期动态配置
        """
        self.model = model
        self.config = config
        self.dynamics_detector = DynamicsDetector(config)
        self.adaptive_lr = AdaptiveLearningRate()
        self.optimizer = None
        self.loss_history = []
        self.change_history = []
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
        
        self.adaptive_lr.base_lr = learning_rate
    
    def compute_dynamics_loss(self, predictions: Dict[str, torch.Tensor], 
                            targets: torch.Tensor,
                            change_labels: torch.Tensor = None,
                            dynamics_weights: Dict[str, float] = None) -> torch.Tensor:
        """
        计算动态损失
        
        Args:
            predictions: 动态预测结果
            targets: 目标值 [batch_size, output_dim]
            change_labels: 变化标签 [batch_size, 1]
            dynamics_weights: 动态权重
            
        Returns:
            Tensor: 总损失
        """
        if dynamics_weights is None:
            dynamics_weights = {dynamics_type.value: 1.0 for dynamics_type in self.model.dynamics_types}
        
        total_loss = torch.tensor(0.0, device=targets.device)
        
        # 动态预测损失
        for dynamics_type, weight in dynamics_weights.items():
            if dynamics_type in predictions['dynamics_predictions']:
                dynamics_pred = predictions['dynamics_predictions'][dynamics_type]
                dynamics_loss = nn.MSELoss()(dynamics_pred, targets)
                total_loss += weight * dynamics_loss
        
        # 变化检测损失
        if change_labels is not None:
            change_pred = predictions['change_probability']
            change_loss = nn.BCELoss()(change_pred, change_labels.float())
            total_loss += 0.5 * change_loss
        
        # 时序一致性损失
        fused_features = predictions['fused_features']
        if fused_features.shape[1] > 1:
            # 相邻时间步的特征应该相似（除非有突变）
            temporal_diff = torch.abs(fused_features[:, 1:] - fused_features[:, :-1])
            temporal_consistency_loss = torch.mean(temporal_diff)
            total_loss += 0.1 * temporal_consistency_loss
        
        # 注意力正则化
        attention_weights = predictions['attention_weights']
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        attention_reg = -torch.mean(attention_entropy)
        total_loss += 0.01 * attention_reg
        
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
        epoch_changes = []
        
        for batch_idx, batch_data in enumerate(data_loader):
            if len(batch_data) == 2:
                inputs, targets = batch_data
                change_labels = None
            else:
                inputs, targets, change_labels = batch_data
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(inputs)
            
            # 计算损失
            loss = self.compute_dynamics_loss(predictions, targets, change_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录变化检测结果
            change_prob = torch.mean(predictions['change_probability']).item()
            change_detected = change_prob > 0.5
            epoch_changes.append(change_detected)
            
            # 自适应学习率调整
            adjusted_lr = self.adaptive_lr.update(change_detected, loss.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjusted_lr
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, "
                    f"Change Prob: {change_prob:.3f}, LR: {adjusted_lr:.6f}"
                )
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        self.change_history.extend(epoch_changes)
        
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
        total_change_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    change_labels = None
                else:
                    inputs, targets, change_labels = batch_data
                
                predictions = self.model(inputs)
                
                # 计算损失
                loss = self.compute_dynamics_loss(predictions, targets, change_labels)
                total_loss += loss.item()
                
                # 计算MAE（使用主要动态预测）
                main_dynamics = list(predictions['dynamics_predictions'].values())[0]
                mae = torch.mean(torch.abs(main_dynamics - targets))
                total_mae += mae.item()
                
                # 计算变化检测准确率
                if change_labels is not None:
                    change_pred = (predictions['change_probability'] > 0.5).float()
                    change_accuracy = torch.mean((change_pred == change_labels.float()).float())
                    total_change_accuracy += change_accuracy.item()
                
                num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
        
        if change_labels is not None:
            metrics['val_change_accuracy'] = total_change_accuracy / num_batches
        
        return metrics
    
    def analyze_dynamics(self, data: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        分析数据中的短期动态
        
        Args:
            data: 时间序列数据
            time_points: 时间点
            
        Returns:
            Dict: 动态分析结果
        """
        # 使用检测器分析
        change_points = self.dynamics_detector.detect_change_points(data, time_points)
        frequency_analysis = self.dynamics_detector.analyze_frequency_content(data, time_points)
        oscillation_analysis = self.dynamics_detector.detect_oscillations(data, time_points)
        
        # 使用模型预测
        self.model.eval()
        with torch.no_grad():
            # 准备数据
            window_size = min(self.config.time_window, len(data))
            if window_size < len(data):
                # 使用滑动窗口
                data_windows = []
                for i in range(len(data) - window_size + 1):
                    data_windows.append(data[i:i+window_size])
                data_tensor = torch.FloatTensor(np.array(data_windows))
            else:
                data_tensor = torch.FloatTensor(data).unsqueeze(0)
            
            model_predictions = self.model(data_tensor)
        
        analysis_results = {
            'change_points': change_points,
            'frequency_analysis': frequency_analysis,
            'oscillation_analysis': oscillation_analysis,
            'model_predictions': {
                k: v.numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in model_predictions.items() 
                if isinstance(v, torch.Tensor) and v.dim() <= 2
            },
            'dynamics_summary': self._summarize_dynamics(
                change_points, frequency_analysis, oscillation_analysis
            )
        }
        
        return analysis_results
    
    def _summarize_dynamics(self, change_points: List[Dict], 
                          frequency_analysis: Dict, 
                          oscillation_analysis: Dict) -> Dict[str, Any]:
        """
        总结动态特征
        
        Args:
            change_points: 变化点列表
            frequency_analysis: 频率分析结果
            oscillation_analysis: 振荡分析结果
            
        Returns:
            Dict: 动态总结
        """
        summary = {
            'change_frequency': len(change_points),
            'dominant_dynamics': [],
            'stability_level': 'stable',
            'response_characteristics': {}
        }
        
        # 分析变化频率
        if len(change_points) > 5:
            summary['stability_level'] = 'highly_dynamic'
        elif len(change_points) > 2:
            summary['stability_level'] = 'moderately_dynamic'
        
        # 分析主导动态类型
        for feature_key, freq_info in frequency_analysis.items():
            dominant_freq = freq_info['dominant_frequency']
            
            if dominant_freq > 0.5:  # 高频
                dynamics_type = 'rapid_oscillation'
            elif dominant_freq > 0.1:  # 中频
                dynamics_type = 'periodic_variation'
            else:  # 低频
                dynamics_type = 'slow_drift'
            
            summary['dominant_dynamics'].append({
                'feature': feature_key,
                'type': dynamics_type,
                'frequency': dominant_freq
            })
        
        # 分析响应特征
        oscillating_features = 0
        total_regularity = 0
        
        for feature_key, osc_info in oscillation_analysis.items():
            if osc_info['has_oscillation']:
                oscillating_features += 1
                total_regularity += osc_info['regularity']
        
        if oscillating_features > 0:
            avg_regularity = total_regularity / oscillating_features
            summary['response_characteristics'] = {
                'oscillating_features': oscillating_features,
                'average_regularity': avg_regularity,
                'response_type': 'oscillatory' if avg_regularity > 0.7 else 'irregular'
            }
        
        return summary

def create_shortterm_dynamics_trainer(
    input_dim: int,
    hidden_dims: List[int] = None,
    output_dim: int = None,
    dynamics_types: List[DynamicsType] = None,
    config: ShortTermConfig = None
) -> ShortTermDynamicsTrainer:
    """
    创建短期动态训练器
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度
        output_dim: 输出维度
        dynamics_types: 动态类型
        config: 配置
        
    Returns:
        ShortTermDynamicsTrainer: 训练器实例
    """
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]
    
    if output_dim is None:
        output_dim = input_dim
    
    if dynamics_types is None:
        dynamics_types = [DynamicsType.TRANSIENT, DynamicsType.OSCILLATORY, DynamicsType.IMPULSE]
    
    if config is None:
        config = ShortTermConfig()
    
    # 创建模型
    model = ShortTermDynamicsLearner(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dynamics_types=dynamics_types
    )
    
    # 创建训练器
    trainer = ShortTermDynamicsTrainer(model, config)
    
    return trainer

if __name__ == "__main__":
    # 测试短期动态学习
    
    # 创建配置
    config = ShortTermConfig(
        time_window=24,
        sampling_rate=1.0,
        detection_threshold=0.1,
        response_time=6.0
    )
    
    # 创建训练器
    trainer = create_shortterm_dynamics_trainer(
        input_dim=3,
        hidden_dims=[32, 16],
        output_dim=3,
        dynamics_types=[DynamicsType.TRANSIENT, DynamicsType.OSCILLATORY],
        config=config
    )
    
    # 生成测试数据
    time_points = np.linspace(0, 48, 200)  # 48小时，每0.24小时一个点
    
    # 模拟短期动态数据
    data = np.zeros((200, 3))
    for i in range(3):
        # 基础信号
        base_signal = np.sin(2 * np.pi * time_points / 12)  # 12小时周期
        
        # 添加瞬态响应
        transient_times = [10, 25, 40]  # 瞬态发生时间
        transient_signal = np.zeros_like(time_points)
        for t_time in transient_times:
            transient_signal += 0.5 * np.exp(-(time_points - t_time)**2 / 4) * \
                              np.heaviside(time_points - t_time, 0)
        
        # 添加高频振荡
        high_freq_osc = 0.2 * np.sin(2 * np.pi * time_points / 2)  # 2小时周期
        
        # 噪声
        noise = 0.1 * np.random.randn(200)
        
        data[:, i] = base_signal + transient_signal + high_freq_osc + noise
    
    # 分析短期动态
    analysis_results = trainer.analyze_dynamics(data, time_points)
    
    print("=== 短期动态分析结果 ===")
    dynamics_summary = analysis_results['dynamics_summary']
    
    print(f"变化点数量: {dynamics_summary['change_frequency']}")
    print(f"稳定性水平: {dynamics_summary['stability_level']}")
    
    print("\n主导动态类型:")
    for dynamics in dynamics_summary['dominant_dynamics']:
        print(f"  {dynamics['feature']}: {dynamics['type']} (频率: {dynamics['frequency']:.3f})")
    
    if 'response_characteristics' in dynamics_summary:
        resp_char = dynamics_summary['response_characteristics']
        print(f"\n响应特征:")
        print(f"  振荡特征数量: {resp_char.get('oscillating_features', 0)}")
        print(f"  平均规律性: {resp_char.get('average_regularity', 0):.3f}")
        print(f"  响应类型: {resp_char.get('response_type', 'unknown')}")
    
    print(f"\n检测到的变化点数量: {len(analysis_results['change_points'])}")
    
    print("\n短期动态学习测试完成！")