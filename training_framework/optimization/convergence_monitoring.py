#!/usr/bin/env python3
"""
收敛监控

实现训练过程的收敛监控，包括：
- 损失收敛监控
- 梯度收敛监控
- 参数收敛监控
- 早停机制
- 收敛诊断和可视化
- 物理信息神经网络专用收敛指标

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from scipy import stats
from scipy.signal import savgol_filter
import warnings

class ConvergenceMetric(Enum):
    """收敛指标枚举"""
    LOSS = "loss"  # 损失
    GRADIENT_NORM = "gradient_norm"  # 梯度范数
    PARAMETER_CHANGE = "parameter_change"  # 参数变化
    VALIDATION_LOSS = "validation_loss"  # 验证损失
    PHYSICS_RESIDUAL = "physics_residual"  # 物理残差
    DATA_FITTING = "data_fitting"  # 数据拟合
    BOUNDARY_LOSS = "boundary_loss"  # 边界损失
    LEARNING_RATE = "learning_rate"  # 学习率
    RELATIVE_IMPROVEMENT = "relative_improvement"  # 相对改善

class ConvergenceStatus(Enum):
    """收敛状态枚举"""
    CONVERGED = "converged"  # 已收敛
    DIVERGED = "diverged"  # 发散
    STAGNATED = "stagnated"  # 停滞
    OSCILLATING = "oscillating"  # 振荡
    IMPROVING = "improving"  # 改善中
    UNSTABLE = "unstable"  # 不稳定

class EarlyStopReason(Enum):
    """早停原因枚举"""
    LOSS_CONVERGED = "loss_converged"  # 损失收敛
    NO_IMPROVEMENT = "no_improvement"  # 无改善
    GRADIENT_VANISHED = "gradient_vanished"  # 梯度消失
    GRADIENT_EXPLODED = "gradient_exploded"  # 梯度爆炸
    LOSS_DIVERGED = "loss_diverged"  # 损失发散
    MAX_PATIENCE = "max_patience"  # 达到最大耐心值
    MANUAL_STOP = "manual_stop"  # 手动停止

@dataclass
class ConvergenceConfig:
    """收敛监控配置"""
    # 基础配置
    monitor_frequency: int = 1  # 监控频率
    window_size: int = 50  # 滑动窗口大小
    smoothing_factor: float = 0.9  # 平滑因子
    
    # 收敛阈值
    loss_threshold: float = 1e-6  # 损失阈值
    gradient_threshold: float = 1e-8  # 梯度阈值
    parameter_threshold: float = 1e-10  # 参数变化阈值
    relative_threshold: float = 1e-4  # 相对改善阈值
    
    # 早停配置
    enable_early_stopping: bool = True
    patience: int = 100  # 耐心值
    min_delta: float = 1e-6  # 最小改善
    restore_best_weights: bool = True
    
    # 发散检测
    divergence_threshold: float = 1e6  # 发散阈值
    gradient_explosion_threshold: float = 1e3  # 梯度爆炸阈值
    
    # 振荡检测
    oscillation_threshold: float = 0.1  # 振荡阈值
    oscillation_window: int = 20  # 振荡检测窗口
    
    # 停滞检测
    stagnation_threshold: float = 1e-8  # 停滞阈值
    stagnation_window: int = 50  # 停滞检测窗口
    
    # 物理信息配置
    physics_weight: float = 1.0  # 物理损失权重
    data_weight: float = 1.0  # 数据损失权重
    boundary_weight: float = 1.0  # 边界损失权重
    
    # 可视化配置
    enable_plotting: bool = True
    plot_frequency: int = 100  # 绘图频率
    save_plots: bool = False
    plot_directory: str = "./convergence_plots"
    
    # 日志配置
    enable_logging: bool = True
    log_level: str = "INFO"
    save_history: bool = True
    history_file: str = "convergence_history.json"

class ConvergenceMonitorBase(ABC):
    """
    收敛监控器基类
    
    定义收敛监控器的通用接口
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        初始化收敛监控器
        
        Args:
            config: 配置
        """
        self.config = config
        self.step_count = 0
        self.epoch_count = 0
        self.history = defaultdict(list)
        self.smoothed_history = defaultdict(list)
        self.best_values = {}
        self.best_step = 0
        self.patience_counter = 0
        self.should_stop = False
        self.stop_reason = None
        self.convergence_status = ConvergenceStatus.IMPROVING
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # 创建绘图目录
        if config.save_plots:
            Path(config.plot_directory).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """更新监控状态"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'convergence_status': self.convergence_status.value,
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason.value if self.stop_reason else None,
            'patience_counter': self.patience_counter,
            'best_step': self.best_step,
            'best_values': self.best_values.copy()
        }
    
    def save_history(self, filepath: Optional[str] = None) -> None:
        """
        保存历史记录
        
        Args:
            filepath: 文件路径
        """
        if not self.config.save_history:
            return
        
        if filepath is None:
            filepath = self.config.history_file
        
        # 转换为可序列化的格式
        serializable_history = {}
        for key, values in self.history.items():
            if isinstance(values, list):
                serializable_history[key] = [float(v) if isinstance(v, (int, float, np.number)) else str(v) for v in values]
            else:
                serializable_history[key] = str(values)
        
        data = {
            'history': serializable_history,
            'config': {
                'monitor_frequency': self.config.monitor_frequency,
                'window_size': self.config.window_size,
                'patience': self.config.patience,
                'loss_threshold': self.config.loss_threshold
            },
            'status': self.get_status()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"历史记录已保存到 {filepath}")
        except Exception as e:
            self.logger.error(f"保存历史记录失败: {e}")
    
    def load_history(self, filepath: str) -> None:
        """
        加载历史记录
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 恢复历史记录
            for key, values in data.get('history', {}).items():
                self.history[key] = values
            
            # 恢复状态
            status = data.get('status', {})
            self.step_count = status.get('step_count', 0)
            self.epoch_count = status.get('epoch_count', 0)
            self.best_step = status.get('best_step', 0)
            self.best_values = status.get('best_values', {})
            self.patience_counter = status.get('patience_counter', 0)
            
            self.logger.info(f"历史记录已从 {filepath} 加载")
        except Exception as e:
            self.logger.error(f"加载历史记录失败: {e}")

class LossConvergenceMonitor(ConvergenceMonitorBase):
    """损失收敛监控器"""
    
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        更新损失收敛监控
        
        Args:
            metrics: 指标字典
            model: 模型
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        self.step_count += 1
        
        # 获取损失值
        loss = metrics.get('loss', float('inf'))
        val_loss = metrics.get('validation_loss', None)
        
        # 记录历史
        self.history['loss'].append(loss)
        if val_loss is not None:
            self.history['validation_loss'].append(val_loss)
        
        # 计算平滑值
        if len(self.history['loss']) > 1:
            smoothed_loss = self._exponential_smoothing('loss')
            self.smoothed_history['loss'].append(smoothed_loss)
        
        # 检查是否为最佳值
        is_best = False
        if 'loss' not in self.best_values or loss < self.best_values['loss']:
            self.best_values['loss'] = loss
            self.best_step = self.step_count
            self.patience_counter = 0
            is_best = True
        else:
            self.patience_counter += 1
        
        # 检查收敛状态
        self._check_convergence_status()
        
        # 检查早停条件
        self._check_early_stopping()
        
        result = {
            'loss': loss,
            'smoothed_loss': self.smoothed_history['loss'][-1] if self.smoothed_history['loss'] else loss,
            'is_best': is_best,
            'improvement': self._calculate_improvement(),
            'convergence_status': self.convergence_status.value,
            'should_stop': self.should_stop
        }
        
        # 记录日志
        if self.step_count % self.config.monitor_frequency == 0:
            self.logger.info(
                f"Step {self.step_count}: Loss = {loss:.6e}, "
                f"Best = {self.best_values.get('loss', float('inf')):.6e}, "
                f"Patience = {self.patience_counter}/{self.config.patience}, "
                f"Status = {self.convergence_status.value}"
            )
        
        return result
    
    def _exponential_smoothing(self, metric: str) -> float:
        """指数平滑"""
        values = self.history[metric]
        if len(values) == 1:
            return values[0]
        
        alpha = 1 - self.config.smoothing_factor
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _calculate_improvement(self) -> float:
        """计算改善程度"""
        if len(self.history['loss']) < 2:
            return 0.0
        
        current_loss = self.history['loss'][-1]
        previous_loss = self.history['loss'][-2]
        
        if previous_loss == 0:
            return 0.0
        
        return (previous_loss - current_loss) / abs(previous_loss)
    
    def _check_convergence_status(self) -> None:
        """检查收敛状态"""
        if len(self.history['loss']) < self.config.window_size:
            self.convergence_status = ConvergenceStatus.IMPROVING
            return
        
        recent_losses = self.history['loss'][-self.config.window_size:]
        
        # 检查发散
        if recent_losses[-1] > self.config.divergence_threshold:
            self.convergence_status = ConvergenceStatus.DIVERGED
            return
        
        # 检查收敛
        loss_std = np.std(recent_losses)
        if loss_std < self.config.loss_threshold:
            self.convergence_status = ConvergenceStatus.CONVERGED
            return
        
        # 检查停滞
        if len(recent_losses) >= self.config.stagnation_window:
            recent_improvement = abs(recent_losses[-1] - recent_losses[-self.config.stagnation_window])
            if recent_improvement < self.config.stagnation_threshold:
                self.convergence_status = ConvergenceStatus.STAGNATED
                return
        
        # 检查振荡
        if self._detect_oscillation(recent_losses):
            self.convergence_status = ConvergenceStatus.OSCILLATING
            return
        
        # 检查改善趋势
        if len(recent_losses) >= 10:
            slope, _, _, p_value, _ = stats.linregress(range(len(recent_losses)), recent_losses)
            if slope < 0 and p_value < 0.05:
                self.convergence_status = ConvergenceStatus.IMPROVING
            else:
                self.convergence_status = ConvergenceStatus.UNSTABLE
    
    def _detect_oscillation(self, values: List[float]) -> bool:
        """检测振荡"""
        if len(values) < self.config.oscillation_window:
            return False
        
        # 计算相对变化
        relative_changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                relative_changes.append(abs(values[i] - values[i-1]) / abs(values[i-1]))
        
        if not relative_changes:
            return False
        
        # 检查是否存在大幅振荡
        avg_change = np.mean(relative_changes)
        return avg_change > self.config.oscillation_threshold
    
    def _check_early_stopping(self) -> None:
        """检查早停条件"""
        if not self.config.enable_early_stopping:
            return
        
        # 检查耐心值
        if self.patience_counter >= self.config.patience:
            self.should_stop = True
            self.stop_reason = EarlyStopReason.MAX_PATIENCE
            return
        
        # 检查收敛状态
        if self.convergence_status == ConvergenceStatus.CONVERGED:
            self.should_stop = True
            self.stop_reason = EarlyStopReason.LOSS_CONVERGED
        elif self.convergence_status == ConvergenceStatus.DIVERGED:
            self.should_stop = True
            self.stop_reason = EarlyStopReason.LOSS_DIVERGED

class GradientConvergenceMonitor(ConvergenceMonitorBase):
    """梯度收敛监控器"""
    
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        更新梯度收敛监控
        
        Args:
            metrics: 指标字典
            model: 模型
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        self.step_count += 1
        
        # 计算梯度范数
        if model is not None:
            grad_norm = self._compute_gradient_norm(model)
            metrics['gradient_norm'] = grad_norm
        else:
            grad_norm = metrics.get('gradient_norm', 0.0)
        
        # 记录历史
        self.history['gradient_norm'].append(grad_norm)
        
        # 计算平滑值
        if len(self.history['gradient_norm']) > 1:
            smoothed_grad = self._exponential_smoothing('gradient_norm')
            self.smoothed_history['gradient_norm'].append(smoothed_grad)
        
        # 检查梯度状态
        grad_status = self._check_gradient_status(grad_norm)
        
        result = {
            'gradient_norm': grad_norm,
            'smoothed_gradient_norm': self.smoothed_history['gradient_norm'][-1] if self.smoothed_history['gradient_norm'] else grad_norm,
            'gradient_status': grad_status,
            'should_stop': self.should_stop
        }
        
        # 记录日志
        if self.step_count % self.config.monitor_frequency == 0:
            self.logger.info(
                f"Step {self.step_count}: Gradient Norm = {grad_norm:.6e}, "
                f"Status = {grad_status}"
            )
        
        return result
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _exponential_smoothing(self, metric: str) -> float:
        """指数平滑"""
        values = self.history[metric]
        if len(values) == 1:
            return values[0]
        
        alpha = 1 - self.config.smoothing_factor
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _check_gradient_status(self, grad_norm: float) -> str:
        """检查梯度状态"""
        # 检查梯度爆炸
        if grad_norm > self.config.gradient_explosion_threshold:
            self.should_stop = True
            self.stop_reason = EarlyStopReason.GRADIENT_EXPLODED
            return "exploded"
        
        # 检查梯度消失
        if grad_norm < self.config.gradient_threshold:
            if len(self.history['gradient_norm']) >= 10:
                recent_grads = self.history['gradient_norm'][-10:]
                if all(g < self.config.gradient_threshold for g in recent_grads):
                    self.should_stop = True
                    self.stop_reason = EarlyStopReason.GRADIENT_VANISHED
                    return "vanished"
        
        # 正常状态
        if self.config.gradient_threshold < grad_norm < self.config.gradient_explosion_threshold:
            return "normal"
        elif grad_norm <= self.config.gradient_threshold:
            return "small"
        else:
            return "large"

class ParameterConvergenceMonitor(ConvergenceMonitorBase):
    """参数收敛监控器"""
    
    def __init__(self, config: ConvergenceConfig):
        super().__init__(config)
        self.previous_params = None
    
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        更新参数收敛监控
        
        Args:
            metrics: 指标字典
            model: 模型
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        self.step_count += 1
        
        if model is None:
            return {'parameter_change': 0.0, 'should_stop': False}
        
        # 计算参数变化
        param_change = self._compute_parameter_change(model)
        
        # 记录历史
        self.history['parameter_change'].append(param_change)
        
        # 检查参数收敛
        param_converged = param_change < self.config.parameter_threshold
        
        if param_converged and len(self.history['parameter_change']) >= self.config.window_size:
            recent_changes = self.history['parameter_change'][-self.config.window_size:]
            if all(c < self.config.parameter_threshold for c in recent_changes):
                self.convergence_status = ConvergenceStatus.CONVERGED
        
        result = {
            'parameter_change': param_change,
            'parameter_converged': param_converged,
            'convergence_status': self.convergence_status.value,
            'should_stop': self.should_stop
        }
        
        # 记录日志
        if self.step_count % self.config.monitor_frequency == 0:
            self.logger.info(
                f"Step {self.step_count}: Parameter Change = {param_change:.6e}, "
                f"Converged = {param_converged}"
            )
        
        return result
    
    def _compute_parameter_change(self, model: nn.Module) -> float:
        """计算参数变化"""
        current_params = []
        for param in model.parameters():
            current_params.append(param.data.clone().flatten())
        
        current_params = torch.cat(current_params)
        
        if self.previous_params is None:
            self.previous_params = current_params.clone()
            return 0.0
        
        # 计算L2范数变化
        param_diff = current_params - self.previous_params
        param_change = torch.norm(param_diff).item()
        
        # 更新前一步参数
        self.previous_params = current_params.clone()
        
        return param_change

class PhysicsInformedConvergenceMonitor(ConvergenceMonitorBase):
    """物理信息收敛监控器"""
    
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        更新物理信息收敛监控
        
        Args:
            metrics: 指标字典
            model: 模型
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        self.step_count += 1
        
        # 获取各类损失
        physics_loss = metrics.get('physics_loss', 0.0)
        data_loss = metrics.get('data_loss', 0.0)
        boundary_loss = metrics.get('boundary_loss', 0.0)
        total_loss = metrics.get('loss', physics_loss + data_loss + boundary_loss)
        
        # 记录历史
        self.history['physics_loss'].append(physics_loss)
        self.history['data_loss'].append(data_loss)
        self.history['boundary_loss'].append(boundary_loss)
        self.history['total_loss'].append(total_loss)
        
        # 计算损失平衡
        balance_score = self._compute_balance_score(physics_loss, data_loss, boundary_loss)
        self.history['balance_score'].append(balance_score)
        
        # 计算物理残差
        physics_residual = self._compute_physics_residual(physics_loss)
        self.history['physics_residual'].append(physics_residual)
        
        # 检查物理收敛
        physics_converged = self._check_physics_convergence()
        
        result = {
            'physics_loss': physics_loss,
            'data_loss': data_loss,
            'boundary_loss': boundary_loss,
            'balance_score': balance_score,
            'physics_residual': physics_residual,
            'physics_converged': physics_converged,
            'should_stop': self.should_stop
        }
        
        # 记录日志
        if self.step_count % self.config.monitor_frequency == 0:
            self.logger.info(
                f"Step {self.step_count}: Physics = {physics_loss:.6e}, "
                f"Data = {data_loss:.6e}, Boundary = {boundary_loss:.6e}, "
                f"Balance = {balance_score:.4f}, Residual = {physics_residual:.6e}"
            )
        
        return result
    
    def _compute_balance_score(self, physics_loss: float, data_loss: float, boundary_loss: float) -> float:
        """计算损失平衡分数"""
        total = physics_loss + data_loss + boundary_loss
        if total == 0:
            return 1.0
        
        # 计算各损失的比例
        physics_ratio = physics_loss / total
        data_ratio = data_loss / total
        boundary_ratio = boundary_loss / total
        
        # 理想情况下，各损失应该平衡
        target_ratio = 1.0 / 3.0
        
        # 计算与理想比例的偏差
        physics_dev = abs(physics_ratio - target_ratio)
        data_dev = abs(data_ratio - target_ratio)
        boundary_dev = abs(boundary_ratio - target_ratio)
        
        # 平衡分数：偏差越小，分数越高
        total_dev = physics_dev + data_dev + boundary_dev
        balance_score = 1.0 / (1.0 + total_dev * 3)
        
        return balance_score
    
    def _compute_physics_residual(self, physics_loss: float) -> float:
        """计算物理残差"""
        # 简化的物理残差计算
        # 实际应用中应该基于具体的物理方程
        return np.sqrt(physics_loss)
    
    def _check_physics_convergence(self) -> bool:
        """检查物理收敛"""
        if len(self.history['physics_residual']) < self.config.window_size:
            return False
        
        recent_residuals = self.history['physics_residual'][-self.config.window_size:]
        
        # 检查物理残差是否足够小
        avg_residual = np.mean(recent_residuals)
        residual_std = np.std(recent_residuals)
        
        physics_threshold = self.config.loss_threshold * 10  # 物理损失阈值稍微宽松
        
        return avg_residual < physics_threshold and residual_std < physics_threshold

class ConvergenceMonitor:
    """
    综合收敛监控器
    
    整合所有收敛监控功能
    """
    
    def __init__(self, config: ConvergenceConfig = None):
        """
        初始化收敛监控器
        
        Args:
            config: 配置
        """
        if config is None:
            config = ConvergenceConfig()
        
        self.config = config
        self.monitors = {
            'loss': LossConvergenceMonitor(config),
            'gradient': GradientConvergenceMonitor(config),
            'parameter': ParameterConvergenceMonitor(config),
            'physics': PhysicsInformedConvergenceMonitor(config)
        }
        
        self.global_step = 0
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        # 创建绘图目录
        if config.save_plots:
            Path(config.plot_directory).mkdir(parents=True, exist_ok=True)
    
    def update(self, metrics: Dict[str, float], model: Optional[nn.Module] = None, epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        更新所有监控器
        
        Args:
            metrics: 指标字典
            model: 模型
            epoch: 当前epoch
            
        Returns:
            Dict[str, Any]: 更新结果
        """
        self.global_step += 1
        
        if epoch is not None:
            for monitor in self.monitors.values():
                monitor.epoch_count = epoch
        
        # 更新所有监控器
        results = {}
        for name, monitor in self.monitors.items():
            try:
                result = monitor.update(metrics, model)
                results[name] = result
            except Exception as e:
                self.logger.error(f"监控器 {name} 更新失败: {e}")
                results[name] = {'error': str(e)}
        
        # 检查全局停止条件
        should_stop = any(monitor.should_stop for monitor in self.monitors.values())
        stop_reasons = [monitor.stop_reason for monitor in self.monitors.values() if monitor.stop_reason]
        
        # 综合结果
        global_result = {
            'global_step': self.global_step,
            'elapsed_time': time.time() - self.start_time,
            'should_stop': should_stop,
            'stop_reasons': [reason.value for reason in stop_reasons],
            'monitors': results
        }
        
        # 绘图
        if self.config.enable_plotting and self.global_step % self.config.plot_frequency == 0:
            self.plot_convergence()
        
        # 保存历史
        if self.global_step % 100 == 0:  # 每100步保存一次
            self.save_history()
        
        return global_result
    
    def should_stop(self) -> bool:
        """检查是否应该停止训练"""
        return any(monitor.should_stop for monitor in self.monitors.values())
    
    def get_stop_reasons(self) -> List[str]:
        """获取停止原因"""
        reasons = []
        for monitor in self.monitors.values():
            if monitor.stop_reason:
                reasons.append(monitor.stop_reason.value)
        return reasons
    
    def get_best_values(self) -> Dict[str, Any]:
        """获取最佳值"""
        best_values = {}
        for name, monitor in self.monitors.items():
            best_values[name] = monitor.best_values.copy()
        return best_values
    
    def reset(self) -> None:
        """重置所有监控器"""
        for monitor in self.monitors.values():
            monitor.step_count = 0
            monitor.epoch_count = 0
            monitor.history.clear()
            monitor.smoothed_history.clear()
            monitor.best_values.clear()
            monitor.best_step = 0
            monitor.patience_counter = 0
            monitor.should_stop = False
            monitor.stop_reason = None
            monitor.convergence_status = ConvergenceStatus.IMPROVING
        
        self.global_step = 0
        self.start_time = time.time()
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        绘制收敛图
        
        Args:
            save_path: 保存路径
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # 损失曲线
            if 'loss' in self.monitors['loss'].history:
                losses = self.monitors['loss'].history['loss']
                if losses:
                    axes[0].plot(losses, label='Loss', alpha=0.7)
                    if self.monitors['loss'].smoothed_history['loss']:
                        axes[0].plot(self.monitors['loss'].smoothed_history['loss'], label='Smoothed Loss', linewidth=2)
                    axes[0].set_title('Loss Convergence')
                    axes[0].set_xlabel('Steps')
                    axes[0].set_ylabel('Loss')
                    axes[0].set_yscale('log')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
            
            # 梯度范数
            if 'gradient_norm' in self.monitors['gradient'].history:
                grad_norms = self.monitors['gradient'].history['gradient_norm']
                if grad_norms:
                    axes[1].plot(grad_norms, label='Gradient Norm', alpha=0.7)
                    if self.monitors['gradient'].smoothed_history['gradient_norm']:
                        axes[1].plot(self.monitors['gradient'].smoothed_history['gradient_norm'], label='Smoothed', linewidth=2)
                    axes[1].set_title('Gradient Norm')
                    axes[1].set_xlabel('Steps')
                    axes[1].set_ylabel('Gradient Norm')
                    axes[1].set_yscale('log')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
            
            # 参数变化
            if 'parameter_change' in self.monitors['parameter'].history:
                param_changes = self.monitors['parameter'].history['parameter_change']
                if param_changes:
                    axes[2].plot(param_changes, label='Parameter Change')
                    axes[2].set_title('Parameter Change')
                    axes[2].set_xlabel('Steps')
                    axes[2].set_ylabel('Parameter Change')
                    axes[2].set_yscale('log')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
            
            # 物理损失分解
            physics_monitor = self.monitors['physics']
            if 'physics_loss' in physics_monitor.history:
                physics_losses = physics_monitor.history['physics_loss']
                data_losses = physics_monitor.history['data_loss']
                boundary_losses = physics_monitor.history['boundary_loss']
                
                if physics_losses:
                    axes[3].plot(physics_losses, label='Physics Loss', alpha=0.7)
                if data_losses:
                    axes[3].plot(data_losses, label='Data Loss', alpha=0.7)
                if boundary_losses:
                    axes[3].plot(boundary_losses, label='Boundary Loss', alpha=0.7)
                
                axes[3].set_title('Loss Components')
                axes[3].set_xlabel('Steps')
                axes[3].set_ylabel('Loss')
                axes[3].set_yscale('log')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            
            # 平衡分数
            if 'balance_score' in physics_monitor.history:
                balance_scores = physics_monitor.history['balance_score']
                if balance_scores:
                    axes[4].plot(balance_scores, label='Balance Score')
                    axes[4].set_title('Loss Balance Score')
                    axes[4].set_xlabel('Steps')
                    axes[4].set_ylabel('Balance Score')
                    axes[4].set_ylim(0, 1)
                    axes[4].legend()
                    axes[4].grid(True, alpha=0.3)
            
            # 物理残差
            if 'physics_residual' in physics_monitor.history:
                residuals = physics_monitor.history['physics_residual']
                if residuals:
                    axes[5].plot(residuals, label='Physics Residual')
                    axes[5].set_title('Physics Residual')
                    axes[5].set_xlabel('Steps')
                    axes[5].set_ylabel('Residual')
                    axes[5].set_yscale('log')
                    axes[5].legend()
                    axes[5].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path or self.config.save_plots:
                if save_path is None:
                    save_path = Path(self.config.plot_directory) / f"convergence_step_{self.global_step}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制收敛图失败: {e}")
    
    def save_history(self, filepath: Optional[str] = None) -> None:
        """保存所有监控器的历史记录"""
        for name, monitor in self.monitors.items():
            if filepath:
                monitor_filepath = f"{filepath}_{name}.json"
            else:
                monitor_filepath = f"{self.config.history_file.replace('.json', '')}_{name}.json"
            monitor.save_history(monitor_filepath)
    
    def load_history(self, filepath_pattern: str) -> None:
        """加载所有监控器的历史记录"""
        for name, monitor in self.monitors.items():
            monitor_filepath = filepath_pattern.replace('{}', name)
            if Path(monitor_filepath).exists():
                monitor.load_history(monitor_filepath)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取收敛总结"""
        summary = {
            'global_step': self.global_step,
            'elapsed_time': time.time() - self.start_time,
            'should_stop': self.should_stop(),
            'stop_reasons': self.get_stop_reasons(),
            'best_values': self.get_best_values()
        }
        
        # 添加各监控器的状态
        for name, monitor in self.monitors.items():
            summary[f"{name}_status"] = monitor.get_status()
        
        return summary

def create_convergence_monitor(config: ConvergenceConfig = None) -> ConvergenceMonitor:
    """
    创建收敛监控器
    
    Args:
        config: 配置
        
    Returns:
        ConvergenceMonitor: 收敛监控器实例
    """
    return ConvergenceMonitor(config)

if __name__ == "__main__":
    # 测试收敛监控器
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建收敛监控配置
    config = ConvergenceConfig(
        monitor_frequency=10,
        window_size=20,
        patience=50,
        enable_early_stopping=True,
        enable_plotting=True,
        plot_frequency=50
    )
    
    # 创建收敛监控器
    convergence_monitor = create_convergence_monitor(config)
    
    print("=== 收敛监控器测试 ===")
    
    # 模拟训练过程
    for step in range(200):
        # 生成随机数据
        inputs = torch.randn(32, 3)
        targets = torch.sin(inputs.sum(dim=1, keepdim=True))
        
        # 前向传播
        predictions = model(inputs)
        
        # 计算损失
        data_loss = nn.MSELoss()(predictions, targets)
        physics_loss = torch.mean(predictions ** 2) * 0.1
        boundary_loss = torch.mean(torch.abs(predictions)) * 0.05
        total_loss = data_loss + physics_loss + boundary_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 准备指标
        metrics = {
            'loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'boundary_loss': boundary_loss.item()
        }
        
        # 更新收敛监控
        result = convergence_monitor.update(metrics, model, epoch=step//10)
        
        # 检查是否应该停止
        if result['should_stop']:
            print(f"\n训练在步骤 {step} 停止")
            print(f"停止原因: {result['stop_reasons']}")
            break
        
        # 打印进度
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Data Loss: {data_loss.item():.6f}")
            print(f"  Physics Loss: {physics_loss.item():.6f}")
            print(f"  Boundary Loss: {boundary_loss.item():.6f}")
            
            # 打印监控状态
            for monitor_name, monitor_result in result['monitors'].items():
                if isinstance(monitor_result, dict) and 'convergence_status' in monitor_result:
                    print(f"  {monitor_name.title()} Status: {monitor_result.get('convergence_status', 'unknown')}")
    
    # 获取最终总结
    summary = convergence_monitor.get_summary()
    print(f"\n=== 收敛总结 ===")
    print(f"总步数: {summary['global_step']}")
    print(f"训练时间: {summary['elapsed_time']:.2f} 秒")
    print(f"是否停止: {summary['should_stop']}")
    print(f"停止原因: {summary['stop_reasons']}")
    
    # 打印最佳值
    best_values = summary['best_values']
    for monitor_name, values in best_values.items():
        if values:
            print(f"\n{monitor_name.title()} 最佳值:")
            for metric, value in values.items():
                print(f"  {metric}: {value:.6e}")
    
    print("\n收敛监控器测试完成！")