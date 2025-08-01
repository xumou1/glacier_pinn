#!/usr/bin/env python3
"""
损失权重模块

实现PINNs训练中各种损失项的动态权重调整策略，
包括物理损失、数据损失、边界条件损失等的平衡。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class LossWeightingStrategy(ABC):
    """
    损失权重策略基类
    
    定义损失权重调整的通用接口，包括：
    - 静态权重策略
    - 动态权重策略
    - 自适应权重策略
    - 多阶段权重策略
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 weighting_config: Dict[str, Any]):
        """
        初始化权重策略
        
        Args:
            initial_weights: 初始权重字典
            weighting_config: 权重配置
        """
        self.initial_weights = initial_weights.copy()
        self.current_weights = initial_weights.copy()
        self.weighting_config = weighting_config
        self.step_count = 0
        self.loss_history = {}
        
    @abstractmethod
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int) -> Dict[str, float]:
        """
        更新损失权重
        
        Args:
            loss_components: 各损失组件的值
            step: 当前步数
            
        Returns:
            更新后的权重字典
        """
        pass
        
    def get_current_weights(self) -> Dict[str, float]:
        """
        获取当前权重
        
        Returns:
            当前权重字典
        """
        return self.current_weights.copy()
        
    def reset(self):
        """
        重置权重策略
        """
        self.current_weights = self.initial_weights.copy()
        self.step_count = 0
        self.loss_history = {}

class StaticWeightingStrategy(LossWeightingStrategy):
    """
    静态权重策略
    
    使用固定的权重，不随训练过程变化。
    适用于权重已经调优的情况。
    """
    
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int) -> Dict[str, float]:
        """
        静态权重不变
        """
        self.step_count = step
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
            
        return self.current_weights

class GradNormWeightingStrategy(LossWeightingStrategy):
    """
    基于梯度范数的权重策略
    
    根据各损失项的梯度范数动态调整权重，
    确保不同损失项对参数更新的贡献相对平衡。
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 weighting_config: Dict[str, Any]):
        super().__init__(initial_weights, weighting_config)
        
        # GradNorm参数
        self.alpha = weighting_config.get('alpha', 0.12)  # 恢复速率
        self.update_frequency = weighting_config.get('update_frequency', 10)
        self.target_ratios = weighting_config.get('target_ratios', None)
        
        # 内部状态
        self.initial_losses = None
        self.gradient_norms = {}
        
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int,
                      gradients: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        基于梯度范数更新权重
        
        Args:
            loss_components: 各损失组件的值
            step: 当前步数
            gradients: 各损失项对应的梯度（可选）
            
        Returns:
            更新后的权重字典
        """
        self.step_count = step
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
            
        # 记录初始损失
        if self.initial_losses is None:
            self.initial_losses = loss_components.copy()
            
        # 只在指定频率更新权重
        if step % self.update_frequency != 0 or gradients is None:
            return self.current_weights
            
        # 计算各损失项的梯度范数
        for loss_name in loss_components.keys():
            if loss_name in gradients:
                grad_norm = self._compute_gradient_norm(gradients[loss_name])
                self.gradient_norms[loss_name] = grad_norm
                
        # 计算相对损失率
        relative_losses = self._compute_relative_losses(loss_components)
        
        # 计算目标梯度范数
        target_grad_norms = self._compute_target_gradient_norms(
            relative_losses, self.gradient_norms
        )
        
        # 更新权重
        self._update_weights_gradnorm(target_grad_norms)
        
        return self.current_weights
        
    def _compute_gradient_norm(self, gradients: Dict) -> float:
        """
        计算梯度范数
        """
        total_norm = 0.0
        
        def add_norm(x):
            nonlocal total_norm
            total_norm += jnp.sum(x**2)
            
        jax.tree_map(add_norm, gradients)
        
        return jnp.sqrt(total_norm)
        
    def _compute_relative_losses(self, loss_components: Dict[str, float]) -> Dict[str, float]:
        """
        计算相对损失率
        """
        relative_losses = {}
        
        for key, current_loss in loss_components.items():
            if key in self.initial_losses and self.initial_losses[key] > 0:
                relative_losses[key] = current_loss / self.initial_losses[key]
            else:
                relative_losses[key] = 1.0
                
        return relative_losses
        
    def _compute_target_gradient_norms(self, 
                                      relative_losses: Dict[str, float],
                                      gradient_norms: Dict[str, float]) -> Dict[str, float]:
        """
        计算目标梯度范数
        """
        # 计算平均梯度范数
        avg_grad_norm = jnp.mean(jnp.array(list(gradient_norms.values())))
        
        target_grad_norms = {}
        
        for key in gradient_norms.keys():
            if self.target_ratios and key in self.target_ratios:
                # 使用指定的目标比率
                target_ratio = self.target_ratios[key]
            else:
                # 使用相对损失率的倒数作为目标比率
                target_ratio = 1.0 / (relative_losses.get(key, 1.0) + 1e-8)
                
            target_grad_norms[key] = avg_grad_norm * target_ratio
            
        return target_grad_norms
        
    def _update_weights_gradnorm(self, target_grad_norms: Dict[str, float]):
        """
        使用GradNorm算法更新权重
        """
        for key in self.current_weights.keys():
            if key in self.gradient_norms and key in target_grad_norms:
                current_norm = self.gradient_norms[key]
                target_norm = target_grad_norms[key]
                
                if current_norm > 0:
                    # GradNorm更新规则
                    ratio = target_norm / current_norm
                    
                    # 使用指数移动平均更新权重
                    self.current_weights[key] = (
                        (1 - self.alpha) * self.current_weights[key] + 
                        self.alpha * self.current_weights[key] * ratio
                    )
                    
                    # 限制权重范围
                    self.current_weights[key] = jnp.clip(
                        self.current_weights[key], 0.01, 100.0
                    )

class AdaptiveWeightingStrategy(LossWeightingStrategy):
    """
    自适应权重策略
    
    根据各损失项的收敛情况动态调整权重：
    - 收敛慢的损失项增加权重
    - 收敛快的损失项减少权重
    - 震荡的损失项稳定权重
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 weighting_config: Dict[str, Any]):
        super().__init__(initial_weights, weighting_config)
        
        # 自适应参数
        self.adaptation_rate = weighting_config.get('adaptation_rate', 0.01)
        self.window_size = weighting_config.get('window_size', 50)
        self.convergence_threshold = weighting_config.get('convergence_threshold', 1e-6)
        self.oscillation_threshold = weighting_config.get('oscillation_threshold', 0.1)
        
        # 权重限制
        self.min_weight = weighting_config.get('min_weight', 0.01)
        self.max_weight = weighting_config.get('max_weight', 10.0)
        
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int) -> Dict[str, float]:
        """
        自适应权重更新
        """
        self.step_count = step
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
            
        # 需要足够的历史数据才能进行自适应调整
        if step < self.window_size:
            return self.current_weights
            
        # 分析各损失项的收敛情况
        convergence_analysis = self._analyze_convergence()
        
        # 根据收敛分析调整权重
        self._adjust_weights_based_on_convergence(convergence_analysis)
        
        return self.current_weights
        
    def _analyze_convergence(self) -> Dict[str, Dict[str, float]]:
        """
        分析各损失项的收敛情况
        
        Returns:
            包含收敛速度、稳定性等信息的字典
        """
        analysis = {}
        
        for key, history in self.loss_history.items():
            if len(history) < self.window_size:
                continue
                
            recent_history = jnp.array(history[-self.window_size:])
            
            # 计算收敛速度（损失下降率）
            convergence_rate = self._compute_convergence_rate(recent_history)
            
            # 计算稳定性（方差）
            stability = self._compute_stability(recent_history)
            
            # 检测震荡
            oscillation = self._detect_oscillation(recent_history)
            
            analysis[key] = {
                'convergence_rate': convergence_rate,
                'stability': stability,
                'oscillation': oscillation,
                'current_loss': recent_history[-1]
            }
            
        return analysis
        
    def _compute_convergence_rate(self, history: jnp.ndarray) -> float:
        """
        计算收敛速度
        """
        if len(history) < 2:
            return 0.0
            
        # 使用线性回归计算趋势
        x = jnp.arange(len(history))
        y = jnp.log(history + 1e-12)  # 对数尺度
        
        # 简单线性回归
        n = len(x)
        sum_x = jnp.sum(x)
        sum_y = jnp.sum(y)
        sum_xy = jnp.sum(x * y)
        sum_x2 = jnp.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-12)
        
        return float(-slope)  # 负斜率表示下降
        
    def _compute_stability(self, history: jnp.ndarray) -> float:
        """
        计算稳定性（基于相对方差）
        """
        mean_loss = jnp.mean(history)
        var_loss = jnp.var(history)
        
        # 相对方差
        relative_variance = var_loss / (mean_loss**2 + 1e-12)
        
        return float(relative_variance)
        
    def _detect_oscillation(self, history: jnp.ndarray) -> float:
        """
        检测震荡程度
        """
        if len(history) < 3:
            return 0.0
            
        # 计算相邻差值的符号变化
        diffs = jnp.diff(history)
        sign_changes = jnp.sum(jnp.abs(jnp.diff(jnp.sign(diffs))))
        
        # 归一化震荡指标
        oscillation_index = sign_changes / (len(diffs) - 1)
        
        return float(oscillation_index)
        
    def _adjust_weights_based_on_convergence(self, analysis: Dict[str, Dict[str, float]]):
        """
        基于收敛分析调整权重
        """
        for key, info in analysis.items():
            if key not in self.current_weights:
                continue
                
            convergence_rate = info['convergence_rate']
            stability = info['stability']
            oscillation = info['oscillation']
            
            # 权重调整因子
            adjustment_factor = 1.0
            
            # 收敛慢的损失项增加权重
            if convergence_rate < self.convergence_threshold:
                adjustment_factor *= (1 + self.adaptation_rate)
                
            # 不稳定的损失项减少权重
            if stability > self.oscillation_threshold:
                adjustment_factor *= (1 - self.adaptation_rate * 0.5)
                
            # 震荡严重的损失项减少权重
            if oscillation > self.oscillation_threshold:
                adjustment_factor *= (1 - self.adaptation_rate * 0.3)
                
            # 应用调整
            self.current_weights[key] *= adjustment_factor
            
            # 限制权重范围
            self.current_weights[key] = jnp.clip(
                self.current_weights[key], 
                self.min_weight, 
                self.max_weight
            )

class MultiStageWeightingStrategy(LossWeightingStrategy):
    """
    多阶段权重策略
    
    为PINNs训练的不同阶段使用不同的权重配置：
    - 预训练阶段：重点关注数据拟合
    - 物理集成阶段：平衡物理和数据损失
    - 耦合优化阶段：重点关注物理一致性
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 weighting_config: Dict[str, Any]):
        super().__init__(initial_weights, weighting_config)
        
        # 阶段配置
        self.stages = weighting_config.get('stages', {
            'pretraining': {
                'duration': 1000,
                'weights': {'data_loss': 1.0, 'physics_loss': 0.1, 'boundary_loss': 0.5}
            },
            'physics_integration': {
                'duration': 2000,
                'weights': {'data_loss': 0.5, 'physics_loss': 1.0, 'boundary_loss': 1.0}
            },
            'coupled_optimization': {
                'duration': 2000,
                'weights': {'data_loss': 0.3, 'physics_loss': 1.5, 'boundary_loss': 1.2}
            }
        })
        
        # 当前阶段
        self.current_stage = 'pretraining'
        self.stage_start_step = 0
        
        # 平滑过渡参数
        self.transition_steps = weighting_config.get('transition_steps', 100)
        
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int) -> Dict[str, float]:
        """
        多阶段权重更新
        """
        self.step_count = step
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
            
        # 更新当前阶段
        self._update_stage(step)
        
        # 获取当前阶段的目标权重
        target_weights = self._get_stage_weights(self.current_stage)
        
        # 平滑过渡到目标权重
        self._smooth_transition_to_target(target_weights, step)
        
        return self.current_weights
        
    def _update_stage(self, step: int):
        """
        更新当前训练阶段
        """
        cumulative_steps = 0
        
        for stage_name, stage_config in self.stages.items():
            cumulative_steps += stage_config['duration']
            
            if step < cumulative_steps:
                if self.current_stage != stage_name:
                    print(f"Switching to stage: {stage_name} at step {step}")
                    self.current_stage = stage_name
                    self.stage_start_step = cumulative_steps - stage_config['duration']
                break
                
    def _get_stage_weights(self, stage_name: str) -> Dict[str, float]:
        """
        获取指定阶段的权重
        """
        if stage_name in self.stages:
            stage_weights = self.stages[stage_name]['weights']
            
            # 确保所有权重键都存在
            target_weights = self.current_weights.copy()
            for key, weight in stage_weights.items():
                if key in target_weights:
                    target_weights[key] = weight
                    
            return target_weights
        else:
            return self.current_weights.copy()
            
    def _smooth_transition_to_target(self, target_weights: Dict[str, float], step: int):
        """
        平滑过渡到目标权重
        """
        # 计算过渡进度
        steps_in_stage = step - self.stage_start_step
        transition_progress = min(steps_in_stage / self.transition_steps, 1.0)
        
        # 线性插值
        for key in self.current_weights.keys():
            if key in target_weights:
                current_weight = self.current_weights[key]
                target_weight = target_weights[key]
                
                self.current_weights[key] = (
                    current_weight * (1 - transition_progress) + 
                    target_weight * transition_progress
                )
                
    def set_stage(self, stage_name: str, step: int):
        """
        手动设置训练阶段
        
        Args:
            stage_name: 阶段名称
            step: 当前步数
        """
        if stage_name in self.stages:
            self.current_stage = stage_name
            self.stage_start_step = step
            print(f"Manually set stage to: {stage_name} at step {step}")
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

class UncertaintyWeightingStrategy(LossWeightingStrategy):
    """
    基于不确定性的权重策略
    
    根据模型对各损失项的不确定性动态调整权重，
    不确定性高的损失项获得更高权重。
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 weighting_config: Dict[str, Any]):
        super().__init__(initial_weights, weighting_config)
        
        # 不确定性参数
        self.uncertainty_window = weighting_config.get('uncertainty_window', 20)
        self.uncertainty_threshold = weighting_config.get('uncertainty_threshold', 0.1)
        self.adaptation_strength = weighting_config.get('adaptation_strength', 0.1)
        
    def update_weights(self, 
                      loss_components: Dict[str, float],
                      step: int) -> Dict[str, float]:
        """
        基于不确定性更新权重
        """
        self.step_count = step
        
        # 记录损失历史
        for key, value in loss_components.items():
            if key not in self.loss_history:
                self.loss_history[key] = []
            self.loss_history[key].append(value)
            
        # 需要足够的历史数据
        if step < self.uncertainty_window:
            return self.current_weights
            
        # 计算各损失项的不确定性
        uncertainties = self._compute_uncertainties()
        
        # 基于不确定性调整权重
        self._adjust_weights_based_on_uncertainty(uncertainties)
        
        return self.current_weights
        
    def _compute_uncertainties(self) -> Dict[str, float]:
        """
        计算各损失项的不确定性
        """
        uncertainties = {}
        
        for key, history in self.loss_history.items():
            if len(history) < self.uncertainty_window:
                uncertainties[key] = 0.0
                continue
                
            recent_history = jnp.array(history[-self.uncertainty_window:])
            
            # 使用标准差作为不确定性度量
            mean_loss = jnp.mean(recent_history)
            std_loss = jnp.std(recent_history)
            
            # 相对不确定性
            relative_uncertainty = std_loss / (mean_loss + 1e-12)
            uncertainties[key] = float(relative_uncertainty)
            
        return uncertainties
        
    def _adjust_weights_based_on_uncertainty(self, uncertainties: Dict[str, float]):
        """
        基于不确定性调整权重
        """
        # 计算平均不确定性
        avg_uncertainty = jnp.mean(jnp.array(list(uncertainties.values())))
        
        for key, uncertainty in uncertainties.items():
            if key not in self.current_weights:
                continue
                
            # 不确定性高于平均值的损失项增加权重
            if uncertainty > avg_uncertainty + self.uncertainty_threshold:
                adjustment = 1 + self.adaptation_strength * (
                    uncertainty - avg_uncertainty
                ) / avg_uncertainty
            # 不确定性低于平均值的损失项减少权重
            elif uncertainty < avg_uncertainty - self.uncertainty_threshold:
                adjustment = 1 - self.adaptation_strength * (
                    avg_uncertainty - uncertainty
                ) / avg_uncertainty
            else:
                adjustment = 1.0
                
            self.current_weights[key] *= adjustment
            
            # 限制权重范围
            self.current_weights[key] = jnp.clip(
                self.current_weights[key], 0.01, 10.0
            )

def create_weighting_strategy(strategy_type: str,
                            initial_weights: Dict[str, float],
                            weighting_config: Dict[str, Any]) -> LossWeightingStrategy:
    """
    工厂函数：创建损失权重策略
    
    Args:
        strategy_type: 策略类型
        initial_weights: 初始权重
        weighting_config: 权重配置
        
    Returns:
        权重策略实例
    """
    if strategy_type == 'static':
        return StaticWeightingStrategy(initial_weights, weighting_config)
    elif strategy_type == 'gradnorm':
        return GradNormWeightingStrategy(initial_weights, weighting_config)
    elif strategy_type == 'adaptive':
        return AdaptiveWeightingStrategy(initial_weights, weighting_config)
    elif strategy_type == 'multi_stage':
        return MultiStageWeightingStrategy(initial_weights, weighting_config)
    elif strategy_type == 'uncertainty':
        return UncertaintyWeightingStrategy(initial_weights, weighting_config)
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy_type}")

def compute_weighted_loss(loss_components: Dict[str, float],
                         weights: Dict[str, float]) -> float:
    """
    计算加权总损失
    
    Args:
        loss_components: 各损失组件
        weights: 权重字典
        
    Returns:
        加权总损失
    """
    total_loss = 0.0
    
    for key, loss_value in loss_components.items():
        weight = weights.get(key, 1.0)
        total_loss += weight * loss_value
        
    return total_loss

if __name__ == "__main__":
    # 测试代码
    print("Loss weighting strategy module loaded successfully")
    
    # 测试不同类型的权重策略
    initial_weights = {
        'data_loss': 1.0,
        'physics_loss': 1.0,
        'boundary_loss': 0.5,
        'initial_loss': 0.3
    }
    
    # 静态权重策略
    static_strategy = create_weighting_strategy(
        'static', 
        initial_weights, 
        {}
    )
    
    # 自适应权重策略
    adaptive_strategy = create_weighting_strategy(
        'adaptive',
        initial_weights,
        {
            'adaptation_rate': 0.01,
            'window_size': 50,
            'convergence_threshold': 1e-6
        }
    )
    
    # 多阶段权重策略
    multi_stage_strategy = create_weighting_strategy(
        'multi_stage',
        initial_weights,
        {
            'stages': {
                'pretraining': {
                    'duration': 1000,
                    'weights': {'data_loss': 1.0, 'physics_loss': 0.1}
                },
                'physics_integration': {
                    'duration': 2000,
                    'weights': {'data_loss': 0.5, 'physics_loss': 1.0}
                }
            }
        }
    )
    
    print(f"Created strategies:")
    print(f"Static strategy: {type(static_strategy).__name__}")
    print(f"Adaptive strategy: {type(adaptive_strategy).__name__}")
    print(f"Multi-stage strategy: {type(multi_stage_strategy).__name__}")
    
    # 模拟训练过程
    print("\nSimulating training process:")
    for step in range(0, 100, 10):
        # 模拟损失组件
        loss_components = {
            'data_loss': 1.0 / (step + 1),
            'physics_loss': 0.5 / (step + 1),
            'boundary_loss': 0.3 / (step + 1)
        }
        
        weights = adaptive_strategy.update_weights(loss_components, step)
        total_loss = compute_weighted_loss(loss_components, weights)
        
        print(f"Step {step}: Total loss = {total_loss:.4f}")
        print(f"  Weights: {weights}")