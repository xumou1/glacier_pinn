#!/usr/bin/env python3
"""
自适应优化器

实现各种自适应优化算法，包括：
- 自适应学习率优化器
- 动量自适应优化器
- 二阶优化器
- 物理信息神经网络专用优化器

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
import math
from collections import defaultdict, deque

class OptimizerType(Enum):
    """优化器类型枚举"""
    ADAPTIVE_SGD = "adaptive_sgd"
    ADAPTIVE_ADAM = "adaptive_adam"
    ADAPTIVE_ADAMW = "adaptive_adamw"
    LBFGS_ADAPTIVE = "lbfgs_adaptive"
    PHYSICS_INFORMED = "physics_informed"
    MULTI_OBJECTIVE = "multi_objective"
    GRADIENT_CENTRALIZATION = "gradient_centralization"
    LOOKAHEAD = "lookahead"

class AdaptationStrategy(Enum):
    """自适应策略枚举"""
    LOSS_BASED = "loss_based"  # 基于损失的自适应
    GRADIENT_BASED = "gradient_based"  # 基于梯度的自适应
    MOMENTUM_BASED = "momentum_based"  # 基于动量的自适应
    CURVATURE_BASED = "curvature_based"  # 基于曲率的自适应
    PHYSICS_BASED = "physics_based"  # 基于物理的自适应
    HYBRID = "hybrid"  # 混合策略

@dataclass
class AdaptiveOptimizerConfig:
    """自适应优化器配置"""
    optimizer_type: OptimizerType = OptimizerType.ADAPTIVE_ADAM
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.LOSS_BASED
    base_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-1
    adaptation_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    
    # 自适应参数
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 5
    factor: float = 0.5
    
    # 物理信息特定参数
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 0.5
    
    # 高级参数
    gradient_clipping: float = 1.0
    use_amsgrad: bool = False
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

class AdaptiveOptimizerBase(ABC):
    """
    自适应优化器基类
    
    定义自适应优化器的通用接口
    """
    
    def __init__(self, params, config: AdaptiveOptimizerConfig):
        """
        初始化自适应优化器
        
        Args:
            params: 模型参数
            config: 配置
        """
        self.param_groups = list(params) if not isinstance(params, list) else params
        self.config = config
        self.state = defaultdict(dict)
        self.step_count = 0
        self.loss_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        self.lr_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 自适应状态
        self.current_lr = config.base_lr
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.best_loss = float('inf')
        
        self._initialize_state()
    
    @abstractmethod
    def _initialize_state(self) -> None:
        """初始化优化器状态"""
        pass
    
    @abstractmethod
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行优化步骤"""
        pass
    
    @abstractmethod
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """自适应调整学习率"""
        pass
    
    def zero_grad(self) -> None:
        """清零梯度"""
        for group in self.param_groups:
            for p in group['params'] if isinstance(group, dict) else group:
                if p.grad is not None:
                    p.grad.zero_()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'state': dict(self.state),
            'step_count': self.step_count,
            'current_lr': self.current_lr,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'cooldown_counter': self.cooldown_counter,
            'loss_history': list(self.loss_history),
            'lr_history': self.lr_history
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.state = defaultdict(dict, state_dict['state'])
        self.step_count = state_dict['step_count']
        self.current_lr = state_dict['current_lr']
        self.best_loss = state_dict['best_loss']
        self.patience_counter = state_dict['patience_counter']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.loss_history = deque(state_dict['loss_history'], maxlen=100)
        self.lr_history = state_dict['lr_history']

class AdaptiveAdam(AdaptiveOptimizerBase):
    """
    自适应Adam优化器
    
    基于损失和梯度信息自适应调整学习率的Adam优化器
    """
    
    def _initialize_state(self) -> None:
        """初始化Adam状态"""
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.requires_grad:
                    param_state = self.state[id(p)]
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    if self.config.use_amsgrad:
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p.data)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行Adam优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 收集梯度信息
        gradients = {}
        total_grad_norm = 0.0
        
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is not None:
                    gradients[id(p)] = p.grad.data.clone()
                    total_grad_norm += p.grad.data.norm().item() ** 2
        
        total_grad_norm = math.sqrt(total_grad_norm)
        self.gradient_history.append(total_grad_norm)
        
        # 自适应学习率
        if loss is not None:
            self.loss_history.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            self._adapt_learning_rate(loss, gradients)
        
        # Adam更新
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is None:
                    continue
                
                param_state = self.state[id(p)]
                
                grad = p.grad.data
                if self.config.gradient_clipping > 0:
                    grad = torch.clamp(grad, -self.config.gradient_clipping, self.config.gradient_clipping)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                beta1, beta2 = self.config.beta1, self.config.beta2
                
                param_state['step'] += 1
                
                # 指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if self.config.use_amsgrad:
                    max_exp_avg_sq = param_state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(self.config.eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(self.config.eps)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                step_size = self.current_lr * math.sqrt(bias_correction2) / bias_correction1
                
                # 参数更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # 权重衰减
                if self.config.weight_decay > 0:
                    p.data.add_(p.data, alpha=-self.config.weight_decay * self.current_lr)
        
        self.step_count += 1
        self.lr_history.append(self.current_lr)
        
        return loss
    
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """自适应调整学习率"""
        current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # 冷却期检查
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # 基于损失的自适应
        if self.config.adaptation_strategy in [AdaptationStrategy.LOSS_BASED, AdaptationStrategy.HYBRID]:
            if current_loss < self.best_loss - self.config.threshold:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.config.factor, self.config.min_lr)
                self.patience_counter = 0
                self.cooldown_counter = self.config.cooldown
                
                self.logger.info(f"学习率调整: {old_lr:.6f} -> {self.current_lr:.6f}")
        
        # 基于梯度的自适应
        if self.config.adaptation_strategy in [AdaptationStrategy.GRADIENT_BASED, AdaptationStrategy.HYBRID]:
            if len(self.gradient_history) >= 5:
                recent_grads = list(self.gradient_history)[-5:]
                grad_variance = np.var(recent_grads)
                grad_mean = np.mean(recent_grads)
                
                # 如果梯度变化很小，增加学习率
                if grad_variance < 1e-6 and grad_mean < 1e-3:
                    self.current_lr = min(self.current_lr * 1.1, self.config.max_lr)
                # 如果梯度变化很大，减少学习率
                elif grad_variance > 1e-2:
                    self.current_lr = max(self.current_lr * 0.9, self.config.min_lr)

class AdaptiveSGD(AdaptiveOptimizerBase):
    """
    自适应SGD优化器
    
    带动量和自适应学习率的SGD优化器
    """
    
    def _initialize_state(self) -> None:
        """初始化SGD状态"""
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.requires_grad:
                    param_state = self.state[id(p)]
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行SGD优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 收集梯度信息
        gradients = {}
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is not None:
                    gradients[id(p)] = p.grad.data.clone()
        
        # 自适应学习率
        if loss is not None:
            self.loss_history.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            self._adapt_learning_rate(loss, gradients)
        
        # SGD更新
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is None:
                    continue
                
                param_state = self.state[id(p)]
                
                grad = p.grad.data
                if self.config.gradient_clipping > 0:
                    grad = torch.clamp(grad, -self.config.gradient_clipping, self.config.gradient_clipping)
                
                # 权重衰减
                if self.config.weight_decay > 0:
                    grad = grad.add(p.data, alpha=self.config.weight_decay)
                
                # 动量
                buf = param_state['momentum_buffer']
                buf.mul_(self.config.momentum).add_(grad)
                
                # 参数更新
                p.data.add_(buf, alpha=-self.current_lr)
        
        self.step_count += 1
        self.lr_history.append(self.current_lr)
        
        return loss
    
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """自适应调整学习率"""
        current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # 简单的基于损失改进的自适应策略
        if len(self.loss_history) >= 2:
            loss_improvement = self.loss_history[-2] - current_loss
            
            if loss_improvement > 0:
                # 损失在改善，可以稍微增加学习率
                self.current_lr = min(self.current_lr * 1.01, self.config.max_lr)
            elif loss_improvement < -self.config.threshold:
                # 损失在恶化，减少学习率
                self.current_lr = max(self.current_lr * 0.95, self.config.min_lr)

class PhysicsInformedOptimizer(AdaptiveOptimizerBase):
    """
    物理信息神经网络专用优化器
    
    针对PINNs的多目标优化需求设计的优化器
    """
    
    def __init__(self, params, config: AdaptiveOptimizerConfig):
        super().__init__(params, config)
        self.loss_components = {'physics': [], 'data': [], 'boundary': []}
        self.adaptive_weights = {
            'physics': config.physics_weight,
            'data': config.data_weight,
            'boundary': config.boundary_weight
        }
    
    def _initialize_state(self) -> None:
        """初始化物理信息优化器状态"""
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.requires_grad:
                    param_state = self.state[id(p)]
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    param_state['physics_grad'] = torch.zeros_like(p.data)
                    param_state['data_grad'] = torch.zeros_like(p.data)
    
    def step(self, closure: Optional[Callable] = None, loss_components: Dict[str, float] = None) -> Optional[float]:
        """执行物理信息优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 记录损失组件
        if loss_components:
            for component, value in loss_components.items():
                if component in self.loss_components:
                    self.loss_components[component].append(value)
        
        # 自适应权重调整
        self._adapt_physics_weights()
        
        # 收集梯度信息
        gradients = {}
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is not None:
                    gradients[id(p)] = p.grad.data.clone()
        
        # 自适应学习率
        if loss is not None:
            self.loss_history.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            self._adapt_learning_rate(loss, gradients)
        
        # Adam-like更新
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is None:
                    continue
                
                param_state = self.state[id(p)]
                
                grad = p.grad.data
                if self.config.gradient_clipping > 0:
                    grad = torch.clamp(grad, -self.config.gradient_clipping, self.config.gradient_clipping)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                beta1, beta2 = self.config.beta1, self.config.beta2
                
                param_state['step'] += 1
                
                # 指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(self.config.eps)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                step_size = self.current_lr * math.sqrt(bias_correction2) / bias_correction1
                
                # 参数更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # 权重衰减
                if self.config.weight_decay > 0:
                    p.data.add_(p.data, alpha=-self.config.weight_decay * self.current_lr)
        
        self.step_count += 1
        self.lr_history.append(self.current_lr)
        
        return loss
    
    def _adapt_physics_weights(self) -> None:
        """自适应调整物理权重"""
        if len(self.loss_components['physics']) < 10:
            return
        
        # 计算各组件的相对变化
        for component in ['physics', 'data', 'boundary']:
            if len(self.loss_components[component]) >= 10:
                recent_losses = self.loss_components[component][-10:]
                loss_trend = (recent_losses[-1] - recent_losses[0]) / (recent_losses[0] + 1e-10)
                
                # 如果某个组件损失下降缓慢，增加其权重
                if loss_trend > -0.01:  # 改善不足1%
                    self.adaptive_weights[component] *= 1.1
                elif loss_trend < -0.1:  # 改善超过10%
                    self.adaptive_weights[component] *= 0.95
                
                # 限制权重范围
                self.adaptive_weights[component] = max(0.1, min(10.0, self.adaptive_weights[component]))
    
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """自适应调整学习率"""
        current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # 基于物理损失和数据损失的平衡调整学习率
        if (len(self.loss_components['physics']) >= 5 and 
            len(self.loss_components['data']) >= 5):
            
            physics_trend = np.mean(self.loss_components['physics'][-5:])
            data_trend = np.mean(self.loss_components['data'][-5:])
            
            # 如果物理损失和数据损失不平衡，调整学习率
            ratio = physics_trend / (data_trend + 1e-10)
            
            if ratio > 10:  # 物理损失过大
                self.current_lr = max(self.current_lr * 0.9, self.config.min_lr)
            elif ratio < 0.1:  # 数据损失过大
                self.current_lr = max(self.current_lr * 0.9, self.config.min_lr)
            else:
                # 平衡状态，可以稍微增加学习率
                self.current_lr = min(self.current_lr * 1.01, self.config.max_lr)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """获取当前自适应权重"""
        return self.adaptive_weights.copy()

class LookaheadOptimizer(AdaptiveOptimizerBase):
    """
    Lookahead优化器
    
    结合快速权重和慢速权重的优化策略
    """
    
    def __init__(self, params, config: AdaptiveOptimizerConfig, base_optimizer_class=AdaptiveAdam):
        super().__init__(params, config)
        self.base_optimizer = base_optimizer_class(params, config)
        self.k = config.lookahead_k
        self.alpha = config.lookahead_alpha
        self.step_count = 0
    
    def _initialize_state(self) -> None:
        """初始化Lookahead状态"""
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.requires_grad:
                    param_state = self.state[id(p)]
                    param_state['slow_weights'] = p.data.clone()
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行Lookahead优化步骤"""
        # 执行基础优化器步骤
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        
        # 每k步更新慢速权重
        if self.step_count % self.k == 0:
            for group in self.param_groups:
                params = group['params'] if isinstance(group, dict) else group
                for p in params:
                    if p.requires_grad:
                        param_state = self.state[id(p)]
                        slow_weights = param_state['slow_weights']
                        
                        # 更新慢速权重
                        slow_weights.add_(p.data - slow_weights, alpha=self.alpha)
                        
                        # 将快速权重设置为慢速权重
                        p.data.copy_(slow_weights)
        
        return loss
    
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """委托给基础优化器"""
        self.base_optimizer._adapt_learning_rate(loss, gradients)
        self.current_lr = self.base_optimizer.current_lr
    
    def zero_grad(self) -> None:
        """清零梯度"""
        self.base_optimizer.zero_grad()

class GradientCentralizationOptimizer(AdaptiveOptimizerBase):
    """
    梯度中心化优化器
    
    通过梯度中心化提高训练稳定性
    """
    
    def __init__(self, params, config: AdaptiveOptimizerConfig, base_optimizer_class=AdaptiveAdam):
        super().__init__(params, config)
        self.base_optimizer = base_optimizer_class(params, config)
    
    def _initialize_state(self) -> None:
        """初始化状态"""
        self.base_optimizer._initialize_state()
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行梯度中心化优化步骤"""
        # 梯度中心化
        self._centralize_gradients()
        
        # 执行基础优化器步骤
        loss = self.base_optimizer.step(closure)
        
        self.step_count = self.base_optimizer.step_count
        self.current_lr = self.base_optimizer.current_lr
        
        return loss
    
    def _centralize_gradients(self) -> None:
        """梯度中心化"""
        for group in self.param_groups:
            params = group['params'] if isinstance(group, dict) else group
            for p in params:
                if p.grad is not None and len(p.grad.shape) > 1:
                    # 对于多维参数，减去梯度的均值
                    grad = p.grad.data
                    grad_mean = grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
                    p.grad.data = grad - grad_mean
    
    def _adapt_learning_rate(self, loss: float, gradients: Dict[str, torch.Tensor]) -> None:
        """委托给基础优化器"""
        self.base_optimizer._adapt_learning_rate(loss, gradients)
        self.current_lr = self.base_optimizer.current_lr
    
    def zero_grad(self) -> None:
        """清零梯度"""
        self.base_optimizer.zero_grad()

def create_adaptive_optimizer(
    params,
    optimizer_type: OptimizerType = OptimizerType.ADAPTIVE_ADAM,
    config: AdaptiveOptimizerConfig = None
) -> AdaptiveOptimizerBase:
    """
    创建自适应优化器
    
    Args:
        params: 模型参数
        optimizer_type: 优化器类型
        config: 配置
        
    Returns:
        AdaptiveOptimizerBase: 优化器实例
    """
    if config is None:
        config = AdaptiveOptimizerConfig(optimizer_type=optimizer_type)
    
    if optimizer_type == OptimizerType.ADAPTIVE_ADAM:
        return AdaptiveAdam(params, config)
    elif optimizer_type == OptimizerType.ADAPTIVE_ADAMW:
        config.weight_decay = max(config.weight_decay, 1e-2)  # AdamW需要更大的权重衰减
        return AdaptiveAdam(params, config)
    elif optimizer_type == OptimizerType.ADAPTIVE_SGD:
        return AdaptiveSGD(params, config)
    elif optimizer_type == OptimizerType.PHYSICS_INFORMED:
        return PhysicsInformedOptimizer(params, config)
    elif optimizer_type == OptimizerType.LOOKAHEAD:
        return LookaheadOptimizer(params, config)
    elif optimizer_type == OptimizerType.GRADIENT_CENTRALIZATION:
        return GradientCentralizationOptimizer(params, config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

class OptimizerScheduler:
    """
    优化器调度器
    
    管理多个优化器的协调使用
    """
    
    def __init__(self, optimizers: Dict[str, AdaptiveOptimizerBase]):
        """
        初始化优化器调度器
        
        Args:
            optimizers: 优化器字典
        """
        self.optimizers = optimizers
        self.current_optimizer = list(optimizers.keys())[0]
        self.switch_history = []
        self.performance_history = defaultdict(list)
    
    def step(self, optimizer_name: str = None, closure: Optional[Callable] = None, **kwargs) -> Optional[float]:
        """
        执行优化步骤
        
        Args:
            optimizer_name: 优化器名称
            closure: 闭包函数
            **kwargs: 其他参数
            
        Returns:
            Optional[float]: 损失值
        """
        if optimizer_name is None:
            optimizer_name = self.current_optimizer
        
        optimizer = self.optimizers[optimizer_name]
        loss = optimizer.step(closure, **kwargs)
        
        # 记录性能
        if loss is not None:
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            self.performance_history[optimizer_name].append(loss_value)
        
        return loss
    
    def switch_optimizer(self, new_optimizer: str, reason: str = "manual") -> None:
        """
        切换优化器
        
        Args:
            new_optimizer: 新优化器名称
            reason: 切换原因
        """
        if new_optimizer in self.optimizers:
            old_optimizer = self.current_optimizer
            self.current_optimizer = new_optimizer
            
            self.switch_history.append({
                'from': old_optimizer,
                'to': new_optimizer,
                'reason': reason,
                'step': sum(opt.step_count for opt in self.optimizers.values())
            })
    
    def auto_switch(self, patience: int = 50) -> bool:
        """
        自动切换优化器
        
        Args:
            patience: 耐心值
            
        Returns:
            bool: 是否进行了切换
        """
        current_opt = self.optimizers[self.current_optimizer]
        
        # 如果当前优化器性能停滞，尝试切换
        if len(current_opt.loss_history) >= patience:
            recent_losses = list(current_opt.loss_history)[-patience:]
            improvement = (recent_losses[0] - recent_losses[-1]) / (recent_losses[0] + 1e-10)
            
            if improvement < 0.01:  # 改善不足1%
                # 寻找表现更好的优化器
                best_optimizer = None
                best_recent_loss = float('inf')
                
                for name, opt in self.optimizers.items():
                    if name != self.current_optimizer and len(opt.loss_history) > 0:
                        recent_loss = list(opt.loss_history)[-10:] if len(opt.loss_history) >= 10 else list(opt.loss_history)
                        avg_recent_loss = np.mean(recent_loss)
                        
                        if avg_recent_loss < best_recent_loss:
                            best_recent_loss = avg_recent_loss
                            best_optimizer = name
                
                if best_optimizer is not None:
                    self.switch_optimizer(best_optimizer, "auto_performance")
                    return True
        
        return False
    
    def zero_grad(self) -> None:
        """清零所有优化器的梯度"""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def get_current_optimizer(self) -> AdaptiveOptimizerBase:
        """获取当前优化器"""
        return self.optimizers[self.current_optimizer]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        summary = {}
        
        for name, optimizer in self.optimizers.items():
            if len(optimizer.loss_history) > 0:
                losses = list(optimizer.loss_history)
                summary[name] = {
                    'total_steps': optimizer.step_count,
                    'current_lr': optimizer.current_lr,
                    'best_loss': min(losses),
                    'final_loss': losses[-1],
                    'improvement': (losses[0] - losses[-1]) / (losses[0] + 1e-10) if len(losses) > 1 else 0.0
                }
        
        summary['switch_history'] = self.switch_history
        summary['current_optimizer'] = self.current_optimizer
        
        return summary

if __name__ == "__main__":
    # 测试自适应优化器
    
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
    
    # 测试不同的优化器
    optimizers_to_test = [
        OptimizerType.ADAPTIVE_ADAM,
        OptimizerType.ADAPTIVE_SGD,
        OptimizerType.PHYSICS_INFORMED,
        OptimizerType.LOOKAHEAD
    ]
    
    print("=== 自适应优化器测试 ===")
    
    for opt_type in optimizers_to_test:
        print(f"\n测试 {opt_type.value}:")
        
        # 创建配置
        config = AdaptiveOptimizerConfig(
            optimizer_type=opt_type,
            base_lr=1e-3,
            adaptation_strategy=AdaptationStrategy.LOSS_BASED
        )
        
        # 创建优化器
        optimizer = create_adaptive_optimizer(model.parameters(), opt_type, config)
        
        # 简单训练循环
        for epoch in range(20):
            # 生成随机数据
            inputs = torch.randn(32, 3)
            targets = torch.sin(inputs.sum(dim=1, keepdim=True))
            
            # 前向传播
            predictions = model(inputs)
            loss = nn.MSELoss()(predictions, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 优化步骤
            if opt_type == OptimizerType.PHYSICS_INFORMED:
                # 模拟物理损失组件
                loss_components = {
                    'physics': loss.item() * 0.3,
                    'data': loss.item() * 0.7,
                    'boundary': loss.item() * 0.1
                }
                optimizer.step(loss_components=loss_components)
            else:
                optimizer.step()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.get_lr():.6f}")
        
        print(f"  最终损失: {loss.item():.6f}")
        print(f"  最终学习率: {optimizer.get_lr():.6f}")
    
    # 测试优化器调度器
    print(f"\n=== 优化器调度器测试 ===")
    
    # 创建多个优化器
    optimizers = {
        'adam': create_adaptive_optimizer(model.parameters(), OptimizerType.ADAPTIVE_ADAM),
        'sgd': create_adaptive_optimizer(model.parameters(), OptimizerType.ADAPTIVE_SGD),
        'physics': create_adaptive_optimizer(model.parameters(), OptimizerType.PHYSICS_INFORMED)
    }
    
    scheduler = OptimizerScheduler(optimizers)
    
    # 训练并自动切换
    for epoch in range(30):
        inputs = torch.randn(32, 3)
        targets = torch.sin(inputs.sum(dim=1, keepdim=True))
        
        def closure():
            predictions = model(inputs)
            return nn.MSELoss()(predictions, targets)
        
        loss = scheduler.step(closure=closure)
        
        # 每10步尝试自动切换
        if epoch % 10 == 0 and epoch > 0:
            switched = scheduler.auto_switch(patience=10)
            if switched:
                print(f"  Epoch {epoch}: 自动切换到 {scheduler.current_optimizer}")
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, 当前优化器 = {scheduler.current_optimizer}")
    
    # 性能总结
    summary = scheduler.get_performance_summary()
    print(f"\n=== 性能总结 ===")
    for name, stats in summary.items():
        if isinstance(stats, dict) and 'total_steps' in stats:
            print(f"  {name}: 步数={stats['total_steps']}, 最佳损失={stats['best_loss']:.6f}, 改善={stats['improvement']:.2%}")
    
    print("\n自适应优化器测试完成！")