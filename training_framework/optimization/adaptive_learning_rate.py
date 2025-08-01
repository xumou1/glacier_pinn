#!/usr/bin/env python3
"""
自适应学习率模块

实现多种自适应学习率调度策略，用于PINNs训练的不同阶段。
包括基于损失的调度、基于梯度的调度和多阶段调度。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from abc import ABC, abstractmethod

class AdaptiveLearningRateScheduler(ABC):
    """
    自适应学习率调度器基类
    
    定义自适应学习率调度的通用接口，包括：
    - 基于训练进度的调度
    - 基于损失变化的调度
    - 基于梯度信息的调度
    - 多阶段协调调度
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        """
        初始化学习率调度器
        
        Args:
            initial_learning_rate: 初始学习率
            scheduler_config: 调度器配置
        """
        self.initial_learning_rate = initial_learning_rate
        self.scheduler_config = scheduler_config
        self.current_learning_rate = initial_learning_rate
        self.step_count = 0
        self.loss_history = []
        self.gradient_history = []
        
    @abstractmethod
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        更新学习率
        
        Args:
            loss: 当前损失值
            gradients: 当前梯度
            step: 当前步数
            
        Returns:
            更新后的学习率
        """
        pass
        
    def get_current_learning_rate(self) -> float:
        """
        获取当前学习率
        
        Returns:
            当前学习率
        """
        return self.current_learning_rate
        
    def reset(self):
        """
        重置调度器状态
        """
        self.current_learning_rate = self.initial_learning_rate
        self.step_count = 0
        self.loss_history = []
        self.gradient_history = []

class LossBasedScheduler(AdaptiveLearningRateScheduler):
    """
    基于损失的自适应学习率调度器
    
    根据损失的变化趋势调整学习率：
    - 损失下降缓慢时降低学习率
    - 损失震荡时降低学习率
    - 损失快速下降时保持或增加学习率
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        super().__init__(initial_learning_rate, scheduler_config)
        
        # 损失监控参数
        self.patience = scheduler_config.get('patience', 50)
        self.min_delta = scheduler_config.get('min_delta', 1e-6)
        self.factor = scheduler_config.get('factor', 0.5)
        self.min_lr = scheduler_config.get('min_lr', 1e-8)
        self.max_lr = scheduler_config.get('max_lr', 1e-2)
        
        # 内部状态
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.cooldown = scheduler_config.get('cooldown', 10)
        
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        基于损失更新学习率
        """
        self.step_count = step
        self.loss_history.append(loss)
        
        # 冷却期间不调整
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_learning_rate
            
        # 检查是否有改善
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
            
            # 如果损失快速下降，可以尝试增加学习率
            if self._is_fast_improvement():
                self.current_learning_rate = min(
                    self.current_learning_rate * 1.1,
                    self.max_lr
                )
        else:
            self.wait += 1
            
        # 如果等待时间超过耐心值，降低学习率
        if self.wait >= self.patience:
            old_lr = self.current_learning_rate
            self.current_learning_rate = max(
                self.current_learning_rate * self.factor,
                self.min_lr
            )
            
            if self.current_learning_rate < old_lr:
                self.wait = 0
                self.cooldown_counter = self.cooldown
                print(f"Reducing learning rate to {self.current_learning_rate:.2e}")
                
        # 检查损失震荡
        if self._is_oscillating():
            self.current_learning_rate = max(
                self.current_learning_rate * 0.8,
                self.min_lr
            )
            print(f"Detected oscillation, reducing learning rate to {self.current_learning_rate:.2e}")
            
        return self.current_learning_rate
        
    def _is_fast_improvement(self) -> bool:
        """
        检查是否快速改善
        """
        if len(self.loss_history) < 10:
            return False
            
        recent_losses = self.loss_history[-10:]
        improvement_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        return improvement_rate > 0.1  # 10%的改善
        
    def _is_oscillating(self) -> bool:
        """
        检查损失是否震荡
        """
        if len(self.loss_history) < 20:
            return False
            
        recent_losses = jnp.array(self.loss_history[-20:])
        
        # 计算损失的变化方向
        diffs = jnp.diff(recent_losses)
        sign_changes = jnp.sum(jnp.diff(jnp.sign(diffs)) != 0)
        
        # 如果符号变化太频繁，认为是震荡
        return sign_changes > len(diffs) * 0.6

class GradientBasedScheduler(AdaptiveLearningRateScheduler):
    """
    基于梯度的自适应学习率调度器
    
    根据梯度的特性调整学习率：
    - 梯度范数大时降低学习率
    - 梯度范数小时增加学习率
    - 梯度方向变化大时降低学习率
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        super().__init__(initial_learning_rate, scheduler_config)
        
        # 梯度监控参数
        self.grad_norm_threshold_high = scheduler_config.get('grad_norm_threshold_high', 10.0)
        self.grad_norm_threshold_low = scheduler_config.get('grad_norm_threshold_low', 0.01)
        self.direction_change_threshold = scheduler_config.get('direction_change_threshold', 0.5)
        
        # 调整因子
        self.increase_factor = scheduler_config.get('increase_factor', 1.05)
        self.decrease_factor = scheduler_config.get('decrease_factor', 0.9)
        
        # 限制
        self.min_lr = scheduler_config.get('min_lr', 1e-8)
        self.max_lr = scheduler_config.get('max_lr', 1e-2)
        
        # 内部状态
        self.prev_grad_direction = None
        
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        基于梯度更新学习率
        """
        self.step_count = step
        self.loss_history.append(loss)
        
        # 计算梯度范数
        grad_norm = self._compute_gradient_norm(gradients)
        self.gradient_history.append(grad_norm)
        
        # 计算梯度方向变化
        direction_change = self._compute_direction_change(gradients)
        
        # 基于梯度范数调整
        if grad_norm > self.grad_norm_threshold_high:
            # 梯度太大，降低学习率
            self.current_learning_rate = max(
                self.current_learning_rate * self.decrease_factor,
                self.min_lr
            )
        elif grad_norm < self.grad_norm_threshold_low:
            # 梯度太小，增加学习率
            self.current_learning_rate = min(
                self.current_learning_rate * self.increase_factor,
                self.max_lr
            )
            
        # 基于方向变化调整
        if direction_change > self.direction_change_threshold:
            # 方向变化大，降低学习率
            self.current_learning_rate = max(
                self.current_learning_rate * 0.95,
                self.min_lr
            )
            
        return self.current_learning_rate
        
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
        
    def _compute_direction_change(self, gradients: Dict) -> float:
        """
        计算梯度方向变化
        """
        # 将梯度展平为向量
        current_direction = self._flatten_gradients(gradients)
        
        if self.prev_grad_direction is None:
            self.prev_grad_direction = current_direction
            return 0.0
            
        # 计算余弦相似度
        dot_product = jnp.dot(current_direction, self.prev_grad_direction)
        norm_current = jnp.linalg.norm(current_direction)
        norm_prev = jnp.linalg.norm(self.prev_grad_direction)
        
        if norm_current == 0 or norm_prev == 0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = dot_product / (norm_current * norm_prev)
            
        # 方向变化 = 1 - 余弦相似度
        direction_change = 1.0 - cosine_similarity
        
        self.prev_grad_direction = current_direction
        
        return direction_change
        
    def _flatten_gradients(self, gradients: Dict) -> jnp.ndarray:
        """
        将梯度字典展平为向量
        """
        flat_grads = []
        
        def collect_grads(x):
            flat_grads.append(x.flatten())
            
        jax.tree_map(collect_grads, gradients)
        
        return jnp.concatenate(flat_grads)

class MultiStageScheduler(AdaptiveLearningRateScheduler):
    """
    多阶段自适应学习率调度器
    
    为PINNs训练的不同阶段使用不同的调度策略：
    - 预训练阶段：较高学习率，快速收敛
    - 物理集成阶段：中等学习率，平衡物理和数据
    - 耦合优化阶段：较低学习率，精细调优
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        super().__init__(initial_learning_rate, scheduler_config)
        
        # 阶段配置
        self.stages = scheduler_config.get('stages', {
            'pretraining': {'duration': 1000, 'lr_factor': 1.0},
            'physics_integration': {'duration': 2000, 'lr_factor': 0.5},
            'coupled_optimization': {'duration': 2000, 'lr_factor': 0.1}
        })
        
        # 当前阶段
        self.current_stage = 'pretraining'
        self.stage_start_step = 0
        
        # 每个阶段的调度器
        self.stage_schedulers = self._create_stage_schedulers()
        
    def _create_stage_schedulers(self) -> Dict[str, AdaptiveLearningRateScheduler]:
        """
        为每个阶段创建调度器
        """
        schedulers = {}
        
        for stage_name, stage_config in self.stages.items():
            stage_lr = self.initial_learning_rate * stage_config['lr_factor']
            
            if stage_name == 'pretraining':
                # 预训练阶段使用基于损失的调度
                schedulers[stage_name] = LossBasedScheduler(
                    stage_lr,
                    {
                        'patience': 30,
                        'factor': 0.8,
                        'min_lr': stage_lr * 0.01,
                        'max_lr': stage_lr * 2.0
                    }
                )
            elif stage_name == 'physics_integration':
                # 物理集成阶段使用梯度基调度
                schedulers[stage_name] = GradientBasedScheduler(
                    stage_lr,
                    {
                        'grad_norm_threshold_high': 5.0,
                        'grad_norm_threshold_low': 0.1,
                        'min_lr': stage_lr * 0.1,
                        'max_lr': stage_lr * 1.5
                    }
                )
            else:
                # 耦合优化阶段使用保守的基于损失调度
                schedulers[stage_name] = LossBasedScheduler(
                    stage_lr,
                    {
                        'patience': 100,
                        'factor': 0.9,
                        'min_lr': stage_lr * 0.1,
                        'max_lr': stage_lr * 1.2
                    }
                )
                
        return schedulers
        
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        多阶段学习率更新
        """
        self.step_count = step
        
        # 检查是否需要切换阶段
        self._update_stage(step)
        
        # 使用当前阶段的调度器
        current_scheduler = self.stage_schedulers[self.current_stage]
        self.current_learning_rate = current_scheduler.update_learning_rate(
            loss, gradients, step - self.stage_start_step
        )
        
        return self.current_learning_rate
        
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

class CyclicalScheduler(AdaptiveLearningRateScheduler):
    """
    循环学习率调度器
    
    实现循环学习率策略，在最小和最大学习率之间循环，
    有助于跳出局部最优解。
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        super().__init__(initial_learning_rate, scheduler_config)
        
        # 循环参数
        self.base_lr = scheduler_config.get('base_lr', initial_learning_rate * 0.1)
        self.max_lr = scheduler_config.get('max_lr', initial_learning_rate)
        self.step_size = scheduler_config.get('step_size', 1000)
        self.mode = scheduler_config.get('mode', 'triangular')  # 'triangular', 'triangular2', 'exp_range'
        self.gamma = scheduler_config.get('gamma', 0.99994)  # for exp_range mode
        
        # 内部状态
        self.cycle = 0
        self.x = 0
        
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        循环学习率更新
        """
        self.step_count = step
        self.loss_history.append(loss)
        
        # 计算循环位置
        cycle = jnp.floor(1 + step / (2 * self.step_size))
        x = jnp.abs(step / self.step_size - 2 * cycle + 1)
        
        # 根据模式计算学习率
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * jnp.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * jnp.maximum(0, (1 - x)) / (2**(cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * jnp.maximum(0, (1 - x)) * (self.gamma**step)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * jnp.maximum(0, (1 - x))
            
        self.current_learning_rate = float(lr)
        
        return self.current_learning_rate

class WarmupScheduler(AdaptiveLearningRateScheduler):
    """
    预热学习率调度器
    
    在训练初期使用较小的学习率，然后逐渐增加到目标学习率，
    之后使用其他调度策略。
    """
    
    def __init__(self, 
                 initial_learning_rate: float,
                 scheduler_config: Dict[str, Any]):
        super().__init__(initial_learning_rate, scheduler_config)
        
        # 预热参数
        self.warmup_steps = scheduler_config.get('warmup_steps', 500)
        self.warmup_start_lr = scheduler_config.get('warmup_start_lr', initial_learning_rate * 0.01)
        self.target_lr = initial_learning_rate
        
        # 预热后的调度器
        post_warmup_config = scheduler_config.get('post_warmup_scheduler', {
            'type': 'loss_based',
            'patience': 50,
            'factor': 0.5
        })
        
        self.post_warmup_scheduler = self._create_post_warmup_scheduler(
            post_warmup_config
        )
        
        # 状态
        self.warmup_completed = False
        
    def _create_post_warmup_scheduler(self, config: Dict[str, Any]) -> AdaptiveLearningRateScheduler:
        """
        创建预热后的调度器
        """
        scheduler_type = config.get('type', 'loss_based')
        
        if scheduler_type == 'loss_based':
            return LossBasedScheduler(self.target_lr, config)
        elif scheduler_type == 'gradient_based':
            return GradientBasedScheduler(self.target_lr, config)
        elif scheduler_type == 'cyclical':
            return CyclicalScheduler(self.target_lr, config)
        else:
            return LossBasedScheduler(self.target_lr, config)
            
    def update_learning_rate(self, 
                           loss: float,
                           gradients: Dict,
                           step: int) -> float:
        """
        预热学习率更新
        """
        self.step_count = step
        self.loss_history.append(loss)
        
        if step < self.warmup_steps:
            # 预热阶段：线性增加学习率
            progress = step / self.warmup_steps
            self.current_learning_rate = (
                self.warmup_start_lr + 
                (self.target_lr - self.warmup_start_lr) * progress
            )
        else:
            # 预热完成，使用后续调度器
            if not self.warmup_completed:
                print(f"Warmup completed at step {step}, switching to post-warmup scheduler")
                self.warmup_completed = True
                
            self.current_learning_rate = self.post_warmup_scheduler.update_learning_rate(
                loss, gradients, step - self.warmup_steps
            )
            
        return self.current_learning_rate

def create_adaptive_scheduler(scheduler_type: str,
                            initial_learning_rate: float,
                            scheduler_config: Dict[str, Any]) -> AdaptiveLearningRateScheduler:
    """
    工厂函数：创建自适应学习率调度器
    
    Args:
        scheduler_type: 调度器类型
        initial_learning_rate: 初始学习率
        scheduler_config: 调度器配置
        
    Returns:
        学习率调度器实例
    """
    if scheduler_type == 'loss_based':
        return LossBasedScheduler(initial_learning_rate, scheduler_config)
    elif scheduler_type == 'gradient_based':
        return GradientBasedScheduler(initial_learning_rate, scheduler_config)
    elif scheduler_type == 'multi_stage':
        return MultiStageScheduler(initial_learning_rate, scheduler_config)
    elif scheduler_type == 'cyclical':
        return CyclicalScheduler(initial_learning_rate, scheduler_config)
    elif scheduler_type == 'warmup':
        return WarmupScheduler(initial_learning_rate, scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def create_optax_schedule_from_adaptive(scheduler: AdaptiveLearningRateScheduler) -> optax.Schedule:
    """
    将自适应调度器转换为Optax调度
    
    Args:
        scheduler: 自适应调度器
        
    Returns:
        Optax调度函数
    """
    def schedule_fn(step):
        # 这是一个简化版本，实际使用时需要传递损失和梯度信息
        return scheduler.get_current_learning_rate()
        
    return schedule_fn

if __name__ == "__main__":
    # 测试代码
    print("Adaptive learning rate scheduler module loaded successfully")
    
    # 测试不同类型的调度器
    initial_lr = 1e-3
    
    # 基于损失的调度器
    loss_scheduler = create_adaptive_scheduler(
        'loss_based', 
        initial_lr, 
        {
            'patience': 50,
            'factor': 0.5,
            'min_lr': 1e-8,
            'max_lr': 1e-2
        }
    )
    
    # 基于梯度的调度器
    grad_scheduler = create_adaptive_scheduler(
        'gradient_based',
        initial_lr,
        {
            'grad_norm_threshold_high': 10.0,
            'grad_norm_threshold_low': 0.01,
            'min_lr': 1e-8,
            'max_lr': 1e-2
        }
    )
    
    # 多阶段调度器
    multi_stage_scheduler = create_adaptive_scheduler(
        'multi_stage',
        initial_lr,
        {
            'stages': {
                'pretraining': {'duration': 1000, 'lr_factor': 1.0},
                'physics_integration': {'duration': 2000, 'lr_factor': 0.5},
                'coupled_optimization': {'duration': 2000, 'lr_factor': 0.1}
            }
        }
    )
    
    print(f"Created schedulers:")
    print(f"Loss-based scheduler: {type(loss_scheduler).__name__}")
    print(f"Gradient-based scheduler: {type(grad_scheduler).__name__}")
    print(f"Multi-stage scheduler: {type(multi_stage_scheduler).__name__}")
    
    # 模拟训练过程
    print("\nSimulating training process:")
    for step in range(0, 100, 10):
        # 模拟损失和梯度
        loss = 1.0 / (step + 1)  # 递减损失
        fake_gradients = {'params': jnp.ones((10, 10)) * 0.1}
        
        lr = loss_scheduler.update_learning_rate(loss, fake_gradients, step)
        print(f"Step {step}: Loss = {loss:.4f}, LR = {lr:.2e}")