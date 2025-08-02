#!/usr/bin/env python3
"""
学习率调度

实现各种学习率调度策略，包括：
- 基础调度器（阶梯式、指数式、余弦式）
- 自适应调度器（基于性能、梯度、损失）
- 物理信息神经网络专用调度器
- 多阶段调度器

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

class SchedulerType(Enum):
    """调度器类型枚举"""
    STEP_LR = "step_lr"  # 阶梯式
    EXPONENTIAL_LR = "exponential_lr"  # 指数式
    COSINE_LR = "cosine_lr"  # 余弦式
    COSINE_RESTART = "cosine_restart"  # 余弦重启
    PLATEAU = "plateau"  # 平台期
    CYCLIC_LR = "cyclic_lr"  # 循环式
    ONE_CYCLE = "one_cycle"  # 单周期
    POLYNOMIAL = "polynomial"  # 多项式
    ADAPTIVE_LOSS = "adaptive_loss"  # 自适应损失
    ADAPTIVE_GRADIENT = "adaptive_gradient"  # 自适应梯度
    PHYSICS_INFORMED = "physics_informed"  # 物理信息
    MULTI_STAGE = "multi_stage"  # 多阶段
    WARMUP_COSINE = "warmup_cosine"  # 预热余弦

class AdaptationMetric(Enum):
    """自适应指标枚举"""
    LOSS = "loss"  # 损失
    GRADIENT_NORM = "gradient_norm"  # 梯度范数
    LOSS_IMPROVEMENT = "loss_improvement"  # 损失改善
    VALIDATION_LOSS = "validation_loss"  # 验证损失
    PHYSICS_RESIDUAL = "physics_residual"  # 物理残差
    DATA_FITTING = "data_fitting"  # 数据拟合
    BOUNDARY_LOSS = "boundary_loss"  # 边界损失

@dataclass
class SchedulerConfig:
    """调度器配置"""
    scheduler_type: SchedulerType = SchedulerType.COSINE_LR
    base_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-1
    
    # 基础调度器参数
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 1e-6
    
    # 平台期调度器参数
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 5
    factor: float = 0.5
    
    # 循环调度器参数
    base_lr_cycle: float = 1e-4
    max_lr_cycle: float = 1e-2
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"  # triangular, triangular2, exp_range
    
    # 单周期调度器参数
    max_lr_one_cycle: float = 1e-2
    total_steps: int = 1000
    pct_start: float = 0.3
    anneal_strategy: str = "cos"  # cos, linear
    
    # 多项式调度器参数
    power: float = 1.0
    
    # 自适应参数
    adaptation_metric: AdaptationMetric = AdaptationMetric.LOSS
    adaptation_window: int = 10
    adaptation_threshold: float = 0.01
    adaptation_factor: float = 0.8
    
    # 物理信息参数
    physics_weight_schedule: bool = True
    data_weight_schedule: bool = True
    boundary_weight_schedule: bool = True
    
    # 预热参数
    warmup_steps: int = 100
    warmup_factor: float = 0.1
    
    # 多阶段参数
    stage_milestones: List[int] = None
    stage_factors: List[float] = None
    
    def __post_init__(self):
        if self.step_size_down is None:
            self.step_size_down = self.step_size_up
        
        if self.stage_milestones is None:
            self.stage_milestones = [300, 600, 900]
        
        if self.stage_factors is None:
            self.stage_factors = [1.0, 0.5, 0.1, 0.01]

class LRSchedulerBase(ABC):
    """
    学习率调度器基类
    
    定义学习率调度器的通用接口
    """
    
    def __init__(self, optimizer, config: SchedulerConfig):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            config: 配置
        """
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        self.epoch_count = 0
        self.lr_history = []
        self.metric_history = deque(maxlen=config.adaptation_window * 2)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 记录初始学习率
        self.base_lrs = []
        for group in self.optimizer.param_groups:
            self.base_lrs.append(group['lr'])
        
        self._initialize_scheduler()
    
    @abstractmethod
    def _initialize_scheduler(self) -> None:
        """初始化调度器"""
        pass
    
    @abstractmethod
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        pass
    
    def step(self, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        """
        执行调度步骤
        
        Args:
            metrics: 指标字典
            epoch: 当前epoch
        """
        if epoch is not None:
            self.epoch_count = epoch
        else:
            self.epoch_count += 1
        
        self.step_count += 1
        
        # 记录指标
        if metrics:
            self.metric_history.append(metrics)
        
        # 获取新的学习率
        new_lrs = self.get_lr()
        
        # 更新优化器学习率
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        
        # 记录学习率历史
        self.lr_history.append(new_lrs[0] if new_lrs else 0.0)
    
    def get_last_lr(self) -> List[float]:
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'lr_history': self.lr_history,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.step_count = state_dict['step_count']
        self.epoch_count = state_dict['epoch_count']
        self.lr_history = state_dict['lr_history']
        self.base_lrs = state_dict['base_lrs']

class StepLRScheduler(LRSchedulerBase):
    """阶梯式学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化阶梯式调度器"""
        self.step_size = self.config.step_size
        self.gamma = self.config.gamma
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        decay_factor = self.gamma ** (self.epoch_count // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]

class ExponentialLRScheduler(LRSchedulerBase):
    """指数式学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化指数式调度器"""
        self.gamma = self.config.gamma
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        decay_factor = self.gamma ** self.epoch_count
        return [base_lr * decay_factor for base_lr in self.base_lrs]

class CosineAnnealingLRScheduler(LRSchedulerBase):
    """余弦退火学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化余弦退火调度器"""
        self.T_max = self.config.T_max
        self.eta_min = self.config.eta_min
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.epoch_count == 0:
            return self.base_lrs
        
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.epoch_count / self.T_max)
            ) / 2
            lrs.append(lr)
        
        return lrs

class CosineAnnealingWarmRestartsScheduler(LRSchedulerBase):
    """余弦退火热重启学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化余弦退火热重启调度器"""
        self.T_0 = self.config.T_max
        self.T_mult = 2
        self.eta_min = self.config.eta_min
        self.T_cur = 0
        self.T_i = self.T_0
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.epoch_count == 0:
            return self.base_lrs
        
        # 检查是否需要重启
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)
            ) / 2
            lrs.append(lr)
        
        self.T_cur += 1
        return lrs

class ReduceLROnPlateauScheduler(LRSchedulerBase):
    """平台期学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化平台期调度器"""
        self.patience = self.config.patience
        self.threshold = self.config.threshold
        self.cooldown = self.config.cooldown
        self.factor = self.config.factor
        self.min_lr = self.config.min_lr
        
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        """执行调度步骤"""
        if metrics is None or self.config.adaptation_metric.value not in metrics:
            super().step(metrics, epoch)
            return
        
        current_metric = metrics[self.config.adaptation_metric.value]
        
        # 冷却期检查
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            super().step(metrics, epoch)
            return
        
        # 检查是否有改善
        if current_metric < self.best_metric - self.threshold:
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # 检查是否需要降低学习率
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
        
        super().step(metrics, epoch)
    
    def _reduce_lr(self) -> None:
        """降低学习率"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if old_lr != new_lr:
                self.logger.info(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")

class CyclicLRScheduler(LRSchedulerBase):
    """循环学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化循环调度器"""
        self.base_lr = self.config.base_lr_cycle
        self.max_lr = self.config.max_lr_cycle
        self.step_size_up = self.config.step_size_up
        self.step_size_down = self.config.step_size_down
        self.mode = self.config.mode
        
        self.cycle_size = self.step_size_up + self.step_size_down
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        cycle = math.floor(1 + self.step_count / self.cycle_size)
        x = abs(self.step_count / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = 0.99999 ** self.step_count
        else:
            scale_factor = 1.0
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale_factor
        
        return [lr] * len(self.base_lrs)

class OneCycleLRScheduler(LRSchedulerBase):
    """单周期学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化单周期调度器"""
        self.max_lr = self.config.max_lr_one_cycle
        self.total_steps = self.config.total_steps
        self.pct_start = self.config.pct_start
        self.anneal_strategy = self.config.anneal_strategy
        
        self.step_size_up = int(self.total_steps * self.pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.step_count <= self.step_size_up:
            # 上升阶段
            pct = self.step_count / self.step_size_up
            lr = self.config.base_lr + (self.max_lr - self.config.base_lr) * pct
        else:
            # 下降阶段
            pct = (self.step_count - self.step_size_up) / self.step_size_down
            
            if self.anneal_strategy == "cos":
                lr = self.config.min_lr + (self.max_lr - self.config.min_lr) * (
                    1 + math.cos(math.pi * pct)
                ) / 2
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.config.min_lr) * pct
        
        return [lr] * len(self.base_lrs)

class AdaptiveLossScheduler(LRSchedulerBase):
    """自适应损失学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化自适应损失调度器"""
        self.adaptation_window = self.config.adaptation_window
        self.adaptation_threshold = self.config.adaptation_threshold
        self.adaptation_factor = self.config.adaptation_factor
        self.min_lr = self.config.min_lr
        self.max_lr = self.config.max_lr
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if len(self.metric_history) < self.adaptation_window:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # 计算损失趋势
        recent_metrics = list(self.metric_history)[-self.adaptation_window:]
        losses = [m.get('loss', float('inf')) for m in recent_metrics]
        
        if len(losses) < 2:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # 计算损失改善率
        loss_improvement = (losses[0] - losses[-1]) / (losses[0] + 1e-10)
        
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        new_lrs = []
        
        for current_lr in current_lrs:
            if loss_improvement < self.adaptation_threshold:
                # 改善不足，降低学习率
                new_lr = max(current_lr * self.adaptation_factor, self.min_lr)
            elif loss_improvement > self.adaptation_threshold * 3:
                # 改善很好，可以稍微增加学习率
                new_lr = min(current_lr * (1 / self.adaptation_factor), self.max_lr)
            else:
                # 保持当前学习率
                new_lr = current_lr
            
            new_lrs.append(new_lr)
        
        return new_lrs

class PhysicsInformedScheduler(LRSchedulerBase):
    """物理信息学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化物理信息调度器"""
        self.physics_weight = self.config.physics_weight_schedule
        self.data_weight = self.config.data_weight_schedule
        self.boundary_weight = self.config.boundary_weight_schedule
        
        self.loss_balance_history = deque(maxlen=20)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if len(self.metric_history) < 5:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # 分析物理损失和数据损失的平衡
        recent_metrics = list(self.metric_history)[-5:]
        
        physics_losses = [m.get('physics_loss', 0) for m in recent_metrics]
        data_losses = [m.get('data_loss', 0) for m in recent_metrics]
        
        if not physics_losses or not data_losses:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        avg_physics = np.mean(physics_losses)
        avg_data = np.mean(data_losses)
        
        # 计算损失比例
        if avg_data > 0:
            loss_ratio = avg_physics / avg_data
        else:
            loss_ratio = 1.0
        
        self.loss_balance_history.append(loss_ratio)
        
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        new_lrs = []
        
        for current_lr in current_lrs:
            # 根据损失平衡调整学习率
            if loss_ratio > 10:  # 物理损失过大
                new_lr = current_lr * 0.9
            elif loss_ratio < 0.1:  # 数据损失过大
                new_lr = current_lr * 0.9
            elif 0.5 <= loss_ratio <= 2.0:  # 平衡状态，可以增加学习率
                new_lr = min(current_lr * 1.01, self.config.max_lr)
            else:
                new_lr = current_lr
            
            new_lr = max(new_lr, self.config.min_lr)
            new_lrs.append(new_lr)
        
        return new_lrs

class MultiStageScheduler(LRSchedulerBase):
    """多阶段学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化多阶段调度器"""
        self.milestones = self.config.stage_milestones
        self.factors = self.config.stage_factors
        self.current_stage = 0
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        # 确定当前阶段
        stage = 0
        for i, milestone in enumerate(self.milestones):
            if self.epoch_count >= milestone:
                stage = i + 1
            else:
                break
        
        if stage != self.current_stage:
            self.current_stage = stage
            self.logger.info(f"进入阶段 {stage + 1}")
        
        # 获取当前阶段的因子
        if stage < len(self.factors):
            factor = self.factors[stage]
        else:
            factor = self.factors[-1]
        
        return [base_lr * factor for base_lr in self.base_lrs]

class WarmupCosineScheduler(LRSchedulerBase):
    """预热余弦学习率调度器"""
    
    def _initialize_scheduler(self) -> None:
        """初始化预热余弦调度器"""
        self.warmup_steps = self.config.warmup_steps
        self.warmup_factor = self.config.warmup_factor
        self.T_max = self.config.T_max
        self.eta_min = self.config.eta_min
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.step_count < self.warmup_steps:
            # 预热阶段
            warmup_factor = self.warmup_factor + (1 - self.warmup_factor) * self.step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            adjusted_step = self.step_count - self.warmup_steps
            adjusted_T_max = self.T_max - self.warmup_steps
            
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.eta_min + (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * adjusted_step / adjusted_T_max)
                ) / 2
                lrs.append(lr)
            
            return lrs

def create_lr_scheduler(
    optimizer,
    scheduler_type: SchedulerType = SchedulerType.COSINE_LR,
    config: SchedulerConfig = None
) -> LRSchedulerBase:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        config: 配置
        
    Returns:
        LRSchedulerBase: 调度器实例
    """
    if config is None:
        config = SchedulerConfig(scheduler_type=scheduler_type)
    
    scheduler_map = {
        SchedulerType.STEP_LR: StepLRScheduler,
        SchedulerType.EXPONENTIAL_LR: ExponentialLRScheduler,
        SchedulerType.COSINE_LR: CosineAnnealingLRScheduler,
        SchedulerType.COSINE_RESTART: CosineAnnealingWarmRestartsScheduler,
        SchedulerType.PLATEAU: ReduceLROnPlateauScheduler,
        SchedulerType.CYCLIC_LR: CyclicLRScheduler,
        SchedulerType.ONE_CYCLE: OneCycleLRScheduler,
        SchedulerType.ADAPTIVE_LOSS: AdaptiveLossScheduler,
        SchedulerType.PHYSICS_INFORMED: PhysicsInformedScheduler,
        SchedulerType.MULTI_STAGE: MultiStageScheduler,
        SchedulerType.WARMUP_COSINE: WarmupCosineScheduler
    }
    
    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler_map[scheduler_type](optimizer, config)

class SchedulerManager:
    """
    调度器管理器
    
    管理多个调度器的协调使用
    """
    
    def __init__(self, schedulers: Dict[str, LRSchedulerBase]):
        """
        初始化调度器管理器
        
        Args:
            schedulers: 调度器字典
        """
        self.schedulers = schedulers
        self.current_scheduler = list(schedulers.keys())[0]
        self.switch_history = []
        self.performance_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def step(self, scheduler_name: str = None, metrics: Optional[Dict[str, float]] = None, epoch: Optional[int] = None) -> None:
        """
        执行调度步骤
        
        Args:
            scheduler_name: 调度器名称
            metrics: 指标字典
            epoch: 当前epoch
        """
        if scheduler_name is None:
            scheduler_name = self.current_scheduler
        
        scheduler = self.schedulers[scheduler_name]
        scheduler.step(metrics, epoch)
        
        # 记录性能
        if metrics:
            self.performance_history[scheduler_name].append(metrics)
    
    def switch_scheduler(self, new_scheduler: str, reason: str = "manual") -> None:
        """
        切换调度器
        
        Args:
            new_scheduler: 新调度器名称
            reason: 切换原因
        """
        if new_scheduler in self.schedulers:
            old_scheduler = self.current_scheduler
            self.current_scheduler = new_scheduler
            
            self.switch_history.append({
                'from': old_scheduler,
                'to': new_scheduler,
                'reason': reason,
                'step': sum(s.step_count for s in self.schedulers.values())
            })
            
            self.logger.info(f"调度器切换: {old_scheduler} -> {new_scheduler} (原因: {reason})")
    
    def auto_switch(self, patience: int = 50) -> bool:
        """
        自动切换调度器
        
        Args:
            patience: 耐心值
            
        Returns:
            bool: 是否进行了切换
        """
        current_scheduler = self.schedulers[self.current_scheduler]
        
        # 如果当前调度器性能停滞，尝试切换
        if len(current_scheduler.lr_history) >= patience:
            recent_lrs = current_scheduler.lr_history[-patience:]
            lr_variance = np.var(recent_lrs)
            
            # 如果学习率变化很小且性能没有改善，考虑切换
            if lr_variance < 1e-10:
                # 寻找可能更好的调度器
                for name, scheduler in self.schedulers.items():
                    if name != self.current_scheduler:
                        self.switch_scheduler(name, "auto_stagnation")
                        return True
        
        return False
    
    def get_current_lr(self) -> List[float]:
        """获取当前学习率"""
        return self.schedulers[self.current_scheduler].get_last_lr()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        summary = {}
        
        for name, scheduler in self.schedulers.items():
            summary[name] = {
                'total_steps': scheduler.step_count,
                'total_epochs': scheduler.epoch_count,
                'current_lr': scheduler.get_last_lr(),
                'lr_range': (min(scheduler.lr_history), max(scheduler.lr_history)) if scheduler.lr_history else (0, 0)
            }
        
        summary['switch_history'] = self.switch_history
        summary['current_scheduler'] = self.current_scheduler
        
        return summary
    
    def plot_lr_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制学习率历史
        
        Args:
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 8))
        
        for name, scheduler in self.schedulers.items():
            if scheduler.lr_history:
                plt.plot(scheduler.lr_history, label=f"{name} (steps: {scheduler.step_count})")
        
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # 测试学习率调度器
    
    # 创建简单模型和优化器
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 测试不同的调度器
    schedulers_to_test = [
        SchedulerType.COSINE_LR,
        SchedulerType.STEP_LR,
        SchedulerType.PLATEAU,
        SchedulerType.ONE_CYCLE,
        SchedulerType.ADAPTIVE_LOSS
    ]
    
    print("=== 学习率调度器测试 ===")
    
    for scheduler_type in schedulers_to_test:
        print(f"\n测试 {scheduler_type.value}:")
        
        # 创建配置
        config = SchedulerConfig(
            scheduler_type=scheduler_type,
            base_lr=1e-3,
            T_max=50,
            total_steps=100
        )
        
        # 创建调度器
        scheduler = create_lr_scheduler(optimizer, scheduler_type, config)
        
        # 模拟训练过程
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
            optimizer.step()
            
            # 调度器步骤
            metrics = {'loss': loss.item()}
            if scheduler_type == SchedulerType.PHYSICS_INFORMED:
                metrics.update({
                    'physics_loss': loss.item() * 0.3,
                    'data_loss': loss.item() * 0.7
                })
            
            scheduler.step(metrics, epoch)
            
            if epoch % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, LR = {current_lr:.6f}")
        
        final_lr = scheduler.get_last_lr()[0]
        print(f"  最终学习率: {final_lr:.6f}")
    
    # 测试调度器管理器
    print(f"\n=== 调度器管理器测试 ===")
    
    # 创建多个调度器
    schedulers = {
        'cosine': create_lr_scheduler(optimizer, SchedulerType.COSINE_LR),
        'step': create_lr_scheduler(optimizer, SchedulerType.STEP_LR),
        'adaptive': create_lr_scheduler(optimizer, SchedulerType.ADAPTIVE_LOSS)
    }
    
    manager = SchedulerManager(schedulers)
    
    # 训练并自动切换
    for epoch in range(30):
        inputs = torch.randn(32, 3)
        targets = torch.sin(inputs.sum(dim=1, keepdim=True))
        
        predictions = model(inputs)
        loss = nn.MSELoss()(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics = {'loss': loss.item()}
        manager.step(metrics=metrics, epoch=epoch)
        
        # 每10步尝试自动切换
        if epoch % 10 == 0 and epoch > 0:
            switched = manager.auto_switch(patience=10)
            if switched:
                print(f"  Epoch {epoch}: 自动切换到 {manager.current_scheduler}")
        
        if epoch % 10 == 0:
            current_lr = manager.get_current_lr()[0]
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}, LR = {current_lr:.6f}, 调度器 = {manager.current_scheduler}")
    
    # 性能总结
    summary = manager.get_performance_summary()
    print(f"\n=== 性能总结 ===")
    for name, stats in summary.items():
        if isinstance(stats, dict) and 'total_steps' in stats:
            lr_range = stats['lr_range']
            print(f"  {name}: 步数={stats['total_steps']}, LR范围=[{lr_range[0]:.6f}, {lr_range[1]:.6f}]")
    
    print("\n学习率调度器测试完成！")