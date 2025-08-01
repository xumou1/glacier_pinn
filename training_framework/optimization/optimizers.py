#!/usr/bin/env python3
"""
优化器模块

实现各种优化算法用于PINNs训练，包括：
- 自适应优化器
- 物理感知优化器
- 多目标优化器
- 约束优化器
- 学习率调度器

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
import math
from abc import ABC, abstractmethod
from collections import defaultdict

class AdaptiveOptimizer(optim.Optimizer):
    """
    自适应优化器
    
    根据训练进度自适应调整优化策略
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, adaptation_rate: float = 0.1):
        """
        初始化自适应优化器
        
        Args:
            params: 模型参数
            lr: 学习率
            betas: Adam的beta参数
            eps: 数值稳定性参数
            weight_decay: 权重衰减
            adaptation_rate: 自适应率
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       adaptation_rate=adaptation_rate)
        super(AdaptiveOptimizer, self).__init__(params, defaults)
        
        self.loss_history = []
        self.gradient_norms = []
        self.adaptation_threshold = 0.01
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行优化步骤
        
        Args:
            closure: 闭包函数
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # 计算梯度范数
        total_grad_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = math.sqrt(total_grad_norm)
        self.gradient_norms.append(total_grad_norm)
        
        # 自适应调整学习率
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            
            for group in self.param_groups:
                if loss_trend > self.adaptation_threshold:
                    # 损失增加，降低学习率
                    group['lr'] *= (1 - group['adaptation_rate'])
                elif loss_trend < -self.adaptation_threshold:
                    # 损失减少，可以适当增加学习率
                    group['lr'] *= (1 + group['adaptation_rate'] * 0.5)
        
        # 执行Adam更新
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # 指数移动平均
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # 更新参数
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), 
                               value=-step_size)
        
        if loss is not None:
            self.loss_history.append(loss.item())
        
        return loss

class PhysicsAwareOptimizer(optim.Optimizer):
    """
    物理感知优化器
    
    根据物理损失和数据损失的平衡调整优化策略
    """
    
    def __init__(self, params, lr: float = 1e-3, physics_weight: float = 1.0,
                 balance_factor: float = 0.1):
        """
        初始化物理感知优化器
        
        Args:
            params: 模型参数
            lr: 学习率
            physics_weight: 物理损失权重
            balance_factor: 平衡因子
        """
        defaults = dict(lr=lr, physics_weight=physics_weight, balance_factor=balance_factor)
        super(PhysicsAwareOptimizer, self).__init__(params, defaults)
        
        self.data_loss_history = []
        self.physics_loss_history = []
        self.base_optimizer = optim.Adam(params, lr=lr)
    
    def update_loss_history(self, data_loss: float, physics_loss: float):
        """
        更新损失历史
        
        Args:
            data_loss: 数据损失
            physics_loss: 物理损失
        """
        self.data_loss_history.append(data_loss)
        self.physics_loss_history.append(physics_loss)
    
    def compute_adaptive_weights(self) -> Tuple[float, float]:
        """
        计算自适应权重
        
        Returns:
            Tuple: (数据权重, 物理权重)
        """
        if len(self.data_loss_history) < 10 or len(self.physics_loss_history) < 10:
            return 1.0, self.defaults['physics_weight']
        
        # 计算最近的损失趋势
        recent_data = self.data_loss_history[-10:]
        recent_physics = self.physics_loss_history[-10:]
        
        data_trend = (recent_data[-1] - recent_data[0]) / len(recent_data)
        physics_trend = (recent_physics[-1] - recent_physics[0]) / len(recent_physics)
        
        # 自适应调整权重
        balance_factor = self.defaults['balance_factor']
        
        if data_trend > physics_trend:
            # 数据损失下降较慢，增加数据权重
            data_weight = 1.0 + balance_factor
            physics_weight = self.defaults['physics_weight'] * (1 - balance_factor)
        else:
            # 物理损失下降较慢，增加物理权重
            data_weight = 1.0 * (1 - balance_factor)
            physics_weight = self.defaults['physics_weight'] * (1 + balance_factor)
        
        return data_weight, physics_weight
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行优化步骤
        
        Args:
            closure: 闭包函数
        """
        return self.base_optimizer.step(closure)
    
    def zero_grad(self):
        """清零梯度"""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """获取状态字典"""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.base_optimizer.load_state_dict(state_dict)

class MultiObjectiveOptimizer:
    """
    多目标优化器
    
    处理多个目标函数的优化问题
    """
    
    def __init__(self, model: nn.Module, optimizers: Dict[str, optim.Optimizer],
                 weights: Dict[str, float] = None, method: str = 'weighted_sum'):
        """
        初始化多目标优化器
        
        Args:
            model: 模型
            optimizers: 优化器字典
            weights: 权重字典
            method: 多目标优化方法
        """
        self.model = model
        self.optimizers = optimizers
        self.weights = weights or {name: 1.0 for name in optimizers.keys()}
        self.method = method
        
        self.loss_history = defaultdict(list)
        self.pareto_front = []
    
    def step(self, losses: Dict[str, torch.Tensor]):
        """
        执行多目标优化步骤
        
        Args:
            losses: 损失字典
        """
        if self.method == 'weighted_sum':
            self._weighted_sum_step(losses)
        elif self.method == 'pareto':
            self._pareto_step(losses)
        elif self.method == 'gradient_surgery':
            self._gradient_surgery_step(losses)
        else:
            raise ValueError(f"未知的多目标优化方法: {self.method}")
        
        # 更新损失历史
        for name, loss in losses.items():
            self.loss_history[name].append(loss.item())
    
    def _weighted_sum_step(self, losses: Dict[str, torch.Tensor]):
        """
        加权和方法
        
        Args:
            losses: 损失字典
        """
        # 清零所有梯度
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # 计算加权总损失
        total_loss = sum(self.weights[name] * loss for name, loss in losses.items())
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def _pareto_step(self, losses: Dict[str, torch.Tensor]):
        """
        帕累托优化方法
        
        Args:
            losses: 损失字典
        """
        # TODO: 实现帕累托优化
        # 这是一个复杂的多目标优化方法，需要维护帕累托前沿
        pass
    
    def _gradient_surgery_step(self, losses: Dict[str, torch.Tensor]):
        """
        梯度手术方法
        
        Args:
            losses: 损失字典
        """
        # 清零梯度
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # 分别计算每个损失的梯度
        gradients = {}
        for name, loss in losses.items():
            # 计算梯度
            grad = torch.autograd.grad(loss, self.model.parameters(), 
                                     retain_graph=True, create_graph=False)
            gradients[name] = grad
        
        # 梯度手术：解决梯度冲突
        final_gradients = self._resolve_gradient_conflicts(gradients)
        
        # 应用最终梯度
        for param, grad in zip(self.model.parameters(), final_gradients):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.data = grad.clone()
        
        # 更新参数
        for optimizer in self.optimizers.values():
            optimizer.step()
            break  # 只需要一个优化器来更新
    
    def _resolve_gradient_conflicts(self, gradients: Dict[str, Tuple]) -> List[torch.Tensor]:
        """
        解决梯度冲突
        
        Args:
            gradients: 梯度字典
            
        Returns:
            List: 解决冲突后的梯度
        """
        # 简化的梯度手术实现
        grad_names = list(gradients.keys())
        if len(grad_names) < 2:
            return list(gradients.values())[0]
        
        # 计算梯度之间的余弦相似度
        grad1 = gradients[grad_names[0]]
        grad2 = gradients[grad_names[1]]
        
        # 展平梯度
        flat_grad1 = torch.cat([g.flatten() for g in grad1])
        flat_grad2 = torch.cat([g.flatten() for g in grad2])
        
        # 计算余弦相似度
        cos_sim = torch.dot(flat_grad1, flat_grad2) / (
            torch.norm(flat_grad1) * torch.norm(flat_grad2) + 1e-8
        )
        
        if cos_sim < 0:  # 梯度冲突
            # 投影梯度2到梯度1的正交空间
            projection = torch.dot(flat_grad2, flat_grad1) / (torch.norm(flat_grad1) ** 2 + 1e-8)
            flat_grad2_corrected = flat_grad2 - projection * flat_grad1
            
            # 重新整形
            corrected_grad2 = []
            start_idx = 0
            for g in grad2:
                end_idx = start_idx + g.numel()
                corrected_grad2.append(flat_grad2_corrected[start_idx:end_idx].reshape(g.shape))
                start_idx = end_idx
            
            # 平均梯度
            final_gradients = []
            for g1, g2 in zip(grad1, corrected_grad2):
                final_gradients.append((g1 + g2) / 2)
        else:
            # 无冲突，直接平均
            final_gradients = []
            for g1, g2 in zip(grad1, grad2):
                final_gradients.append((g1 + g2) / 2)
        
        return final_gradients

class ConstrainedOptimizer:
    """
    约束优化器
    
    处理带约束的优化问题
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 constraints: List[Callable] = None, penalty_weight: float = 1.0):
        """
        初始化约束优化器
        
        Args:
            model: 模型
            optimizer: 基础优化器
            constraints: 约束函数列表
            penalty_weight: 惩罚权重
        """
        self.model = model
        self.optimizer = optimizer
        self.constraints = constraints or []
        self.penalty_weight = penalty_weight
        
        self.constraint_violations = []
    
    def add_constraint(self, constraint: Callable):
        """
        添加约束
        
        Args:
            constraint: 约束函数
        """
        self.constraints.append(constraint)
    
    def compute_constraint_penalty(self, x: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        计算约束惩罚
        
        Args:
            x: 输入
            predictions: 预测
            
        Returns:
            torch.Tensor: 约束惩罚
        """
        total_penalty = torch.tensor(0.0, device=predictions.device)
        
        for constraint in self.constraints:
            violation = constraint(x, predictions)
            penalty = torch.maximum(violation, torch.zeros_like(violation))
            total_penalty += torch.mean(penalty ** 2)
        
        return total_penalty
    
    def step(self, loss: torch.Tensor, x: torch.Tensor, predictions: torch.Tensor):
        """
        执行约束优化步骤
        
        Args:
            loss: 基础损失
            x: 输入
            predictions: 预测
        """
        self.optimizer.zero_grad()
        
        # 计算约束惩罚
        constraint_penalty = self.compute_constraint_penalty(x, predictions)
        
        # 总损失
        total_loss = loss + self.penalty_weight * constraint_penalty
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        self.optimizer.step()
        
        # 记录约束违反
        self.constraint_violations.append(constraint_penalty.item())
        
        return total_loss

class LearningRateScheduler:
    """
    学习率调度器
    
    实现各种学习率调度策略
    """
    
    def __init__(self, optimizer: optim.Optimizer, schedule_type: str = 'cosine',
                 **schedule_params):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            schedule_type: 调度类型
            **schedule_params: 调度参数
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params
        
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self, epoch: int = None, loss: float = None):
        """
        更新学习率
        
        Args:
            epoch: 当前轮数
            loss: 当前损失
        """
        self.step_count += 1
        
        if self.schedule_type == 'cosine':
            self._cosine_schedule()
        elif self.schedule_type == 'exponential':
            self._exponential_schedule()
        elif self.schedule_type == 'plateau':
            self._plateau_schedule(loss)
        elif self.schedule_type == 'warmup_cosine':
            self._warmup_cosine_schedule()
        elif self.schedule_type == 'cyclic':
            self._cyclic_schedule()
    
    def _cosine_schedule(self):
        """余弦退火调度"""
        T_max = self.schedule_params.get('T_max', 1000)
        eta_min = self.schedule_params.get('eta_min', 0)
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = eta_min + (self.initial_lrs[i] - eta_min) * \
                         (1 + math.cos(math.pi * self.step_count / T_max)) / 2
    
    def _exponential_schedule(self):
        """指数衰减调度"""
        gamma = self.schedule_params.get('gamma', 0.95)
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.initial_lrs[i] * (gamma ** self.step_count)
    
    def _plateau_schedule(self, loss: float):
        """平台调度"""
        # 简化实现，实际应该维护损失历史
        if not hasattr(self, 'best_loss'):
            self.best_loss = float('inf')
            self.patience_count = 0
        
        patience = self.schedule_params.get('patience', 10)
        factor = self.schedule_params.get('factor', 0.5)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_count = 0
        else:
            self.patience_count += 1
            
            if self.patience_count >= patience:
                for group in self.optimizer.param_groups:
                    group['lr'] *= factor
                self.patience_count = 0
    
    def _warmup_cosine_schedule(self):
        """预热余弦调度"""
        warmup_steps = self.schedule_params.get('warmup_steps', 100)
        T_max = self.schedule_params.get('T_max', 1000)
        
        if self.step_count < warmup_steps:
            # 预热阶段
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.initial_lrs[i] * self.step_count / warmup_steps
        else:
            # 余弦退火阶段
            adjusted_step = self.step_count - warmup_steps
            adjusted_T_max = T_max - warmup_steps
            
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.initial_lrs[i] * \
                             (1 + math.cos(math.pi * adjusted_step / adjusted_T_max)) / 2
    
    def _cyclic_schedule(self):
        """循环学习率调度"""
        base_lr = self.schedule_params.get('base_lr', 1e-5)
        max_lr = self.schedule_params.get('max_lr', 1e-2)
        step_size = self.schedule_params.get('step_size', 100)
        
        cycle = math.floor(1 + self.step_count / (2 * step_size))
        x = abs(self.step_count / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        
        for group in self.optimizer.param_groups:
            group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """
        获取当前学习率
        
        Returns:
            List: 学习率列表
        """
        return [group['lr'] for group in self.optimizer.param_groups]

class OptimizerFactory:
    """
    优化器工厂
    
    创建和配置各种优化器
    """
    
    @staticmethod
    def create_optimizer(optimizer_type: str, model: nn.Module, **kwargs) -> optim.Optimizer:
        """
        创建优化器
        
        Args:
            optimizer_type: 优化器类型
            model: 模型
            **kwargs: 优化器参数
            
        Returns:
            optim.Optimizer: 优化器
        """
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), **kwargs)
        elif optimizer_type == 'adamw':
            return optim.AdamW(model.parameters(), **kwargs)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), **kwargs)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(model.parameters(), **kwargs)
        elif optimizer_type == 'adaptive':
            return AdaptiveOptimizer(model.parameters(), **kwargs)
        elif optimizer_type == 'physics_aware':
            return PhysicsAwareOptimizer(model.parameters(), **kwargs)
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str, **kwargs) -> LearningRateScheduler:
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型
            **kwargs: 调度器参数
            
        Returns:
            LearningRateScheduler: 学习率调度器
        """
        return LearningRateScheduler(optimizer, scheduler_type, **kwargs)

if __name__ == "__main__":
    # 测试优化器
    torch.manual_seed(42)
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # 测试数据
    x = torch.randn(100, 2)
    y = torch.randn(100, 1)
    
    # 测试自适应优化器
    print("测试自适应优化器...")
    adaptive_opt = AdaptiveOptimizer(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        def closure():
            adaptive_opt.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            return loss
        
        loss = adaptive_opt.step(closure)
        print(f"轮次 {epoch}: 损失 = {loss.item():.4f}, 学习率 = {adaptive_opt.param_groups[0]['lr']:.6f}")
    
    # 测试物理感知优化器
    print("\n测试物理感知优化器...")
    physics_opt = PhysicsAwareOptimizer(model.parameters(), lr=0.01)
    
    for epoch in range(5):
        physics_opt.zero_grad()
        pred = model(x)
        data_loss = nn.MSELoss()(pred, y)
        physics_loss = torch.mean(pred**2)  # 简单的物理损失
        
        total_loss = data_loss + physics_loss
        total_loss.backward()
        physics_opt.step()
        
        physics_opt.update_loss_history(data_loss.item(), physics_loss.item())
        data_weight, physics_weight = physics_opt.compute_adaptive_weights()
        
        print(f"轮次 {epoch}: 数据损失 = {data_loss.item():.4f}, 物理损失 = {physics_loss.item():.4f}")
        print(f"  权重: 数据 = {data_weight:.3f}, 物理 = {physics_weight:.3f}")
    
    # 测试多目标优化器
    print("\n测试多目标优化器...")
    optimizers = {
        'data': optim.Adam(model.parameters(), lr=0.01),
        'physics': optim.Adam(model.parameters(), lr=0.01)
    }
    multi_opt = MultiObjectiveOptimizer(model, optimizers, method='weighted_sum')
    
    for epoch in range(5):
        pred = model(x)
        losses = {
            'data': nn.MSELoss()(pred, y),
            'physics': torch.mean(pred**2)
        }
        
        multi_opt.step(losses)
        print(f"轮次 {epoch}: 数据损失 = {losses['data'].item():.4f}, 物理损失 = {losses['physics'].item():.4f}")
    
    # 测试学习率调度器
    print("\n测试学习率调度器...")
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = LearningRateScheduler(optimizer, 'cosine', T_max=20)
    
    print("余弦调度学习率变化:")
    for step in range(20):
        scheduler.step()
        print(f"步骤 {step}: 学习率 = {scheduler.get_lr()[0]:.6f}")
    
    # 测试优化器工厂
    print("\n测试优化器工厂...")
    factory_opt = OptimizerFactory.create_optimizer('adaptive', model, lr=0.01)
    factory_scheduler = OptimizerFactory.create_scheduler(factory_opt, 'warmup_cosine', 
                                                        warmup_steps=5, T_max=15)
    
    print("工厂创建的优化器和调度器:")
    for step in range(10):
        factory_scheduler.step()
        print(f"步骤 {step}: 学习率 = {factory_scheduler.get_lr()[0]:.6f}")