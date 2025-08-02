#!/usr/bin/env python3
"""
阶段3：耦合优化训练

实现多目标耦合优化的训练阶段，包括：
- 多目标优化策略
- 约束耦合处理
- 自适应权重调整
- 收敛性分析

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
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

class OptimizationType(Enum):
    """优化类型枚举"""
    PARETO = "pareto"  # 帕累托优化
    WEIGHTED_SUM = "weighted_sum"  # 加权和
    CONSTRAINT = "constraint"  # 约束优化
    LEXICOGRAPHIC = "lexicographic"  # 字典序优化
    GOAL_PROGRAMMING = "goal_programming"  # 目标规划
    EVOLUTIONARY = "evolutionary"  # 进化算法

class ConstraintType(Enum):
    """约束类型枚举"""
    EQUALITY = "equality"  # 等式约束
    INEQUALITY = "inequality"  # 不等式约束
    BOUND = "bound"  # 边界约束
    PHYSICS = "physics"  # 物理约束
    DATA = "data"  # 数据约束
    TEMPORAL = "temporal"  # 时间约束
    SPATIAL = "spatial"  # 空间约束

@dataclass
class CoupledOptimizationConfig:
    """耦合优化配置"""
    optimization_type: OptimizationType = OptimizationType.WEIGHTED_SUM
    max_iterations: int = 1000
    tolerance: float = 1e-6
    constraint_penalty: float = 1.0
    weight_adaptation_rate: float = 0.01
    pareto_population_size: int = 50
    convergence_window: int = 10
    objective_weights: Dict[str, float] = None
    constraint_weights: Dict[str, float] = None
    adaptive_weights: bool = True
    
    def __post_init__(self):
        if self.objective_weights is None:
            self.objective_weights = {
                'physics_loss': 1.0,
                'data_loss': 1.0,
                'boundary_loss': 0.5,
                'regularization_loss': 0.1
            }
        
        if self.constraint_weights is None:
            self.constraint_weights = {
                'equality': 10.0,
                'inequality': 5.0,
                'physics': 2.0,
                'data': 1.0
            }

class ObjectiveFunction:
    """
    目标函数基类
    
    定义多目标优化中的单个目标
    """
    
    def __init__(self, name: str, weight: float = 1.0, minimize: bool = True):
        """
        初始化目标函数
        
        Args:
            name: 目标函数名称
            weight: 权重
            minimize: 是否最小化
        """
        self.name = name
        self.weight = weight
        self.minimize = minimize
        self.history = []
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: nn.Module = None, **kwargs) -> torch.Tensor:
        """
        计算目标函数值
        
        Args:
            predictions: 预测值
            targets: 目标值
            model: 模型（可选）
            **kwargs: 其他参数
            
        Returns:
            Tensor: 目标函数值
        """
        raise NotImplementedError
    
    def update_history(self, value: float) -> None:
        """更新历史记录"""
        self.history.append(value)
        if len(self.history) > 1000:  # 限制历史长度
            self.history.pop(0)

class PhysicsObjective(ObjectiveFunction):
    """物理目标函数"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: nn.Module = None, **kwargs) -> torch.Tensor:
        # 物理方程残差
        physics_residual = kwargs.get('physics_residual', torch.tensor(0.0))
        return torch.mean(physics_residual ** 2)

class DataObjective(ObjectiveFunction):
    """数据拟合目标函数"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: nn.Module = None, **kwargs) -> torch.Tensor:
        return nn.MSELoss()(predictions, targets)

class BoundaryObjective(ObjectiveFunction):
    """边界条件目标函数"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: nn.Module = None, **kwargs) -> torch.Tensor:
        boundary_residual = kwargs.get('boundary_residual', torch.tensor(0.0))
        return torch.mean(boundary_residual ** 2)

class RegularizationObjective(ObjectiveFunction):
    """正则化目标函数"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: nn.Module = None, **kwargs) -> torch.Tensor:
        if model is None:
            return torch.tensor(0.0)
        
        l2_reg = torch.tensor(0.0)
        for param in model.parameters():
            l2_reg += torch.norm(param) ** 2
        
        return l2_reg

class ConstraintFunction:
    """
    约束函数基类
    """
    
    def __init__(self, name: str, constraint_type: ConstraintType, 
                 weight: float = 1.0, tolerance: float = 1e-6):
        """
        初始化约束函数
        
        Args:
            name: 约束名称
            constraint_type: 约束类型
            weight: 权重
            tolerance: 容忍度
        """
        self.name = name
        self.constraint_type = constraint_type
        self.weight = weight
        self.tolerance = tolerance
        self.violation_history = []
    
    def __call__(self, predictions: torch.Tensor, model: nn.Module = None, 
                 **kwargs) -> torch.Tensor:
        """
        计算约束违反程度
        
        Args:
            predictions: 预测值
            model: 模型
            **kwargs: 其他参数
            
        Returns:
            Tensor: 约束违反程度
        """
        raise NotImplementedError
    
    def update_violation_history(self, violation: float) -> None:
        """更新违反历史"""
        self.violation_history.append(violation)
        if len(self.violation_history) > 1000:
            self.violation_history.pop(0)

class EqualityConstraint(ConstraintFunction):
    """等式约束"""
    
    def __init__(self, name: str, target_value: float = 0.0, **kwargs):
        super().__init__(name, ConstraintType.EQUALITY, **kwargs)
        self.target_value = target_value
    
    def __call__(self, predictions: torch.Tensor, model: nn.Module = None, 
                 **kwargs) -> torch.Tensor:
        constraint_value = kwargs.get('constraint_value', predictions)
        violation = torch.abs(constraint_value - self.target_value)
        return torch.mean(violation)

class InequalityConstraint(ConstraintFunction):
    """不等式约束"""
    
    def __init__(self, name: str, upper_bound: float = None, 
                 lower_bound: float = None, **kwargs):
        super().__init__(name, ConstraintType.INEQUALITY, **kwargs)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
    def __call__(self, predictions: torch.Tensor, model: nn.Module = None, 
                 **kwargs) -> torch.Tensor:
        constraint_value = kwargs.get('constraint_value', predictions)
        violation = torch.tensor(0.0)
        
        if self.upper_bound is not None:
            upper_violation = torch.clamp(constraint_value - self.upper_bound, min=0)
            violation += torch.mean(upper_violation)
        
        if self.lower_bound is not None:
            lower_violation = torch.clamp(self.lower_bound - constraint_value, min=0)
            violation += torch.mean(lower_violation)
        
        return violation

class AdaptiveWeightManager:
    """
    自适应权重管理器
    
    根据优化过程动态调整目标函数和约束的权重
    """
    
    def __init__(self, config: CoupledOptimizationConfig):
        """
        初始化权重管理器
        
        Args:
            config: 耦合优化配置
        """
        self.config = config
        self.objective_weights = config.objective_weights.copy()
        self.constraint_weights = config.constraint_weights.copy()
        self.adaptation_history = []
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
    
    def update_weights(self, objective_values: Dict[str, float], 
                      constraint_violations: Dict[str, float],
                      iteration: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        更新权重
        
        Args:
            objective_values: 目标函数值
            constraint_violations: 约束违反程度
            iteration: 当前迭代次数
            
        Returns:
            Tuple: 更新后的目标权重和约束权重
        """
        if not self.config.adaptive_weights:
            return self.objective_weights, self.constraint_weights
        
        # 记录性能历史
        total_objective = sum(objective_values.values())
        total_violation = sum(constraint_violations.values())
        self.performance_history.append({
            'iteration': iteration,
            'total_objective': total_objective,
            'total_violation': total_violation,
            'objectives': objective_values.copy(),
            'violations': constraint_violations.copy()
        })
        
        # 保持历史长度
        if len(self.performance_history) > self.config.convergence_window * 2:
            self.performance_history.pop(0)
        
        # 计算改进率
        if len(self.performance_history) >= self.config.convergence_window:
            recent_performance = self.performance_history[-self.config.convergence_window:]
            
            # 目标函数改进率
            objective_improvement = self._calculate_improvement_rate(
                [p['total_objective'] for p in recent_performance]
            )
            
            # 约束违反改进率
            violation_improvement = self._calculate_improvement_rate(
                [p['total_violation'] for p in recent_performance]
            )
            
            # 调整权重
            self._adapt_objective_weights(objective_values, objective_improvement)
            self._adapt_constraint_weights(constraint_violations, violation_improvement)
        
        return self.objective_weights, self.constraint_weights
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """计算改进率"""
        if len(values) < 2:
            return 0.0
        
        # 使用线性回归计算趋势
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) < 1e-10:  # 避免除零
            return 0.0
        
        # 计算斜率（标准化）
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / (np.mean(y) + 1e-10)
        
        return -normalized_slope  # 负斜率表示改进
    
    def _adapt_objective_weights(self, objective_values: Dict[str, float], 
                                improvement_rate: float) -> None:
        """调整目标权重"""
        adaptation_rate = self.config.weight_adaptation_rate
        
        # 如果改进缓慢，增加表现差的目标的权重
        if improvement_rate < 0.01:
            max_value = max(objective_values.values()) if objective_values else 1.0
            
            for name, value in objective_values.items():
                if name in self.objective_weights:
                    # 表现差的目标增加权重
                    relative_performance = value / (max_value + 1e-10)
                    weight_adjustment = adaptation_rate * relative_performance
                    self.objective_weights[name] *= (1 + weight_adjustment)
        
        # 权重归一化
        total_weight = sum(self.objective_weights.values())
        if total_weight > 0:
            for name in self.objective_weights:
                self.objective_weights[name] /= total_weight
                self.objective_weights[name] *= len(self.objective_weights)
    
    def _adapt_constraint_weights(self, constraint_violations: Dict[str, float], 
                                 improvement_rate: float) -> None:
        """调整约束权重"""
        adaptation_rate = self.config.weight_adaptation_rate
        
        # 如果约束违反严重，增加约束权重
        for name, violation in constraint_violations.items():
            if violation > 1e-6:  # 有显著违反
                if name in self.constraint_weights:
                    # 违反越严重，权重增加越多
                    weight_multiplier = 1 + adaptation_rate * np.log(1 + violation)
                    self.constraint_weights[name] *= weight_multiplier
            else:
                # 约束满足良好，可以适度减少权重
                if name in self.constraint_weights:
                    self.constraint_weights[name] *= (1 - adaptation_rate * 0.1)
        
        # 确保权重不会过小
        for name in self.constraint_weights:
            self.constraint_weights[name] = max(self.constraint_weights[name], 0.01)

class ParetoOptimizer:
    """
    帕累托优化器
    
    实现多目标帕累托优化
    """
    
    def __init__(self, objectives: List[ObjectiveFunction], 
                 constraints: List[ConstraintFunction],
                 config: CoupledOptimizationConfig):
        """
        初始化帕累托优化器
        
        Args:
            objectives: 目标函数列表
            constraints: 约束函数列表
            config: 配置
        """
        self.objectives = objectives
        self.constraints = constraints
        self.config = config
        self.pareto_front = []
        self.population = []
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, model: nn.Module, data_loader, 
                 num_generations: int = 100) -> List[Dict[str, Any]]:
        """
        执行帕累托优化
        
        Args:
            model: 模型
            data_loader: 数据加载器
            num_generations: 代数
            
        Returns:
            List: 帕累托前沿解集
        """
        # 初始化种群
        self._initialize_population(model)
        
        for generation in range(num_generations):
            # 评估种群
            fitness_values = self._evaluate_population(model, data_loader)
            
            # 更新帕累托前沿
            self._update_pareto_front(fitness_values)
            
            # 选择和变异
            self._evolve_population()
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}, Pareto front size: {len(self.pareto_front)}")
        
        return self.pareto_front
    
    def _initialize_population(self, model: nn.Module) -> None:
        """初始化种群"""
        # 保存原始参数
        original_params = [param.clone() for param in model.parameters()]
        
        self.population = []
        for _ in range(self.config.pareto_population_size):
            # 随机扰动参数
            individual = []
            for param in model.parameters():
                noise = torch.randn_like(param) * 0.01
                individual.append(param + noise)
            self.population.append(individual)
        
        # 恢复原始参数
        for param, original in zip(model.parameters(), original_params):
            param.data.copy_(original)
    
    def _evaluate_population(self, model: nn.Module, data_loader) -> List[Dict[str, float]]:
        """评估种群适应度"""
        fitness_values = []
        
        # 保存原始参数
        original_params = [param.clone() for param in model.parameters()]
        
        for individual in self.population:
            # 设置个体参数
            for param, individual_param in zip(model.parameters(), individual):
                param.data.copy_(individual_param)
            
            # 评估目标函数
            objective_values = {}
            constraint_violations = {}
            
            model.eval()
            with torch.no_grad():
                total_objectives = {obj.name: 0.0 for obj in self.objectives}
                total_constraints = {const.name: 0.0 for const in self.constraints}
                num_batches = 0
                
                for batch_data in data_loader:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    else:
                        inputs, targets = batch_data[0], batch_data[1]
                    
                    predictions = model(inputs)
                    
                    # 计算目标函数
                    for obj in self.objectives:
                        obj_value = obj(predictions, targets, model).item()
                        total_objectives[obj.name] += obj_value
                    
                    # 计算约束违反
                    for const in self.constraints:
                        const_violation = const(predictions, model).item()
                        total_constraints[const.name] += const_violation
                    
                    num_batches += 1
                
                # 平均化
                for name in total_objectives:
                    objective_values[name] = total_objectives[name] / num_batches
                
                for name in total_constraints:
                    constraint_violations[name] = total_constraints[name] / num_batches
            
            fitness_values.append({
                'objectives': objective_values,
                'constraints': constraint_violations,
                'individual': [param.clone() for param in individual]
            })
        
        # 恢复原始参数
        for param, original in zip(model.parameters(), original_params):
            param.data.copy_(original)
        
        return fitness_values
    
    def _update_pareto_front(self, fitness_values: List[Dict[str, Any]]) -> None:
        """更新帕累托前沿"""
        # 合并当前种群和帕累托前沿
        all_solutions = fitness_values + self.pareto_front
        
        # 找到非支配解
        pareto_solutions = []
        
        for i, solution_i in enumerate(all_solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(all_solutions):
                if i != j and self._dominates(solution_j, solution_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution_i)
        
        self.pareto_front = pareto_solutions
    
    def _dominates(self, solution_a: Dict[str, Any], solution_b: Dict[str, Any]) -> bool:
        """判断解A是否支配解B"""
        objectives_a = solution_a['objectives']
        objectives_b = solution_b['objectives']
        constraints_a = solution_a['constraints']
        constraints_b = solution_b['constraints']
        
        # 首先检查约束违反
        total_violation_a = sum(constraints_a.values())
        total_violation_b = sum(constraints_b.values())
        
        # 如果A可行而B不可行，A支配B
        if total_violation_a <= 1e-6 and total_violation_b > 1e-6:
            return True
        
        # 如果B可行而A不可行，A不支配B
        if total_violation_b <= 1e-6 and total_violation_a > 1e-6:
            return False
        
        # 比较目标函数
        better_in_all = True
        better_in_at_least_one = False
        
        for obj_name in objectives_a:
            if obj_name in objectives_b:
                if objectives_a[obj_name] > objectives_b[obj_name]:  # 假设最小化
                    better_in_all = False
                elif objectives_a[obj_name] < objectives_b[obj_name]:
                    better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one
    
    def _evolve_population(self) -> None:
        """进化种群"""
        # 简单的进化策略：从帕累托前沿选择父代，进行变异
        if len(self.pareto_front) == 0:
            return
        
        new_population = []
        
        for _ in range(self.config.pareto_population_size):
            # 随机选择父代
            parent = np.random.choice(self.pareto_front)
            
            # 变异
            offspring = []
            for param in parent['individual']:
                mutation = torch.randn_like(param) * 0.001
                offspring.append(param + mutation)
            
            new_population.append(offspring)
        
        self.population = new_population

class CoupledOptimizationTrainer:
    """
    耦合优化训练器
    
    管理多目标耦合优化的训练过程
    """
    
    def __init__(self, model: nn.Module, objectives: List[ObjectiveFunction],
                 constraints: List[ConstraintFunction], config: CoupledOptimizationConfig):
        """
        初始化耦合优化训练器
        
        Args:
            model: 模型
            objectives: 目标函数列表
            constraints: 约束函数列表
            config: 配置
        """
        self.model = model
        self.objectives = objectives
        self.constraints = constraints
        self.config = config
        self.weight_manager = AdaptiveWeightManager(config)
        self.optimizer = None
        self.training_history = []
        self.convergence_metrics = []
        self.logger = logging.getLogger(__name__)
        
        # 根据优化类型初始化特定优化器
        if config.optimization_type == OptimizationType.PARETO:
            self.pareto_optimizer = ParetoOptimizer(objectives, constraints, config)
    
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
    
    def compute_total_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                          iteration: int, **kwargs) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        计算总损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            iteration: 迭代次数
            **kwargs: 其他参数
            
        Returns:
            Tuple: (总损失, 目标函数值, 约束违反程度)
        """
        # 计算各个目标函数
        objective_values = {}
        for obj in self.objectives:
            obj_value = obj(predictions, targets, self.model, **kwargs)
            objective_values[obj.name] = obj_value.item()
        
        # 计算约束违反
        constraint_violations = {}
        for const in self.constraints:
            const_violation = const(predictions, self.model, **kwargs)
            constraint_violations[const.name] = const_violation.item()
        
        # 更新权重
        obj_weights, const_weights = self.weight_manager.update_weights(
            objective_values, constraint_violations, iteration
        )
        
        # 计算加权总损失
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # 目标函数部分
        for obj in self.objectives:
            obj_value = obj(predictions, targets, self.model, **kwargs)
            weight = obj_weights.get(obj.name, obj.weight)
            total_loss += weight * obj_value
            
            # 更新目标函数历史
            obj.update_history(obj_value.item())
        
        # 约束部分
        for const in self.constraints:
            const_violation = const(predictions, self.model, **kwargs)
            weight = const_weights.get(const.name, const.weight)
            
            if const.constraint_type == ConstraintType.EQUALITY:
                penalty = weight * const_violation
            else:  # 不等式约束
                penalty = weight * torch.clamp(const_violation, min=0)
            
            total_loss += penalty
            
            # 更新约束违反历史
            const.update_violation_history(const_violation.item())
        
        return total_loss, objective_values, constraint_violations
    
    def train_epoch(self, data_loader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            data_loader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict: 训练指标
        """
        self.model.train()
        
        epoch_objectives = {obj.name: 0.0 for obj in self.objectives}
        epoch_constraints = {const.name: 0.0 for const in self.constraints}
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(data_loader):
            if len(batch_data) == 2:
                inputs, targets = batch_data
                kwargs = {}
            else:
                inputs, targets = batch_data[0], batch_data[1]
                kwargs = batch_data[2] if len(batch_data) > 2 else {}
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(inputs)
            
            # 计算损失
            iteration = epoch * len(data_loader) + batch_idx
            total_loss, objective_values, constraint_violations = self.compute_total_loss(
                predictions, targets, iteration, **kwargs
            )
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累积指标
            epoch_total_loss += total_loss.item()
            for name, value in objective_values.items():
                epoch_objectives[name] += value
            for name, value in constraint_violations.items():
                epoch_constraints[name] += value
            
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Total Loss: {total_loss.item():.6f}"
                )
        
        # 计算平均值
        metrics = {
            'total_loss': epoch_total_loss / num_batches,
            'objectives': {name: value / num_batches for name, value in epoch_objectives.items()},
            'constraints': {name: value / num_batches for name, value in epoch_constraints.items()}
        }
        
        # 记录训练历史
        self.training_history.append(metrics)
        
        # 分析收敛性
        convergence_info = self._analyze_convergence()
        metrics['convergence'] = convergence_info
        
        return metrics
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        分析收敛性
        
        Returns:
            Dict: 收敛性分析结果
        """
        if len(self.training_history) < self.config.convergence_window:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        recent_history = self.training_history[-self.config.convergence_window:]
        
        # 检查总损失收敛
        total_losses = [h['total_loss'] for h in recent_history]
        loss_std = np.std(total_losses)
        loss_mean = np.mean(total_losses)
        
        loss_converged = loss_std / (loss_mean + 1e-10) < self.config.tolerance
        
        # 检查目标函数收敛
        objective_converged = True
        for obj_name in recent_history[0]['objectives']:
            obj_values = [h['objectives'][obj_name] for h in recent_history]
            obj_std = np.std(obj_values)
            obj_mean = np.mean(obj_values)
            
            if obj_std / (obj_mean + 1e-10) > self.config.tolerance:
                objective_converged = False
                break
        
        # 检查约束满足
        constraints_satisfied = True
        max_violation = 0.0
        
        for const_name in recent_history[-1]['constraints']:
            violation = recent_history[-1]['constraints'][const_name]
            max_violation = max(max_violation, violation)
            
            if violation > self.config.tolerance:
                constraints_satisfied = False
        
        # 综合判断
        converged = loss_converged and objective_converged and constraints_satisfied
        
        convergence_info = {
            'converged': converged,
            'loss_converged': loss_converged,
            'objective_converged': objective_converged,
            'constraints_satisfied': constraints_satisfied,
            'max_constraint_violation': max_violation,
            'loss_stability': loss_std / (loss_mean + 1e-10)
        }
        
        self.convergence_metrics.append(convergence_info)
        
        return convergence_info
    
    def validate(self, data_loader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            data_loader: 验证数据加载器
            
        Returns:
            Dict: 验证指标
        """
        self.model.eval()
        
        val_objectives = {obj.name: 0.0 for obj in self.objectives}
        val_constraints = {const.name: 0.0 for const in self.constraints}
        val_total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    kwargs = {}
                else:
                    inputs, targets = batch_data[0], batch_data[1]
                    kwargs = batch_data[2] if len(batch_data) > 2 else {}
                
                predictions = self.model(inputs)
                
                # 计算损失（不更新权重）
                total_loss, objective_values, constraint_violations = self.compute_total_loss(
                    predictions, targets, 0, **kwargs  # iteration=0 for validation
                )
                
                val_total_loss += total_loss.item()
                for name, value in objective_values.items():
                    val_objectives[name] += value
                for name, value in constraint_violations.items():
                    val_constraints[name] += value
                
                num_batches += 1
        
        return {
            'val_total_loss': val_total_loss / num_batches,
            'val_objectives': {name: value / num_batches for name, value in val_objectives.items()},
            'val_constraints': {name: value / num_batches for name, value in val_constraints.items()}
        }
    
    def optimize_pareto(self, data_loader, num_generations: int = 100) -> List[Dict[str, Any]]:
        """
        执行帕累托优化
        
        Args:
            data_loader: 数据加载器
            num_generations: 代数
            
        Returns:
            List: 帕累托前沿解集
        """
        if self.config.optimization_type != OptimizationType.PARETO:
            raise ValueError("Pareto optimization requires PARETO optimization type")
        
        return self.pareto_optimizer.optimize(self.model, data_loader, num_generations)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        获取优化总结
        
        Returns:
            Dict: 优化总结
        """
        if not self.training_history:
            return {'status': 'no_training_data'}
        
        latest_metrics = self.training_history[-1]
        
        summary = {
            'optimization_type': self.config.optimization_type.value,
            'total_epochs': len(self.training_history),
            'final_loss': latest_metrics['total_loss'],
            'final_objectives': latest_metrics['objectives'],
            'final_constraints': latest_metrics['constraints'],
            'convergence_status': latest_metrics.get('convergence', {}),
            'objective_weights': self.weight_manager.objective_weights,
            'constraint_weights': self.weight_manager.constraint_weights
        }
        
        # 计算改进统计
        if len(self.training_history) > 1:
            initial_loss = self.training_history[0]['total_loss']
            final_loss = latest_metrics['total_loss']
            improvement = (initial_loss - final_loss) / (initial_loss + 1e-10)
            summary['loss_improvement'] = improvement
        
        return summary

def create_coupled_optimization_trainer(
    model: nn.Module,
    objective_types: List[str] = None,
    constraint_types: List[str] = None,
    config: CoupledOptimizationConfig = None
) -> CoupledOptimizationTrainer:
    """
    创建耦合优化训练器
    
    Args:
        model: 模型
        objective_types: 目标类型列表
        constraint_types: 约束类型列表
        config: 配置
        
    Returns:
        CoupledOptimizationTrainer: 训练器实例
    """
    if objective_types is None:
        objective_types = ['physics', 'data', 'boundary', 'regularization']
    
    if constraint_types is None:
        constraint_types = ['equality', 'inequality']
    
    if config is None:
        config = CoupledOptimizationConfig()
    
    # 创建目标函数
    objectives = []
    for obj_type in objective_types:
        if obj_type == 'physics':
            objectives.append(PhysicsObjective('physics_loss'))
        elif obj_type == 'data':
            objectives.append(DataObjective('data_loss'))
        elif obj_type == 'boundary':
            objectives.append(BoundaryObjective('boundary_loss'))
        elif obj_type == 'regularization':
            objectives.append(RegularizationObjective('regularization_loss'))
    
    # 创建约束函数
    constraints = []
    for const_type in constraint_types:
        if const_type == 'equality':
            constraints.append(EqualityConstraint('mass_conservation'))
        elif const_type == 'inequality':
            constraints.append(InequalityConstraint('physical_bounds', upper_bound=1.0, lower_bound=0.0))
    
    # 创建训练器
    trainer = CoupledOptimizationTrainer(model, objectives, constraints, config)
    
    return trainer

if __name__ == "__main__":
    # 测试耦合优化
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    
    # 创建配置
    config = CoupledOptimizationConfig(
        optimization_type=OptimizationType.WEIGHTED_SUM,
        max_iterations=100,
        tolerance=1e-4,
        adaptive_weights=True
    )
    
    # 创建训练器
    trainer = create_coupled_optimization_trainer(
        model=model,
        objective_types=['physics', 'data', 'regularization'],
        constraint_types=['equality', 'inequality'],
        config=config
    )
    
    # 设置优化器
    trainer.setup_optimizer(learning_rate=1e-3)
    
    # 生成测试数据
    def generate_test_data(batch_size=32, num_batches=10):
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, 3)
            targets = torch.sin(inputs.sum(dim=1, keepdim=True)).repeat(1, 3)
            
            # 添加物理约束信息
            kwargs = {
                'physics_residual': torch.randn(batch_size, 3) * 0.1,
                'boundary_residual': torch.randn(batch_size, 3) * 0.05,
                'constraint_value': torch.sigmoid(inputs)  # 确保在[0,1]范围内
            }
            
            yield inputs, targets, kwargs
    
    # 训练几个epoch
    print("=== 耦合优化训练测试 ===")
    
    for epoch in range(5):
        data_loader = generate_test_data()
        metrics = trainer.train_epoch(data_loader, epoch)
        
        print(f"\nEpoch {epoch}:")
        print(f"  总损失: {metrics['total_loss']:.6f}")
        print(f"  目标函数: {metrics['objectives']}")
        print(f"  约束违反: {metrics['constraints']}")
        
        convergence = metrics['convergence']
        print(f"  收敛状态: {convergence['converged']}")
        if convergence['converged']:
            print("  优化已收敛！")
            break
    
    # 获取优化总结
    summary = trainer.get_optimization_summary()
    print(f"\n=== 优化总结 ===")
    print(f"优化类型: {summary['optimization_type']}")
    print(f"总训练轮数: {summary['total_epochs']}")
    print(f"最终损失: {summary['final_loss']:.6f}")
    print(f"损失改进: {summary.get('loss_improvement', 0):.2%}")
    print(f"收敛状态: {summary['convergence_status']}")
    
    print("\n耦合优化训练测试完成！")