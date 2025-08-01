#!/usr/bin/env python3
"""
约束管理器模块

统一管理物理约束和数据约束，提供约束组合、
权重调整、损失计算等功能。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from .physics_constraints import (
    PhysicsConstraint, PhysicsConstraintManager,
    create_physics_constraint
)
from .data_constraints import (
    DataConstraint, DataConstraintManager,
    create_data_constraint
)

class UnifiedConstraintManager:
    """
    统一约束管理器
    
    管理物理约束和数据约束的组合，提供：
    - 约束权重自适应调整
    - 多目标损失平衡
    - 约束满足度监控
    - 训练阶段约束调度
    """
    
    def __init__(self, 
                 physics_constraints: List[PhysicsConstraint],
                 data_constraints: List[DataConstraint],
                 constraint_config: Dict[str, Any]):
        """
        初始化统一约束管理器
        
        Args:
            physics_constraints: 物理约束列表
            data_constraints: 数据约束列表
            constraint_config: 约束配置参数
        """
        self.physics_manager = PhysicsConstraintManager(physics_constraints)
        self.data_manager = DataConstraintManager(data_constraints)
        
        self.constraint_config = constraint_config
        
        # 权重调整参数
        self.physics_weight = constraint_config.get('physics_weight', 1.0)
        self.data_weight = constraint_config.get('data_weight', 1.0)
        self.adaptive_weighting = constraint_config.get('adaptive_weighting', True)
        
        # 权重调整策略
        self.weighting_strategy = constraint_config.get('weighting_strategy', 'gradnorm')
        self.weight_update_frequency = constraint_config.get('weight_update_frequency', 100)
        
        # 约束调度参数
        self.constraint_scheduling = constraint_config.get('constraint_scheduling', False)
        self.scheduling_strategy = constraint_config.get('scheduling_strategy', 'curriculum')
        
        # 历史记录
        self.loss_history = {
            'physics_losses': [],
            'data_losses': [],
            'total_losses': [],
            'constraint_weights': []
        }
        
        self.step_count = 0
        
    def compute_total_loss(self, 
                          model_outputs: Dict[str, jnp.ndarray],
                          inputs: Dict[str, jnp.ndarray],
                          observed_data: Dict[str, jnp.ndarray],
                          model_fn: Callable,
                          data_weights: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """
        计算总的约束损失
        
        Args:
            model_outputs: 模型输出
            inputs: 输入变量
            observed_data: 观测数据
            model_fn: 模型函数
            data_weights: 数据权重
            
        Returns:
            损失字典
        """
        # 计算物理约束损失
        physics_loss = self.physics_manager.compute_total_loss(
            model_outputs, inputs, model_fn
        )
        
        # 计算数据约束损失
        data_loss = self.data_manager.compute_total_data_loss(
            model_outputs, observed_data, data_weights
        )
        
        # 应用权重
        weighted_physics_loss = self.physics_weight * physics_loss
        weighted_data_loss = self.data_weight * data_loss
        
        # 总损失
        total_loss = weighted_physics_loss + weighted_data_loss
        
        # 记录损失历史
        self.loss_history['physics_losses'].append(float(physics_loss))
        self.loss_history['data_losses'].append(float(data_loss))
        self.loss_history['total_losses'].append(float(total_loss))
        self.loss_history['constraint_weights'].append({
            'physics_weight': float(self.physics_weight),
            'data_weight': float(self.data_weight)
        })
        
        self.step_count += 1
        
        # 自适应权重调整
        if (self.adaptive_weighting and 
            self.step_count % self.weight_update_frequency == 0):
            self._update_constraint_weights(
                physics_loss, data_loss, model_outputs, inputs, observed_data, model_fn
            )
            
        return {
            'total_loss': total_loss,
            'physics_loss': physics_loss,
            'data_loss': data_loss,
            'weighted_physics_loss': weighted_physics_loss,
            'weighted_data_loss': weighted_data_loss
        }
        
    def compute_detailed_losses(self, 
                               model_outputs: Dict[str, jnp.ndarray],
                               inputs: Dict[str, jnp.ndarray],
                               observed_data: Dict[str, jnp.ndarray],
                               model_fn: Callable,
                               data_weights: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """
        计算详细的约束损失
        
        Args:
            model_outputs: 模型输出
            inputs: 输入变量
            observed_data: 观测数据
            model_fn: 模型函数
            data_weights: 数据权重
            
        Returns:
            详细损失字典
        """
        # 获取各个物理约束的损失
        physics_losses = self.physics_manager.compute_individual_losses(
            model_outputs, inputs, model_fn
        )
        
        # 获取各个数据约束的损失
        data_losses = self.data_manager.compute_individual_data_losses(
            model_outputs, observed_data, data_weights
        )
        
        # 合并损失字典
        detailed_losses = {}
        detailed_losses.update(physics_losses)
        detailed_losses.update(data_losses)
        
        # 添加总损失
        total_losses = self.compute_total_loss(
            model_outputs, inputs, observed_data, model_fn, data_weights
        )
        detailed_losses.update(total_losses)
        
        return detailed_losses
        
    def _update_constraint_weights(self, 
                                  physics_loss: float,
                                  data_loss: float,
                                  model_outputs: Dict[str, jnp.ndarray],
                                  inputs: Dict[str, jnp.ndarray],
                                  observed_data: Dict[str, jnp.ndarray],
                                  model_fn: Callable):
        """
        更新约束权重
        
        Args:
            physics_loss: 物理损失
            data_loss: 数据损失
            model_outputs: 模型输出
            inputs: 输入变量
            observed_data: 观测数据
            model_fn: 模型函数
        """
        if self.weighting_strategy == 'gradnorm':
            self._update_weights_gradnorm(
                physics_loss, data_loss, model_outputs, inputs, observed_data, model_fn
            )
        elif self.weighting_strategy == 'uncertainty':
            self._update_weights_uncertainty(physics_loss, data_loss)
        elif self.weighting_strategy == 'adaptive':
            self._update_weights_adaptive(physics_loss, data_loss)
        elif self.weighting_strategy == 'curriculum':
            self._update_weights_curriculum()
            
    def _update_weights_gradnorm(self, 
                                physics_loss: float,
                                data_loss: float,
                                model_outputs: Dict[str, jnp.ndarray],
                                inputs: Dict[str, jnp.ndarray],
                                observed_data: Dict[str, jnp.ndarray],
                                model_fn: Callable):
        """
        基于GradNorm的权重更新
        """
        # 简化的GradNorm实现
        # 计算损失梯度的范数
        
        # 这里需要实际的梯度计算，简化处理
        physics_grad_norm = jnp.sqrt(physics_loss + 1e-8)
        data_grad_norm = jnp.sqrt(data_loss + 1e-8)
        
        # 计算相对梯度范数
        total_grad_norm = physics_grad_norm + data_grad_norm
        
        if total_grad_norm > 0:
            # 调整权重以平衡梯度
            target_physics_ratio = 0.5  # 目标比例
            current_physics_ratio = physics_grad_norm / total_grad_norm
            
            # 权重调整
            adjustment_factor = 0.1  # 调整强度
            
            if current_physics_ratio > target_physics_ratio:
                self.physics_weight *= (1 - adjustment_factor)
                self.data_weight *= (1 + adjustment_factor)
            else:
                self.physics_weight *= (1 + adjustment_factor)
                self.data_weight *= (1 - adjustment_factor)
                
            # 确保权重为正
            self.physics_weight = max(self.physics_weight, 0.01)
            self.data_weight = max(self.data_weight, 0.01)
            
    def _update_weights_uncertainty(self, physics_loss: float, data_loss: float):
        """
        基于不确定性的权重更新
        """
        # 计算损失的不确定性（基于历史方差）
        if len(self.loss_history['physics_losses']) > 10:
            physics_losses = jnp.array(self.loss_history['physics_losses'][-10:])
            data_losses = jnp.array(self.loss_history['data_losses'][-10:])
            
            physics_uncertainty = jnp.std(physics_losses)
            data_uncertainty = jnp.std(data_losses)
            
            # 不确定性高的损失获得更高权重
            total_uncertainty = physics_uncertainty + data_uncertainty
            
            if total_uncertainty > 0:
                self.physics_weight = physics_uncertainty / total_uncertainty
                self.data_weight = data_uncertainty / total_uncertainty
                
                # 归一化权重
                total_weight = self.physics_weight + self.data_weight
                self.physics_weight /= total_weight
                self.data_weight /= total_weight
                
    def _update_weights_adaptive(self, physics_loss: float, data_loss: float):
        """
        自适应权重更新
        """
        # 基于损失比例的自适应调整
        total_loss = physics_loss + data_loss
        
        if total_loss > 0:
            physics_ratio = physics_loss / total_loss
            data_ratio = data_loss / total_loss
            
            # 目标比例（可配置）
            target_physics_ratio = self.constraint_config.get('target_physics_ratio', 0.3)
            target_data_ratio = 1.0 - target_physics_ratio
            
            # 权重调整
            learning_rate = 0.05
            
            physics_adjustment = learning_rate * (target_physics_ratio - physics_ratio)
            data_adjustment = learning_rate * (target_data_ratio - data_ratio)
            
            self.physics_weight += physics_adjustment
            self.data_weight += data_adjustment
            
            # 确保权重为正
            self.physics_weight = max(self.physics_weight, 0.01)
            self.data_weight = max(self.data_weight, 0.01)
            
    def _update_weights_curriculum(self):
        """
        课程学习权重更新
        """
        # 基于训练阶段的权重调度
        curriculum_stages = self.constraint_config.get('curriculum_stages', {
            'stage1': {'steps': 1000, 'physics_weight': 0.1, 'data_weight': 1.0},
            'stage2': {'steps': 2000, 'physics_weight': 0.5, 'data_weight': 1.0},
            'stage3': {'steps': float('inf'), 'physics_weight': 1.0, 'data_weight': 1.0}
        })
        
        cumulative_steps = 0
        for stage_name, stage_config in curriculum_stages.items():
            cumulative_steps += stage_config['steps']
            
            if self.step_count <= cumulative_steps:
                self.physics_weight = stage_config['physics_weight']
                self.data_weight = stage_config['data_weight']
                break
                
    def check_constraint_satisfaction(self, 
                                    model_outputs: Dict[str, jnp.ndarray],
                                    inputs: Dict[str, jnp.ndarray],
                                    observed_data: Dict[str, jnp.ndarray],
                                    model_fn: Callable) -> Dict[str, Any]:
        """
        检查约束满足情况
        
        Args:
            model_outputs: 模型输出
            inputs: 输入变量
            observed_data: 观测数据
            model_fn: 模型函数
            
        Returns:
            约束满足情况报告
        """
        # 检查物理约束满足情况
        physics_satisfaction = self.physics_manager.check_all_constraints(
            model_outputs, inputs, model_fn
        )
        
        # 计算数据拟合质量
        data_losses = self.data_manager.compute_individual_data_losses(
            model_outputs, observed_data
        )
        
        # 数据拟合质量评估（基于损失阈值）
        data_satisfaction = {}
        for constraint_name, loss_value in data_losses.items():
            # 简单的阈值判断
            threshold = self.constraint_config.get(f'{constraint_name}_threshold', 1.0)
            data_satisfaction[constraint_name] = loss_value < threshold
            
        # 综合报告
        satisfaction_report = {
            'physics_constraints': physics_satisfaction,
            'data_constraints': data_satisfaction,
            'overall_physics_satisfaction': all(physics_satisfaction.values()),
            'overall_data_satisfaction': all(data_satisfaction.values()),
            'constraint_weights': {
                'physics_weight': self.physics_weight,
                'data_weight': self.data_weight
            },
            'step_count': self.step_count
        }
        
        return satisfaction_report
        
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """
        获取约束统计信息
        
        Returns:
            约束统计信息
        """
        if not self.loss_history['total_losses']:
            return {}
            
        physics_losses = jnp.array(self.loss_history['physics_losses'])
        data_losses = jnp.array(self.loss_history['data_losses'])
        total_losses = jnp.array(self.loss_history['total_losses'])
        
        statistics = {
            'loss_statistics': {
                'physics_loss': {
                    'mean': float(jnp.mean(physics_losses)),
                    'std': float(jnp.std(physics_losses)),
                    'min': float(jnp.min(physics_losses)),
                    'max': float(jnp.max(physics_losses)),
                    'current': float(physics_losses[-1])
                },
                'data_loss': {
                    'mean': float(jnp.mean(data_losses)),
                    'std': float(jnp.std(data_losses)),
                    'min': float(jnp.min(data_losses)),
                    'max': float(jnp.max(data_losses)),
                    'current': float(data_losses[-1])
                },
                'total_loss': {
                    'mean': float(jnp.mean(total_losses)),
                    'std': float(jnp.std(total_losses)),
                    'min': float(jnp.min(total_losses)),
                    'max': float(jnp.max(total_losses)),
                    'current': float(total_losses[-1])
                }
            },
            'weight_statistics': {
                'current_physics_weight': self.physics_weight,
                'current_data_weight': self.data_weight,
                'weight_history': self.loss_history['constraint_weights'][-10:]  # 最近10步
            },
            'training_progress': {
                'total_steps': self.step_count,
                'convergence_trend': self._compute_convergence_trend()
            }
        }
        
        return statistics
        
    def _compute_convergence_trend(self) -> Dict[str, float]:
        """
        计算收敛趋势
        
        Returns:
            收敛趋势指标
        """
        if len(self.loss_history['total_losses']) < 10:
            return {'trend': 0.0, 'stability': 0.0}
            
        recent_losses = jnp.array(self.loss_history['total_losses'][-10:])
        
        # 计算趋势（线性回归斜率）
        x = jnp.arange(len(recent_losses))
        trend = jnp.corrcoef(x, recent_losses)[0, 1]
        
        # 计算稳定性（变异系数）
        stability = jnp.std(recent_losses) / (jnp.mean(recent_losses) + 1e-8)
        
        return {
            'trend': float(trend),  # 负值表示下降趋势
            'stability': float(stability)  # 越小越稳定
        }
        
    def reset_history(self):
        """
        重置历史记录
        """
        self.loss_history = {
            'physics_losses': [],
            'data_losses': [],
            'total_losses': [],
            'constraint_weights': []
        }
        self.step_count = 0
        
    def save_constraint_state(self, filepath: str):
        """
        保存约束状态
        
        Args:
            filepath: 保存路径
        """
        import pickle
        
        state = {
            'constraint_config': self.constraint_config,
            'physics_weight': self.physics_weight,
            'data_weight': self.data_weight,
            'loss_history': self.loss_history,
            'step_count': self.step_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load_constraint_state(self, filepath: str):
        """
        加载约束状态
        
        Args:
            filepath: 加载路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.constraint_config.update(state['constraint_config'])
        self.physics_weight = state['physics_weight']
        self.data_weight = state['data_weight']
        self.loss_history = state['loss_history']
        self.step_count = state['step_count']

def create_unified_constraint_manager(
    physics_constraint_configs: List[Dict[str, Any]],
    data_constraint_configs: List[Dict[str, Any]],
    manager_config: Dict[str, Any]
) -> UnifiedConstraintManager:
    """
    工厂函数：创建统一约束管理器
    
    Args:
        physics_constraint_configs: 物理约束配置列表
        data_constraint_configs: 数据约束配置列表
        manager_config: 管理器配置
        
    Returns:
        统一约束管理器实例
    """
    # 创建物理约束
    physics_constraints = []
    for config in physics_constraint_configs:
        constraint_type = config['type']
        constraint = create_physics_constraint(constraint_type, config)
        physics_constraints.append(constraint)
        
    # 创建数据约束
    data_constraints = []
    for config in data_constraint_configs:
        constraint_type = config['type']
        constraint = create_data_constraint(constraint_type, config)
        data_constraints.append(constraint)
        
    # 创建统一管理器
    manager = UnifiedConstraintManager(
        physics_constraints, data_constraints, manager_config
    )
    
    return manager

class ConstraintScheduler:
    """
    约束调度器
    
    管理训练过程中约束的动态调整
    """
    
    def __init__(self, scheduler_config: Dict[str, Any]):
        """
        初始化约束调度器
        
        Args:
            scheduler_config: 调度器配置
        """
        self.scheduler_config = scheduler_config
        self.scheduling_strategy = scheduler_config.get('strategy', 'linear')
        self.total_steps = scheduler_config.get('total_steps', 10000)
        
    def get_constraint_weights(self, current_step: int) -> Dict[str, float]:
        """
        获取当前步骤的约束权重
        
        Args:
            current_step: 当前训练步骤
            
        Returns:
            约束权重字典
        """
        progress = min(current_step / self.total_steps, 1.0)
        
        if self.scheduling_strategy == 'linear':
            return self._linear_scheduling(progress)
        elif self.scheduling_strategy == 'exponential':
            return self._exponential_scheduling(progress)
        elif self.scheduling_strategy == 'cosine':
            return self._cosine_scheduling(progress)
        elif self.scheduling_strategy == 'curriculum':
            return self._curriculum_scheduling(current_step)
        else:
            return {'physics_weight': 1.0, 'data_weight': 1.0}
            
    def _linear_scheduling(self, progress: float) -> Dict[str, float]:
        """
        线性调度
        """
        initial_physics = self.scheduler_config.get('initial_physics_weight', 0.1)
        final_physics = self.scheduler_config.get('final_physics_weight', 1.0)
        
        physics_weight = initial_physics + progress * (final_physics - initial_physics)
        data_weight = 1.0  # 保持数据权重不变
        
        return {'physics_weight': physics_weight, 'data_weight': data_weight}
        
    def _exponential_scheduling(self, progress: float) -> Dict[str, float]:
        """
        指数调度
        """
        initial_physics = self.scheduler_config.get('initial_physics_weight', 0.1)
        final_physics = self.scheduler_config.get('final_physics_weight', 1.0)
        decay_rate = self.scheduler_config.get('decay_rate', 2.0)
        
        physics_weight = initial_physics + (final_physics - initial_physics) * (1 - jnp.exp(-decay_rate * progress))
        data_weight = 1.0
        
        return {'physics_weight': float(physics_weight), 'data_weight': data_weight}
        
    def _cosine_scheduling(self, progress: float) -> Dict[str, float]:
        """
        余弦调度
        """
        initial_physics = self.scheduler_config.get('initial_physics_weight', 0.1)
        final_physics = self.scheduler_config.get('final_physics_weight', 1.0)
        
        physics_weight = initial_physics + 0.5 * (final_physics - initial_physics) * (1 + jnp.cos(jnp.pi * (1 - progress)))
        data_weight = 1.0
        
        return {'physics_weight': float(physics_weight), 'data_weight': data_weight}
        
    def _curriculum_scheduling(self, current_step: int) -> Dict[str, float]:
        """
        课程学习调度
        """
        stages = self.scheduler_config.get('curriculum_stages', [
            {'steps': 1000, 'physics_weight': 0.1, 'data_weight': 1.0},
            {'steps': 3000, 'physics_weight': 0.5, 'data_weight': 1.0},
            {'steps': float('inf'), 'physics_weight': 1.0, 'data_weight': 1.0}
        ])
        
        cumulative_steps = 0
        for stage in stages:
            cumulative_steps += stage['steps']
            if current_step <= cumulative_steps:
                return {
                    'physics_weight': stage['physics_weight'],
                    'data_weight': stage['data_weight']
                }
                
        # 默认返回最后阶段的权重
        return {
            'physics_weight': stages[-1]['physics_weight'],
            'data_weight': stages[-1]['data_weight']
        }

if __name__ == "__main__":
    # 测试代码
    print("Constraint manager module loaded successfully")
    
    # 创建示例配置
    physics_configs = [
        {'type': 'mass_conservation', 'weight': 1.0},
        {'type': 'momentum_conservation', 'weight': 0.8},
        {'type': 'constitutive_relation', 'weight': 0.6}
    ]
    
    data_configs = [
        {'type': 'velocity', 'weight': 1.0, 'measurement_error': 0.1},
        {'type': 'thickness', 'weight': 0.8, 'measurement_error': 5.0}
    ]
    
    manager_config = {
        'physics_weight': 1.0,
        'data_weight': 1.0,
        'adaptive_weighting': True,
        'weighting_strategy': 'gradnorm'
    }
    
    # 创建统一约束管理器
    constraint_manager = create_unified_constraint_manager(
        physics_configs, data_configs, manager_config
    )
    
    print(f"Created unified constraint manager with:")
    print(f"Physics constraints: {len(constraint_manager.physics_manager.constraints)}")
    print(f"Data constraints: {len(constraint_manager.data_manager.constraints)}")
    print(f"Adaptive weighting: {constraint_manager.adaptive_weighting}")
    print(f"Weighting strategy: {constraint_manager.weighting_strategy}")
    
    # 创建约束调度器
    scheduler_config = {
        'strategy': 'curriculum',
        'total_steps': 5000,
        'curriculum_stages': [
            {'steps': 1000, 'physics_weight': 0.1, 'data_weight': 1.0},
            {'steps': 2000, 'physics_weight': 0.5, 'data_weight': 1.0},
            {'steps': float('inf'), 'physics_weight': 1.0, 'data_weight': 1.0}
        ]
    }
    
    scheduler = ConstraintScheduler(scheduler_config)
    
    print(f"\nCreated constraint scheduler with strategy: {scheduler.scheduling_strategy}")
    
    # 测试调度器
    for step in [500, 1500, 3000, 5000]:
        weights = scheduler.get_constraint_weights(step)
        print(f"Step {step}: physics_weight={weights['physics_weight']:.2f}, data_weight={weights['data_weight']:.2f}")