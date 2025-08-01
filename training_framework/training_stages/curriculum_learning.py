#!/usr/bin/env python3
"""
课程学习模块

实现PINNs的课程学习策略，包括：
- 基础课程学习框架
- 难度递增策略
- 多阶段训练
- 自适应课程调整
- 冰川物理课程设计

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math

class DifficultyLevel(Enum):
    """
    难度等级枚举
    """
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

@dataclass
class CurriculumStage:
    """
    课程阶段配置
    """
    name: str
    difficulty: DifficultyLevel
    duration_epochs: int
    learning_rate: float
    loss_weights: Dict[str, float]
    constraints_active: List[str]
    data_complexity: float = 1.0
    physics_complexity: float = 1.0
    boundary_complexity: float = 1.0
    success_threshold: float = 1e-3
    min_epochs: int = 10
    max_epochs: int = 1000

class BaseCurriculumStrategy(ABC):
    """
    基础课程策略
    
    所有课程学习策略的基类
    """
    
    def __init__(self, stages: List[CurriculumStage]):
        """
        初始化课程策略
        
        Args:
            stages: 课程阶段列表
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.stage_history = []
        self.performance_history = []
    
    @abstractmethod
    def should_advance_stage(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            metrics: 性能指标
            epoch: 当前轮次
            
        Returns:
            bool: 是否进入下一阶段
        """
        pass
    
    def get_current_stage(self) -> CurriculumStage:
        """
        获取当前阶段
        
        Returns:
            CurriculumStage: 当前阶段
        """
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        else:
            return self.stages[-1]  # 返回最后一个阶段
    
    def advance_stage(self) -> bool:
        """
        进入下一阶段
        
        Returns:
            bool: 是否成功进入下一阶段
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.stage_history.append(self.current_stage_idx)
            self.current_stage_idx += 1
            return True
        return False
    
    def reset_to_stage(self, stage_idx: int):
        """
        重置到指定阶段
        
        Args:
            stage_idx: 阶段索引
        """
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
    
    def is_completed(self) -> bool:
        """
        检查课程是否完成
        
        Returns:
            bool: 是否完成
        """
        return self.current_stage_idx >= len(self.stages) - 1
    
    def get_progress(self) -> float:
        """
        获取课程进度
        
        Returns:
            float: 进度百分比 (0-1)
        """
        return (self.current_stage_idx + 1) / len(self.stages)

class ProgressiveCurriculum(BaseCurriculumStrategy):
    """
    渐进式课程学习
    
    基于性能指标逐步增加难度
    """
    
    def __init__(self, stages: List[CurriculumStage], patience: int = 10):
        """
        初始化渐进式课程
        
        Args:
            stages: 课程阶段列表
            patience: 耐心值（连续多少轮无改善后考虑进入下一阶段）
        """
        super().__init__(stages)
        self.patience = patience
        self.no_improvement_count = 0
        self.best_loss = float('inf')
    
    def should_advance_stage(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            metrics: 性能指标
            epoch: 当前轮次
            
        Returns:
            bool: 是否进入下一阶段
        """
        current_stage = self.get_current_stage()
        current_loss = metrics.get('total_loss', float('inf'))
        
        # 记录性能历史
        self.performance_history.append({
            'epoch': epoch,
            'stage': self.current_stage_idx,
            'metrics': metrics.copy()
        })
        
        # 检查最小轮次要求
        stage_epochs = len([h for h in self.performance_history 
                           if h['stage'] == self.current_stage_idx])
        
        if stage_epochs < current_stage.min_epochs:
            return False
        
        # 检查成功阈值
        if current_loss < current_stage.success_threshold:
            return True
        
        # 检查最大轮次
        if stage_epochs >= current_stage.max_epochs:
            return True
        
        # 检查改善情况
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 如果长时间无改善，考虑进入下一阶段
        if self.no_improvement_count >= self.patience:
            return True
        
        return False

class AdaptiveCurriculum(BaseCurriculumStrategy):
    """
    自适应课程学习
    
    根据学习进度动态调整课程
    """
    
    def __init__(self, stages: List[CurriculumStage], 
                 adaptation_rate: float = 0.1):
        """
        初始化自适应课程
        
        Args:
            stages: 课程阶段列表
            adaptation_rate: 自适应率
        """
        super().__init__(stages)
        self.adaptation_rate = adaptation_rate
        self.learning_rates = []
        self.difficulty_adjustments = []
    
    def should_advance_stage(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            metrics: 性能指标
            epoch: 当前轮次
            
        Returns:
            bool: 是否进入下一阶段
        """
        current_stage = self.get_current_stage()
        
        # 记录性能历史
        self.performance_history.append({
            'epoch': epoch,
            'stage': self.current_stage_idx,
            'metrics': metrics.copy()
        })
        
        # 自适应调整当前阶段参数
        self._adapt_current_stage(metrics)
        
        # 检查是否满足进入下一阶段的条件
        stage_epochs = len([h for h in self.performance_history 
                           if h['stage'] == self.current_stage_idx])
        
        if stage_epochs < current_stage.min_epochs:
            return False
        
        current_loss = metrics.get('total_loss', float('inf'))
        
        # 动态调整成功阈值
        adapted_threshold = self._get_adapted_threshold()
        
        if current_loss < adapted_threshold or stage_epochs >= current_stage.max_epochs:
            return True
        
        return False
    
    def _adapt_current_stage(self, metrics: Dict[str, float]):
        """
        自适应调整当前阶段参数
        
        Args:
            metrics: 性能指标
        """
        current_stage = self.get_current_stage()
        
        # 根据损失趋势调整学习率
        if len(self.performance_history) > 5:
            recent_losses = [h['metrics'].get('total_loss', float('inf')) 
                           for h in self.performance_history[-5:]]
            
            loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            
            if loss_trend > 0:  # 损失增加
                current_stage.learning_rate *= (1 - self.adaptation_rate)
            elif loss_trend < -0.01:  # 损失显著减少
                current_stage.learning_rate *= (1 + self.adaptation_rate * 0.5)
            
            # 限制学习率范围
            current_stage.learning_rate = max(1e-6, min(1e-1, current_stage.learning_rate))
        
        self.learning_rates.append(current_stage.learning_rate)
    
    def _get_adapted_threshold(self) -> float:
        """
        获取自适应阈值
        
        Returns:
            float: 自适应阈值
        """
        current_stage = self.get_current_stage()
        base_threshold = current_stage.success_threshold
        
        # 根据历史性能调整阈值
        if len(self.performance_history) > 10:
            recent_losses = [h['metrics'].get('total_loss', float('inf')) 
                           for h in self.performance_history[-10:]]
            
            avg_recent_loss = np.mean(recent_losses)
            
            # 如果平均损失远高于阈值，放宽阈值
            if avg_recent_loss > base_threshold * 10:
                return base_threshold * 5
            elif avg_recent_loss > base_threshold * 5:
                return base_threshold * 2
        
        return base_threshold

class GlacierPhysicsCurriculum:
    """
    冰川物理课程设计
    
    专门为冰川物理问题设计的课程学习策略
    """
    
    def __init__(self):
        """
        初始化冰川物理课程
        """
        self.stages = self._create_glacier_curriculum()
        self.strategy = ProgressiveCurriculum(self.stages, patience=15)
    
    def _create_glacier_curriculum(self) -> List[CurriculumStage]:
        """
        创建冰川物理课程阶段
        
        Returns:
            List[CurriculumStage]: 课程阶段列表
        """
        stages = []
        
        # 阶段1: 基础几何学习
        stage1 = CurriculumStage(
            name="基础几何",
            difficulty=DifficultyLevel.EASY,
            duration_epochs=100,
            learning_rate=1e-2,
            loss_weights={
                'data': 1.0,
                'physics': 0.1,
                'boundary': 0.5
            },
            constraints_active=['surface_boundary', 'bed_boundary'],
            data_complexity=0.3,
            physics_complexity=0.1,
            boundary_complexity=0.5,
            success_threshold=1e-2,
            min_epochs=50,
            max_epochs=200
        )
        stages.append(stage1)
        
        # 阶段2: 简单流动
        stage2 = CurriculumStage(
            name="简单流动",
            difficulty=DifficultyLevel.MEDIUM,
            duration_epochs=150,
            learning_rate=5e-3,
            loss_weights={
                'data': 1.0,
                'physics': 0.3,
                'boundary': 0.8
            },
            constraints_active=['surface_boundary', 'bed_boundary', 'momentum_balance'],
            data_complexity=0.5,
            physics_complexity=0.3,
            boundary_complexity=0.8,
            success_threshold=5e-3,
            min_epochs=75,
            max_epochs=300
        )
        stages.append(stage2)
        
        # 阶段3: 质量平衡
        stage3 = CurriculumStage(
            name="质量平衡",
            difficulty=DifficultyLevel.MEDIUM,
            duration_epochs=200,
            learning_rate=2e-3,
            loss_weights={
                'data': 1.0,
                'physics': 0.6,
                'boundary': 1.0
            },
            constraints_active=[
                'surface_boundary', 'bed_boundary', 
                'momentum_balance', 'mass_balance'
            ],
            data_complexity=0.7,
            physics_complexity=0.6,
            boundary_complexity=1.0,
            success_threshold=2e-3,
            min_epochs=100,
            max_epochs=400
        )
        stages.append(stage3)
        
        # 阶段4: Glen流动定律
        stage4 = CurriculumStage(
            name="Glen流动定律",
            difficulty=DifficultyLevel.HARD,
            duration_epochs=250,
            learning_rate=1e-3,
            loss_weights={
                'data': 1.0,
                'physics': 0.8,
                'boundary': 1.0
            },
            constraints_active=[
                'surface_boundary', 'bed_boundary',
                'momentum_balance', 'mass_balance', 'glen_flow_law'
            ],
            data_complexity=0.8,
            physics_complexity=0.8,
            boundary_complexity=1.0,
            success_threshold=1e-3,
            min_epochs=125,
            max_epochs=500
        )
        stages.append(stage4)
        
        # 阶段5: 完整物理
        stage5 = CurriculumStage(
            name="完整物理",
            difficulty=DifficultyLevel.EXPERT,
            duration_epochs=300,
            learning_rate=5e-4,
            loss_weights={
                'data': 1.0,
                'physics': 1.0,
                'boundary': 1.0,
                'regularization': 0.1
            },
            constraints_active=[
                'surface_boundary', 'bed_boundary',
                'momentum_balance', 'mass_balance', 
                'glen_flow_law', 'temperature_evolution'
            ],
            data_complexity=1.0,
            physics_complexity=1.0,
            boundary_complexity=1.0,
            success_threshold=5e-4,
            min_epochs=150,
            max_epochs=600
        )
        stages.append(stage5)
        
        return stages
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """
        获取当前阶段配置
        
        Returns:
            Dict: 当前配置
        """
        current_stage = self.strategy.get_current_stage()
        
        return {
            'stage_name': current_stage.name,
            'difficulty': current_stage.difficulty.name,
            'learning_rate': current_stage.learning_rate,
            'loss_weights': current_stage.loss_weights,
            'constraints_active': current_stage.constraints_active,
            'complexity_factors': {
                'data': current_stage.data_complexity,
                'physics': current_stage.physics_complexity,
                'boundary': current_stage.boundary_complexity
            },
            'success_threshold': current_stage.success_threshold
        }
    
    def update_training(self, metrics: Dict[str, float], epoch: int) -> Dict[str, Any]:
        """
        更新训练配置
        
        Args:
            metrics: 性能指标
            epoch: 当前轮次
            
        Returns:
            Dict: 更新信息
        """
        old_stage_idx = self.strategy.current_stage_idx
        
        # 检查是否应该进入下一阶段
        if self.strategy.should_advance_stage(metrics, epoch):
            advanced = self.strategy.advance_stage()
            
            if advanced:
                new_stage = self.strategy.get_current_stage()
                return {
                    'stage_changed': True,
                    'old_stage': old_stage_idx,
                    'new_stage': self.strategy.current_stage_idx,
                    'new_stage_name': new_stage.name,
                    'new_configuration': self.get_current_configuration()
                }
        
        return {
            'stage_changed': False,
            'current_stage': self.strategy.current_stage_idx,
            'current_configuration': self.get_current_configuration()
        }
    
    def get_progress_report(self) -> str:
        """
        生成进度报告
        
        Returns:
            str: 进度报告
        """
        current_stage = self.strategy.get_current_stage()
        progress = self.strategy.get_progress()
        
        report = f"冰川物理课程学习进度报告\n"
        report += "=" * 40 + "\n"
        report += f"当前阶段: {current_stage.name} ({current_stage.difficulty.name})\n"
        report += f"总体进度: {progress:.1%}\n"
        report += f"阶段进度: {self.strategy.current_stage_idx + 1}/{len(self.stages)}\n"
        
        if self.strategy.performance_history:
            recent_metrics = self.strategy.performance_history[-1]['metrics']
            report += f"\n最新性能指标:\n"
            for key, value in recent_metrics.items():
                report += f"  {key}: {value:.2e}\n"
        
        report += f"\n当前配置:\n"
        config = self.get_current_configuration()
        report += f"  学习率: {config['learning_rate']:.2e}\n"
        report += f"  损失权重: {config['loss_weights']}\n"
        report += f"  激活约束: {len(config['constraints_active'])} 个\n"
        
        return report

class CurriculumManager:
    """
    课程管理器
    
    管理和协调不同的课程学习策略
    """
    
    def __init__(self, curriculum_type: str = 'glacier_physics'):
        """
        初始化课程管理器
        
        Args:
            curriculum_type: 课程类型
        """
        self.curriculum_type = curriculum_type
        self.curriculum = self._create_curriculum(curriculum_type)
        self.training_history = []
        self.stage_transitions = []
    
    def _create_curriculum(self, curriculum_type: str):
        """
        创建课程
        
        Args:
            curriculum_type: 课程类型
            
        Returns:
            课程实例
        """
        if curriculum_type == 'glacier_physics':
            return GlacierPhysicsCurriculum()
        else:
            raise ValueError(f"未知的课程类型: {curriculum_type}")
    
    def step(self, metrics: Dict[str, float], epoch: int) -> Dict[str, Any]:
        """
        执行一步课程学习
        
        Args:
            metrics: 性能指标
            epoch: 当前轮次
            
        Returns:
            Dict: 更新信息
        """
        # 更新课程
        update_info = self.curriculum.update_training(metrics, epoch)
        
        # 记录训练历史
        self.training_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': self.curriculum.strategy.current_stage_idx,
            'configuration': self.curriculum.get_current_configuration()
        })
        
        # 记录阶段转换
        if update_info.get('stage_changed', False):
            self.stage_transitions.append({
                'epoch': epoch,
                'from_stage': update_info['old_stage'],
                'to_stage': update_info['new_stage'],
                'metrics_at_transition': metrics.copy()
            })
        
        return update_info
    
    def get_current_training_config(self) -> Dict[str, Any]:
        """
        获取当前训练配置
        
        Returns:
            Dict: 训练配置
        """
        return self.curriculum.get_current_configuration()
    
    def is_curriculum_completed(self) -> bool:
        """
        检查课程是否完成
        
        Returns:
            bool: 是否完成
        """
        return self.curriculum.strategy.is_completed()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            Dict: 训练统计
        """
        if not self.training_history:
            return {}
        
        total_epochs = len(self.training_history)
        n_stages = len(self.curriculum.stages)
        n_transitions = len(self.stage_transitions)
        
        # 计算每个阶段的平均轮次
        stage_epochs = {}
        for entry in self.training_history:
            stage = entry['stage']
            stage_epochs[stage] = stage_epochs.get(stage, 0) + 1
        
        # 计算损失趋势
        losses = [entry['metrics'].get('total_loss', 0) for entry in self.training_history]
        loss_improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
        
        return {
            'total_epochs': total_epochs,
            'total_stages': n_stages,
            'completed_transitions': n_transitions,
            'current_stage': self.curriculum.strategy.current_stage_idx,
            'progress': self.curriculum.strategy.get_progress(),
            'stage_epochs': stage_epochs,
            'loss_improvement': loss_improvement,
            'average_epochs_per_stage': total_epochs / max(1, n_transitions + 1)
        }
    
    def generate_curriculum_report(self) -> str:
        """
        生成课程报告
        
        Returns:
            str: 课程报告
        """
        report = self.curriculum.get_progress_report()
        
        # 添加统计信息
        stats = self.get_training_statistics()
        if stats:
            report += "\n" + "=" * 40 + "\n"
            report += "训练统计:\n"
            report += f"  总轮次: {stats['total_epochs']}\n"
            report += f"  阶段转换: {stats['completed_transitions']}\n"
            report += f"  损失改善: {stats['loss_improvement']:.1%}\n"
            report += f"  平均每阶段轮次: {stats['average_epochs_per_stage']:.1f}\n"
        
        # 添加阶段转换历史
        if self.stage_transitions:
            report += "\n阶段转换历史:\n"
            for i, transition in enumerate(self.stage_transitions[-5:]):  # 最近5次转换
                report += f"  {i+1}. 轮次 {transition['epoch']}: "
                report += f"阶段 {transition['from_stage']} → {transition['to_stage']}\n"
        
        return report

if __name__ == "__main__":
    # 测试课程学习
    torch.manual_seed(42)
    
    print("测试冰川物理课程学习...")
    
    # 创建课程管理器
    manager = CurriculumManager('glacier_physics')
    
    # 模拟训练过程
    for epoch in range(100):
        # 模拟性能指标
        base_loss = 1.0 * math.exp(-epoch / 50)  # 指数衰减
        noise = 0.1 * np.random.randn()
        
        metrics = {
            'total_loss': base_loss + noise,
            'data_loss': base_loss * 0.6 + noise * 0.5,
            'physics_loss': base_loss * 0.3 + noise * 0.3,
            'boundary_loss': base_loss * 0.1 + noise * 0.2
        }
        
        # 更新课程
        update_info = manager.step(metrics, epoch)
        
        # 打印阶段转换信息
        if update_info.get('stage_changed', False):
            print(f"\n轮次 {epoch}: 阶段转换")
            print(f"  从阶段 {update_info['old_stage']} 转换到阶段 {update_info['new_stage']}")
            print(f"  新阶段: {update_info['new_stage_name']}")
            print(f"  新学习率: {update_info['new_configuration']['learning_rate']:.2e}")
        
        # 每20轮打印一次进度
        if epoch % 20 == 0:
            config = manager.get_current_training_config()
            print(f"\n轮次 {epoch}:")
            print(f"  当前阶段: {config['stage_name']}")
            print(f"  总损失: {metrics['total_loss']:.2e}")
            print(f"  学习率: {config['learning_rate']:.2e}")
    
    # 生成最终报告
    print("\n" + "=" * 50)
    print(manager.generate_curriculum_report())
    
    # 获取训练统计
    stats = manager.get_training_statistics()
    print("\n详细统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试自适应课程
    print("\n测试自适应课程...")
    
    # 创建简单的自适应课程
    simple_stages = [
        CurriculumStage(
            name="简单",
            difficulty=DifficultyLevel.EASY,
            duration_epochs=50,
            learning_rate=1e-2,
            loss_weights={'data': 1.0},
            constraints_active=[],
            success_threshold=1e-2
        ),
        CurriculumStage(
            name="复杂",
            difficulty=DifficultyLevel.HARD,
            duration_epochs=100,
            learning_rate=1e-3,
            loss_weights={'data': 1.0, 'physics': 1.0},
            constraints_active=['physics'],
            success_threshold=1e-3
        )
    ]
    
    adaptive_curriculum = AdaptiveCurriculum(simple_stages)
    
    print("自适应课程测试:")
    for epoch in range(30):
        metrics = {'total_loss': 1.0 * math.exp(-epoch / 20) + 0.05 * np.random.randn()}
        
        should_advance = adaptive_curriculum.should_advance_stage(metrics, epoch)
        
        if should_advance and not adaptive_curriculum.is_completed():
            adaptive_curriculum.advance_stage()
            current_stage = adaptive_curriculum.get_current_stage()
            print(f"  轮次 {epoch}: 进入阶段 '{current_stage.name}'")
            print(f"    学习率: {current_stage.learning_rate:.2e}")
    
    print("\n课程学习测试完成！")