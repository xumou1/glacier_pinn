#!/usr/bin/env python3
"""
多阶段训练模块

实现PINNs的多阶段训练策略，包括：
- 预训练阶段
- 微调阶段
- 联合训练阶段
- 阶段间的平滑过渡
- 冰川特定的多阶段策略

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import copy
import math
from pathlib import Path

class TrainingStage(Enum):
    """
    训练阶段枚举
    """
    PRETRAINING = "pretraining"
    PHYSICS_LEARNING = "physics_learning"
    FINE_TUNING = "fine_tuning"
    JOINT_TRAINING = "joint_training"
    REFINEMENT = "refinement"
    VALIDATION = "validation"

@dataclass
class StageConfig:
    """
    训练阶段配置
    """
    name: str
    stage_type: TrainingStage
    epochs: int
    learning_rate: float
    optimizer_type: str = "Adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    scheduler_type: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    loss_weights: Dict[str, float] = field(default_factory=dict)
    active_losses: List[str] = field(default_factory=list)
    data_sampling_strategy: str = "uniform"
    batch_size: Optional[int] = None
    gradient_clipping: Optional[float] = None
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-6
    checkpoint_frequency: int = 10
    validation_frequency: int = 5
    freeze_layers: List[str] = field(default_factory=list)
    unfreeze_layers: List[str] = field(default_factory=list)
    custom_callbacks: List[Callable] = field(default_factory=list)
    stage_specific_params: Dict[str, Any] = field(default_factory=dict)

class BaseStageTrainer(ABC):
    """
    基础阶段训练器
    
    所有阶段训练器的基类
    """
    
    def __init__(self, config: StageConfig):
        """
        初始化阶段训练器
        
        Args:
            config: 阶段配置
        """
        self.config = config
        self.training_history = []
        self.best_metrics = {}
        self.early_stopping_counter = 0
        self.should_stop_early = False
    
    @abstractmethod
    def setup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        设置训练阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 设置信息
        """
        pass
    
    @abstractmethod
    def train_epoch(self, model: nn.Module, epoch: int, **kwargs) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            model: 神经网络模型
            epoch: 当前轮次
            **kwargs: 其他参数
            
        Returns:
            Dict: 训练指标
        """
        pass
    
    @abstractmethod
    def cleanup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        清理训练阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 清理信息
        """
        pass
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        检查是否应该早停
        
        Args:
            metrics: 当前指标
            
        Returns:
            bool: 是否应该早停
        """
        if not self.config.early_stopping:
            return False
        
        # 使用总损失作为早停指标
        current_loss = metrics.get('total_loss', float('inf'))
        
        # 初始化最佳指标
        if 'total_loss' not in self.best_metrics:
            self.best_metrics['total_loss'] = current_loss
            return False
        
        # 检查是否有改善
        improvement = self.best_metrics['total_loss'] - current_loss
        
        if improvement > self.config.early_stopping_threshold:
            self.best_metrics['total_loss'] = current_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        # 判断是否应该早停
        if self.early_stopping_counter >= self.config.early_stopping_patience:
            self.should_stop_early = True
            return True
        
        return False
    
    def update_training_history(self, epoch: int, metrics: Dict[str, float]):
        """
        更新训练历史
        
        Args:
            epoch: 轮次
            metrics: 指标
        """
        self.training_history.append({
            'epoch': epoch,
            'stage': self.config.name,
            'metrics': metrics.copy()
        })
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        获取阶段总结
        
        Returns:
            Dict: 阶段总结
        """
        if not self.training_history:
            return {}
        
        # 计算统计信息
        losses = [entry['metrics'].get('total_loss', 0) for entry in self.training_history]
        
        return {
            'stage_name': self.config.name,
            'stage_type': self.config.stage_type.value,
            'total_epochs': len(self.training_history),
            'initial_loss': losses[0] if losses else 0,
            'final_loss': losses[-1] if losses else 0,
            'best_loss': min(losses) if losses else 0,
            'loss_reduction': (losses[0] - losses[-1]) / losses[0] if losses and losses[0] > 0 else 0,
            'early_stopped': self.should_stop_early,
            'best_metrics': self.best_metrics.copy()
        }

class PretrainingStage(BaseStageTrainer):
    """
    预训练阶段
    
    专注于数据拟合，不包含物理约束
    """
    
    def setup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        设置预训练阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 设置信息
        """
        # 冻结指定层
        for layer_name in self.config.freeze_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False
        
        # 解冻指定层
        for layer_name in self.config.unfreeze_layers:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
        
        return {
            'stage_type': 'pretraining',
            'frozen_layers': self.config.freeze_layers,
            'unfrozen_layers': self.config.unfreeze_layers,
            'focus': 'data_fitting'
        }
    
    def train_epoch(self, model: nn.Module, epoch: int, **kwargs) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            model: 神经网络模型
            epoch: 当前轮次
            **kwargs: 其他参数
            
        Returns:
            Dict: 训练指标
        """
        # 获取训练组件
        optimizer = kwargs.get('optimizer')
        data_loader = kwargs.get('data_loader')
        loss_function = kwargs.get('loss_function')
        
        if not all([optimizer, data_loader, loss_function]):
            raise ValueError("缺少必要的训练组件")
        
        model.train()
        epoch_losses = []
        
        for batch_idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_data['inputs'])
            
            # 计算数据损失（预训练阶段只关注数据拟合）
            data_loss = loss_function(predictions, batch_data['targets'])
            
            # 反向传播
            data_loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
            
            optimizer.step()
            
            epoch_losses.append(data_loss.item())
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        
        metrics = {
            'total_loss': avg_loss,
            'data_loss': avg_loss,
            'physics_loss': 0.0,  # 预训练阶段无物理损失
            'boundary_loss': 0.0
        }
        
        # 更新训练历史
        self.update_training_history(epoch, metrics)
        
        return metrics
    
    def cleanup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        清理预训练阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 清理信息
        """
        # 解冻所有层，为下一阶段做准备
        for param in model.parameters():
            param.requires_grad = True
        
        return {
            'stage_completed': True,
            'all_layers_unfrozen': True,
            'summary': self.get_stage_summary()
        }

class PhysicsLearningStage(BaseStageTrainer):
    """
    物理学习阶段
    
    引入物理约束，平衡数据和物理损失
    """
    
    def setup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        设置物理学习阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 设置信息
        """
        # 设置损失权重
        default_weights = {
            'data': 1.0,
            'physics': 0.1,  # 初始物理权重较小
            'boundary': 0.5
        }
        
        self.loss_weights = {**default_weights, **self.config.loss_weights}
        
        return {
            'stage_type': 'physics_learning',
            'loss_weights': self.loss_weights,
            'focus': 'physics_introduction'
        }
    
    def train_epoch(self, model: nn.Module, epoch: int, **kwargs) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            model: 神经网络模型
            epoch: 当前轮次
            **kwargs: 其他参数
            
        Returns:
            Dict: 训练指标
        """
        optimizer = kwargs.get('optimizer')
        data_loader = kwargs.get('data_loader')
        physics_loss_fn = kwargs.get('physics_loss_fn')
        boundary_loss_fn = kwargs.get('boundary_loss_fn')
        data_loss_fn = kwargs.get('data_loss_fn')
        
        model.train()
        epoch_losses = {'data': [], 'physics': [], 'boundary': [], 'total': []}
        
        for batch_idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_data['inputs'])
            
            # 计算各种损失
            data_loss = data_loss_fn(predictions, batch_data['targets']) if data_loss_fn else 0
            physics_loss = physics_loss_fn(model, batch_data) if physics_loss_fn else 0
            boundary_loss = boundary_loss_fn(model, batch_data) if boundary_loss_fn else 0
            
            # 加权总损失
            total_loss = (self.loss_weights['data'] * data_loss + 
                         self.loss_weights['physics'] * physics_loss + 
                         self.loss_weights['boundary'] * boundary_loss)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
            
            optimizer.step()
            
            # 记录损失
            epoch_losses['data'].append(data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss)
            epoch_losses['physics'].append(physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss)
            epoch_losses['boundary'].append(boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss)
            epoch_losses['total'].append(total_loss.item())
        
        # 计算平均损失
        metrics = {
            'total_loss': np.mean(epoch_losses['total']),
            'data_loss': np.mean(epoch_losses['data']),
            'physics_loss': np.mean(epoch_losses['physics']),
            'boundary_loss': np.mean(epoch_losses['boundary'])
        }
        
        # 动态调整物理损失权重
        self._adjust_physics_weight(epoch, metrics)
        
        # 更新训练历史
        self.update_training_history(epoch, metrics)
        
        return metrics
    
    def _adjust_physics_weight(self, epoch: int, metrics: Dict[str, float]):
        """
        动态调整物理损失权重
        
        Args:
            epoch: 当前轮次
            metrics: 当前指标
        """
        # 逐渐增加物理损失权重
        max_physics_weight = self.config.stage_specific_params.get('max_physics_weight', 1.0)
        weight_increase_rate = self.config.stage_specific_params.get('weight_increase_rate', 0.01)
        
        current_weight = self.loss_weights['physics']
        new_weight = min(max_physics_weight, current_weight + weight_increase_rate)
        
        self.loss_weights['physics'] = new_weight
    
    def cleanup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        清理物理学习阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 清理信息
        """
        return {
            'stage_completed': True,
            'final_loss_weights': self.loss_weights.copy(),
            'summary': self.get_stage_summary()
        }

class FineTuningStage(BaseStageTrainer):
    """
    微调阶段
    
    使用较小的学习率进行精细调整
    """
    
    def setup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        设置微调阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 设置信息
        """
        # 设置平衡的损失权重
        default_weights = {
            'data': 1.0,
            'physics': 1.0,
            'boundary': 1.0
        }
        
        self.loss_weights = {**default_weights, **self.config.loss_weights}
        
        return {
            'stage_type': 'fine_tuning',
            'loss_weights': self.loss_weights,
            'focus': 'precision_optimization'
        }
    
    def train_epoch(self, model: nn.Module, epoch: int, **kwargs) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            model: 神经网络模型
            epoch: 当前轮次
            **kwargs: 其他参数
            
        Returns:
            Dict: 训练指标
        """
        optimizer = kwargs.get('optimizer')
        data_loader = kwargs.get('data_loader')
        physics_loss_fn = kwargs.get('physics_loss_fn')
        boundary_loss_fn = kwargs.get('boundary_loss_fn')
        data_loss_fn = kwargs.get('data_loss_fn')
        
        model.train()
        epoch_losses = {'data': [], 'physics': [], 'boundary': [], 'total': []}
        
        for batch_idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_data['inputs'])
            
            # 计算各种损失
            data_loss = data_loss_fn(predictions, batch_data['targets']) if data_loss_fn else 0
            physics_loss = physics_loss_fn(model, batch_data) if physics_loss_fn else 0
            boundary_loss = boundary_loss_fn(model, batch_data) if boundary_loss_fn else 0
            
            # 加权总损失
            total_loss = (self.loss_weights['data'] * data_loss + 
                         self.loss_weights['physics'] * physics_loss + 
                         self.loss_weights['boundary'] * boundary_loss)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（微调阶段使用更严格的梯度裁剪）
            clip_value = self.config.gradient_clipping or 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            # 记录损失
            epoch_losses['data'].append(data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss)
            epoch_losses['physics'].append(physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss)
            epoch_losses['boundary'].append(boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss)
            epoch_losses['total'].append(total_loss.item())
        
        # 计算平均损失
        metrics = {
            'total_loss': np.mean(epoch_losses['total']),
            'data_loss': np.mean(epoch_losses['data']),
            'physics_loss': np.mean(epoch_losses['physics']),
            'boundary_loss': np.mean(epoch_losses['boundary'])
        }
        
        # 更新训练历史
        self.update_training_history(epoch, metrics)
        
        return metrics
    
    def cleanup_stage(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        清理微调阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 其他参数
            
        Returns:
            Dict: 清理信息
        """
        return {
            'stage_completed': True,
            'model_fine_tuned': True,
            'summary': self.get_stage_summary()
        }

class GlacierMultiStageTrainer:
    """
    冰川多阶段训练器
    
    专门为冰川物理问题设计的多阶段训练策略
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化冰川多阶段训练器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.stages = self._create_glacier_stages()
        self.current_stage_idx = 0
        self.training_history = []
        self.stage_trainers = {}
        self.global_best_metrics = {}
    
    def _create_glacier_stages(self) -> List[StageConfig]:
        """
        创建冰川训练阶段
        
        Returns:
            List[StageConfig]: 阶段配置列表
        """
        stages = []
        
        # 阶段1: 几何预训练
        geometry_stage = StageConfig(
            name="几何预训练",
            stage_type=TrainingStage.PRETRAINING,
            epochs=200,
            learning_rate=1e-2,
            optimizer_type="Adam",
            optimizer_params={'betas': (0.9, 0.999), 'weight_decay': 1e-4},
            scheduler_type="StepLR",
            scheduler_params={'step_size': 50, 'gamma': 0.8},
            loss_weights={'data': 1.0},
            active_losses=['data'],
            data_sampling_strategy="uniform",
            gradient_clipping=1.0,
            early_stopping=True,
            early_stopping_patience=20,
            checkpoint_frequency=20,
            validation_frequency=10
        )
        stages.append(geometry_stage)
        
        # 阶段2: 物理引入
        physics_intro_stage = StageConfig(
            name="物理引入",
            stage_type=TrainingStage.PHYSICS_LEARNING,
            epochs=300,
            learning_rate=5e-3,
            optimizer_type="Adam",
            optimizer_params={'betas': (0.9, 0.999), 'weight_decay': 1e-4},
            scheduler_type="CosineAnnealingLR",
            scheduler_params={'T_max': 300, 'eta_min': 1e-5},
            loss_weights={'data': 1.0, 'physics': 0.1, 'boundary': 0.5},
            active_losses=['data', 'physics', 'boundary'],
            data_sampling_strategy="adaptive",
            gradient_clipping=0.5,
            early_stopping=True,
            early_stopping_patience=30,
            checkpoint_frequency=25,
            validation_frequency=15,
            stage_specific_params={
                'max_physics_weight': 1.0,
                'weight_increase_rate': 0.005
            }
        )
        stages.append(physics_intro_stage)
        
        # 阶段3: 联合训练
        joint_training_stage = StageConfig(
            name="联合训练",
            stage_type=TrainingStage.JOINT_TRAINING,
            epochs=400,
            learning_rate=2e-3,
            optimizer_type="AdamW",
            optimizer_params={'betas': (0.9, 0.999), 'weight_decay': 1e-3},
            scheduler_type="CosineAnnealingWarmRestarts",
            scheduler_params={'T_0': 50, 'T_mult': 2, 'eta_min': 1e-6},
            loss_weights={'data': 1.0, 'physics': 1.0, 'boundary': 1.0},
            active_losses=['data', 'physics', 'boundary', 'conservation'],
            data_sampling_strategy="multi_scale",
            gradient_clipping=0.3,
            early_stopping=True,
            early_stopping_patience=40,
            checkpoint_frequency=30,
            validation_frequency=20
        )
        stages.append(joint_training_stage)
        
        # 阶段4: 精细微调
        fine_tuning_stage = StageConfig(
            name="精细微调",
            stage_type=TrainingStage.FINE_TUNING,
            epochs=200,
            learning_rate=1e-4,
            optimizer_type="LBFGS",
            optimizer_params={'max_iter': 20, 'tolerance_grad': 1e-7, 'tolerance_change': 1e-9},
            loss_weights={'data': 1.0, 'physics': 1.0, 'boundary': 1.0, 'regularization': 0.1},
            active_losses=['data', 'physics', 'boundary', 'conservation', 'regularization'],
            data_sampling_strategy="uncertainty_based",
            gradient_clipping=0.1,
            early_stopping=True,
            early_stopping_patience=25,
            checkpoint_frequency=15,
            validation_frequency=10
        )
        stages.append(fine_tuning_stage)
        
        return stages
    
    def get_current_stage(self) -> StageConfig:
        """
        获取当前阶段配置
        
        Returns:
            StageConfig: 当前阶段配置
        """
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        else:
            return self.stages[-1]
    
    def create_stage_trainer(self, stage_config: StageConfig) -> BaseStageTrainer:
        """
        创建阶段训练器
        
        Args:
            stage_config: 阶段配置
            
        Returns:
            BaseStageTrainer: 阶段训练器
        """
        if stage_config.stage_type == TrainingStage.PRETRAINING:
            return PretrainingStage(stage_config)
        elif stage_config.stage_type == TrainingStage.PHYSICS_LEARNING:
            return PhysicsLearningStage(stage_config)
        elif stage_config.stage_type == TrainingStage.FINE_TUNING:
            return FineTuningStage(stage_config)
        elif stage_config.stage_type == TrainingStage.JOINT_TRAINING:
            return PhysicsLearningStage(stage_config)  # 使用物理学习阶段的逻辑
        else:
            raise ValueError(f"未知的阶段类型: {stage_config.stage_type}")
    
    def train_stage(self, model: nn.Module, stage_idx: int, **kwargs) -> Dict[str, Any]:
        """
        训练指定阶段
        
        Args:
            model: 神经网络模型
            stage_idx: 阶段索引
            **kwargs: 训练参数
            
        Returns:
            Dict: 训练结果
        """
        if stage_idx >= len(self.stages):
            raise ValueError(f"阶段索引超出范围: {stage_idx}")
        
        stage_config = self.stages[stage_idx]
        stage_trainer = self.create_stage_trainer(stage_config)
        
        print(f"\n开始训练阶段: {stage_config.name}")
        print(f"阶段类型: {stage_config.stage_type.value}")
        print(f"计划轮次: {stage_config.epochs}")
        print(f"学习率: {stage_config.learning_rate:.2e}")
        
        # 设置阶段
        setup_info = stage_trainer.setup_stage(model, **kwargs)
        print(f"阶段设置: {setup_info}")
        
        # 创建优化器和调度器
        optimizer = self._create_optimizer(model, stage_config)
        scheduler = self._create_scheduler(optimizer, stage_config)
        
        # 训练循环
        stage_results = []
        
        for epoch in range(stage_config.epochs):
            # 训练一个轮次
            metrics = stage_trainer.train_epoch(
                model, epoch, 
                optimizer=optimizer, 
                **kwargs
            )
            
            # 更新调度器
            if scheduler:
                scheduler.step()
            
            # 记录结果
            stage_results.append({
                'epoch': epoch,
                'stage': stage_config.name,
                'metrics': metrics
            })
            
            # 打印进度
            if epoch % 20 == 0 or epoch == stage_config.epochs - 1:
                print(f"  轮次 {epoch:3d}: 总损失 = {metrics['total_loss']:.2e}, "
                      f"数据损失 = {metrics.get('data_loss', 0):.2e}, "
                      f"物理损失 = {metrics.get('physics_loss', 0):.2e}")
            
            # 检查早停
            if stage_trainer.check_early_stopping(metrics):
                print(f"  早停于轮次 {epoch}")
                break
            
            # 保存检查点
            if (epoch + 1) % stage_config.checkpoint_frequency == 0:
                self._save_checkpoint(model, stage_idx, epoch, metrics)
        
        # 清理阶段
        cleanup_info = stage_trainer.cleanup_stage(model, **kwargs)
        
        # 保存阶段训练器
        self.stage_trainers[stage_idx] = stage_trainer
        
        # 更新全局最佳指标
        final_metrics = stage_results[-1]['metrics']
        if 'total_loss' not in self.global_best_metrics or \
           final_metrics['total_loss'] < self.global_best_metrics['total_loss']:
            self.global_best_metrics = final_metrics.copy()
            self.global_best_metrics['best_stage'] = stage_idx
            self.global_best_metrics['best_epoch'] = len(stage_results) - 1
        
        stage_summary = stage_trainer.get_stage_summary()
        
        print(f"阶段 '{stage_config.name}' 完成")
        print(f"  最终损失: {final_metrics['total_loss']:.2e}")
        print(f"  损失减少: {stage_summary.get('loss_reduction', 0):.1%}")
        
        return {
            'stage_config': stage_config,
            'stage_results': stage_results,
            'stage_summary': stage_summary,
            'cleanup_info': cleanup_info,
            'final_metrics': final_metrics
        }
    
    def train_all_stages(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        训练所有阶段
        
        Args:
            model: 神经网络模型
            **kwargs: 训练参数
            
        Returns:
            Dict: 完整训练结果
        """
        print("开始冰川多阶段训练")
        print(f"总共 {len(self.stages)} 个阶段")
        
        all_results = []
        
        for stage_idx in range(len(self.stages)):
            self.current_stage_idx = stage_idx
            
            try:
                stage_result = self.train_stage(model, stage_idx, **kwargs)
                all_results.append(stage_result)
                
                # 更新全局训练历史
                self.training_history.extend(stage_result['stage_results'])
                
            except Exception as e:
                print(f"阶段 {stage_idx} 训练失败: {e}")
                break
        
        print("\n多阶段训练完成")
        print(f"全局最佳损失: {self.global_best_metrics.get('total_loss', 'N/A')}")
        print(f"最佳阶段: {self.global_best_metrics.get('best_stage', 'N/A')}")
        
        return {
            'all_stage_results': all_results,
            'global_best_metrics': self.global_best_metrics,
            'training_history': self.training_history,
            'stage_summaries': [trainer.get_stage_summary() 
                              for trainer in self.stage_trainers.values()]
        }
    
    def _create_optimizer(self, model: nn.Module, stage_config: StageConfig):
        """
        创建优化器
        
        Args:
            model: 神经网络模型
            stage_config: 阶段配置
            
        Returns:
            优化器
        """
        optimizer_type = stage_config.optimizer_type
        lr = stage_config.learning_rate
        params = stage_config.optimizer_params
        
        if optimizer_type == "Adam":
            return torch.optim.Adam(model.parameters(), lr=lr, **params)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(model.parameters(), lr=lr, **params)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(model.parameters(), lr=lr, **params)
        elif optimizer_type == "LBFGS":
            return torch.optim.LBFGS(model.parameters(), lr=lr, **params)
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self, optimizer, stage_config: StageConfig):
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器
            stage_config: 阶段配置
            
        Returns:
            调度器或None
        """
        scheduler_type = stage_config.scheduler_type
        
        if not scheduler_type:
            return None
        
        params = stage_config.scheduler_params
        
        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer, **params)
        elif scheduler_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
        elif scheduler_type == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        else:
            raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    def _save_checkpoint(self, model: nn.Module, stage_idx: int, epoch: int, metrics: Dict[str, float]):
        """
        保存检查点
        
        Args:
            model: 神经网络模型
            stage_idx: 阶段索引
            epoch: 轮次
            metrics: 指标
        """
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stage_idx': stage_idx,
            'epoch': epoch,
            'metrics': metrics,
            'global_best_metrics': self.global_best_metrics
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_stage_{stage_idx}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def generate_training_report(self) -> str:
        """
        生成训练报告
        
        Returns:
            str: 训练报告
        """
        report = "冰川多阶段训练报告\n"
        report += "=" * 50 + "\n"
        
        # 总体统计
        total_epochs = len(self.training_history)
        report += f"总训练轮次: {total_epochs}\n"
        report += f"完成阶段数: {len(self.stage_trainers)}\n"
        
        if self.global_best_metrics:
            report += f"全局最佳损失: {self.global_best_metrics['total_loss']:.2e}\n"
            report += f"最佳阶段: {self.global_best_metrics.get('best_stage', 'N/A')}\n"
        
        # 各阶段总结
        report += "\n各阶段详情:\n"
        report += "-" * 30 + "\n"
        
        for stage_idx, trainer in self.stage_trainers.items():
            summary = trainer.get_stage_summary()
            stage_config = self.stages[stage_idx]
            
            report += f"\n阶段 {stage_idx + 1}: {summary['stage_name']}\n"
            report += f"  类型: {summary['stage_type']}\n"
            report += f"  轮次: {summary['total_epochs']}\n"
            report += f"  初始损失: {summary['initial_loss']:.2e}\n"
            report += f"  最终损失: {summary['final_loss']:.2e}\n"
            report += f"  最佳损失: {summary['best_loss']:.2e}\n"
            report += f"  损失减少: {summary['loss_reduction']:.1%}\n"
            report += f"  早停: {'是' if summary['early_stopped'] else '否'}\n"
        
        return report

if __name__ == "__main__":
    # 测试多阶段训练
    torch.manual_seed(42)
    
    print("测试冰川多阶段训练...")
    
    # 创建简单的测试模型
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleTestModel()
    
    # 创建多阶段训练器
    trainer = GlacierMultiStageTrainer()
    
    # 模拟训练数据
    def create_mock_data_loader():
        # 简单的数据生成器
        for _ in range(10):  # 每个epoch 10个batch
            inputs = torch.randn(32, 2)  # batch_size=32, input_dim=2
            targets = torch.randn(32, 1)  # batch_size=32, output_dim=1
            yield {
                'inputs': inputs,
                'targets': targets
            }
    
    # 模拟损失函数
    def mock_data_loss_fn(predictions, targets):
        return nn.MSELoss()(predictions, targets)
    
    def mock_physics_loss_fn(model, batch_data):
        # 简单的物理损失（梯度惩罚）
        inputs = batch_data['inputs']
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        gradients = torch.autograd.grad(
            outputs.sum(), inputs, 
            create_graph=True, retain_graph=True
        )[0]
        
        return torch.mean(gradients**2)
    
    def mock_boundary_loss_fn(model, batch_data):
        # 简单的边界损失
        boundary_inputs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        boundary_outputs = model(boundary_inputs)
        target_values = torch.tensor([[0.0], [1.0]])
        return nn.MSELoss()(boundary_outputs, target_values)
    
    # 训练参数
    training_kwargs = {
        'data_loader': create_mock_data_loader(),
        'data_loss_fn': mock_data_loss_fn,
        'physics_loss_fn': mock_physics_loss_fn,
        'boundary_loss_fn': mock_boundary_loss_fn
    }
    
    # 测试单个阶段训练
    print("\n测试单个阶段训练...")
    stage_result = trainer.train_stage(model, 0, **training_kwargs)
    print(f"阶段结果: {stage_result['stage_summary']}")
    
    # 生成报告
    print("\n" + "=" * 50)
    print(trainer.generate_training_report())
    
    print("\n多阶段训练测试完成！")