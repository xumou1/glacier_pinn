#!/usr/bin/env python3
"""
渐进训练管理器

实现多阶段渐进训练的管理和协调，包括：
- 训练阶段管理
- 阶段间过渡控制
- 性能监控和评估
- 自适应阶段调整

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
from pathlib import Path
import json
import time
from collections import defaultdict

class TrainingStage(Enum):
    """训练阶段枚举"""
    LONGTERM_TRENDS = "longterm_trends"  # 长期趋势学习
    SHORTTERM_DYNAMICS = "shortterm_dynamics"  # 短期动态学习
    COUPLED_OPTIMIZATION = "coupled_optimization"  # 耦合优化
    FINE_TUNING = "fine_tuning"  # 微调
    VALIDATION = "validation"  # 验证

class StageStatus(Enum):
    """阶段状态枚举"""
    PENDING = "pending"  # 待执行
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过

class TransitionCriteria(Enum):
    """过渡条件枚举"""
    LOSS_THRESHOLD = "loss_threshold"  # 损失阈值
    IMPROVEMENT_RATE = "improvement_rate"  # 改进率
    CONVERGENCE = "convergence"  # 收敛
    MAX_EPOCHS = "max_epochs"  # 最大轮数
    MANUAL = "manual"  # 手动控制
    PERFORMANCE_PLATEAU = "performance_plateau"  # 性能平台期

@dataclass
class StageConfig:
    """阶段配置"""
    stage: TrainingStage
    max_epochs: int = 100
    min_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    transition_criteria: Dict[TransitionCriteria, float] = None
    stage_specific_params: Dict[str, Any] = None
    checkpoint_frequency: int = 10
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        if self.transition_criteria is None:
            self.transition_criteria = {
                TransitionCriteria.LOSS_THRESHOLD: 1e-4,
                TransitionCriteria.IMPROVEMENT_RATE: 1e-5,
                TransitionCriteria.MAX_EPOCHS: self.max_epochs
            }
        
        if self.stage_specific_params is None:
            self.stage_specific_params = {}

@dataclass
class ProgressiveTrainingConfig:
    """渐进训练配置"""
    stages: List[StageConfig] = None
    global_max_epochs: int = 1000
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_best_model: bool = True
    enable_stage_skipping: bool = True
    performance_window: int = 10
    transition_smoothing: bool = True
    adaptive_scheduling: bool = True
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                StageConfig(
                    stage=TrainingStage.LONGTERM_TRENDS,
                    max_epochs=200,
                    learning_rate=1e-3
                ),
                StageConfig(
                    stage=TrainingStage.SHORTTERM_DYNAMICS,
                    max_epochs=150,
                    learning_rate=5e-4
                ),
                StageConfig(
                    stage=TrainingStage.COUPLED_OPTIMIZATION,
                    max_epochs=100,
                    learning_rate=1e-4
                ),
                StageConfig(
                    stage=TrainingStage.FINE_TUNING,
                    max_epochs=50,
                    learning_rate=1e-5
                )
            ]

class StageTrainer(ABC):
    """
    阶段训练器基类
    
    定义每个训练阶段的通用接口
    """
    
    def __init__(self, stage: TrainingStage, config: StageConfig):
        """
        初始化阶段训练器
        
        Args:
            stage: 训练阶段
            config: 阶段配置
        """
        self.stage = stage
        self.config = config
        self.status = StageStatus.PENDING
        self.metrics_history = []
        self.best_metrics = None
        self.logger = logging.getLogger(f"{__name__}.{stage.value}")
    
    @abstractmethod
    def setup(self, model: nn.Module, **kwargs) -> None:
        """
        设置阶段训练器
        
        Args:
            model: 模型
            **kwargs: 其他参数
        """
        pass
    
    @abstractmethod
    def train_epoch(self, model: nn.Module, data_loader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            model: 模型
            data_loader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict: 训练指标
        """
        pass
    
    @abstractmethod
    def validate(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            model: 模型
            data_loader: 验证数据加载器
            
        Returns:
            Dict: 验证指标
        """
        pass
    
    @abstractmethod
    def check_transition_criteria(self) -> Tuple[bool, str]:
        """
        检查过渡条件
        
        Returns:
            Tuple: (是否满足过渡条件, 原因)
        """
        pass
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """更新指标历史"""
        self.metrics_history.append(metrics)
        
        # 更新最佳指标
        if self.best_metrics is None or metrics.get('val_loss', float('inf')) < self.best_metrics.get('val_loss', float('inf')):
            self.best_metrics = metrics.copy()
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """获取阶段总结"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        return {
            'stage': self.stage.value,
            'status': self.status.value,
            'total_epochs': len(self.metrics_history),
            'best_metrics': self.best_metrics,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'improvement': self._calculate_improvement()
        }
    
    def _calculate_improvement(self) -> float:
        """计算改进程度"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        initial_loss = self.metrics_history[0].get('val_loss', 0)
        final_loss = self.metrics_history[-1].get('val_loss', 0)
        
        if initial_loss == 0:
            return 0.0
        
        return (initial_loss - final_loss) / initial_loss

class LongTermTrendsTrainer(StageTrainer):
    """长期趋势训练器"""
    
    def __init__(self, config: StageConfig):
        super().__init__(TrainingStage.LONGTERM_TRENDS, config)
        self.optimizer = None
        self.scheduler = None
    
    def setup(self, model: nn.Module, **kwargs) -> None:
        """设置训练器"""
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    def train_epoch(self, model: nn.Module, data_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        total_trend_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            inputs, targets = batch_data[:2]
            
            self.optimizer.zero_grad()
            
            predictions = model(inputs)
            
            # 基础MSE损失
            mse_loss = nn.MSELoss()(predictions, targets)
            
            # 长期趋势损失（平滑性约束）
            if predictions.shape[1] > 1:
                trend_loss = torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1]))
            else:
                trend_loss = torch.tensor(0.0)
            
            total_loss_batch = mse_loss + 0.1 * trend_loss
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_trend_loss += trend_loss.item()
            num_batches += 1
        
        metrics = {
            'train_loss': total_loss / num_batches,
            'trend_loss': total_trend_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets = batch_data[:2]
                predictions = model(inputs)
                
                loss = nn.MSELoss()(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
        
        # 更新学习率调度器
        self.scheduler.step(val_metrics['val_loss'])
        
        return val_metrics
    
    def check_transition_criteria(self) -> Tuple[bool, str]:
        """检查过渡条件"""
        if len(self.metrics_history) < self.config.min_epochs:
            return False, "minimum_epochs_not_reached"
        
        # 检查损失阈值
        current_loss = self.metrics_history[-1].get('val_loss', float('inf'))
        loss_threshold = self.config.transition_criteria.get(TransitionCriteria.LOSS_THRESHOLD, 1e-4)
        
        if current_loss < loss_threshold:
            return True, "loss_threshold_reached"
        
        # 检查改进率
        if len(self.metrics_history) >= 10:
            recent_losses = [m.get('val_loss', float('inf')) for m in self.metrics_history[-10:]]
            improvement_rate = (recent_losses[0] - recent_losses[-1]) / (recent_losses[0] + 1e-10)
            
            min_improvement = self.config.transition_criteria.get(TransitionCriteria.IMPROVEMENT_RATE, 1e-5)
            
            if improvement_rate < min_improvement:
                return True, "improvement_rate_too_low"
        
        # 检查最大轮数
        max_epochs = self.config.transition_criteria.get(TransitionCriteria.MAX_EPOCHS, self.config.max_epochs)
        if len(self.metrics_history) >= max_epochs:
            return True, "max_epochs_reached"
        
        return False, "criteria_not_met"

class ShortTermDynamicsTrainer(StageTrainer):
    """短期动态训练器"""
    
    def __init__(self, config: StageConfig):
        super().__init__(TrainingStage.SHORTTERM_DYNAMICS, config)
        self.optimizer = None
        self.adaptive_lr = None
    
    def setup(self, model: nn.Module, **kwargs) -> None:
        """设置训练器"""
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def train_epoch(self, model: nn.Module, data_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            inputs, targets = batch_data[:2]
            
            self.optimizer.zero_grad()
            
            predictions = model(inputs)
            
            # 基础损失
            base_loss = nn.MSELoss()(predictions, targets)
            
            # 动态变化损失（高频成分）
            if predictions.shape[1] > 2:
                # 计算二阶差分（加速度）
                second_diff = predictions[:, 2:] - 2*predictions[:, 1:-1] + predictions[:, :-2]
                dynamics_loss = torch.mean(second_diff ** 2)
            else:
                dynamics_loss = torch.tensor(0.0)
            
            total_loss_batch = base_loss + 0.05 * dynamics_loss
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_dynamics_loss += dynamics_loss.item()
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'dynamics_loss': total_dynamics_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets = batch_data[:2]
                predictions = model(inputs)
                
                loss = nn.MSELoss()(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def check_transition_criteria(self) -> Tuple[bool, str]:
        """检查过渡条件"""
        if len(self.metrics_history) < self.config.min_epochs:
            return False, "minimum_epochs_not_reached"
        
        # 检查收敛性
        if len(self.metrics_history) >= 5:
            recent_losses = [m.get('val_loss', float('inf')) for m in self.metrics_history[-5:]]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std / (loss_mean + 1e-10) < 0.01:  # 变异系数小于1%
                return True, "convergence_detected"
        
        # 检查最大轮数
        if len(self.metrics_history) >= self.config.max_epochs:
            return True, "max_epochs_reached"
        
        return False, "criteria_not_met"

class CoupledOptimizationTrainer(StageTrainer):
    """耦合优化训练器"""
    
    def __init__(self, config: StageConfig):
        super().__init__(TrainingStage.COUPLED_OPTIMIZATION, config)
        self.optimizer = None
        self.objective_weights = {'physics': 1.0, 'data': 1.0, 'boundary': 0.5}
    
    def setup(self, model: nn.Module, **kwargs) -> None:
        """设置训练器"""
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def train_epoch(self, model: nn.Module, data_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        physics_loss_total = 0.0
        data_loss_total = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            inputs, targets = batch_data[:2]
            
            self.optimizer.zero_grad()
            
            predictions = model(inputs)
            
            # 数据拟合损失
            data_loss = nn.MSELoss()(predictions, targets)
            
            # 物理约束损失（简化版）
            physics_loss = torch.mean((predictions.sum(dim=-1, keepdim=True) - targets.sum(dim=-1, keepdim=True)) ** 2)
            
            # 加权总损失
            total_loss_batch = (self.objective_weights['data'] * data_loss + 
                              self.objective_weights['physics'] * physics_loss)
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            physics_loss_total += physics_loss.item()
            data_loss_total += data_loss.item()
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'physics_loss': physics_loss_total / num_batches,
            'data_loss': data_loss_total / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets = batch_data[:2]
                predictions = model(inputs)
                
                loss = nn.MSELoss()(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def check_transition_criteria(self) -> Tuple[bool, str]:
        """检查过渡条件"""
        if len(self.metrics_history) < self.config.min_epochs:
            return False, "minimum_epochs_not_reached"
        
        # 检查多目标收敛
        if len(self.metrics_history) >= 10:
            recent_metrics = self.metrics_history[-10:]
            
            # 检查各个损失的稳定性
            physics_losses = [m.get('physics_loss', 0) for m in recent_metrics]
            data_losses = [m.get('data_loss', 0) for m in recent_metrics]
            
            physics_stable = np.std(physics_losses) / (np.mean(physics_losses) + 1e-10) < 0.05
            data_stable = np.std(data_losses) / (np.mean(data_losses) + 1e-10) < 0.05
            
            if physics_stable and data_stable:
                return True, "multi_objective_convergence"
        
        if len(self.metrics_history) >= self.config.max_epochs:
            return True, "max_epochs_reached"
        
        return False, "criteria_not_met"

class FineTuningTrainer(StageTrainer):
    """微调训练器"""
    
    def __init__(self, config: StageConfig):
        super().__init__(TrainingStage.FINE_TUNING, config)
        self.optimizer = None
    
    def setup(self, model: nn.Module, **kwargs) -> None:
        """设置训练器"""
        # 使用更小的学习率进行微调
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def train_epoch(self, model: nn.Module, data_loader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            inputs, targets = batch_data[:2]
            
            self.optimizer.zero_grad()
            
            predictions = model(inputs)
            loss = nn.MSELoss()(predictions, targets)
            
            # 添加L2正则化
            l2_reg = torch.tensor(0.0)
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            
            total_loss_batch = loss + 1e-5 * l2_reg
            total_loss_batch.backward()
            
            # 更小的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, model: nn.Module, data_loader) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets = batch_data[:2]
                predictions = model(inputs)
                
                loss = nn.MSELoss()(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def check_transition_criteria(self) -> Tuple[bool, str]:
        """检查过渡条件"""
        if len(self.metrics_history) >= self.config.max_epochs:
            return True, "max_epochs_reached"
        
        # 微调阶段通常运行固定轮数
        return False, "fine_tuning_in_progress"

class ProgressiveTrainer:
    """
    渐进训练管理器
    
    协调和管理多阶段训练过程
    """
    
    def __init__(self, model: nn.Module, config: ProgressiveTrainingConfig):
        """
        初始化渐进训练管理器
        
        Args:
            model: 模型
            config: 配置
        """
        self.model = model
        self.config = config
        self.current_stage_idx = 0
        self.stage_trainers = self._create_stage_trainers()
        self.training_history = []
        self.best_model_state = None
        self.best_overall_loss = float('inf')
        self.logger = logging.getLogger(__name__)
        
        # 创建目录
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_stage_trainers(self) -> List[StageTrainer]:
        """创建阶段训练器"""
        trainers = []
        
        for stage_config in self.config.stages:
            if stage_config.stage == TrainingStage.LONGTERM_TRENDS:
                trainer = LongTermTrendsTrainer(stage_config)
            elif stage_config.stage == TrainingStage.SHORTTERM_DYNAMICS:
                trainer = ShortTermDynamicsTrainer(stage_config)
            elif stage_config.stage == TrainingStage.COUPLED_OPTIMIZATION:
                trainer = CoupledOptimizationTrainer(stage_config)
            elif stage_config.stage == TrainingStage.FINE_TUNING:
                trainer = FineTuningTrainer(stage_config)
            else:
                raise ValueError(f"Unknown training stage: {stage_config.stage}")
            
            trainers.append(trainer)
        
        return trainers
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        执行渐进训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            Dict: 训练结果
        """
        self.logger.info("开始渐进训练")
        
        total_epochs = 0
        
        for stage_idx, trainer in enumerate(self.stage_trainers):
            self.current_stage_idx = stage_idx
            
            self.logger.info(f"开始阶段 {stage_idx + 1}: {trainer.stage.value}")
            
            # 设置阶段训练器
            trainer.setup(self.model)
            trainer.status = StageStatus.RUNNING
            
            stage_start_time = time.time()
            
            # 阶段训练循环
            for epoch in range(trainer.config.max_epochs):
                epoch_start_time = time.time()
                
                # 训练
                train_metrics = trainer.train_epoch(self.model, train_loader, epoch)
                
                # 验证
                val_metrics = trainer.validate(self.model, val_loader)
                
                # 合并指标
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_metrics['epoch'] = epoch
                epoch_metrics['stage'] = trainer.stage.value
                epoch_metrics['total_epochs'] = total_epochs
                epoch_metrics['epoch_time'] = time.time() - epoch_start_time
                
                # 更新训练器指标
                trainer.update_metrics(epoch_metrics)
                
                # 记录全局历史
                self.training_history.append(epoch_metrics)
                
                # 保存最佳模型
                if self.config.save_best_model and val_metrics.get('val_loss', float('inf')) < self.best_overall_loss:
                    self.best_overall_loss = val_metrics['val_loss']
                    self.best_model_state = self.model.state_dict().copy()
                
                # 检查过渡条件
                should_transition, reason = trainer.check_transition_criteria()
                
                if should_transition:
                    self.logger.info(f"阶段 {stage_idx + 1} 完成，原因: {reason}，共 {epoch + 1} 轮")
                    trainer.status = StageStatus.COMPLETED
                    break
                
                # 定期保存检查点
                if epoch % trainer.config.checkpoint_frequency == 0:
                    self._save_checkpoint(stage_idx, epoch)
                
                # 日志输出
                if epoch % 10 == 0:
                    self.logger.info(
                        f"阶段 {stage_idx + 1}, Epoch {epoch}: "
                        f"Train Loss: {train_metrics.get('train_loss', 0):.6f}, "
                        f"Val Loss: {val_metrics.get('val_loss', 0):.6f}"
                    )
                
                total_epochs += 1
            
            stage_duration = time.time() - stage_start_time
            self.logger.info(f"阶段 {stage_idx + 1} 耗时: {stage_duration:.2f} 秒")
            
            # 阶段间过渡处理
            if stage_idx < len(self.stage_trainers) - 1:
                self._handle_stage_transition(stage_idx)
        
        # 训练完成
        training_summary = self._generate_training_summary()
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"已恢复最佳模型 (验证损失: {self.best_overall_loss:.6f})")
        
        return training_summary
    
    def _handle_stage_transition(self, completed_stage_idx: int) -> None:
        """
        处理阶段间过渡
        
        Args:
            completed_stage_idx: 已完成的阶段索引
        """
        if not self.config.transition_smoothing:
            return
        
        # 平滑过渡：逐渐调整学习率
        next_trainer = self.stage_trainers[completed_stage_idx + 1]
        current_lr = self.stage_trainers[completed_stage_idx].optimizer.param_groups[0]['lr']
        target_lr = next_trainer.config.learning_rate
        
        # 如果学习率差异较大，进行平滑过渡
        if abs(current_lr - target_lr) / current_lr > 0.5:
            self.logger.info(f"学习率平滑过渡: {current_lr:.6f} -> {target_lr:.6f}")
            
            # 可以在这里添加更复杂的过渡逻辑
            # 例如：权重预热、渐进式学习率调整等
    
    def _save_checkpoint(self, stage_idx: int, epoch: int) -> None:
        """
        保存检查点
        
        Args:
            stage_idx: 阶段索引
            epoch: 轮数
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'stage_idx': stage_idx,
            'epoch': epoch,
            'training_history': self.training_history,
            'best_overall_loss': self.best_overall_loss,
            'config': self.config
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_stage_{stage_idx}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def _generate_training_summary(self) -> Dict[str, Any]:
        """
        生成训练总结
        
        Returns:
            Dict: 训练总结
        """
        stage_summaries = [trainer.get_stage_summary() for trainer in self.stage_trainers]
        
        total_epochs = len(self.training_history)
        total_improvement = 0.0
        
        if self.training_history:
            initial_loss = self.training_history[0].get('val_loss', 0)
            final_loss = self.training_history[-1].get('val_loss', 0)
            
            if initial_loss > 0:
                total_improvement = (initial_loss - final_loss) / initial_loss
        
        summary = {
            'total_epochs': total_epochs,
            'total_improvement': total_improvement,
            'best_overall_loss': self.best_overall_loss,
            'stage_summaries': stage_summaries,
            'final_metrics': self.training_history[-1] if self.training_history else None,
            'training_time_per_stage': self._calculate_stage_times()
        }
        
        return summary
    
    def _calculate_stage_times(self) -> Dict[str, float]:
        """计算各阶段训练时间"""
        stage_times = {}
        
        for stage_name in [stage.value for stage in TrainingStage]:
            stage_epochs = [h for h in self.training_history if h.get('stage') == stage_name]
            total_time = sum(h.get('epoch_time', 0) for h in stage_epochs)
            stage_times[stage_name] = total_time
        
        return stage_times
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_stage_idx = checkpoint['stage_idx']
        self.training_history = checkpoint['training_history']
        self.best_overall_loss = checkpoint['best_overall_loss']
        
        self.logger.info(f"已加载检查点: {checkpoint_path}")
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """
        获取当前阶段信息
        
        Returns:
            Dict: 当前阶段信息
        """
        if self.current_stage_idx < len(self.stage_trainers):
            current_trainer = self.stage_trainers[self.current_stage_idx]
            return {
                'stage_index': self.current_stage_idx,
                'stage_name': current_trainer.stage.value,
                'stage_status': current_trainer.status.value,
                'epochs_completed': len(current_trainer.metrics_history),
                'max_epochs': current_trainer.config.max_epochs,
                'best_metrics': current_trainer.best_metrics
            }
        else:
            return {'status': 'training_completed'}

def create_progressive_trainer(
    model: nn.Module,
    stage_configs: List[StageConfig] = None,
    config: ProgressiveTrainingConfig = None
) -> ProgressiveTrainer:
    """
    创建渐进训练管理器
    
    Args:
        model: 模型
        stage_configs: 阶段配置列表
        config: 全局配置
        
    Returns:
        ProgressiveTrainer: 训练器实例
    """
    if config is None:
        config = ProgressiveTrainingConfig()
    
    if stage_configs is not None:
        config.stages = stage_configs
    
    trainer = ProgressiveTrainer(model, config)
    
    return trainer

if __name__ == "__main__":
    # 测试渐进训练
    
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
    stage_configs = [
        StageConfig(
            stage=TrainingStage.LONGTERM_TRENDS,
            max_epochs=20,
            min_epochs=5,
            learning_rate=1e-3
        ),
        StageConfig(
            stage=TrainingStage.SHORTTERM_DYNAMICS,
            max_epochs=15,
            min_epochs=5,
            learning_rate=5e-4
        ),
        StageConfig(
            stage=TrainingStage.COUPLED_OPTIMIZATION,
            max_epochs=10,
            min_epochs=3,
            learning_rate=1e-4
        )
    ]
    
    config = ProgressiveTrainingConfig(
        stages=stage_configs,
        checkpoint_dir="./test_checkpoints",
        save_best_model=True
    )
    
    # 创建训练器
    trainer = create_progressive_trainer(model, config=config)
    
    # 生成测试数据
    def generate_test_data(batch_size=32, num_batches=10):
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, 3)
            targets = torch.sin(inputs.sum(dim=1, keepdim=True)).repeat(1, 3) + 0.1 * torch.randn(batch_size, 3)
            yield inputs, targets
    
    # 执行训练
    print("=== 渐进训练测试 ===")
    
    train_loader = generate_test_data()
    val_loader = generate_test_data(batch_size=16, num_batches=5)
    
    training_summary = trainer.train(train_loader, val_loader)
    
    print(f"\n=== 训练总结 ===")
    print(f"总训练轮数: {training_summary['total_epochs']}")
    print(f"总体改进: {training_summary['total_improvement']:.2%}")
    print(f"最佳验证损失: {training_summary['best_overall_loss']:.6f}")
    
    print(f"\n各阶段总结:")
    for i, stage_summary in enumerate(training_summary['stage_summaries']):
        print(f"  阶段 {i+1} ({stage_summary['stage']}):")
        print(f"    状态: {stage_summary['status']}")
        print(f"    训练轮数: {stage_summary['total_epochs']}")
        print(f"    改进程度: {stage_summary['improvement']:.2%}")
        if stage_summary['best_metrics']:
            print(f"    最佳验证损失: {stage_summary['best_metrics'].get('val_loss', 'N/A')}")
    
    print("\n渐进训练测试完成！")