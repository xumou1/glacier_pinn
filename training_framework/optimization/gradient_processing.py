#!/usr/bin/env python3
"""
梯度处理

实现各种梯度处理技术，包括：
- 梯度裁剪（范数裁剪、值裁剪、自适应裁剪）
- 梯度累积和平均
- 梯度噪声和正则化
- 梯度分析和监控
- 物理信息神经网络专用梯度处理

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

class GradientClipType(Enum):
    """梯度裁剪类型枚举"""
    NORM = "norm"  # 范数裁剪
    VALUE = "value"  # 值裁剪
    ADAPTIVE = "adaptive"  # 自适应裁剪
    PERCENTILE = "percentile"  # 百分位裁剪
    GLOBAL_NORM = "global_norm"  # 全局范数裁剪
    LAYER_WISE = "layer_wise"  # 逐层裁剪

class GradientNoiseType(Enum):
    """梯度噪声类型枚举"""
    GAUSSIAN = "gaussian"  # 高斯噪声
    UNIFORM = "uniform"  # 均匀噪声
    DROPOUT = "dropout"  # 梯度dropout
    LANGEVIN = "langevin"  # Langevin噪声
    ADAPTIVE = "adaptive"  # 自适应噪声

class GradientNormType(Enum):
    """梯度范数类型枚举"""
    L1 = "l1"  # L1范数
    L2 = "l2"  # L2范数
    LINF = "linf"  # L∞范数
    FROBENIUS = "frobenius"  # Frobenius范数

@dataclass
class GradientProcessingConfig:
    """梯度处理配置"""
    # 梯度裁剪配置
    enable_clipping: bool = True
    clip_type: GradientClipType = GradientClipType.NORM
    clip_value: float = 1.0
    clip_norm: float = 1.0
    adaptive_clip_factor: float = 0.1
    percentile_threshold: float = 95.0
    
    # 梯度累积配置
    enable_accumulation: bool = False
    accumulation_steps: int = 1
    normalize_accumulated: bool = True
    
    # 梯度噪声配置
    enable_noise: bool = False
    noise_type: GradientNoiseType = GradientNoiseType.GAUSSIAN
    noise_scale: float = 0.01
    noise_decay: float = 0.99
    
    # 梯度正则化配置
    enable_regularization: bool = False
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    spectral_reg: float = 0.0
    
    # 梯度监控配置
    enable_monitoring: bool = True
    monitor_frequency: int = 10
    save_gradients: bool = False
    gradient_history_size: int = 100
    
    # 物理信息配置
    physics_gradient_weighting: bool = False
    data_gradient_weighting: bool = False
    boundary_gradient_weighting: bool = False
    
    # 高级配置
    enable_gradient_centralization: bool = False
    enable_gradient_standardization: bool = False
    enable_gradient_surgery: bool = False
    surgery_threshold: float = 0.1

class GradientProcessorBase(ABC):
    """
    梯度处理器基类
    
    定义梯度处理器的通用接口
    """
    
    def __init__(self, config: GradientProcessingConfig):
        """
        初始化梯度处理器
        
        Args:
            config: 配置
        """
        self.config = config
        self.step_count = 0
        self.gradient_history = deque(maxlen=config.gradient_history_size)
        self.statistics = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """处理梯度"""
        pass
    
    def get_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """
        获取梯度统计信息
        
        Args:
            model: 模型
            
        Returns:
            Dict[str, float]: 梯度统计
        """
        stats = {}
        total_norm = 0.0
        total_params = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                param_norm = grad.norm().item()
                total_norm += param_norm ** 2
                total_params += grad.numel()
                
                param_max = grad.abs().max().item()
                param_min = grad.abs().min().item()
                
                max_grad = max(max_grad, param_max)
                min_grad = min(min_grad, param_min)
                
                stats[f"{name}_norm"] = param_norm
                stats[f"{name}_mean"] = grad.mean().item()
                stats[f"{name}_std"] = grad.std().item()
        
        total_norm = total_norm ** 0.5
        
        stats.update({
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad if min_grad != float('inf') else 0.0,
            'total_params': total_params
        })
        
        return stats
    
    def log_statistics(self, stats: Dict[str, float]) -> None:
        """
        记录统计信息
        
        Args:
            stats: 统计信息
        """
        for key, value in stats.items():
            self.statistics[key].append(value)
        
        if self.step_count % self.config.monitor_frequency == 0:
            self.logger.info(
                f"Step {self.step_count}: "
                f"Total Norm = {stats.get('total_norm', 0):.6f}, "
                f"Max Grad = {stats.get('max_grad', 0):.6f}"
            )

class GradientClipper(GradientProcessorBase):
    """梯度裁剪器"""
    
    def __init__(self, config: GradientProcessingConfig):
        super().__init__(config)
        self.adaptive_threshold = config.clip_norm
        self.threshold_history = deque(maxlen=100)
    
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理梯度裁剪
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.step_count += 1
        
        # 获取裁剪前的统计信息
        pre_clip_stats = self.get_gradient_stats(model)
        
        # 执行梯度裁剪
        if self.config.clip_type == GradientClipType.NORM:
            clipped_norm = self._clip_by_norm(model)
        elif self.config.clip_type == GradientClipType.VALUE:
            clipped_norm = self._clip_by_value(model)
        elif self.config.clip_type == GradientClipType.ADAPTIVE:
            clipped_norm = self._adaptive_clip(model)
        elif self.config.clip_type == GradientClipType.PERCENTILE:
            clipped_norm = self._clip_by_percentile(model)
        elif self.config.clip_type == GradientClipType.GLOBAL_NORM:
            clipped_norm = self._clip_global_norm(model)
        elif self.config.clip_type == GradientClipType.LAYER_WISE:
            clipped_norm = self._clip_layer_wise(model)
        else:
            clipped_norm = pre_clip_stats['total_norm']
        
        # 获取裁剪后的统计信息
        post_clip_stats = self.get_gradient_stats(model)
        
        # 记录统计信息
        self.log_statistics(post_clip_stats)
        
        result = {
            'pre_clip_norm': pre_clip_stats['total_norm'],
            'post_clip_norm': post_clip_stats['total_norm'],
            'clipped': clipped_norm < pre_clip_stats['total_norm'],
            'clip_ratio': post_clip_stats['total_norm'] / (pre_clip_stats['total_norm'] + 1e-10)
        }
        
        return result
    
    def _clip_by_norm(self, model: nn.Module) -> float:
        """按范数裁剪"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
    
    def _clip_by_value(self, model: nn.Module) -> float:
        """按值裁剪"""
        torch.nn.utils.clip_grad_value_(model.parameters(), self.config.clip_value)
        return self.get_gradient_stats(model)['total_norm']
    
    def _adaptive_clip(self, model: nn.Module) -> float:
        """自适应裁剪"""
        # 计算当前梯度范数
        current_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                current_norm += param.grad.data.norm() ** 2
        current_norm = current_norm ** 0.5
        
        # 更新自适应阈值
        if len(self.threshold_history) > 0:
            avg_norm = np.mean(self.threshold_history)
            self.adaptive_threshold = avg_norm * (1 + self.config.adaptive_clip_factor)
        
        self.threshold_history.append(current_norm)
        
        # 执行裁剪
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.adaptive_threshold)
    
    def _clip_by_percentile(self, model: nn.Module) -> float:
        """按百分位裁剪"""
        # 收集所有梯度
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.data.flatten().tolist())
        
        if not all_grads:
            return 0.0
        
        # 计算百分位阈值
        threshold = np.percentile(np.abs(all_grads), self.config.percentile_threshold)
        
        # 执行裁剪
        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)
        return self.get_gradient_stats(model)['total_norm']
    
    def _clip_global_norm(self, model: nn.Module) -> float:
        """全局范数裁剪"""
        # 计算全局范数
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm()
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 如果超过阈值，按比例缩放所有梯度
        if total_norm > self.config.clip_norm:
            clip_coef = self.config.clip_norm / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return min(total_norm, self.config.clip_norm)
    
    def _clip_layer_wise(self, model: nn.Module) -> float:
        """逐层裁剪"""
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 为每层设置不同的裁剪阈值
                if 'weight' in name:
                    layer_threshold = self.config.clip_norm
                elif 'bias' in name:
                    layer_threshold = self.config.clip_norm * 0.1
                else:
                    layer_threshold = self.config.clip_norm * 0.5
                
                # 裁剪该层梯度
                param_norm = param.grad.data.norm()
                if param_norm > layer_threshold:
                    param.grad.data.mul_(layer_threshold / (param_norm + 1e-6))
                
                total_norm += param.grad.data.norm().item() ** 2
        
        return total_norm ** 0.5

class GradientAccumulator(GradientProcessorBase):
    """梯度累积器"""
    
    def __init__(self, config: GradientProcessingConfig):
        super().__init__(config)
        self.accumulated_gradients = {}
        self.accumulation_count = 0
    
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理梯度累积
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.step_count += 1
        self.accumulation_count += 1
        
        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = torch.zeros_like(param.grad)
                
                self.accumulated_gradients[name] += param.grad.data
        
        # 检查是否需要应用累积的梯度
        should_apply = self.accumulation_count >= self.config.accumulation_steps
        
        if should_apply:
            # 应用累积的梯度
            for name, param in model.named_parameters():
                if name in self.accumulated_gradients:
                    if self.config.normalize_accumulated:
                        # 归一化累积梯度
                        param.grad.data = self.accumulated_gradients[name] / self.accumulation_count
                    else:
                        param.grad.data = self.accumulated_gradients[name]
            
            # 重置累积
            self.accumulated_gradients.clear()
            self.accumulation_count = 0
        else:
            # 清零当前梯度，等待累积完成
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
        
        # 获取统计信息
        stats = self.get_gradient_stats(model)
        self.log_statistics(stats)
        
        result = {
            'accumulated_steps': self.accumulation_count,
            'should_apply': should_apply,
            'gradient_norm': stats['total_norm']
        }
        
        return result

class GradientNoiseInjector(GradientProcessorBase):
    """梯度噪声注入器"""
    
    def __init__(self, config: GradientProcessingConfig):
        super().__init__(config)
        self.current_noise_scale = config.noise_scale
    
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理梯度噪声注入
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.step_count += 1
        
        # 获取噪声前的统计信息
        pre_noise_stats = self.get_gradient_stats(model)
        
        # 注入噪声
        if self.config.noise_type == GradientNoiseType.GAUSSIAN:
            self._add_gaussian_noise(model)
        elif self.config.noise_type == GradientNoiseType.UNIFORM:
            self._add_uniform_noise(model)
        elif self.config.noise_type == GradientNoiseType.DROPOUT:
            self._apply_gradient_dropout(model)
        elif self.config.noise_type == GradientNoiseType.LANGEVIN:
            self._add_langevin_noise(model)
        elif self.config.noise_type == GradientNoiseType.ADAPTIVE:
            self._add_adaptive_noise(model, pre_noise_stats)
        
        # 衰减噪声尺度
        self.current_noise_scale *= self.config.noise_decay
        
        # 获取噪声后的统计信息
        post_noise_stats = self.get_gradient_stats(model)
        self.log_statistics(post_noise_stats)
        
        result = {
            'pre_noise_norm': pre_noise_stats['total_norm'],
            'post_noise_norm': post_noise_stats['total_norm'],
            'noise_scale': self.current_noise_scale,
            'noise_ratio': (post_noise_stats['total_norm'] - pre_noise_stats['total_norm']) / (pre_noise_stats['total_norm'] + 1e-10)
        }
        
        return result
    
    def _add_gaussian_noise(self, model: nn.Module) -> None:
        """添加高斯噪声"""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.current_noise_scale
                param.grad.data.add_(noise)
    
    def _add_uniform_noise(self, model: nn.Module) -> None:
        """添加均匀噪声"""
        for param in model.parameters():
            if param.grad is not None:
                noise = (torch.rand_like(param.grad) - 0.5) * 2 * self.current_noise_scale
                param.grad.data.add_(noise)
    
    def _apply_gradient_dropout(self, model: nn.Module) -> None:
        """应用梯度dropout"""
        dropout_prob = self.current_noise_scale
        for param in model.parameters():
            if param.grad is not None:
                mask = torch.rand_like(param.grad) > dropout_prob
                param.grad.data.mul_(mask.float())
    
    def _add_langevin_noise(self, model: nn.Module) -> None:
        """添加Langevin噪声"""
        for param in model.parameters():
            if param.grad is not None:
                # Langevin噪声与梯度范数成比例
                grad_norm = param.grad.norm()
                noise_scale = self.current_noise_scale * torch.sqrt(grad_norm + 1e-10)
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.data.add_(noise)
    
    def _add_adaptive_noise(self, model: nn.Module, stats: Dict[str, float]) -> None:
        """添加自适应噪声"""
        # 根据梯度统计信息调整噪声
        total_norm = stats['total_norm']
        adaptive_scale = self.current_noise_scale * (1.0 + total_norm)
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * adaptive_scale
                param.grad.data.add_(noise)

class GradientRegularizer(GradientProcessorBase):
    """梯度正则化器"""
    
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理梯度正则化
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.step_count += 1
        
        # 获取正则化前的统计信息
        pre_reg_stats = self.get_gradient_stats(model)
        
        # 应用L1正则化
        if self.config.l1_reg > 0:
            self._apply_l1_regularization(model)
        
        # 应用L2正则化
        if self.config.l2_reg > 0:
            self._apply_l2_regularization(model)
        
        # 应用谱正则化
        if self.config.spectral_reg > 0:
            self._apply_spectral_regularization(model)
        
        # 获取正则化后的统计信息
        post_reg_stats = self.get_gradient_stats(model)
        self.log_statistics(post_reg_stats)
        
        result = {
            'pre_reg_norm': pre_reg_stats['total_norm'],
            'post_reg_norm': post_reg_stats['total_norm'],
            'l1_penalty': self._compute_l1_penalty(model),
            'l2_penalty': self._compute_l2_penalty(model)
        }
        
        return result
    
    def _apply_l1_regularization(self, model: nn.Module) -> None:
        """应用L1正则化"""
        for param in model.parameters():
            if param.grad is not None:
                l1_grad = torch.sign(param.data) * self.config.l1_reg
                param.grad.data.add_(l1_grad)
    
    def _apply_l2_regularization(self, model: nn.Module) -> None:
        """应用L2正则化"""
        for param in model.parameters():
            if param.grad is not None:
                l2_grad = param.data * self.config.l2_reg
                param.grad.data.add_(l2_grad)
    
    def _apply_spectral_regularization(self, model: nn.Module) -> None:
        """应用谱正则化"""
        for name, param in model.named_parameters():
            if param.grad is not None and len(param.shape) >= 2:
                # 对权重矩阵应用谱正则化
                U, S, V = torch.svd(param.data)
                spectral_grad = torch.mm(torch.mm(U, torch.diag(torch.ones_like(S))), V.t()) * self.config.spectral_reg
                param.grad.data.add_(spectral_grad)
    
    def _compute_l1_penalty(self, model: nn.Module) -> float:
        """计算L1惩罚"""
        l1_penalty = 0.0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param)).item()
        return l1_penalty * self.config.l1_reg
    
    def _compute_l2_penalty(self, model: nn.Module) -> float:
        """计算L2惩罚"""
        l2_penalty = 0.0
        for param in model.parameters():
            l2_penalty += torch.sum(param ** 2).item()
        return l2_penalty * self.config.l2_reg * 0.5

class PhysicsInformedGradientProcessor(GradientProcessorBase):
    """物理信息梯度处理器"""
    
    def __init__(self, config: GradientProcessingConfig):
        super().__init__(config)
        self.loss_balance_history = deque(maxlen=50)
    
    def process_gradients(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理物理信息梯度
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self.step_count += 1
        
        if loss_dict is None:
            loss_dict = {}
        
        # 获取各类损失
        physics_loss = loss_dict.get('physics_loss', torch.tensor(0.0))
        data_loss = loss_dict.get('data_loss', torch.tensor(0.0))
        boundary_loss = loss_dict.get('boundary_loss', torch.tensor(0.0))
        
        # 分析损失平衡
        total_loss = physics_loss + data_loss + boundary_loss
        if total_loss > 0:
            physics_ratio = physics_loss / total_loss
            data_ratio = data_loss / total_loss
            boundary_ratio = boundary_loss / total_loss
            
            self.loss_balance_history.append({
                'physics': physics_ratio.item(),
                'data': data_ratio.item(),
                'boundary': boundary_ratio.item()
            })
        
        # 获取梯度统计信息
        stats = self.get_gradient_stats(model)
        
        # 应用物理信息梯度加权
        if self.config.physics_gradient_weighting:
            self._apply_physics_weighting(model, loss_dict)
        
        # 应用梯度手术（解决冲突梯度）
        if self.config.enable_gradient_surgery:
            self._apply_gradient_surgery(model, loss_dict)
        
        # 记录统计信息
        self.log_statistics(stats)
        
        result = {
            'gradient_norm': stats['total_norm'],
            'physics_ratio': physics_ratio.item() if 'physics_ratio' in locals() else 0.0,
            'data_ratio': data_ratio.item() if 'data_ratio' in locals() else 0.0,
            'boundary_ratio': boundary_ratio.item() if 'boundary_ratio' in locals() else 0.0,
            'loss_balance_score': self._compute_balance_score()
        }
        
        return result
    
    def _apply_physics_weighting(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor]) -> None:
        """应用物理信息梯度加权"""
        if len(self.loss_balance_history) < 10:
            return
        
        # 计算平均损失比例
        avg_ratios = {
            'physics': np.mean([h['physics'] for h in self.loss_balance_history]),
            'data': np.mean([h['data'] for h in self.loss_balance_history]),
            'boundary': np.mean([h['boundary'] for h in self.loss_balance_history])
        }
        
        # 计算加权因子
        target_ratio = 1.0 / 3.0  # 目标是平衡的
        physics_weight = target_ratio / (avg_ratios['physics'] + 1e-10)
        data_weight = target_ratio / (avg_ratios['data'] + 1e-10)
        boundary_weight = target_ratio / (avg_ratios['boundary'] + 1e-10)
        
        # 限制权重范围
        physics_weight = np.clip(physics_weight, 0.1, 10.0)
        data_weight = np.clip(data_weight, 0.1, 10.0)
        boundary_weight = np.clip(boundary_weight, 0.1, 10.0)
        
        # 应用权重（这里简化处理，实际应用中需要分别计算各部分梯度）
        for param in model.parameters():
            if param.grad is not None:
                # 简化的加权方案
                param.grad.data.mul_(0.5 * (physics_weight + data_weight))
    
    def _apply_gradient_surgery(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor]) -> None:
        """应用梯度手术解决冲突梯度"""
        # 这里需要分别计算物理损失和数据损失的梯度
        # 简化实现，实际中需要更复杂的梯度分离和冲突检测
        
        # 检测梯度冲突
        conflicts_detected = self._detect_gradient_conflicts(model)
        
        if conflicts_detected:
            # 应用梯度投影来解决冲突
            self._project_conflicting_gradients(model)
    
    def _detect_gradient_conflicts(self, model: nn.Module) -> bool:
        """检测梯度冲突"""
        # 简化的冲突检测：检查梯度方向的一致性
        if len(self.gradient_history) < 2:
            return False
        
        current_grads = []
        for param in model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.data.flatten())
        
        if not current_grads:
            return False
        
        current_grad_vector = torch.cat(current_grads)
        
        # 与历史梯度比较
        if len(self.gradient_history) > 0:
            prev_grad_vector = self.gradient_history[-1]
            cosine_sim = torch.cosine_similarity(current_grad_vector.unsqueeze(0), prev_grad_vector.unsqueeze(0))
            
            # 如果余弦相似度小于阈值，认为存在冲突
            return cosine_sim.item() < self.config.surgery_threshold
        
        return False
    
    def _project_conflicting_gradients(self, model: nn.Module) -> None:
        """投影冲突梯度"""
        # 简化的梯度投影实现
        for param in model.parameters():
            if param.grad is not None:
                # 应用简单的梯度平滑
                if len(self.gradient_history) > 0:
                    param.grad.data.mul_(0.7).add_(self.gradient_history[-1][:param.grad.numel()].view_as(param.grad), alpha=0.3)
    
    def _compute_balance_score(self) -> float:
        """计算损失平衡分数"""
        if len(self.loss_balance_history) < 5:
            return 1.0
        
        recent_ratios = self.loss_balance_history[-5:]
        
        # 计算方差作为不平衡的度量
        physics_var = np.var([h['physics'] for h in recent_ratios])
        data_var = np.var([h['data'] for h in recent_ratios])
        boundary_var = np.var([h['boundary'] for h in recent_ratios])
        
        total_var = physics_var + data_var + boundary_var
        
        # 平衡分数：方差越小，平衡性越好
        balance_score = 1.0 / (1.0 + total_var * 10)
        
        return balance_score

class GradientProcessor:
    """
    综合梯度处理器
    
    整合所有梯度处理功能
    """
    
    def __init__(self, config: GradientProcessingConfig):
        """
        初始化梯度处理器
        
        Args:
            config: 配置
        """
        self.config = config
        self.processors = []
        
        # 根据配置创建处理器
        if config.enable_clipping:
            self.processors.append(GradientClipper(config))
        
        if config.enable_accumulation:
            self.processors.append(GradientAccumulator(config))
        
        if config.enable_noise:
            self.processors.append(GradientNoiseInjector(config))
        
        if config.enable_regularization:
            self.processors.append(GradientRegularizer(config))
        
        if config.physics_gradient_weighting or config.enable_gradient_surgery:
            self.processors.append(PhysicsInformedGradientProcessor(config))
        
        self.logger = logging.getLogger(__name__)
    
    def process(self, model: nn.Module, loss_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        处理梯度
        
        Args:
            model: 模型
            loss_dict: 损失字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        results = {}
        
        # 应用梯度中心化
        if self.config.enable_gradient_centralization:
            self._apply_gradient_centralization(model)
        
        # 应用梯度标准化
        if self.config.enable_gradient_standardization:
            self._apply_gradient_standardization(model)
        
        # 依次应用所有处理器
        for i, processor in enumerate(self.processors):
            processor_result = processor.process_gradients(model, loss_dict)
            results[f"processor_{i}_{processor.__class__.__name__}"] = processor_result
        
        # 保存梯度历史
        if self.config.save_gradients:
            self._save_gradient_snapshot(model)
        
        return results
    
    def _apply_gradient_centralization(self, model: nn.Module) -> None:
        """应用梯度中心化"""
        for param in model.parameters():
            if param.grad is not None and len(param.grad.shape) > 1:
                # 对多维参数应用梯度中心化
                grad_mean = param.grad.mean(dim=tuple(range(1, len(param.grad.shape))), keepdim=True)
                param.grad.data.sub_(grad_mean)
    
    def _apply_gradient_standardization(self, model: nn.Module) -> None:
        """应用梯度标准化"""
        for param in model.parameters():
            if param.grad is not None:
                grad_std = param.grad.std()
                if grad_std > 1e-10:
                    param.grad.data.div_(grad_std)
    
    def _save_gradient_snapshot(self, model: nn.Module) -> None:
        """保存梯度快照"""
        gradient_snapshot = []
        for param in model.parameters():
            if param.grad is not None:
                gradient_snapshot.append(param.grad.data.clone().flatten())
        
        if gradient_snapshot:
            full_gradient = torch.cat(gradient_snapshot)
            
            # 保存到所有处理器的历史中
            for processor in self.processors:
                if hasattr(processor, 'gradient_history'):
                    processor.gradient_history.append(full_gradient)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        
        for i, processor in enumerate(self.processors):
            processor_stats = {}
            if hasattr(processor, 'statistics'):
                for key, values in processor.statistics.items():
                    if values:
                        processor_stats[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'latest': values[-1]
                        }
            
            stats[f"processor_{i}_{processor.__class__.__name__}"] = processor_stats
        
        return stats
    
    def plot_gradient_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制梯度历史
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, processor in enumerate(self.processors[:4]):
            if hasattr(processor, 'statistics') and 'total_norm' in processor.statistics:
                norms = processor.statistics['total_norm']
                if norms:
                    axes[i].plot(norms)
                    axes[i].set_title(f"{processor.__class__.__name__} - Gradient Norm")
                    axes[i].set_xlabel('Steps')
                    axes[i].set_ylabel('Gradient Norm')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def create_gradient_processor(config: GradientProcessingConfig = None) -> GradientProcessor:
    """
    创建梯度处理器
    
    Args:
        config: 配置
        
    Returns:
        GradientProcessor: 梯度处理器实例
    """
    if config is None:
        config = GradientProcessingConfig()
    
    return GradientProcessor(config)

if __name__ == "__main__":
    # 测试梯度处理器
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建梯度处理配置
    config = GradientProcessingConfig(
        enable_clipping=True,
        clip_type=GradientClipType.ADAPTIVE,
        enable_noise=True,
        noise_type=GradientNoiseType.GAUSSIAN,
        enable_regularization=True,
        l2_reg=1e-4,
        enable_monitoring=True,
        physics_gradient_weighting=True
    )
    
    # 创建梯度处理器
    gradient_processor = create_gradient_processor(config)
    
    print("=== 梯度处理器测试 ===")
    
    # 模拟训练过程
    for step in range(50):
        # 生成随机数据
        inputs = torch.randn(32, 3)
        targets = torch.sin(inputs.sum(dim=1, keepdim=True))
        
        # 前向传播
        predictions = model(inputs)
        
        # 计算损失
        data_loss = nn.MSELoss()(predictions, targets)
        physics_loss = torch.mean(predictions ** 2) * 0.1  # 简化的物理损失
        boundary_loss = torch.mean(torch.abs(predictions)) * 0.05  # 简化的边界损失
        
        total_loss = data_loss + physics_loss + boundary_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 处理梯度
        loss_dict = {
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'boundary_loss': boundary_loss
        }
        
        results = gradient_processor.process(model, loss_dict)
        
        # 优化器步骤
        optimizer.step()
        
        # 打印结果
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Data Loss: {data_loss.item():.6f}")
            print(f"  Physics Loss: {physics_loss.item():.6f}")
            print(f"  Boundary Loss: {boundary_loss.item():.6f}")
            
            for key, result in results.items():
                if isinstance(result, dict) and 'gradient_norm' in result:
                    print(f"  {key}: Gradient Norm = {result['gradient_norm']:.6f}")
    
    # 获取统计信息
    stats = gradient_processor.get_statistics()
    print(f"\n=== 统计信息 ===")
    for processor_name, processor_stats in stats.items():
        print(f"\n{processor_name}:")
        for metric_name, metric_stats in processor_stats.items():
            if isinstance(metric_stats, dict):
                print(f"  {metric_name}: mean={metric_stats['mean']:.6f}, std={metric_stats['std']:.6f}")
    
    print("\n梯度处理器测试完成！")