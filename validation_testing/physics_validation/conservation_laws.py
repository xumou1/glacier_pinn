#!/usr/bin/env python3
"""
守恒定律验证

实现物理信息神经网络中守恒定律的验证，包括：
- 质量守恒验证
- 动量守恒验证
- 能量守恒验证
- 角动量守恒验证
- 通用守恒定律验证框架

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
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy import integrate
from scipy.spatial import distance_matrix

class ConservationLawType(Enum):
    """守恒定律类型枚举"""
    MASS = "mass"  # 质量守恒
    MOMENTUM = "momentum"  # 动量守恒
    ENERGY = "energy"  # 能量守恒
    ANGULAR_MOMENTUM = "angular_momentum"  # 角动量守恒
    CHARGE = "charge"  # 电荷守恒
    CUSTOM = "custom"  # 自定义守恒定律

class ValidationLevel(Enum):
    """验证级别枚举"""
    STRICT = "strict"  # 严格验证
    MODERATE = "moderate"  # 中等验证
    RELAXED = "relaxed"  # 宽松验证

@dataclass
class ConservationConfig:
    """守恒定律验证配置"""
    # 基础配置
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    tolerance: float = 1e-6  # 容差
    relative_tolerance: float = 1e-4  # 相对容差
    
    # 采样配置
    num_test_points: int = 1000  # 测试点数量
    spatial_domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)  # 空间域 (x_min, x_max, y_min, y_max)
    temporal_domain: Tuple[float, float] = (0.0, 1.0)  # 时间域 (t_min, t_max)
    
    # 数值微分配置
    finite_diff_step: float = 1e-5  # 有限差分步长
    use_automatic_diff: bool = True  # 使用自动微分
    
    # 积分配置
    integration_method: str = "simpson"  # 积分方法
    integration_points: int = 100  # 积分点数
    
    # 可视化配置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_directory: str = "./conservation_plots"
    
    # 日志配置
    log_level: str = "INFO"
    detailed_logging: bool = False

class ConservationValidatorBase(ABC):
    """
    守恒定律验证器基类
    
    定义守恒定律验证的通用接口
    """
    
    def __init__(self, config: ConservationConfig):
        """
        初始化验证器
        
        Args:
            config: 配置
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # 创建绘图目录
        if config.save_plots:
            Path(config.plot_directory).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """验证守恒定律"""
        pass
    
    def _generate_test_points(self, num_points: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        生成测试点
        
        Args:
            num_points: 测试点数量
            
        Returns:
            Dict[str, torch.Tensor]: 测试点数据
        """
        if num_points is None:
            num_points = self.config.num_test_points
        
        # 空间坐标
        x_min, x_max, y_min, y_max = self.config.spatial_domain
        x = torch.linspace(x_min, x_max, int(np.sqrt(num_points)))
        y = torch.linspace(y_min, y_max, int(np.sqrt(num_points)))
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 时间坐标
        t_min, t_max = self.config.temporal_domain
        t = torch.linspace(t_min, t_max, num_points)
        
        # 随机采样
        indices = torch.randperm(X.numel())[:num_points]
        x_flat = X.flatten()[indices]
        y_flat = Y.flatten()[indices]
        t_sample = t[:len(x_flat)]
        
        return {
            'x': x_flat.unsqueeze(1),
            'y': y_flat.unsqueeze(1),
            't': t_sample.unsqueeze(1)
        }
    
    def _compute_derivatives(self, model: nn.Module, inputs: torch.Tensor, 
                           output_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算导数
        
        Args:
            model: 模型
            inputs: 输入张量 [N, 3] (x, y, t)
            output_idx: 输出索引
            
        Returns:
            Dict[str, torch.Tensor]: 导数字典
        """
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        if outputs.dim() > 1:
            u = outputs[:, output_idx:output_idx+1]
        else:
            u = outputs.unsqueeze(1)
        
        # 一阶导数
        grad_u = torch.autograd.grad(
            outputs=u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]
        
        # 二阶导数
        u_xx = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            outputs=u_y.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        u_xy = torch.autograd.grad(
            outputs=u_x.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        return {
            'u': u,
            'u_x': u_x,
            'u_y': u_y,
            'u_t': u_t,
            'u_xx': u_xx,
            'u_yy': u_yy,
            'u_xy': u_xy
        }
    
    def _check_tolerance(self, residual: torch.Tensor, name: str) -> Dict[str, Any]:
        """
        检查容差
        
        Args:
            residual: 残差
            name: 守恒定律名称
            
        Returns:
            Dict[str, Any]: 检查结果
        """
        abs_residual = torch.abs(residual)
        max_residual = torch.max(abs_residual).item()
        mean_residual = torch.mean(abs_residual).item()
        std_residual = torch.std(abs_residual).item()
        
        # 绝对容差检查
        abs_passed = max_residual < self.config.tolerance
        
        # 相对容差检查（如果有参考值）
        rel_passed = True
        if torch.any(torch.abs(residual) > 0):
            ref_value = torch.mean(torch.abs(residual)).item()
            if ref_value > 0:
                rel_residual = max_residual / ref_value
                rel_passed = rel_residual < self.config.relative_tolerance
        
        # 根据验证级别确定是否通过
        if self.config.validation_level == ValidationLevel.STRICT:
            passed = abs_passed and rel_passed
        elif self.config.validation_level == ValidationLevel.MODERATE:
            passed = abs_passed or rel_passed
        else:  # RELAXED
            passed = max_residual < self.config.tolerance * 10
        
        result = {
            'passed': passed,
            'max_residual': max_residual,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'abs_passed': abs_passed,
            'rel_passed': rel_passed,
            'conservation_law': name
        }
        
        # 记录日志
        status = "PASSED" if passed else "FAILED"
        self.logger.info(
            f"{name} Conservation: {status} - "
            f"Max: {max_residual:.2e}, Mean: {mean_residual:.2e}, Std: {std_residual:.2e}"
        )
        
        return result

class MassConservationValidator(ConservationValidatorBase):
    """质量守恒验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证质量守恒定律
        
        质量守恒方程: ∂ρ/∂t + ∇·(ρv) = 0
        其中 ρ 是密度，v 是速度场
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        inputs.requires_grad_(True)
        
        # 假设模型输出 [密度, 速度x, 速度y]
        outputs = model(inputs)
        
        if outputs.shape[1] < 3:
            self.logger.warning("模型输出维度不足，无法进行质量守恒验证")
            return {'passed': False, 'error': '模型输出维度不足'}
        
        rho = outputs[:, 0:1]  # 密度
        u = outputs[:, 1:2]    # x方向速度
        v = outputs[:, 2:3]    # y方向速度
        
        # 计算密度的时间导数
        rho_t = torch.autograd.grad(
            outputs=rho.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 2:3]
        
        # 计算密度通量的散度
        rho_u = rho * u
        rho_v = rho * v
        
        rho_u_x = torch.autograd.grad(
            outputs=rho_u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        rho_v_y = torch.autograd.grad(
            outputs=rho_v.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # 质量守恒残差
        mass_residual = rho_t + rho_u_x + rho_v_y
        
        # 检查容差
        result = self._check_tolerance(mass_residual, "Mass")
        result['residual'] = mass_residual.detach()
        result['test_points'] = inputs.detach()
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_mass_conservation(inputs.detach(), mass_residual.detach(), result)
        
        return result
    
    def _plot_mass_conservation(self, inputs: torch.Tensor, residual: torch.Tensor, 
                              result: Dict[str, Any]) -> None:
        """绘制质量守恒验证结果"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 残差分布
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            res = residual.squeeze().numpy()
            
            scatter = axes[0].scatter(x, y, c=res, cmap='RdBu_r', alpha=0.6)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].set_title('Mass Conservation Residual')
            plt.colorbar(scatter, ax=axes[0])
            
            # 残差直方图
            axes[1].hist(res, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', label='Perfect Conservation')
            axes[1].set_xlabel('Residual')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residual Distribution')
            axes[1].legend()
            
            # 添加统计信息
            stats_text = f"Max: {result['max_residual']:.2e}\nMean: {result['mean_residual']:.2e}\nStd: {result['std_residual']:.2e}"
            axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "mass_conservation.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制质量守恒图失败: {e}")

class MomentumConservationValidator(ConservationValidatorBase):
    """动量守恒验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证动量守恒定律
        
        动量守恒方程: ∂(ρv)/∂t + ∇·(ρv⊗v) + ∇p = f
        其中 ρ 是密度，v 是速度，p 是压力，f 是外力
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        inputs.requires_grad_(True)
        
        # 假设模型输出 [密度, 速度x, 速度y, 压力]
        outputs = model(inputs)
        
        if outputs.shape[1] < 4:
            self.logger.warning("模型输出维度不足，无法进行动量守恒验证")
            return {'passed': False, 'error': '模型输出维度不足'}
        
        rho = outputs[:, 0:1]  # 密度
        u = outputs[:, 1:2]    # x方向速度
        v = outputs[:, 2:3]    # y方向速度
        p = outputs[:, 3:4]    # 压力
        
        # x方向动量守恒
        rho_u = rho * u
        
        # 时间导数
        rho_u_t = torch.autograd.grad(
            outputs=rho_u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 2:3]
        
        # 对流项
        rho_u_u = rho * u * u
        rho_u_v = rho * u * v
        
        rho_u_u_x = torch.autograd.grad(
            outputs=rho_u_u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        rho_u_v_y = torch.autograd.grad(
            outputs=rho_u_v.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # 压力梯度
        p_x = torch.autograd.grad(
            outputs=p.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        # x方向动量守恒残差（忽略外力）
        momentum_x_residual = rho_u_t + rho_u_u_x + rho_u_v_y + p_x
        
        # y方向动量守恒
        rho_v = rho * v
        
        rho_v_t = torch.autograd.grad(
            outputs=rho_v.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 2:3]
        
        rho_v_u = rho * v * u
        rho_v_v = rho * v * v
        
        rho_v_u_x = torch.autograd.grad(
            outputs=rho_v_u.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        rho_v_v_y = torch.autograd.grad(
            outputs=rho_v_v.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        p_y = torch.autograd.grad(
            outputs=p.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # y方向动量守恒残差
        momentum_y_residual = rho_v_t + rho_v_u_x + rho_v_v_y + p_y
        
        # 总动量守恒残差
        total_momentum_residual = torch.sqrt(momentum_x_residual**2 + momentum_y_residual**2)
        
        # 检查容差
        result = self._check_tolerance(total_momentum_residual, "Momentum")
        result['x_residual'] = momentum_x_residual.detach()
        result['y_residual'] = momentum_y_residual.detach()
        result['total_residual'] = total_momentum_residual.detach()
        result['test_points'] = inputs.detach()
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_momentum_conservation(inputs.detach(), momentum_x_residual.detach(), 
                                           momentum_y_residual.detach(), result)
        
        return result
    
    def _plot_momentum_conservation(self, inputs: torch.Tensor, x_residual: torch.Tensor, 
                                  y_residual: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制动量守恒验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            x_res = x_residual.squeeze().numpy()
            y_res = y_residual.squeeze().numpy()
            
            # x方向动量残差
            scatter1 = axes[0, 0].scatter(x, y, c=x_res, cmap='RdBu_r', alpha=0.6)
            axes[0, 0].set_title('X-Momentum Residual')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[0, 0])
            
            # y方向动量残差
            scatter2 = axes[0, 1].scatter(x, y, c=y_res, cmap='RdBu_r', alpha=0.6)
            axes[0, 1].set_title('Y-Momentum Residual')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[0, 1])
            
            # x方向残差直方图
            axes[1, 0].hist(x_res, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--')
            axes[1, 0].set_title('X-Momentum Residual Distribution')
            axes[1, 0].set_xlabel('Residual')
            axes[1, 0].set_ylabel('Frequency')
            
            # y方向残差直方图
            axes[1, 1].hist(y_res, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Y-Momentum Residual Distribution')
            axes[1, 1].set_xlabel('Residual')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "momentum_conservation.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制动量守恒图失败: {e}")

class EnergyConservationValidator(ConservationValidatorBase):
    """能量守恒验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证能量守恒定律
        
        能量守恒方程: ∂E/∂t + ∇·(Ev + pv) = Q
        其中 E 是总能量密度，v 是速度，p 是压力，Q 是热源
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([test_data['x'], test_data['y'], test_data['t']], dim=1)
        inputs.requires_grad_(True)
        
        # 假设模型输出 [密度, 速度x, 速度y, 压力, 温度]
        outputs = model(inputs)
        
        if outputs.shape[1] < 5:
            self.logger.warning("模型输出维度不足，无法进行能量守恒验证")
            return {'passed': False, 'error': '模型输出维度不足'}
        
        rho = outputs[:, 0:1]  # 密度
        u = outputs[:, 1:2]    # x方向速度
        v = outputs[:, 2:3]    # y方向速度
        p = outputs[:, 3:4]    # 压力
        T = outputs[:, 4:5]    # 温度
        
        # 计算总能量密度（动能 + 内能）
        kinetic_energy = 0.5 * rho * (u**2 + v**2)
        internal_energy = rho * T  # 简化的内能
        total_energy = kinetic_energy + internal_energy
        
        # 能量的时间导数
        E_t = torch.autograd.grad(
            outputs=total_energy.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 2:3]
        
        # 能量通量
        energy_flux_x = total_energy * u + p * u
        energy_flux_y = total_energy * v + p * v
        
        # 能量通量散度
        flux_x_div = torch.autograd.grad(
            outputs=energy_flux_x.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        flux_y_div = torch.autograd.grad(
            outputs=energy_flux_y.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # 能量守恒残差（忽略热源）
        energy_residual = E_t + flux_x_div + flux_y_div
        
        # 检查容差
        result = self._check_tolerance(energy_residual, "Energy")
        result['residual'] = energy_residual.detach()
        result['kinetic_energy'] = kinetic_energy.detach()
        result['internal_energy'] = internal_energy.detach()
        result['total_energy'] = total_energy.detach()
        result['test_points'] = inputs.detach()
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_energy_conservation(inputs.detach(), energy_residual.detach(), 
                                         total_energy.detach(), result)
        
        return result
    
    def _plot_energy_conservation(self, inputs: torch.Tensor, residual: torch.Tensor, 
                                energy: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制能量守恒验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            res = residual.squeeze().numpy()
            eng = energy.squeeze().numpy()
            
            # 能量分布
            scatter1 = axes[0, 0].scatter(x, y, c=eng, cmap='viridis', alpha=0.6)
            axes[0, 0].set_title('Total Energy Distribution')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[0, 0])
            
            # 能量守恒残差
            scatter2 = axes[0, 1].scatter(x, y, c=res, cmap='RdBu_r', alpha=0.6)
            axes[0, 1].set_title('Energy Conservation Residual')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[0, 1])
            
            # 残差直方图
            axes[1, 0].hist(res, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--')
            axes[1, 0].set_title('Residual Distribution')
            axes[1, 0].set_xlabel('Residual')
            axes[1, 0].set_ylabel('Frequency')
            
            # 能量vs残差散点图
            axes[1, 1].scatter(eng, res, alpha=0.6)
            axes[1, 1].set_xlabel('Total Energy')
            axes[1, 1].set_ylabel('Conservation Residual')
            axes[1, 1].set_title('Energy vs Residual')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "energy_conservation.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制能量守恒图失败: {e}")

class ConservationLawValidator:
    """
    守恒定律验证器
    
    整合所有守恒定律验证功能
    """
    
    def __init__(self, config: ConservationConfig = None):
        """
        初始化验证器
        
        Args:
            config: 配置
        """
        if config is None:
            config = ConservationConfig()
        
        self.config = config
        self.validators = {
            ConservationLawType.MASS: MassConservationValidator(config),
            ConservationLawType.MOMENTUM: MomentumConservationValidator(config),
            ConservationLawType.ENERGY: EnergyConservationValidator(config)
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate_all(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证所有守恒定律
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for law_type, validator in self.validators.items():
            try:
                result = validator.validate(model, test_data)
                results[law_type.value] = result
                
                if not result.get('passed', False):
                    overall_passed = False
                    
            except Exception as e:
                self.logger.error(f"验证 {law_type.value} 守恒定律失败: {e}")
                results[law_type.value] = {
                    'passed': False,
                    'error': str(e)
                }
                overall_passed = False
        
        # 综合结果
        summary = {
            'overall_passed': overall_passed,
            'num_laws_tested': len(self.validators),
            'num_laws_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'validation_level': self.config.validation_level.value,
            'tolerance': self.config.tolerance,
            'results': results
        }
        
        self.logger.info(
            f"守恒定律验证完成: {summary['num_laws_passed']}/{summary['num_laws_tested']} 通过, "
            f"总体结果: {'PASSED' if overall_passed else 'FAILED'}"
        )
        
        return summary
    
    def validate_specific(self, model: nn.Module, law_types: List[ConservationLawType], 
                         test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证特定守恒定律
        
        Args:
            model: 模型
            law_types: 要验证的守恒定律类型列表
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for law_type in law_types:
            if law_type in self.validators:
                try:
                    result = self.validators[law_type].validate(model, test_data)
                    results[law_type.value] = result
                    
                    if not result.get('passed', False):
                        overall_passed = False
                        
                except Exception as e:
                    self.logger.error(f"验证 {law_type.value} 守恒定律失败: {e}")
                    results[law_type.value] = {
                        'passed': False,
                        'error': str(e)
                    }
                    overall_passed = False
            else:
                self.logger.warning(f"不支持的守恒定律类型: {law_type.value}")
        
        summary = {
            'overall_passed': overall_passed,
            'num_laws_tested': len(law_types),
            'num_laws_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'results': results
        }
        
        return summary
    
    def generate_report(self, validation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果
            save_path: 保存路径
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "=" * 60,
            "守恒定律验证报告",
            "=" * 60,
            f"验证时间: {torch.datetime.now()}",
            f"验证级别: {self.config.validation_level.value}",
            f"容差设置: 绝对={self.config.tolerance:.2e}, 相对={self.config.relative_tolerance:.2e}",
            "",
            "总体结果:",
            f"  状态: {'通过' if validation_results['overall_passed'] else '失败'}",
            f"  通过率: {validation_results['num_laws_passed']}/{validation_results['num_laws_tested']}",
            ""
        ]
        
        # 详细结果
        report_lines.append("详细结果:")
        for law_name, result in validation_results['results'].items():
            if 'error' in result:
                report_lines.extend([
                    f"  {law_name.upper()} 守恒:",
                    f"    状态: 错误",
                    f"    错误信息: {result['error']}",
                    ""
                ])
            else:
                status = "通过" if result['passed'] else "失败"
                report_lines.extend([
                    f"  {law_name.upper()} 守恒:",
                    f"    状态: {status}",
                    f"    最大残差: {result['max_residual']:.2e}",
                    f"    平均残差: {result['mean_residual']:.2e}",
                    f"    标准差: {result['std_residual']:.2e}",
                    ""
                ])
        
        report_lines.append("=" * 60)
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"验证报告已保存到: {save_path}")
            except Exception as e:
                self.logger.error(f"保存验证报告失败: {e}")
        
        return report_content

def create_conservation_validator(config: ConservationConfig = None) -> ConservationLawValidator:
    """
    创建守恒定律验证器
    
    Args:
        config: 配置
        
    Returns:
        ConservationLawValidator: 验证器实例
    """
    return ConservationLawValidator(config)

if __name__ == "__main__":
    # 测试守恒定律验证器
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 5)  # 输出: [密度, 速度x, 速度y, 压力, 温度]
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # 创建配置
    config = ConservationConfig(
        validation_level=ValidationLevel.MODERATE,
        tolerance=1e-4,
        num_test_points=500,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_conservation_validator(config)
    
    print("=== 守恒定律验证器测试 ===")
    
    # 验证所有守恒定律
    results = validator.validate_all(model)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    # 验证特定守恒定律
    print("\n=== 验证特定守恒定律 ===")
    specific_results = validator.validate_specific(
        model, 
        [ConservationLawType.MASS, ConservationLawType.ENERGY]
    )
    
    print(f"特定验证结果: {specific_results['num_laws_passed']}/{specific_results['num_laws_tested']} 通过")
    
    print("\n守恒定律验证器测试完成！")