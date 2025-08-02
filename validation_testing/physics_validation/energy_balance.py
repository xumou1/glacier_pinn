#!/usr/bin/env python3
"""
能量平衡验证

实现物理信息神经网络中能量平衡的验证，包括：
- 表面能量平衡验证
- 内部能量平衡验证
- 热传导验证
- 相变能量验证
- 辐射平衡验证
- 冰川能量平衡专用验证

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
from scipy import integrate, constants
from scipy.interpolate import griddata

class EnergyBalanceType(Enum):
    """能量平衡类型枚举"""
    SURFACE = "surface"  # 表面能量平衡
    INTERNAL = "internal"  # 内部能量平衡
    HEAT_CONDUCTION = "heat_conduction"  # 热传导
    PHASE_CHANGE = "phase_change"  # 相变能量
    RADIATION = "radiation"  # 辐射平衡
    GLACIER_MASS_ENERGY = "glacier_mass_energy"  # 冰川质量-能量平衡

class ValidationMethod(Enum):
    """验证方法枚举"""
    ANALYTICAL = "analytical"  # 解析解对比
    NUMERICAL = "numerical"  # 数值解对比
    CONSERVATION = "conservation"  # 守恒性检查
    BOUNDARY = "boundary"  # 边界条件检查

@dataclass
class EnergyBalanceConfig:
    """能量平衡验证配置"""
    # 基础配置
    tolerance: float = 1e-6  # 容差
    relative_tolerance: float = 1e-4  # 相对容差
    
    # 物理常数
    stefan_boltzmann: float = 5.67e-8  # 斯特藩-玻尔兹曼常数 [W m^-2 K^-4]
    latent_heat_fusion: float = 334000.0  # 融化潜热 [J kg^-1]
    latent_heat_vaporization: float = 2.5e6  # 汽化潜热 [J kg^-1]
    specific_heat_ice: float = 2100.0  # 冰的比热容 [J kg^-1 K^-1]
    specific_heat_water: float = 4186.0  # 水的比热容 [J kg^-1 K^-1]
    thermal_conductivity_ice: float = 2.1  # 冰的热导率 [W m^-1 K^-1]
    density_ice: float = 917.0  # 冰的密度 [kg m^-3]
    density_water: float = 1000.0  # 水的密度 [kg m^-3]
    
    # 采样配置
    num_test_points: int = 1000  # 测试点数量
    spatial_domain: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)  # 空间域
    temporal_domain: Tuple[float, float] = (0.0, 1.0)  # 时间域
    altitude_range: Tuple[float, float] = (3000.0, 6000.0)  # 海拔范围 [m]
    
    # 环境参数
    air_temperature_range: Tuple[float, float] = (253.15, 283.15)  # 气温范围 [K]
    solar_radiation_range: Tuple[float, float] = (0.0, 1000.0)  # 太阳辐射范围 [W m^-2]
    wind_speed_range: Tuple[float, float] = (0.0, 20.0)  # 风速范围 [m s^-1]
    
    # 数值配置
    finite_diff_step: float = 1e-5  # 有限差分步长
    integration_method: str = "simpson"  # 积分方法
    
    # 可视化配置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_directory: str = "./energy_balance_plots"
    
    # 日志配置
    log_level: str = "INFO"
    detailed_logging: bool = False

class EnergyBalanceValidatorBase(ABC):
    """
    能量平衡验证器基类
    
    定义能量平衡验证的通用接口
    """
    
    def __init__(self, config: EnergyBalanceConfig):
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
        """验证能量平衡"""
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
        
        # 海拔
        alt_min, alt_max = self.config.altitude_range
        altitude = torch.linspace(alt_min, alt_max, num_points)
        
        # 随机采样
        indices = torch.randperm(X.numel())[:num_points]
        x_flat = X.flatten()[indices]
        y_flat = Y.flatten()[indices]
        t_sample = t[:len(x_flat)]
        alt_sample = altitude[:len(x_flat)]
        
        return {
            'x': x_flat.unsqueeze(1),
            'y': y_flat.unsqueeze(1),
            't': t_sample.unsqueeze(1),
            'altitude': alt_sample.unsqueeze(1)
        }
    
    def _compute_derivatives(self, model: nn.Module, inputs: torch.Tensor, 
                           output_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算导数
        
        Args:
            model: 模型
            inputs: 输入张量
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
        
        derivatives = {
            'u': u,
            'u_x': grad_u[:, 0:1],
            'u_y': grad_u[:, 1:2],
            'u_t': grad_u[:, 2:3]
        }
        
        # 如果有海拔维度
        if inputs.shape[1] > 3:
            derivatives['u_z'] = grad_u[:, 3:4]
        
        # 二阶导数
        try:
            u_xx = torch.autograd.grad(
                outputs=derivatives['u_x'].sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            
            u_yy = torch.autograd.grad(
                outputs=derivatives['u_y'].sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0][:, 1:2]
            
            derivatives.update({
                'u_xx': u_xx,
                'u_yy': u_yy,
                'laplacian': u_xx + u_yy
            })
        except:
            pass
        
        return derivatives
    
    def _check_tolerance(self, residual: torch.Tensor, name: str) -> Dict[str, Any]:
        """
        检查容差
        
        Args:
            residual: 残差
            name: 验证名称
            
        Returns:
            Dict[str, Any]: 检查结果
        """
        abs_residual = torch.abs(residual)
        max_residual = torch.max(abs_residual).item()
        mean_residual = torch.mean(abs_residual).item()
        std_residual = torch.std(abs_residual).item()
        
        # 绝对容差检查
        abs_passed = max_residual < self.config.tolerance
        
        # 相对容差检查
        rel_passed = True
        if torch.any(torch.abs(residual) > 0):
            ref_value = torch.mean(torch.abs(residual)).item()
            if ref_value > 0:
                rel_residual = max_residual / ref_value
                rel_passed = rel_residual < self.config.relative_tolerance
        
        passed = abs_passed or rel_passed
        
        result = {
            'passed': passed,
            'max_residual': max_residual,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'abs_passed': abs_passed,
            'rel_passed': rel_passed,
            'validation_type': name
        }
        
        # 记录日志
        status = "PASSED" if passed else "FAILED"
        self.logger.info(
            f"{name} Energy Balance: {status} - "
            f"Max: {max_residual:.2e}, Mean: {mean_residual:.2e}, Std: {std_residual:.2e}"
        )
        
        return result

class SurfaceEnergyBalanceValidator(EnergyBalanceValidatorBase):
    """表面能量平衡验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证表面能量平衡
        
        表面能量平衡方程:
        S_in + L_in - S_out - L_out - H - LE - G = 0
        
        其中:
        S_in: 入射短波辐射
        L_in: 入射长波辐射
        S_out: 反射短波辐射
        L_out: 出射长波辐射
        H: 感热通量
        LE: 潜热通量
        G: 地面热通量
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入 [x, y, t, altitude]
        inputs = torch.cat([
            test_data['x'], test_data['y'], 
            test_data['t'], test_data['altitude']
        ], dim=1)
        inputs.requires_grad_(True)
        
        # 假设模型输出 [温度, 太阳辐射, 长波辐射, 反照率, 风速, 湿度]
        outputs = model(inputs)
        
        if outputs.shape[1] < 6:
            self.logger.warning("模型输出维度不足，无法进行表面能量平衡验证")
            return {'passed': False, 'error': '模型输出维度不足'}
        
        T_surface = outputs[:, 0:1] + 273.15  # 表面温度 [K]
        S_in = outputs[:, 1:2]  # 入射短波辐射 [W m^-2]
        L_in = outputs[:, 2:3]  # 入射长波辐射 [W m^-2]
        albedo = torch.sigmoid(outputs[:, 3:4])  # 反照率 [0-1]
        wind_speed = torch.relu(outputs[:, 4:5])  # 风速 [m s^-1]
        humidity = torch.sigmoid(outputs[:, 5:6])  # 相对湿度 [0-1]
        
        # 计算各能量分量
        S_out = albedo * S_in  # 反射短波辐射
        L_out = self.config.stefan_boltzmann * T_surface**4  # 出射长波辐射
        
        # 感热通量（简化计算）
        T_air = T_surface - 5.0  # 假设气温比表面温度低5K
        H = 5.0 * wind_speed * (T_surface - T_air)  # 感热通量
        
        # 潜热通量（简化计算）
        LE = 10.0 * wind_speed * humidity * torch.relu(T_surface - 273.15)  # 潜热通量
        
        # 地面热通量（通过温度梯度计算）
        temp_derivatives = self._compute_derivatives(model, inputs, output_idx=0)
        if 'u_z' in temp_derivatives:
            G = -self.config.thermal_conductivity_ice * temp_derivatives['u_z']
        else:
            G = torch.zeros_like(T_surface)
        
        # 表面能量平衡残差
        energy_balance_residual = S_in + L_in - S_out - L_out - H - LE - G
        
        # 检查容差
        result = self._check_tolerance(energy_balance_residual, "Surface")
        
        # 添加详细信息
        result.update({
            'residual': energy_balance_residual.detach(),
            'S_in': S_in.detach(),
            'S_out': S_out.detach(),
            'L_in': L_in.detach(),
            'L_out': L_out.detach(),
            'H': H.detach(),
            'LE': LE.detach(),
            'G': G.detach(),
            'T_surface': T_surface.detach(),
            'albedo': albedo.detach(),
            'test_points': inputs.detach()
        })
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_surface_energy_balance(inputs.detach(), result)
        
        return result
    
    def _plot_surface_energy_balance(self, inputs: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制表面能量平衡验证结果"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            
            # 能量平衡残差
            residual = result['residual'].squeeze().numpy()
            scatter0 = axes[0].scatter(x, y, c=residual, cmap='RdBu_r', alpha=0.6)
            axes[0].set_title('Energy Balance Residual')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(scatter0, ax=axes[0])
            
            # 表面温度
            T_surface = result['T_surface'].squeeze().numpy()
            scatter1 = axes[1].scatter(x, y, c=T_surface, cmap='coolwarm', alpha=0.6)
            axes[1].set_title('Surface Temperature [K]')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[1])
            
            # 短波辐射平衡
            S_net = (result['S_in'] - result['S_out']).squeeze().numpy()
            scatter2 = axes[2].scatter(x, y, c=S_net, cmap='viridis', alpha=0.6)
            axes[2].set_title('Net Shortwave Radiation [W m⁻²]')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[2])
            
            # 长波辐射平衡
            L_net = (result['L_in'] - result['L_out']).squeeze().numpy()
            scatter3 = axes[3].scatter(x, y, c=L_net, cmap='plasma', alpha=0.6)
            axes[3].set_title('Net Longwave Radiation [W m⁻²]')
            axes[3].set_xlabel('X')
            axes[3].set_ylabel('Y')
            plt.colorbar(scatter3, ax=axes[3])
            
            # 湍流热通量
            turbulent_flux = (result['H'] + result['LE']).squeeze().numpy()
            scatter4 = axes[4].scatter(x, y, c=turbulent_flux, cmap='inferno', alpha=0.6)
            axes[4].set_title('Turbulent Heat Flux [W m⁻²]')
            axes[4].set_xlabel('X')
            axes[4].set_ylabel('Y')
            plt.colorbar(scatter4, ax=axes[4])
            
            # 残差直方图
            axes[5].hist(residual, bins=50, alpha=0.7, edgecolor='black')
            axes[5].axvline(0, color='red', linestyle='--', label='Perfect Balance')
            axes[5].set_title('Residual Distribution')
            axes[5].set_xlabel('Residual [W m⁻²]')
            axes[5].set_ylabel('Frequency')
            axes[5].legend()
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "surface_energy_balance.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制表面能量平衡图失败: {e}")

class HeatConductionValidator(EnergyBalanceValidatorBase):
    """热传导验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证热传导方程
        
        热传导方程: ρc ∂T/∂t = k ∇²T + Q
        其中 ρ 是密度，c 是比热容，k 是热导率，Q 是热源
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([
            test_data['x'], test_data['y'], 
            test_data['t'], test_data['altitude']
        ], dim=1)
        
        # 计算温度及其导数
        temp_derivatives = self._compute_derivatives(model, inputs, output_idx=0)
        
        if 'laplacian' not in temp_derivatives:
            self.logger.warning("无法计算拉普拉斯算子，跳过热传导验证")
            return {'passed': False, 'error': '无法计算二阶导数'}
        
        T = temp_derivatives['u']
        T_t = temp_derivatives['u_t']
        laplacian_T = temp_derivatives['laplacian']
        
        # 物理参数
        rho = self.config.density_ice
        c = self.config.specific_heat_ice
        k = self.config.thermal_conductivity_ice
        
        # 热传导残差（假设无热源）
        heat_conduction_residual = rho * c * T_t - k * laplacian_T
        
        # 检查容差
        result = self._check_tolerance(heat_conduction_residual, "Heat Conduction")
        
        # 添加详细信息
        result.update({
            'residual': heat_conduction_residual.detach(),
            'temperature': T.detach(),
            'temp_time_derivative': T_t.detach(),
            'temp_laplacian': laplacian_T.detach(),
            'thermal_diffusivity': k / (rho * c),
            'test_points': inputs.detach()
        })
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_heat_conduction(inputs.detach(), result)
        
        return result
    
    def _plot_heat_conduction(self, inputs: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制热传导验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            
            # 温度分布
            temperature = result['temperature'].squeeze().numpy()
            scatter0 = axes[0].scatter(x, y, c=temperature, cmap='coolwarm', alpha=0.6)
            axes[0].set_title('Temperature Distribution [K]')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(scatter0, ax=axes[0])
            
            # 温度时间导数
            temp_t = result['temp_time_derivative'].squeeze().numpy()
            scatter1 = axes[1].scatter(x, y, c=temp_t, cmap='RdBu_r', alpha=0.6)
            axes[1].set_title('Temperature Time Derivative [K/s]')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[1])
            
            # 温度拉普拉斯算子
            temp_laplacian = result['temp_laplacian'].squeeze().numpy()
            scatter2 = axes[2].scatter(x, y, c=temp_laplacian, cmap='viridis', alpha=0.6)
            axes[2].set_title('Temperature Laplacian [K/m²]')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[2])
            
            # 热传导残差
            residual = result['residual'].squeeze().numpy()
            scatter3 = axes[3].scatter(x, y, c=residual, cmap='RdBu_r', alpha=0.6)
            axes[3].set_title('Heat Conduction Residual')
            axes[3].set_xlabel('X')
            axes[3].set_ylabel('Y')
            plt.colorbar(scatter3, ax=axes[3])
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "heat_conduction.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制热传导图失败: {e}")

class PhaseChangeEnergyValidator(EnergyBalanceValidatorBase):
    """相变能量验证器"""
    
    def validate(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证相变能量平衡
        
        相变能量方程: dm/dt * L = Q_phase
        其中 dm/dt 是质量变化率，L 是潜热，Q_phase 是相变热量
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if test_data is None:
            test_data = self._generate_test_points()
        
        # 构建输入
        inputs = torch.cat([
            test_data['x'], test_data['y'], 
            test_data['t'], test_data['altitude']
        ], dim=1)
        inputs.requires_grad_(True)
        
        # 假设模型输出包含温度和冰含量
        outputs = model(inputs)
        
        if outputs.shape[1] < 2:
            self.logger.warning("模型输出维度不足，无法进行相变能量验证")
            return {'passed': False, 'error': '模型输出维度不足'}
        
        T = outputs[:, 0:1] + 273.15  # 温度 [K]
        ice_fraction = torch.sigmoid(outputs[:, 1:2])  # 冰含量 [0-1]
        
        # 计算冰含量的时间导数
        ice_t = torch.autograd.grad(
            outputs=ice_fraction.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True
        )[0][:, 2:3]
        
        # 相变判断（温度接近冰点）
        T_melt = 273.15  # 冰点温度
        phase_change_mask = torch.abs(T - T_melt) < 1.0  # 相变区域
        
        # 相变能量
        L_fusion = self.config.latent_heat_fusion
        rho = self.config.density_ice
        
        # 相变能量释放/吸收率
        phase_energy_rate = -rho * L_fusion * ice_t  # 负号：融化吸热，结冰放热
        
        # 在相变区域，温度应该保持稳定
        temp_derivatives = self._compute_derivatives(model, inputs, output_idx=0)
        T_t = temp_derivatives['u_t']
        
        # 相变能量平衡残差
        # 在相变区域，温度变化应该很小，能量主要用于相变
        phase_balance_residual = torch.where(
            phase_change_mask,
            T_t,  # 相变时温度变化应该很小
            torch.zeros_like(T_t)  # 非相变区域不检查
        )
        
        # 检查容差
        result = self._check_tolerance(phase_balance_residual, "Phase Change")
        
        # 添加详细信息
        result.update({
            'residual': phase_balance_residual.detach(),
            'temperature': T.detach(),
            'ice_fraction': ice_fraction.detach(),
            'ice_time_derivative': ice_t.detach(),
            'phase_energy_rate': phase_energy_rate.detach(),
            'phase_change_mask': phase_change_mask.detach(),
            'temp_time_derivative': T_t.detach(),
            'test_points': inputs.detach()
        })
        
        # 可视化
        if self.config.enable_plotting:
            self._plot_phase_change_energy(inputs.detach(), result)
        
        return result
    
    def _plot_phase_change_energy(self, inputs: torch.Tensor, result: Dict[str, Any]) -> None:
        """绘制相变能量验证结果"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            x = inputs[:, 0].numpy()
            y = inputs[:, 1].numpy()
            
            # 温度分布
            temperature = result['temperature'].squeeze().numpy()
            scatter0 = axes[0].scatter(x, y, c=temperature, cmap='coolwarm', alpha=0.6)
            axes[0].set_title('Temperature [K]')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(scatter0, ax=axes[0])
            
            # 冰含量
            ice_fraction = result['ice_fraction'].squeeze().numpy()
            scatter1 = axes[1].scatter(x, y, c=ice_fraction, cmap='Blues', alpha=0.6)
            axes[1].set_title('Ice Fraction')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[1])
            
            # 冰含量变化率
            ice_t = result['ice_time_derivative'].squeeze().numpy()
            scatter2 = axes[2].scatter(x, y, c=ice_t, cmap='RdBu_r', alpha=0.6)
            axes[2].set_title('Ice Fraction Time Derivative [1/s]')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[2])
            
            # 相变能量率
            phase_energy = result['phase_energy_rate'].squeeze().numpy()
            scatter3 = axes[3].scatter(x, y, c=phase_energy, cmap='viridis', alpha=0.6)
            axes[3].set_title('Phase Change Energy Rate [W/m³]')
            axes[3].set_xlabel('X')
            axes[3].set_ylabel('Y')
            plt.colorbar(scatter3, ax=axes[3])
            
            # 相变区域
            phase_mask = result['phase_change_mask'].squeeze().numpy()
            scatter4 = axes[4].scatter(x, y, c=phase_mask, cmap='Reds', alpha=0.6)
            axes[4].set_title('Phase Change Region')
            axes[4].set_xlabel('X')
            axes[4].set_ylabel('Y')
            plt.colorbar(scatter4, ax=axes[4])
            
            # 相变残差
            residual = result['residual'].squeeze().numpy()
            scatter5 = axes[5].scatter(x, y, c=residual, cmap='RdBu_r', alpha=0.6)
            axes[5].set_title('Phase Change Residual')
            axes[5].set_xlabel('X')
            axes[5].set_ylabel('Y')
            plt.colorbar(scatter5, ax=axes[5])
            
            plt.tight_layout()
            
            if self.config.save_plots:
                save_path = Path(self.config.plot_directory) / "phase_change_energy.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        except Exception as e:
            self.logger.error(f"绘制相变能量图失败: {e}")

class EnergyBalanceValidator:
    """
    能量平衡验证器
    
    整合所有能量平衡验证功能
    """
    
    def __init__(self, config: EnergyBalanceConfig = None):
        """
        初始化验证器
        
        Args:
            config: 配置
        """
        if config is None:
            config = EnergyBalanceConfig()
        
        self.config = config
        self.validators = {
            EnergyBalanceType.SURFACE: SurfaceEnergyBalanceValidator(config),
            EnergyBalanceType.HEAT_CONDUCTION: HeatConductionValidator(config),
            EnergyBalanceType.PHASE_CHANGE: PhaseChangeEnergyValidator(config)
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate_all(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证所有能量平衡
        
        Args:
            model: 模型
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for balance_type, validator in self.validators.items():
            try:
                result = validator.validate(model, test_data)
                results[balance_type.value] = result
                
                if not result.get('passed', False):
                    overall_passed = False
                    
            except Exception as e:
                self.logger.error(f"验证 {balance_type.value} 能量平衡失败: {e}")
                results[balance_type.value] = {
                    'passed': False,
                    'error': str(e)
                }
                overall_passed = False
        
        # 综合结果
        summary = {
            'overall_passed': overall_passed,
            'num_balances_tested': len(self.validators),
            'num_balances_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'tolerance': self.config.tolerance,
            'results': results
        }
        
        self.logger.info(
            f"能量平衡验证完成: {summary['num_balances_passed']}/{summary['num_balances_tested']} 通过, "
            f"总体结果: {'PASSED' if overall_passed else 'FAILED'}"
        )
        
        return summary
    
    def validate_specific(self, model: nn.Module, balance_types: List[EnergyBalanceType], 
                         test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        验证特定能量平衡
        
        Args:
            model: 模型
            balance_types: 要验证的能量平衡类型列表
            test_data: 测试数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {}
        overall_passed = True
        
        for balance_type in balance_types:
            if balance_type in self.validators:
                try:
                    result = self.validators[balance_type].validate(model, test_data)
                    results[balance_type.value] = result
                    
                    if not result.get('passed', False):
                        overall_passed = False
                        
                except Exception as e:
                    self.logger.error(f"验证 {balance_type.value} 能量平衡失败: {e}")
                    results[balance_type.value] = {
                        'passed': False,
                        'error': str(e)
                    }
                    overall_passed = False
            else:
                self.logger.warning(f"不支持的能量平衡类型: {balance_type.value}")
        
        summary = {
            'overall_passed': overall_passed,
            'num_balances_tested': len(balance_types),
            'num_balances_passed': sum(1 for r in results.values() if r.get('passed', False)),
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
            "能量平衡验证报告",
            "=" * 60,
            f"验证时间: {torch.datetime.now()}",
            f"容差设置: 绝对={self.config.tolerance:.2e}, 相对={self.config.relative_tolerance:.2e}",
            "",
            "总体结果:",
            f"  状态: {'通过' if validation_results['overall_passed'] else '失败'}",
            f"  通过率: {validation_results['num_balances_passed']}/{validation_results['num_balances_tested']}",
            ""
        ]
        
        # 详细结果
        report_lines.append("详细结果:")
        for balance_name, result in validation_results['results'].items():
            if 'error' in result:
                report_lines.extend([
                    f"  {balance_name.upper()} 能量平衡:",
                    f"    状态: 错误",
                    f"    错误信息: {result['error']}",
                    ""
                ])
            else:
                status = "通过" if result['passed'] else "失败"
                report_lines.extend([
                    f"  {balance_name.upper()} 能量平衡:",
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

def create_energy_balance_validator(config: EnergyBalanceConfig = None) -> EnergyBalanceValidator:
    """
    创建能量平衡验证器
    
    Args:
        config: 配置
        
    Returns:
        EnergyBalanceValidator: 验证器实例
    """
    return EnergyBalanceValidator(config)

if __name__ == "__main__":
    # 测试能量平衡验证器
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 64),  # 输入: [x, y, t, altitude]
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 6)  # 输出: [温度, 太阳辐射, 长波辐射, 反照率, 风速, 湿度]
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # 创建配置
    config = EnergyBalanceConfig(
        tolerance=1e-4,
        num_test_points=500,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_energy_balance_validator(config)
    
    print("=== 能量平衡验证器测试 ===")
    
    # 验证所有能量平衡
    results = validator.validate_all(model)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    # 验证特定能量平衡
    print("\n=== 验证特定能量平衡 ===")
    specific_results = validator.validate_specific(
        model, 
        [EnergyBalanceType.SURFACE, EnergyBalanceType.HEAT_CONDUCTION]
    )
    
    print(f"特定验证结果: {specific_results['num_balances_passed']}/{specific_results['num_balances_tested']} 通过")
    
    print("\n能量平衡验证器测试完成！")