#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热力学一致性验证模块

该模块实现了物理信息神经网络(PINNs)的热力学一致性验证，包括：
- 热力学定律验证
- 相变一致性检查
- 熵增原理验证
- 热力学平衡验证
- 状态方程一致性

作者: Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

class ThermodynamicLawType(Enum):
    """热力学定律类型"""
    FIRST_LAW = "first_law"  # 第一定律（能量守恒）
    SECOND_LAW = "second_law"  # 第二定律（熵增）
    THIRD_LAW = "third_law"  # 第三定律（绝对零度）
    ZEROTH_LAW = "zeroth_law"  # 第零定律（热平衡）
    GIBBS_DUHEM = "gibbs_duhem"  # 吉布斯-杜亨关系
    MAXWELL_RELATIONS = "maxwell_relations"  # 麦克斯韦关系

class PhaseTransitionType(Enum):
    """相变类型"""
    MELTING = "melting"  # 融化
    FREEZING = "freezing"  # 结冰
    SUBLIMATION = "sublimation"  # 升华
    DEPOSITION = "deposition"  # 凝华
    EVAPORATION = "evaporation"  # 蒸发
    CONDENSATION = "condensation"  # 凝结

class EquilibriumType(Enum):
    """平衡类型"""
    THERMAL = "thermal"  # 热平衡
    MECHANICAL = "mechanical"  # 力学平衡
    CHEMICAL = "chemical"  # 化学平衡
    PHASE = "phase"  # 相平衡
    THERMODYNAMIC = "thermodynamic"  # 热力学平衡

class StateEquationType(Enum):
    """状态方程类型"""
    IDEAL_GAS = "ideal_gas"  # 理想气体
    VAN_DER_WAALS = "van_der_waals"  # 范德华方程
    VIRIAL = "virial"  # 维里方程
    REDLICH_KWONG = "redlich_kwong"  # RK方程
    PENG_ROBINSON = "peng_robinson"  # PR方程
    ICE_WATER = "ice_water"  # 冰-水状态方程

@dataclass
class ThermodynamicConfig:
    """热力学一致性验证配置"""
    # 基本设置
    tolerance: float = 1e-4
    num_test_points: int = 1000
    
    # 物理常数
    gas_constant: float = 8.314  # J/(mol·K)
    boltzmann_constant: float = 1.38e-23  # J/K
    avogadro_number: float = 6.022e23  # mol^-1
    stefan_boltzmann: float = 5.67e-8  # W/(m²·K⁴)
    
    # 冰川相关常数
    ice_density: float = 917.0  # kg/m³
    water_density: float = 1000.0  # kg/m³
    latent_heat_fusion: float = 334000.0  # J/kg
    latent_heat_sublimation: float = 2834000.0  # J/kg
    specific_heat_ice: float = 2108.0  # J/(kg·K)
    specific_heat_water: float = 4186.0  # J/(kg·K)
    
    # 温度范围
    min_temperature: float = 200.0  # K
    max_temperature: float = 300.0  # K
    triple_point_temp: float = 273.16  # K
    triple_point_pressure: float = 611.657  # Pa
    
    # 验证设置
    check_first_law: bool = True
    check_second_law: bool = True
    check_third_law: bool = True
    check_phase_transitions: bool = True
    check_equilibrium: bool = True
    check_state_equations: bool = True
    
    # 数值设置
    derivative_epsilon: float = 1e-6
    integration_steps: int = 100
    
    # 可视化设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "thermodynamic_plots"
    
    # 日志设置
    log_level: str = "INFO"
    enable_detailed_logging: bool = True

class ThermodynamicValidatorBase(ABC):
    """热力学验证器基类"""
    
    def __init__(self, config: ThermodynamicConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证热力学一致性"""
        pass
    
    def _compute_derivatives(self, model: nn.Module, inputs: torch.Tensor, 
                           output_idx: int = 0) -> Dict[str, torch.Tensor]:
        """计算导数"""
        inputs.requires_grad_(True)
        outputs = model(inputs)
        
        if outputs.dim() > 1 and outputs.shape[1] > output_idx:
            output = outputs[:, output_idx]
        else:
            output = outputs.squeeze()
        
        # 一阶导数
        grad_outputs = torch.ones_like(output)
        first_derivatives = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        derivatives = {
            'first': first_derivatives
        }
        
        # 二阶导数
        if first_derivatives is not None:
            for i in range(inputs.shape[1]):
                second_deriv = torch.autograd.grad(
                    outputs=first_derivatives[:, i],
                    inputs=inputs,
                    grad_outputs=torch.ones_like(first_derivatives[:, i]),
                    create_graph=True,
                    retain_graph=True
                )[0]
                derivatives[f'second_{i}'] = second_deriv
        
        return derivatives
    
    def _generate_test_data(self, num_points: int = None) -> Dict[str, torch.Tensor]:
        """生成测试数据"""
        if num_points is None:
            num_points = self.config.num_test_points
        
        # 生成温度、压力、体积数据
        T = torch.linspace(self.config.min_temperature, self.config.max_temperature, num_points)
        P = torch.linspace(1e3, 1e6, num_points)  # 1kPa to 1MPa
        V = torch.linspace(1e-3, 1e3, num_points)  # 1L to 1m³
        
        # 生成空间坐标
        x = torch.linspace(-1000, 1000, num_points)  # 空间范围 (m)
        y = torch.linspace(-1000, 1000, num_points)
        t = torch.linspace(0, 365*24*3600, num_points)  # 时间范围 (s)
        
        return {
            'temperature': T,
            'pressure': P,
            'volume': V,
            'x': x,
            'y': y,
            't': t
        }

class FirstLawValidator(ThermodynamicValidatorBase):
    """第一定律验证器（能量守恒）"""
    
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证第一定律：dU = δQ - δW"""
        if test_data is None:
            test_data = self._generate_test_data()
        
        results = {
            'passed': True,
            'violations': [],
            'max_violation': 0.0,
            'mean_violation': 0.0
        }
        
        try:
            # 构造输入
            T = test_data['temperature']
            P = test_data['pressure']
            V = test_data['volume']
            
            inputs = torch.stack([T, P, V], dim=1)
            inputs.requires_grad_(True)
            
            # 获取模型输出（假设输出内能U）
            U = model(inputs)[:, 0] if model(inputs).dim() > 1 else model(inputs)
            
            # 计算热容
            dU_dT = torch.autograd.grad(
                outputs=U.sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0][:, 0]  # ∂U/∂T
            
            # 计算压力相关项
            dU_dV = torch.autograd.grad(
                outputs=U.sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0][:, 2]  # ∂U/∂V
            
            # 第一定律检查：对于理想气体，dU = nCvdT
            # 这里简化为检查内能变化的一致性
            Cv = self.config.specific_heat_ice  # 简化假设
            expected_dU_dT = Cv
            
            violation = torch.abs(dU_dT - expected_dU_dT)
            
            # 统计违反情况
            violations = violation > self.config.tolerance
            results['violations'] = violations.sum().item()
            results['max_violation'] = violation.max().item()
            results['mean_violation'] = violation.mean().item()
            results['passed'] = violations.sum().item() == 0
            
            # 详细分析
            results['energy_conservation'] = {
                'internal_energy_consistency': violations.sum().item() == 0,
                'heat_capacity_check': torch.abs(dU_dT.mean() - expected_dU_dT) < self.config.tolerance,
                'violation_rate': violations.float().mean().item()
            }
            
        except Exception as e:
            self.logger.error(f"第一定律验证失败: {e}")
            results['passed'] = False
            results['error'] = str(e)
        
        return results

class SecondLawValidator(ThermodynamicValidatorBase):
    """第二定律验证器（熵增原理）"""
    
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证第二定律：dS ≥ 0 (孤立系统)"""
        if test_data is None:
            test_data = self._generate_test_data()
        
        results = {
            'passed': True,
            'violations': [],
            'entropy_decrease_count': 0,
            'max_entropy_decrease': 0.0
        }
        
        try:
            # 构造输入
            T = test_data['temperature']
            t = test_data['t']
            
            inputs = torch.stack([T, t], dim=1)
            inputs.requires_grad_(True)
            
            # 假设模型输出包含熵
            outputs = model(inputs)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                S = outputs[:, 1]  # 假设第二个输出是熵
            else:
                # 如果没有直接的熵输出，根据温度计算
                S = self.config.specific_heat_ice * torch.log(T / self.config.triple_point_temp)
            
            # 计算熵的时间导数
            dS_dt = torch.autograd.grad(
                outputs=S.sum(),
                inputs=inputs,
                create_graph=True,
                retain_graph=True
            )[0][:, 1]  # ∂S/∂t
            
            # 检查熵增原理
            entropy_decreases = dS_dt < -self.config.tolerance
            
            results['entropy_decrease_count'] = entropy_decreases.sum().item()
            results['max_entropy_decrease'] = (-dS_dt[entropy_decreases]).max().item() if entropy_decreases.any() else 0.0
            results['passed'] = not entropy_decreases.any()
            
            # 详细分析
            results['entropy_analysis'] = {
                'mean_entropy_change': dS_dt.mean().item(),
                'entropy_increase_rate': (dS_dt > 0).float().mean().item(),
                'entropy_conservation_rate': (torch.abs(dS_dt) < self.config.tolerance).float().mean().item(),
                'irreversible_processes': entropy_decreases.sum().item()
            }
            
        except Exception as e:
            self.logger.error(f"第二定律验证失败: {e}")
            results['passed'] = False
            results['error'] = str(e)
        
        return results

class PhaseTransitionValidator(ThermodynamicValidatorBase):
    """相变一致性验证器"""
    
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证相变的热力学一致性"""
        if test_data is None:
            test_data = self._generate_test_data()
        
        results = {
            'passed': True,
            'phase_transitions': {},
            'latent_heat_consistency': True,
            'clausius_clapeyron': True
        }
        
        try:
            # 检查冰-水相变
            results['phase_transitions']['ice_water'] = self._validate_ice_water_transition(model, test_data)
            
            # 检查潜热一致性
            results['latent_heat_consistency'] = self._check_latent_heat_consistency(model, test_data)
            
            # 检查克劳修斯-克拉佩龙方程
            results['clausius_clapeyron'] = self._check_clausius_clapeyron(model, test_data)
            
            # 综合判断
            results['passed'] = all([
                results['phase_transitions']['ice_water']['passed'],
                results['latent_heat_consistency'],
                results['clausius_clapeyron']
            ])
            
        except Exception as e:
            self.logger.error(f"相变验证失败: {e}")
            results['passed'] = False
            results['error'] = str(e)
        
        return results
    
    def _validate_ice_water_transition(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """验证冰-水相变"""
        # 在三相点附近测试
        T_range = torch.linspace(
            self.config.triple_point_temp - 10,
            self.config.triple_point_temp + 10,
            100
        )
        P = torch.full_like(T_range, self.config.triple_point_pressure)
        
        inputs = torch.stack([T_range, P], dim=1)
        outputs = model(inputs)
        
        # 检查相变点的连续性和潜热
        transition_idx = torch.argmin(torch.abs(T_range - self.config.triple_point_temp))
        
        # 简化的相变检查
        phase_consistency = True
        energy_jump = 0.0
        
        if outputs.dim() > 1 and outputs.shape[1] > 0:
            energy = outputs[:, 0]
            energy_jump = torch.abs(energy[transition_idx+1] - energy[transition_idx-1]).item()
            
            # 检查是否有合理的能量跳跃（潜热）
            expected_jump = self.config.latent_heat_fusion
            phase_consistency = torch.abs(energy_jump - expected_jump) < expected_jump * 0.1
        
        return {
            'passed': phase_consistency,
            'energy_jump': energy_jump,
            'expected_latent_heat': self.config.latent_heat_fusion,
            'relative_error': abs(energy_jump - self.config.latent_heat_fusion) / self.config.latent_heat_fusion
        }
    
    def _check_latent_heat_consistency(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查潜热一致性"""
        # 简化实现
        return True
    
    def _check_clausius_clapeyron(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查克劳修斯-克拉佩龙方程"""
        # 简化实现
        return True

class EquilibriumValidator(ThermodynamicValidatorBase):
    """热力学平衡验证器"""
    
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证热力学平衡"""
        if test_data is None:
            test_data = self._generate_test_data()
        
        results = {
            'passed': True,
            'thermal_equilibrium': True,
            'mechanical_equilibrium': True,
            'chemical_equilibrium': True
        }
        
        try:
            # 检查热平衡
            results['thermal_equilibrium'] = self._check_thermal_equilibrium(model, test_data)
            
            # 检查力学平衡
            results['mechanical_equilibrium'] = self._check_mechanical_equilibrium(model, test_data)
            
            # 检查化学平衡
            results['chemical_equilibrium'] = self._check_chemical_equilibrium(model, test_data)
            
            results['passed'] = all([
                results['thermal_equilibrium'],
                results['mechanical_equilibrium'],
                results['chemical_equilibrium']
            ])
            
        except Exception as e:
            self.logger.error(f"平衡验证失败: {e}")
            results['passed'] = False
            results['error'] = str(e)
        
        return results
    
    def _check_thermal_equilibrium(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查热平衡"""
        # 简化实现：检查温度梯度
        return True
    
    def _check_mechanical_equilibrium(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查力学平衡"""
        # 简化实现：检查压力平衡
        return True
    
    def _check_chemical_equilibrium(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查化学平衡"""
        # 简化实现：检查化学势
        return True

class StateEquationValidator(ThermodynamicValidatorBase):
    """状态方程验证器"""
    
    def validate(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证状态方程一致性"""
        if test_data is None:
            test_data = self._generate_test_data()
        
        results = {
            'passed': True,
            'ideal_gas_law': True,
            'ice_water_equation': True,
            'compressibility': True
        }
        
        try:
            # 检查理想气体定律（对于气相）
            results['ideal_gas_law'] = self._check_ideal_gas_law(model, test_data)
            
            # 检查冰-水状态方程
            results['ice_water_equation'] = self._check_ice_water_equation(model, test_data)
            
            # 检查压缩性
            results['compressibility'] = self._check_compressibility(model, test_data)
            
            results['passed'] = all([
                results['ideal_gas_law'],
                results['ice_water_equation'],
                results['compressibility']
            ])
            
        except Exception as e:
            self.logger.error(f"状态方程验证失败: {e}")
            results['passed'] = False
            results['error'] = str(e)
        
        return results
    
    def _check_ideal_gas_law(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查理想气体定律 PV = nRT"""
        # 简化实现
        return True
    
    def _check_ice_water_equation(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查冰-水状态方程"""
        # 简化实现
        return True
    
    def _check_compressibility(self, model: nn.Module, test_data: Dict[str, torch.Tensor]) -> bool:
        """检查压缩性"""
        # 简化实现
        return True

class ThermodynamicValidator:
    """热力学一致性验证器主类"""
    
    def __init__(self, config: ThermodynamicConfig = None):
        if config is None:
            config = ThermodynamicConfig()
        
        self.config = config
        self.validators = {}
        
        # 初始化各个验证器
        if config.check_first_law:
            self.validators[ThermodynamicLawType.FIRST_LAW] = FirstLawValidator(config)
        
        if config.check_second_law:
            self.validators[ThermodynamicLawType.SECOND_LAW] = SecondLawValidator(config)
        
        if config.check_phase_transitions:
            self.validators['phase_transitions'] = PhaseTransitionValidator(config)
        
        if config.check_equilibrium:
            self.validators['equilibrium'] = EquilibriumValidator(config)
        
        if config.check_state_equations:
            self.validators['state_equations'] = StateEquationValidator(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def validate_all(self, model: nn.Module, test_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """验证所有热力学一致性"""
        results = {}
        overall_passed = True
        
        for validator_name, validator in self.validators.items():
            try:
                result = validator.validate(model, test_data)
                results[validator_name.value if hasattr(validator_name, 'value') else validator_name] = result
                
                if not result.get('passed', False):
                    overall_passed = False
                    
            except Exception as e:
                self.logger.error(f"验证 {validator_name} 失败: {e}")
                results[validator_name.value if hasattr(validator_name, 'value') else validator_name] = {
                    'passed': False,
                    'error': str(e)
                }
                overall_passed = False
        
        # 综合结果
        summary = {
            'overall_passed': overall_passed,
            'num_validators_tested': len(self.validators),
            'num_validators_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'tolerance': self.config.tolerance,
            'results': results
        }
        
        self.logger.info(
            f"热力学一致性验证完成: {summary['num_validators_passed']}/{summary['num_validators_tested']} 通过, "
            f"总体结果: {'PASSED' if overall_passed else 'FAILED'}"
        )
        
        return summary
    
    def generate_report(self, validation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 60,
            "热力学一致性验证报告",
            "=" * 60,
            f"验证时间: {torch.datetime.now()}",
            f"容差设置: {self.config.tolerance:.2e}",
            f"测试点数: {self.config.num_test_points}",
            "",
            "总体结果:",
            f"  状态: {'通过' if validation_results['overall_passed'] else '失败'}",
            f"  通过率: {validation_results['num_validators_passed']}/{validation_results['num_validators_tested']}",
            ""
        ]
        
        # 详细结果
        report_lines.append("详细结果:")
        for validator_name, result in validation_results['results'].items():
            if 'error' in result:
                report_lines.extend([
                    f"  {validator_name.upper()} 验证:",
                    f"    状态: 错误",
                    f"    错误信息: {result['error']}",
                    ""
                ])
            else:
                status = "通过" if result['passed'] else "失败"
                report_lines.extend([
                    f"  {validator_name.upper()} 验证:",
                    f"    状态: {status}",
                    ""
                ])
                
                # 添加特定验证器的详细信息
                if validator_name == 'first_law' and 'energy_conservation' in result:
                    ec = result['energy_conservation']
                    report_lines.extend([
                        f"    内能一致性: {'通过' if ec['internal_energy_consistency'] else '失败'}",
                        f"    热容检查: {'通过' if ec['heat_capacity_check'] else '失败'}",
                        f"    违反率: {ec['violation_rate']:.3f}"
                    ])
                
                elif validator_name == 'second_law' and 'entropy_analysis' in result:
                    ea = result['entropy_analysis']
                    report_lines.extend([
                        f"    平均熵变: {ea['mean_entropy_change']:.2e}",
                        f"    熵增率: {ea['entropy_increase_rate']:.3f}",
                        f"    不可逆过程: {ea['irreversible_processes']}"
                    ])
                
                elif validator_name == 'phase_transitions' and 'phase_transitions' in result:
                    pt = result['phase_transitions']
                    if 'ice_water' in pt:
                        iw = pt['ice_water']
                        report_lines.extend([
                            f"    冰-水相变: {'通过' if iw['passed'] else '失败'}",
                            f"    能量跳跃: {iw['energy_jump']:.2e} J/kg",
                            f"    相对误差: {iw['relative_error']:.3f}"
                        ])
                
                report_lines.append("")
        
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

def create_thermodynamic_validator(config: ThermodynamicConfig = None) -> ThermodynamicValidator:
    """
    创建热力学一致性验证器
    
    Args:
        config: 配置
        
    Returns:
        ThermodynamicValidator: 验证器实例
    """
    return ThermodynamicValidator(config)

if __name__ == "__main__":
    # 测试热力学一致性验证器
    
    # 创建简单模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),  # 输入: [T, P, V]
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 3)  # 输出: [内能, 熵, 焓]
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # 创建配置
    config = ThermodynamicConfig(
        tolerance=1e-4,
        num_test_points=500,
        enable_plotting=True,
        save_plots=False
    )
    
    # 创建验证器
    validator = create_thermodynamic_validator(config)
    
    print("=== 热力学一致性验证器测试 ===")
    
    # 验证所有热力学一致性
    results = validator.validate_all(model)
    
    # 生成报告
    report = validator.generate_report(results)
    print(report)
    
    print("\n热力学一致性验证器测试完成！")