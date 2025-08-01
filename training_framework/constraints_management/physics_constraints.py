#!/usr/bin/env python3
"""
物理约束模块

实现各种物理约束用于PINNs训练，包括：
- 基础物理约束
- 冰川物理约束
- 边界条件约束
- 守恒定律约束
- 约束验证和监控

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ConstraintConfig:
    """
    约束配置类
    """
    name: str
    weight: float = 1.0
    tolerance: float = 1e-6
    active: bool = True
    adaptive_weight: bool = False
    penalty_type: str = 'l2'  # 'l1', 'l2', 'huber'

class BaseConstraint(ABC):
    """
    基础约束类
    
    所有物理约束的基类
    """
    
    def __init__(self, config: ConstraintConfig):
        """
        初始化基础约束
        
        Args:
            config: 约束配置
        """
        self.config = config
        self.violation_history = []
        self.weight_history = []
    
    @abstractmethod
    def compute_violation(self, x: torch.Tensor, u: torch.Tensor, 
                         derivatives: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        计算约束违反
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 约束违反值
        """
        pass
    
    def compute_penalty(self, violation: torch.Tensor) -> torch.Tensor:
        """
        计算约束惩罚
        
        Args:
            violation: 约束违反
            
        Returns:
            torch.Tensor: 约束惩罚
        """
        if self.config.penalty_type == 'l1':
            penalty = torch.mean(torch.abs(violation))
        elif self.config.penalty_type == 'l2':
            penalty = torch.mean(violation ** 2)
        elif self.config.penalty_type == 'huber':
            delta = self.config.tolerance
            penalty = torch.where(
                torch.abs(violation) <= delta,
                0.5 * violation ** 2,
                delta * (torch.abs(violation) - 0.5 * delta)
            )
            penalty = torch.mean(penalty)
        else:
            raise ValueError(f"未知的惩罚类型: {self.config.penalty_type}")
        
        return self.config.weight * penalty
    
    def update_weight(self, violation: torch.Tensor):
        """
        更新自适应权重
        
        Args:
            violation: 约束违反
        """
        if self.config.adaptive_weight:
            # 简单的自适应权重策略
            avg_violation = torch.mean(torch.abs(violation)).item()
            self.violation_history.append(avg_violation)
            
            if len(self.violation_history) > 10:
                recent_violations = self.violation_history[-10:]
                violation_trend = (recent_violations[-1] - recent_violations[0]) / len(recent_violations)
                
                if violation_trend > 0:  # 违反增加
                    self.config.weight *= 1.1
                elif violation_trend < -self.config.tolerance:  # 违反减少
                    self.config.weight *= 0.95
                
                # 限制权重范围
                self.config.weight = max(0.1, min(10.0, self.config.weight))
            
            self.weight_history.append(self.config.weight)
    
    def is_satisfied(self, violation: torch.Tensor) -> bool:
        """
        检查约束是否满足
        
        Args:
            violation: 约束违反
            
        Returns:
            bool: 是否满足约束
        """
        return torch.mean(torch.abs(violation)).item() < self.config.tolerance

class PDEConstraint(BaseConstraint):
    """
    偏微分方程约束
    
    通用的PDE约束实现
    """
    
    def __init__(self, config: ConstraintConfig, pde_func: Callable):
        """
        初始化PDE约束
        
        Args:
            config: 约束配置
            pde_func: PDE函数
        """
        super().__init__(config)
        self.pde_func = pde_func
    
    def compute_violation(self, x: torch.Tensor, u: torch.Tensor, 
                         derivatives: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        计算PDE约束违反
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 约束违反值
        """
        return self.pde_func(x, u, derivatives)

class BoundaryConstraint(BaseConstraint):
    """
    边界条件约束
    
    处理各种边界条件
    """
    
    def __init__(self, config: ConstraintConfig, boundary_type: str, 
                 boundary_func: Callable = None, boundary_value: float = None):
        """
        初始化边界约束
        
        Args:
            config: 约束配置
            boundary_type: 边界类型 ('dirichlet', 'neumann', 'robin')
            boundary_func: 边界函数
            boundary_value: 边界值
        """
        super().__init__(config)
        self.boundary_type = boundary_type
        self.boundary_func = boundary_func
        self.boundary_value = boundary_value
    
    def compute_violation(self, x: torch.Tensor, u: torch.Tensor, 
                         derivatives: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        计算边界约束违反
        
        Args:
            x: 边界坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 约束违反值
        """
        if self.boundary_type == 'dirichlet':
            # Dirichlet边界条件: u = g
            if self.boundary_func is not None:
                target = self.boundary_func(x)
            else:
                target = torch.full_like(u, self.boundary_value)
            return u - target
        
        elif self.boundary_type == 'neumann':
            # Neumann边界条件: ∂u/∂n = g
            if derivatives is None or 'du_dn' not in derivatives:
                raise ValueError("Neumann边界条件需要法向导数")
            
            du_dn = derivatives['du_dn']
            if self.boundary_func is not None:
                target = self.boundary_func(x)
            else:
                target = torch.full_like(du_dn, self.boundary_value)
            return du_dn - target
        
        elif self.boundary_type == 'robin':
            # Robin边界条件: αu + β∂u/∂n = g
            if derivatives is None or 'du_dn' not in derivatives:
                raise ValueError("Robin边界条件需要法向导数")
            
            alpha, beta = 1.0, 1.0  # 默认系数
            du_dn = derivatives['du_dn']
            
            if self.boundary_func is not None:
                target = self.boundary_func(x)
            else:
                target = torch.full_like(u, self.boundary_value)
            
            return alpha * u + beta * du_dn - target
        
        else:
            raise ValueError(f"未知的边界类型: {self.boundary_type}")

class ConservationConstraint(BaseConstraint):
    """
    守恒定律约束
    
    实现各种守恒定律
    """
    
    def __init__(self, config: ConstraintConfig, conservation_type: str):
        """
        初始化守恒约束
        
        Args:
            config: 约束配置
            conservation_type: 守恒类型 ('mass', 'momentum', 'energy')
        """
        super().__init__(config)
        self.conservation_type = conservation_type
    
    def compute_violation(self, x: torch.Tensor, u: torch.Tensor, 
                         derivatives: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        计算守恒约束违反
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 约束违反值
        """
        if self.conservation_type == 'mass':
            return self._mass_conservation(x, u, derivatives)
        elif self.conservation_type == 'momentum':
            return self._momentum_conservation(x, u, derivatives)
        elif self.conservation_type == 'energy':
            return self._energy_conservation(x, u, derivatives)
        else:
            raise ValueError(f"未知的守恒类型: {self.conservation_type}")
    
    def _mass_conservation(self, x: torch.Tensor, u: torch.Tensor, 
                          derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        质量守恒
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 质量守恒违反
        """
        # ∂ρ/∂t + ∇·(ρv) = 0
        if 'drho_dt' not in derivatives or 'div_rho_v' not in derivatives:
            raise ValueError("质量守恒需要密度时间导数和通量散度")
        
        drho_dt = derivatives['drho_dt']
        div_rho_v = derivatives['div_rho_v']
        
        return drho_dt + div_rho_v
    
    def _momentum_conservation(self, x: torch.Tensor, u: torch.Tensor, 
                              derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        动量守恒
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 动量守恒违反
        """
        # ∂(ρv)/∂t + ∇·(ρvv) = -∇p + ∇·τ + ρg
        # 简化实现
        if 'dv_dt' not in derivatives:
            raise ValueError("动量守恒需要速度时间导数")
        
        # TODO: 实现完整的动量守恒方程
        return torch.zeros_like(derivatives['dv_dt'])
    
    def _energy_conservation(self, x: torch.Tensor, u: torch.Tensor, 
                            derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        能量守恒
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 能量守恒违反
        """
        # ∂E/∂t + ∇·(Ev) = -∇·q + S
        # 简化实现
        if 'dE_dt' not in derivatives:
            raise ValueError("能量守恒需要能量时间导数")
        
        # TODO: 实现完整的能量守恒方程
        return torch.zeros_like(derivatives['dE_dt'])

class GlacierPhysicsConstraints:
    """
    冰川物理约束集合
    
    包含所有冰川相关的物理约束
    """
    
    def __init__(self):
        """
        初始化冰川物理约束
        """
        self.constraints = {}
        self._setup_glacier_constraints()
    
    def _setup_glacier_constraints(self):
        """
        设置冰川约束
        """
        # 质量平衡约束
        mass_balance_config = ConstraintConfig(
            name="mass_balance",
            weight=1.0,
            tolerance=1e-4,
            adaptive_weight=True
        )
        self.constraints['mass_balance'] = PDEConstraint(
            mass_balance_config, 
            self._mass_balance_pde
        )
        
        # 动量平衡约束
        momentum_config = ConstraintConfig(
            name="momentum_balance",
            weight=1.0,
            tolerance=1e-4,
            adaptive_weight=True
        )
        self.constraints['momentum_balance'] = PDEConstraint(
            momentum_config,
            self._momentum_balance_pde
        )
        
        # Glen流动定律约束
        glen_config = ConstraintConfig(
            name="glen_flow_law",
            weight=0.5,
            tolerance=1e-5,
            adaptive_weight=True
        )
        self.constraints['glen_flow_law'] = PDEConstraint(
            glen_config,
            self._glen_flow_law
        )
        
        # 表面边界条件
        surface_bc_config = ConstraintConfig(
            name="surface_boundary",
            weight=2.0,
            tolerance=1e-5
        )
        self.constraints['surface_boundary'] = BoundaryConstraint(
            surface_bc_config,
            'neumann',
            boundary_value=0.0  # 表面应力为零
        )
        
        # 底部边界条件
        bed_bc_config = ConstraintConfig(
            name="bed_boundary",
            weight=2.0,
            tolerance=1e-5
        )
        self.constraints['bed_boundary'] = BoundaryConstraint(
            bed_bc_config,
            'dirichlet',
            boundary_value=0.0  # 底部速度为零（无滑动）
        )
    
    def _mass_balance_pde(self, x: torch.Tensor, u: torch.Tensor, 
                         derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        质量平衡PDE
        
        ∂H/∂t + ∇·(H*v) = M
        
        Args:
            x: 坐标 [x, y, t]
            u: 输出 [H, vx, vy, T] (厚度, 速度, 温度)
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: PDE残差
        """
        H = u[:, 0:1]  # 厚度
        vx = u[:, 1:2]  # x方向速度
        vy = u[:, 2:3]  # y方向速度
        
        # 时间导数
        dH_dt = derivatives.get('dH_dt', torch.zeros_like(H))
        
        # 空间导数
        dH_dx = derivatives.get('dH_dx', torch.zeros_like(H))
        dH_dy = derivatives.get('dH_dy', torch.zeros_like(H))
        dvx_dx = derivatives.get('dvx_dx', torch.zeros_like(vx))
        dvy_dy = derivatives.get('dvy_dy', torch.zeros_like(vy))
        
        # 通量散度
        flux_div = H * (dvx_dx + dvy_dy) + vx * dH_dx + vy * dH_dy
        
        # 质量平衡源项（简化）
        M = torch.zeros_like(H)  # TODO: 实现实际的质量平衡
        
        return dH_dt + flux_div - M
    
    def _momentum_balance_pde(self, x: torch.Tensor, u: torch.Tensor, 
                             derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        动量平衡PDE
        
        Args:
            x: 坐标
            u: 输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: PDE残差
        """
        # 简化的动量平衡方程
        # ∇·σ + ρg = 0
        
        # TODO: 实现完整的动量平衡方程
        return torch.zeros(x.shape[0], 2, device=x.device)
    
    def _glen_flow_law(self, x: torch.Tensor, u: torch.Tensor, 
                      derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Glen流动定律
        
        Args:
            x: 坐标
            u: 输出
            derivatives: 导数字典
            
        Returns:
            torch.Tensor: 流动定律残差
        """
        # Glen流动定律: ε̇ = A * τ^n
        # 其中 ε̇ 是应变率，τ 是有效应力，n 是Glen指数
        
        # TODO: 实现Glen流动定律
        return torch.zeros(x.shape[0], 1, device=x.device)
    
    def compute_total_violation(self, x: torch.Tensor, u: torch.Tensor, 
                               derivatives: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算所有约束的违反
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            
        Returns:
            Dict: 约束违反字典
        """
        violations = {}
        
        for name, constraint in self.constraints.items():
            if constraint.config.active:
                violation = constraint.compute_violation(x, u, derivatives)
                violations[name] = violation
                
                # 更新自适应权重
                constraint.update_weight(violation)
        
        return violations
    
    def compute_total_penalty(self, violations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算总约束惩罚
        
        Args:
            violations: 约束违反字典
            
        Returns:
            torch.Tensor: 总惩罚
        """
        total_penalty = torch.tensor(0.0, device=list(violations.values())[0].device)
        
        for name, violation in violations.items():
            constraint = self.constraints[name]
            penalty = constraint.compute_penalty(violation)
            total_penalty += penalty
        
        return total_penalty
    
    def get_constraint_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取约束状态
        
        Returns:
            Dict: 约束状态字典
        """
        status = {}
        
        for name, constraint in self.constraints.items():
            status[name] = {
                'active': constraint.config.active,
                'weight': constraint.config.weight,
                'tolerance': constraint.config.tolerance,
                'violation_history': constraint.violation_history[-10:],  # 最近10次
                'weight_history': constraint.weight_history[-10:]
            }
        
        return status
    
    def activate_constraint(self, name: str):
        """
        激活约束
        
        Args:
            name: 约束名称
        """
        if name in self.constraints:
            self.constraints[name].config.active = True
    
    def deactivate_constraint(self, name: str):
        """
        停用约束
        
        Args:
            name: 约束名称
        """
        if name in self.constraints:
            self.constraints[name].config.active = False
    
    def update_constraint_weight(self, name: str, weight: float):
        """
        更新约束权重
        
        Args:
            name: 约束名称
            weight: 新权重
        """
        if name in self.constraints:
            self.constraints[name].config.weight = weight

class ConstraintMonitor:
    """
    约束监控器
    
    监控和分析约束满足情况
    """
    
    def __init__(self, constraints: GlacierPhysicsConstraints):
        """
        初始化约束监控器
        
        Args:
            constraints: 约束集合
        """
        self.constraints = constraints
        self.monitoring_history = []
    
    def monitor_step(self, x: torch.Tensor, u: torch.Tensor, 
                    derivatives: Dict[str, torch.Tensor], epoch: int):
        """
        监控一步
        
        Args:
            x: 输入坐标
            u: 网络输出
            derivatives: 导数字典
            epoch: 当前轮次
        """
        violations = self.constraints.compute_total_violation(x, u, derivatives)
        
        step_info = {
            'epoch': epoch,
            'violations': {name: torch.mean(torch.abs(v)).item() 
                          for name, v in violations.items()},
            'total_penalty': self.constraints.compute_total_penalty(violations).item(),
            'constraint_status': self.constraints.get_constraint_status()
        }
        
        self.monitoring_history.append(step_info)
    
    def get_violation_trends(self, window_size: int = 50) -> Dict[str, List[float]]:
        """
        获取违反趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            Dict: 违反趋势字典
        """
        if len(self.monitoring_history) < window_size:
            return {}
        
        recent_history = self.monitoring_history[-window_size:]
        trends = {}
        
        # 获取所有约束名称
        constraint_names = set()
        for step in recent_history:
            constraint_names.update(step['violations'].keys())
        
        # 计算每个约束的趋势
        for name in constraint_names:
            violations = [step['violations'].get(name, 0.0) for step in recent_history]
            trends[name] = violations
        
        return trends
    
    def suggest_weight_adjustments(self) -> Dict[str, float]:
        """
        建议权重调整
        
        Returns:
            Dict: 权重调整建议
        """
        if len(self.monitoring_history) < 20:
            return {}
        
        suggestions = {}
        trends = self.get_violation_trends(20)
        
        for name, violations in trends.items():
            if len(violations) < 10:
                continue
            
            # 计算违反趋势
            recent_avg = np.mean(violations[-5:])
            earlier_avg = np.mean(violations[:5])
            
            if recent_avg > earlier_avg * 1.5:  # 违反增加
                suggestions[name] = 1.2  # 建议增加权重
            elif recent_avg < earlier_avg * 0.5:  # 违反减少
                suggestions[name] = 0.8  # 建议减少权重
        
        return suggestions
    
    def generate_report(self) -> str:
        """
        生成监控报告
        
        Returns:
            str: 监控报告
        """
        if not self.monitoring_history:
            return "无监控数据"
        
        latest = self.monitoring_history[-1]
        report = f"约束监控报告 (轮次 {latest['epoch']})\n"
        report += "=" * 50 + "\n"
        
        # 当前违反情况
        report += "当前约束违反:\n"
        for name, violation in latest['violations'].items():
            status = "✓" if violation < 1e-4 else "✗"
            report += f"  {status} {name}: {violation:.2e}\n"
        
        report += f"\n总惩罚: {latest['total_penalty']:.2e}\n"
        
        # 权重建议
        suggestions = self.suggest_weight_adjustments()
        if suggestions:
            report += "\n权重调整建议:\n"
            for name, factor in suggestions.items():
                action = "增加" if factor > 1 else "减少"
                report += f"  {name}: {action} (因子: {factor:.2f})\n"
        
        return report

if __name__ == "__main__":
    # 测试约束系统
    torch.manual_seed(42)
    
    # 创建测试数据
    x = torch.randn(100, 3, requires_grad=True)  # [x, y, t]
    u = torch.randn(100, 4, requires_grad=True)  # [H, vx, vy, T]
    
    # 创建简单的导数字典
    derivatives = {
        'dH_dt': torch.randn(100, 1),
        'dH_dx': torch.randn(100, 1),
        'dH_dy': torch.randn(100, 1),
        'dvx_dx': torch.randn(100, 1),
        'dvy_dy': torch.randn(100, 1)
    }
    
    # 测试冰川物理约束
    print("测试冰川物理约束...")
    glacier_constraints = GlacierPhysicsConstraints()
    
    # 计算约束违反
    violations = glacier_constraints.compute_total_violation(x, u, derivatives)
    print("约束违反:")
    for name, violation in violations.items():
        avg_violation = torch.mean(torch.abs(violation)).item()
        print(f"  {name}: {avg_violation:.2e}")
    
    # 计算总惩罚
    total_penalty = glacier_constraints.compute_total_penalty(violations)
    print(f"\n总约束惩罚: {total_penalty.item():.2e}")
    
    # 测试约束监控
    print("\n测试约束监控...")
    monitor = ConstraintMonitor(glacier_constraints)
    
    # 模拟训练过程
    for epoch in range(10):
        # 模拟网络输出变化
        u_epoch = u + 0.1 * torch.randn_like(u) * (1 - epoch / 10)
        
        monitor.monitor_step(x, u_epoch, derivatives, epoch)
        
        if epoch % 5 == 0:
            print(f"\n轮次 {epoch}:")
            violations_epoch = glacier_constraints.compute_total_violation(x, u_epoch, derivatives)
            for name, violation in violations_epoch.items():
                avg_violation = torch.mean(torch.abs(violation)).item()
                print(f"  {name}: {avg_violation:.2e}")
    
    # 生成监控报告
    print("\n" + monitor.generate_report())
    
    # 测试约束状态
    print("\n约束状态:")
    status = glacier_constraints.get_constraint_status()
    for name, info in status.items():
        print(f"  {name}: 激活={info['active']}, 权重={info['weight']:.3f}")
    
    # 测试约束操作
    print("\n测试约束操作...")
    glacier_constraints.deactivate_constraint('glen_flow_law')
    glacier_constraints.update_constraint_weight('mass_balance', 2.0)
    
    print("更新后的约束状态:")
    status = glacier_constraints.get_constraint_status()
    for name, info in status.items():
        print(f"  {name}: 激活={info['active']}, 权重={info['weight']:.3f}")