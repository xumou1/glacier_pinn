#!/usr/bin/env python3
"""
阶段3：耦合优化模块

实现PINNs模型的耦合优化阶段，包括多物理场耦合、
多尺度优化和自适应权重调整。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from abc import ABC, abstractmethod

class CoupledOptimizationStage(ABC):
    """
    耦合优化阶段基类
    
    定义耦合优化阶段的通用接口，包括：
    - 多物理场耦合优化
    - 多尺度协调优化
    - 自适应权重调整
    - 收敛性监控
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any],
                 coupling_config: Dict[str, Any]):
        """
        初始化耦合优化阶段
        
        Args:
            model: PINNs模型
            optimizer_config: 优化器配置
            physics_config: 物理约束配置
            coupling_config: 耦合配置
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.physics_config = physics_config
        self.coupling_config = coupling_config
        self.train_state = None
        self.optimization_history = []
        
    @abstractmethod
    def initialize_coupled_optimization(self, 
                                      physics_integrated_state: train_state.TrainState,
                                      sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化耦合优化训练状态
        
        Args:
            physics_integrated_state: 物理集成后的状态
            sample_input: 样本输入
            
        Returns:
            耦合优化训练状态
        """
        pass
        
    @abstractmethod
    def coupled_optimization_step(self, 
                                 state: train_state.TrainState,
                                 batch: Dict[str, jnp.ndarray],
                                 rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行一步耦合优化
        
        Args:
            state: 当前训练状态
            batch: 训练批次
            rng_key: 随机数生成器密钥
            
        Returns:
            更新后的训练状态和损失信息
        """
        pass
        
    def run_coupled_optimization(self, 
                                physics_integrated_state: train_state.TrainState,
                                training_data: Dict[str, jnp.ndarray],
                                n_epochs: int,
                                batch_size: int,
                                rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        运行完整的耦合优化过程
        
        Args:
            physics_integrated_state: 物理集成后的状态
            training_data: 训练数据
            n_epochs: 训练轮数
            batch_size: 批次大小
            rng_key: 随机数生成器密钥
            
        Returns:
            耦合优化完成的状态
        """
        # 初始化耦合优化状态
        sample_input = {k: v[:1] for k, v in training_data.items() if k in ['x', 'y', 't']}
        self.train_state = self.initialize_coupled_optimization(
            physics_integrated_state, sample_input
        )
        
        # 创建数据批次
        n_samples = len(training_data['x'])
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 收敛监控
        convergence_window = self.coupling_config.get('convergence_window', 50)
        convergence_threshold = self.coupling_config.get('convergence_threshold', 1e-6)
        
        for epoch in range(n_epochs):
            epoch_losses = {
                'total': 0.0, 'physics': 0.0, 'data': 0.0, 
                'coupling': 0.0, 'multiscale': 0.0
            }
            
            # 打乱数据
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.random.permutation(shuffle_key, n_samples)
            
            for batch_idx in range(n_batches):
                # 创建批次
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = perm[start_idx:end_idx]
                
                batch = {k: v[batch_indices] for k, v in training_data.items()}
                
                # 执行耦合优化步骤
                rng_key, step_key = jax.random.split(rng_key)
                self.train_state, loss_info = self.coupled_optimization_step(
                    self.train_state, batch, step_key
                )
                
                # 累积损失
                for key in epoch_losses:
                    if key in loss_info:
                        epoch_losses[key] += loss_info[key]
                        
            # 记录优化历史
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            self.optimization_history.append(avg_losses)
            
            # 打印进度
            if epoch % 25 == 0:
                print(f"Coupled Optimization Epoch {epoch}:")
                for key, value in avg_losses.items():
                    print(f"  {key}_loss: {value:.6f}")
                    
            # 检查收敛性
            if self._check_convergence(convergence_window, convergence_threshold):
                print(f"Converged at epoch {epoch}")
                break
                
        return self.train_state
        
    def _check_convergence(self, window: int, threshold: float) -> bool:
        """
        检查优化收敛性
        
        Args:
            window: 收敛检查窗口
            threshold: 收敛阈值
            
        Returns:
            是否收敛
        """
        if len(self.optimization_history) < window:
            return False
            
        recent_losses = [h['total'] for h in self.optimization_history[-window:]]
        
        # 计算损失变化的标准差
        loss_std = jnp.std(jnp.array(recent_losses))
        
        return loss_std < threshold

class MultiPhysicsCoupledOptimization(CoupledOptimizationStage):
    """
    多物理场耦合优化
    
    实现多个物理场之间的耦合优化，包括：
    - 热-力耦合
    - 流-固耦合
    - 质量-动量耦合
    """
    
    def initialize_coupled_optimization(self, 
                                     physics_integrated_state: train_state.TrainState,
                                     sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化多物理场耦合优化的训练状态
        """
        # 使用物理集成后的参数
        params = physics_integrated_state.params
        
        # 创建多阶段优化器
        learning_rate = self.optimizer_config.get('coupled_learning_rate', 1e-4)
        
        # 使用余弦退火调度
        schedule = optax.cosine_onecycle_schedule(
            transition_steps=self.coupling_config.get('total_steps', 5000),
            peak_value=learning_rate,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # 组合优化器
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.optimizer_config.get('max_grad_norm', 1.0)),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.optimizer_config.get('weight_decay', 1e-5),
                b1=0.9,
                b2=0.999
            )
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def coupled_optimization_step(self, 
                                 state: train_state.TrainState,
                                 batch: Dict[str, jnp.ndarray],
                                 rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行多物理场耦合优化步骤
        """
        def loss_fn(params):
            # 前向传播
            predictions = state.apply_fn(params, batch)
            
            # 计算各种耦合损失
            thermal_mechanical_coupling = self._thermal_mechanical_coupling_loss(
                params, batch, state.apply_fn
            )
            
            fluid_solid_coupling = self._fluid_solid_coupling_loss(
                params, batch, state.apply_fn
            )
            
            mass_momentum_coupling = self._mass_momentum_coupling_loss(
                params, batch, state.apply_fn
            )
            
            # 物理一致性损失
            physics_consistency = self._physics_consistency_loss(
                params, batch, state.apply_fn
            )
            
            # 数据拟合损失
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_weighted_data_loss(
                    predictions, batch['observations']
                )
                
            # 耦合损失总和
            coupling_loss = (
                thermal_mechanical_coupling + 
                fluid_solid_coupling + 
                mass_momentum_coupling
            )
            
            # 总损失
            total_loss = (
                self.physics_config.get('coupling_weight', 1.0) * coupling_loss +
                self.physics_config.get('consistency_weight', 0.5) * physics_consistency +
                self.physics_config.get('data_weight', 0.1) * data_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'coupling_loss': coupling_loss,
                'thermal_mechanical_coupling': thermal_mechanical_coupling,
                'fluid_solid_coupling': fluid_solid_coupling,
                'mass_momentum_coupling': mass_momentum_coupling,
                'physics_consistency': physics_consistency,
                'data_loss': data_loss
            }
            
        # 计算梯度
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # 更新参数
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_info
        
    def _thermal_mechanical_coupling_loss(self, 
                                         params: Dict,
                                         batch: Dict[str, jnp.ndarray],
                                         apply_fn: Callable) -> float:
        """
        热-力耦合损失
        
        温度影响冰的流动参数，应力影响温度分布
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 检查模型是否包含温度场
        sample_output = model_outputs(batch['x'][0], batch['y'][0], batch['t'][0])
        if 'temperature' not in sample_output:
            return 0.0
            
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        temperature = outputs['temperature']
        velocity_x = outputs['velocity_x']
        velocity_y = outputs['velocity_y']
        
        # 计算应变率
        def velocity_x_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_x']
            
        def velocity_y_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_y']
            
        dvx_dx = jax.vmap(jax.grad(velocity_x_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dy = jax.vmap(jax.grad(velocity_y_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        dvx_dy = jax.vmap(jax.grad(velocity_x_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dx = jax.vmap(jax.grad(velocity_y_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 应变率第二不变量
        epsilon_xx = dvx_dx
        epsilon_yy = dvy_dy
        epsilon_xy = 0.5 * (dvx_dy + dvy_dx)
        
        epsilon_II = jnp.sqrt(
            epsilon_xx**2 + epsilon_yy**2 + 2 * epsilon_xy**2 + 1e-12
        )
        
        # 温度依赖的流动参数
        def arrhenius_factor(T):
            # Arrhenius关系：A(T) = A0 * exp(-Q/RT)
            T_kelvin = T + 273.15
            Q = 60000.0  # 激活能 (J/mol)
            R = 8.314    # 气体常数 (J/mol/K)
            A0 = 2.4e-24 # 参考流动参数
            
            return A0 * jnp.exp(-Q / (R * T_kelvin))
            
        A_T = jax.vmap(arrhenius_factor)(temperature)
        
        # Glen's law with temperature dependence
        n = 3.0
        tau_e_predicted = (epsilon_II / A_T)**(1/n)
        
        # 简化的应力计算
        eta = 1e12  # 有效粘度
        tau_e_computed = eta * epsilon_II
        
        # 热-力耦合残差
        thermal_mechanical_residual = tau_e_predicted - tau_e_computed
        
        return jnp.mean(thermal_mechanical_residual**2)
        
    def _fluid_solid_coupling_loss(self, 
                                  params: Dict,
                                  batch: Dict[str, jnp.ndarray],
                                  apply_fn: Callable) -> float:
        """
        流-固耦合损失
        
        冰川流动与固体变形的耦合
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        thickness = outputs['thickness']
        velocity_x = outputs['velocity_x']
        velocity_y = outputs['velocity_y']
        
        # 计算厚度梯度
        def thickness_fn(x, y, t):
            return model_outputs(x, y, t)['thickness']
            
        dh_dx = jax.vmap(jax.grad(thickness_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dh_dy = jax.vmap(jax.grad(thickness_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 流-固耦合条件：速度应该与厚度梯度相关
        # 简化的关系：v ∝ ∇h
        coupling_factor = 1e-3
        
        velocity_magnitude = jnp.sqrt(velocity_x**2 + velocity_y**2 + 1e-12)
        thickness_gradient_magnitude = jnp.sqrt(dh_dx**2 + dh_dy**2 + 1e-12)
        
        # 耦合残差
        coupling_residual = velocity_magnitude - coupling_factor * thickness_gradient_magnitude
        
        return jnp.mean(coupling_residual**2)
        
    def _mass_momentum_coupling_loss(self, 
                                    params: Dict,
                                    batch: Dict[str, jnp.ndarray],
                                    apply_fn: Callable) -> float:
        """
        质量-动量耦合损失
        
        质量守恒和动量守恒的耦合一致性
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算质量通量
        def mass_flux_x(x, y, t):
            out = model_outputs(x, y, t)
            return out['thickness'] * out['velocity_x']
            
        def mass_flux_y(x, y, t):
            out = model_outputs(x, y, t)
            return out['thickness'] * out['velocity_y']
            
        # 计算动量
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        rho = 917.0  # 冰密度
        
        momentum_x = rho * outputs['thickness'] * outputs['velocity_x']
        momentum_y = rho * outputs['thickness'] * outputs['velocity_y']
        
        # 质量-动量一致性：动量变化应该与质量通量变化一致
        dflux_x_dt = jax.vmap(jax.grad(mass_flux_x, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        dflux_y_dt = jax.vmap(jax.grad(mass_flux_y, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        def momentum_x_fn(x, y, t):
            out = model_outputs(x, y, t)
            return rho * out['thickness'] * out['velocity_x']
            
        def momentum_y_fn(x, y, t):
            out = model_outputs(x, y, t)
            return rho * out['thickness'] * out['velocity_y']
            
        dmom_x_dt = jax.vmap(jax.grad(momentum_x_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        dmom_y_dt = jax.vmap(jax.grad(momentum_y_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 耦合残差
        coupling_x_residual = dmom_x_dt - rho * dflux_x_dt
        coupling_y_residual = dmom_y_dt - rho * dflux_y_dt
        
        return jnp.mean(coupling_x_residual**2 + coupling_y_residual**2)
        
    def _physics_consistency_loss(self, 
                                 params: Dict,
                                 batch: Dict[str, jnp.ndarray],
                                 apply_fn: Callable) -> float:
        """
        物理一致性损失
        
        确保所有物理量在物理上是一致的
        """
        outputs = apply_fn(params, batch)
        
        consistency_loss = 0.0
        
        # 厚度非负性
        thickness = outputs['thickness']
        thickness_constraint = jnp.mean(jnp.maximum(0.0, -thickness)**2)
        consistency_loss += thickness_constraint
        
        # 速度合理性
        velocity_x = outputs['velocity_x']
        velocity_y = outputs['velocity_y']
        velocity_magnitude = jnp.sqrt(velocity_x**2 + velocity_y**2)
        
        # 限制最大速度（例如，不超过2000 m/year）
        max_velocity = 2000.0
        velocity_constraint = jnp.mean(
            jnp.maximum(0.0, velocity_magnitude - max_velocity)**2
        )
        consistency_loss += velocity_constraint
        
        # 温度合理性（如果存在）
        if 'temperature' in outputs:
            temperature = outputs['temperature']
            # 温度应该在合理范围内（-50°C 到 0°C）
            temp_lower_bound = jnp.mean(jnp.maximum(0.0, -50.0 - temperature)**2)
            temp_upper_bound = jnp.mean(jnp.maximum(0.0, temperature - 0.0)**2)
            consistency_loss += temp_lower_bound + temp_upper_bound
            
        return consistency_loss
        
    def _compute_weighted_data_loss(self, 
                                   predictions: Dict[str, jnp.ndarray],
                                   observations: Dict[str, jnp.ndarray]) -> float:
        """
        计算加权数据损失
        """
        data_loss = 0.0
        
        # 变量权重
        variable_weights = {
            'thickness': 1.0,
            'velocity_x': 0.8,
            'velocity_y': 0.8,
            'surface_elevation': 0.9,
            'temperature': 0.6
        }
        
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                weight = variable_weights.get(key, 1.0)
                
                # 计算相对误差
                relative_error = jnp.abs(pred - obs) / (jnp.abs(obs) + 1e-6)
                data_loss += weight * jnp.mean(relative_error**2)
                
        return data_loss

class MultiscaleCoupledOptimization(CoupledOptimizationStage):
    """
    多尺度耦合优化
    
    实现多时空尺度的耦合优化，确保模型在不同尺度上的一致性。
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any],
                 coupling_config: Dict[str, Any],
                 multiscale_config: Dict[str, Any]):
        super().__init__(model, optimizer_config, physics_config, coupling_config)
        self.multiscale_config = multiscale_config
        self.scale_weights = self._initialize_scale_weights()
        
    def _initialize_scale_weights(self) -> Dict[str, float]:
        """
        初始化多尺度权重
        """
        spatial_scales = self.multiscale_config.get('spatial_scales', [1.0, 10.0, 100.0])
        temporal_scales = self.multiscale_config.get('temporal_scales', [0.1, 1.0, 10.0])
        
        weights = {}
        
        # 空间尺度权重
        for i, scale in enumerate(spatial_scales):
            weights[f'spatial_scale_{i}'] = 1.0 / len(spatial_scales)
            
        # 时间尺度权重
        for i, scale in enumerate(temporal_scales):
            weights[f'temporal_scale_{i}'] = 1.0 / len(temporal_scales)
            
        return weights
        
    def initialize_coupled_optimization(self, 
                                      physics_integrated_state: train_state.TrainState,
                                      sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化多尺度耦合优化的训练状态
        """
        params = physics_integrated_state.params
        
        # 多尺度学习率调度
        base_lr = self.optimizer_config.get('multiscale_learning_rate', 5e-5)
        
        # 使用分段常数调度
        boundaries = [1000, 3000, 5000]
        values = [base_lr, base_lr * 0.5, base_lr * 0.1, base_lr * 0.01]
        
        schedule = optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales={
                1000: 0.5,
                3000: 0.2,
                5000: 0.1
            }
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=1e-6
            )
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def coupled_optimization_step(self, 
                                 state: train_state.TrainState,
                                 batch: Dict[str, jnp.ndarray],
                                 rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行多尺度耦合优化步骤
        """
        def loss_fn(params):
            predictions = state.apply_fn(params, batch)
            
            # 计算多尺度损失
            multiscale_loss = self._compute_multiscale_loss(
                params, batch, state.apply_fn
            )
            
            # 尺度一致性损失
            scale_consistency_loss = self._scale_consistency_loss(
                params, batch, state.apply_fn
            )
            
            # 跨尺度耦合损失
            cross_scale_coupling_loss = self._cross_scale_coupling_loss(
                params, batch, state.apply_fn
            )
            
            # 数据损失
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_multiscale_data_loss(
                    predictions, batch['observations'], batch
                )
                
            total_loss = (
                self.physics_config.get('multiscale_weight', 1.0) * multiscale_loss +
                self.physics_config.get('consistency_weight', 0.5) * scale_consistency_loss +
                self.physics_config.get('cross_scale_weight', 0.3) * cross_scale_coupling_loss +
                self.physics_config.get('data_weight', 0.1) * data_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'multiscale_loss': multiscale_loss,
                'scale_consistency_loss': scale_consistency_loss,
                'cross_scale_coupling_loss': cross_scale_coupling_loss,
                'data_loss': data_loss
            }
            
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        # 更新尺度权重
        self._update_scale_weights(loss_info)
        
        return new_state, loss_info
        
    def _compute_multiscale_loss(self, 
                                params: Dict,
                                batch: Dict[str, jnp.ndarray],
                                apply_fn: Callable) -> float:
        """
        计算多尺度损失
        """
        multiscale_loss = 0.0
        
        # 空间多尺度损失
        spatial_scales = self.multiscale_config.get('spatial_scales', [1.0, 10.0, 100.0])
        
        for i, scale in enumerate(spatial_scales):
            scale_loss = self._compute_spatial_scale_loss(
                params, batch, apply_fn, scale
            )
            weight = self.scale_weights.get(f'spatial_scale_{i}', 1.0)
            multiscale_loss += weight * scale_loss
            
        # 时间多尺度损失
        temporal_scales = self.multiscale_config.get('temporal_scales', [0.1, 1.0, 10.0])
        
        for i, scale in enumerate(temporal_scales):
            scale_loss = self._compute_temporal_scale_loss(
                params, batch, apply_fn, scale
            )
            weight = self.scale_weights.get(f'temporal_scale_{i}', 1.0)
            multiscale_loss += weight * scale_loss
            
        return multiscale_loss
        
    def _compute_spatial_scale_loss(self, 
                                   params: Dict,
                                   batch: Dict[str, jnp.ndarray],
                                   apply_fn: Callable,
                                   scale: float) -> float:
        """
        计算特定空间尺度的损失
        """
        # 根据尺度调整空间导数的权重
        scale_factor = 1.0 / scale
        
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算空间导数
        def thickness_fn(x, y, t):
            return model_outputs(x, y, t)['thickness']
            
        dh_dx = jax.vmap(jax.grad(thickness_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dh_dy = jax.vmap(jax.grad(thickness_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 尺度调整的梯度损失
        gradient_magnitude = jnp.sqrt(dh_dx**2 + dh_dy**2 + 1e-12)
        scale_adjusted_loss = jnp.mean((scale_factor * gradient_magnitude)**2)
        
        return scale_adjusted_loss
        
    def _compute_temporal_scale_loss(self, 
                                    params: Dict,
                                    batch: Dict[str, jnp.ndarray],
                                    apply_fn: Callable,
                                    scale: float) -> float:
        """
        计算特定时间尺度的损失
        """
        # 根据尺度调整时间导数的权重
        scale_factor = 1.0 / scale
        
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算时间导数
        def thickness_fn(x, y, t):
            return model_outputs(x, y, t)['thickness']
            
        dh_dt = jax.vmap(jax.grad(thickness_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 尺度调整的时间导数损失
        temporal_loss = jnp.mean((scale_factor * dh_dt)**2)
        
        return temporal_loss
        
    def _scale_consistency_loss(self, 
                               params: Dict,
                               batch: Dict[str, jnp.ndarray],
                               apply_fn: Callable) -> float:
        """
        尺度一致性损失
        
        确保模型在不同尺度上的预测是一致的
        """
        # 在不同尺度的点上评估模型
        outputs_fine = apply_fn(params, batch)
        
        # 创建粗尺度采样点
        coarse_factor = 2
        coarse_indices = jnp.arange(0, len(batch['x']), coarse_factor)
        
        if len(coarse_indices) == 0:
            return 0.0
            
        coarse_batch = {
            k: v[coarse_indices] for k, v in batch.items() 
            if k in ['x', 'y', 't']
        }
        
        outputs_coarse = apply_fn(params, coarse_batch)
        
        # 比较细尺度和粗尺度的预测
        consistency_loss = 0.0
        
        for key in outputs_fine:
            if key in outputs_coarse:
                fine_values = outputs_fine[key][coarse_indices]
                coarse_values = outputs_coarse[key]
                
                # 计算相对差异
                relative_diff = jnp.abs(fine_values - coarse_values) / (
                    jnp.abs(coarse_values) + 1e-6
                )
                consistency_loss += jnp.mean(relative_diff**2)
                
        return consistency_loss
        
    def _cross_scale_coupling_loss(self, 
                                  params: Dict,
                                  batch: Dict[str, jnp.ndarray],
                                  apply_fn: Callable) -> float:
        """
        跨尺度耦合损失
        
        确保不同尺度的物理过程正确耦合
        """
        # 简化的跨尺度耦合：大尺度趋势应该与小尺度平均一致
        
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        
        # 计算局部平均（模拟大尺度）
        window_size = min(10, len(batch['x']) // 4)
        
        if window_size < 2:
            return 0.0
            
        # 滑动窗口平均
        def moving_average(values, window):
            if len(values) < window:
                return values
            padded = jnp.pad(values, (window//2, window//2), mode='edge')
            return jnp.convolve(padded, jnp.ones(window)/window, mode='valid')
            
        coupling_loss = 0.0
        
        for key in outputs:
            values = outputs[key]
            averaged_values = moving_average(values, window_size)
            
            # 跨尺度一致性
            if len(averaged_values) == len(values):
                scale_coupling = jnp.mean((values - averaged_values)**2)
                coupling_loss += scale_coupling
                
        return coupling_loss
        
    def _compute_multiscale_data_loss(self, 
                                     predictions: Dict[str, jnp.ndarray],
                                     observations: Dict[str, jnp.ndarray],
                                     batch: Dict[str, jnp.ndarray]) -> float:
        """
        计算多尺度数据损失
        """
        data_loss = 0.0
        
        # 根据数据的空间分辨率调整权重
        if 'spatial_resolution' in batch:
            resolution = batch['spatial_resolution']
            # 高分辨率数据权重更高
            resolution_weight = 1.0 / (resolution + 1e-6)
        else:
            resolution_weight = 1.0
            
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                
                # 多尺度加权
                mse = jnp.mean((pred - obs)**2)
                data_loss += resolution_weight * mse
                
        return data_loss
        
    def _update_scale_weights(self, loss_info: Dict[str, float]):
        """
        更新尺度权重
        """
        # 简化的自适应权重更新
        adaptation_rate = self.multiscale_config.get('weight_adaptation_rate', 0.01)
        
        # 如果某个尺度的损失较高，增加其权重
        for key in self.scale_weights:
            if f'{key}_loss' in loss_info:
                loss_value = loss_info[f'{key}_loss']
                if loss_value > 1e-3:  # 阈值
                    self.scale_weights[key] *= (1 + adaptation_rate)
                else:
                    self.scale_weights[key] *= (1 - adaptation_rate * 0.5)
                    
                # 限制权重范围
                self.scale_weights[key] = jnp.clip(
                    self.scale_weights[key], 0.1, 5.0
                )

def create_coupled_optimization_stage(stage_type: str,
                                     model: nn.Module,
                                     optimizer_config: Dict[str, Any],
                                     physics_config: Dict[str, Any],
                                     coupling_config: Dict[str, Any],
                                     **kwargs) -> CoupledOptimizationStage:
    """
    工厂函数：创建耦合优化阶段
    
    Args:
        stage_type: 耦合优化类型 ('multiphysics', 'multiscale')
        model: PINNs模型
        optimizer_config: 优化器配置
        physics_config: 物理约束配置
        coupling_config: 耦合配置
        **kwargs: 额外参数
        
    Returns:
        耦合优化阶段实例
    """
    if stage_type == 'multiphysics':
        return MultiPhysicsCoupledOptimization(
            model, optimizer_config, physics_config, coupling_config
        )
    elif stage_type == 'multiscale':
        return MultiscaleCoupledOptimization(
            model, optimizer_config, physics_config, coupling_config, **kwargs
        )
    else:
        raise ValueError(f"Unknown coupled optimization stage type: {stage_type}")

if __name__ == "__main__":
    # 测试代码
    print("Coupled optimization stage module loaded successfully")
    
    # 配置示例
    optimizer_config = {
        'coupled_learning_rate': 1e-4,
        'multiscale_learning_rate': 5e-5,
        'max_grad_norm': 1.0,
        'weight_decay': 1e-5
    }
    
    physics_config = {
        'coupling_weight': 1.0,
        'consistency_weight': 0.5,
        'multiscale_weight': 1.0,
        'cross_scale_weight': 0.3,
        'data_weight': 0.1
    }
    
    coupling_config = {
        'total_steps': 5000,
        'convergence_window': 50,
        'convergence_threshold': 1e-6
    }
    
    multiscale_config = {
        'spatial_scales': [1.0, 10.0, 100.0],
        'temporal_scales': [0.1, 1.0, 10.0],
        'weight_adaptation_rate': 0.01
    }
    
    print("Configuration loaded:")
    print(f"Optimizer config: {optimizer_config}")
    print(f"Physics config: {physics_config}")
    print(f"Coupling config: {coupling_config}")
    print(f"Multiscale config: {multiscale_config}")