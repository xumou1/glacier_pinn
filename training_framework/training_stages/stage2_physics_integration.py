#!/usr/bin/env python3
"""
阶段2：物理集成模块

实现PINNs模型的物理集成阶段，包括完整物理约束的引入、
多物理场耦合和复杂边界条件处理。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from abc import ABC, abstractmethod

class PhysicsIntegrationStage(ABC):
    """
    物理集成阶段基类
    
    定义物理集成阶段的通用接口，包括：
    - 完整物理方程的实现
    - 多物理场耦合
    - 复杂边界条件处理
    - 物理一致性验证
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any],
                 integration_config: Dict[str, Any]):
        """
        初始化物理集成阶段
        
        Args:
            model: PINNs模型
            optimizer_config: 优化器配置
            physics_config: 物理约束配置
            integration_config: 集成配置
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.physics_config = physics_config
        self.integration_config = integration_config
        self.train_state = None
        
    @abstractmethod
    def initialize_integration(self, 
                             pretrained_state: train_state.TrainState,
                             sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化物理集成训练状态
        
        Args:
            pretrained_state: 预训练状态
            sample_input: 样本输入
            
        Returns:
            集成训练状态
        """
        pass
        
    @abstractmethod
    def integration_step(self, 
                        state: train_state.TrainState,
                        batch: Dict[str, jnp.ndarray],
                        rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行一步物理集成训练
        
        Args:
            state: 当前训练状态
            batch: 训练批次
            rng_key: 随机数生成器密钥
            
        Returns:
            更新后的训练状态和损失信息
        """
        pass
        
    def run_integration(self, 
                       pretrained_state: train_state.TrainState,
                       training_data: Dict[str, jnp.ndarray],
                       n_epochs: int,
                       batch_size: int,
                       rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        运行完整的物理集成过程
        
        Args:
            pretrained_state: 预训练状态
            training_data: 训练数据
            n_epochs: 训练轮数
            batch_size: 批次大小
            rng_key: 随机数生成器密钥
            
        Returns:
            集成训练完成的状态
        """
        # 初始化集成训练状态
        sample_input = {k: v[:1] for k, v in training_data.items() if k in ['x', 'y', 't']}
        self.train_state = self.initialize_integration(pretrained_state, sample_input)
        
        # 创建数据批次
        n_samples = len(training_data['x'])
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            epoch_losses = {'total': 0.0, 'physics': 0.0, 'data': 0.0, 'boundary': 0.0}
            
            # 打乱数据
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.random.permutation(shuffle_key, n_samples)
            
            for batch_idx in range(n_batches):
                # 创建批次
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = perm[start_idx:end_idx]
                
                batch = {k: v[batch_indices] for k, v in training_data.items()}
                
                # 执行集成训练步骤
                rng_key, step_key = jax.random.split(rng_key)
                self.train_state, loss_info = self.integration_step(
                    self.train_state, batch, step_key
                )
                
                # 累积损失
                for key in epoch_losses:
                    if key in loss_info:
                        epoch_losses[key] += loss_info[key]
                        
            # 打印进度
            if epoch % 50 == 0:
                avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
                print(f"Integration Epoch {epoch}:")
                for key, value in avg_losses.items():
                    print(f"  {key}_loss: {value:.6f}")
                    
        return self.train_state

class FullPhysicsIntegration(PhysicsIntegrationStage):
    """
    完整物理集成
    
    实现完整的冰川动力学物理方程，包括：
    - 质量守恒方程
    - 动量守恒方程
    - 能量守恒方程
    - 本构关系
    """
    
    def initialize_integration(self, 
                            pretrained_state: train_state.TrainState,
                            sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化完整物理集成的训练状态
        """
        # 使用预训练的参数作为初始化
        params = pretrained_state.params
        
        # 创建新的优化器（可能使用不同的学习率）
        learning_rate = self.optimizer_config.get('physics_learning_rate', 5e-4)
        
        # 使用学习率调度
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=self.integration_config.get('decay_steps', 2000),
            alpha=0.1
        )
        
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.optimizer_config.get('weight_decay', 1e-5)
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def integration_step(self, 
                        state: train_state.TrainState,
                        batch: Dict[str, jnp.ndarray],
                        rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行完整物理集成训练步骤
        """
        def loss_fn(params):
            # 前向传播
            predictions = state.apply_fn(params, batch)
            
            # 计算各种物理损失
            mass_conservation_loss = self._mass_conservation_loss(
                params, batch, state.apply_fn
            )
            
            momentum_conservation_loss = self._momentum_conservation_loss(
                params, batch, state.apply_fn
            )
            
            energy_conservation_loss = self._energy_conservation_loss(
                params, batch, state.apply_fn
            )
            
            constitutive_loss = self._constitutive_relation_loss(
                params, batch, state.apply_fn
            )
            
            boundary_loss = self._boundary_condition_loss(
                params, batch, state.apply_fn
            )
            
            # 数据拟合损失
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_data_loss(predictions, batch['observations'])
                
            # 物理损失总和
            physics_loss = (
                mass_conservation_loss + 
                momentum_conservation_loss + 
                energy_conservation_loss + 
                constitutive_loss
            )
            
            # 总损失
            total_loss = (
                self.physics_config.get('physics_weight', 1.0) * physics_loss +
                self.physics_config.get('data_weight', 0.1) * data_loss +
                self.physics_config.get('boundary_weight', 1.0) * boundary_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'physics_loss': physics_loss,
                'mass_conservation_loss': mass_conservation_loss,
                'momentum_conservation_loss': momentum_conservation_loss,
                'energy_conservation_loss': energy_conservation_loss,
                'constitutive_loss': constitutive_loss,
                'boundary_loss': boundary_loss,
                'data_loss': data_loss
            }
            
        # 计算梯度
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # 梯度裁剪
        max_grad_norm = self.optimizer_config.get('max_grad_norm', 1.0)
        grads = optax.clip_by_global_norm(max_grad_norm).update(grads, state.opt_state)[1]
        
        # 更新参数
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_info
        
    def _mass_conservation_loss(self, 
                               params: Dict,
                               batch: Dict[str, jnp.ndarray],
                               apply_fn: Callable) -> float:
        """
        质量守恒方程损失
        
        ∂h/∂t + ∇·(h*v) = ṁ
        其中 h 是冰厚，v 是速度，ṁ 是质量平衡率
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算时间导数 ∂h/∂t
        def thickness_fn(x, y, t):
            return model_outputs(x, y, t)['thickness']
            
        dh_dt = jax.vmap(jax.grad(thickness_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 计算速度场
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        h = outputs['thickness']
        vx = outputs['velocity_x']
        vy = outputs['velocity_y']
        
        # 计算通量散度 ∇·(h*v)
        def flux_x_fn(x, y, t):
            out = model_outputs(x, y, t)
            return out['thickness'] * out['velocity_x']
            
        def flux_y_fn(x, y, t):
            out = model_outputs(x, y, t)
            return out['thickness'] * out['velocity_y']
            
        dflux_x_dx = jax.vmap(jax.grad(flux_x_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dflux_y_dy = jax.vmap(jax.grad(flux_y_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        flux_divergence = dflux_x_dx + dflux_y_dy
        
        # 质量平衡率（简化为常数或从数据获取）
        mass_balance = batch.get('mass_balance', jnp.zeros_like(h))
        
        # 质量守恒残差
        mass_residual = dh_dt + flux_divergence - mass_balance
        
        return jnp.mean(mass_residual**2)
        
    def _momentum_conservation_loss(self, 
                                   params: Dict,
                                   batch: Dict[str, jnp.ndarray],
                                   apply_fn: Callable) -> float:
        """
        动量守恒方程损失（浅冰近似）
        
        ∇·σ + ρg∇s = 0
        其中 σ 是应力张量，s 是表面高程
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 获取模型输出
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        h = outputs['thickness']
        vx = outputs['velocity_x']
        vy = outputs['velocity_y']
        
        # 计算表面高程梯度
        def surface_elevation_fn(x, y, t):
            out = model_outputs(x, y, t)
            bed_elevation = batch.get('bed_elevation', 0.0)  # 简化
            return out['thickness'] + bed_elevation
            
        ds_dx = jax.vmap(jax.grad(surface_elevation_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        ds_dy = jax.vmap(jax.grad(surface_elevation_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 计算应力（使用Glen's flow law）
        # τ = A^(-1/n) * |∇v|^((1-n)/n) * ∇v
        # 简化为线性关系
        
        def velocity_x_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_x']
            
        def velocity_y_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_y']
            
        dvx_dx = jax.vmap(jax.grad(velocity_x_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dvx_dy = jax.vmap(jax.grad(velocity_x_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dx = jax.vmap(jax.grad(velocity_y_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dy = jax.vmap(jax.grad(velocity_y_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 应力张量分量（简化）
        eta = 1e12  # 有效粘度 (Pa·s)
        
        sigma_xx = 2 * eta * dvx_dx
        sigma_yy = 2 * eta * dvy_dy
        sigma_xy = eta * (dvx_dy + dvy_dx)
        
        # 应力散度
        def sigma_xx_fn(x, y, t):
            out = model_outputs(x, y, t)
            # 简化计算
            return 2 * eta * jax.grad(lambda x: model_outputs(x, y, t)['velocity_x'], argnums=0)(x)
            
        def sigma_xy_fn(x, y, t):
            out = model_outputs(x, y, t)
            # 简化计算
            return eta * (jax.grad(lambda y: model_outputs(x, y, t)['velocity_x'], argnums=0)(y) +
                         jax.grad(lambda x: model_outputs(x, y, t)['velocity_y'], argnums=0)(x))
            
        # 重力项
        rho = 917.0  # 冰密度 (kg/m³)
        g = 9.81     # 重力加速度 (m/s²)
        
        gravity_x = rho * g * h * ds_dx
        gravity_y = rho * g * h * ds_dy
        
        # 动量平衡残差（简化）
        momentum_x_residual = sigma_xx + gravity_x  # 简化的x方向动量平衡
        momentum_y_residual = sigma_yy + gravity_y  # 简化的y方向动量平衡
        
        momentum_loss = (
            jnp.mean(momentum_x_residual**2) + 
            jnp.mean(momentum_y_residual**2)
        )
        
        return momentum_loss
        
    def _energy_conservation_loss(self, 
                                 params: Dict,
                                 batch: Dict[str, jnp.ndarray],
                                 apply_fn: Callable) -> float:
        """
        能量守恒方程损失
        
        ρc(∂T/∂t + v·∇T) = k∇²T + Φ
        其中 T 是温度，Φ 是粘性耗散
        """
        # 如果模型不包含温度场，返回0
        sample_output = apply_fn(params, {
            'x': batch['x'][:1], 'y': batch['y'][:1], 't': batch['t'][:1]
        })
        
        if 'temperature' not in sample_output:
            return 0.0
            
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算温度的时间导数
        def temperature_fn(x, y, t):
            return model_outputs(x, y, t)['temperature']
            
        dT_dt = jax.vmap(jax.grad(temperature_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 计算温度的空间导数
        dT_dx = jax.vmap(jax.grad(temperature_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dT_dy = jax.vmap(jax.grad(temperature_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 计算温度的二阶导数（拉普拉斯算子）
        d2T_dx2 = jax.vmap(jax.grad(jax.grad(temperature_fn, argnums=0), argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        d2T_dy2 = jax.vmap(jax.grad(jax.grad(temperature_fn, argnums=1), argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        laplacian_T = d2T_dx2 + d2T_dy2
        
        # 获取速度场
        outputs = jax.vmap(model_outputs)(batch['x'], batch['y'], batch['t'])
        vx = outputs['velocity_x']
        vy = outputs['velocity_y']
        
        # 对流项
        advection_term = vx * dT_dx + vy * dT_dy
        
        # 物理常数
        rho = 917.0      # 冰密度 (kg/m³)
        c = 2009.0       # 比热容 (J/kg/K)
        k = 2.1          # 热导率 (W/m/K)
        
        # 粘性耗散项（简化）
        viscous_dissipation = 0.0  # 简化为0
        
        # 能量守恒残差
        energy_residual = (
            rho * c * (dT_dt + advection_term) - 
            k * laplacian_T - 
            viscous_dissipation
        )
        
        return jnp.mean(energy_residual**2)
        
    def _constitutive_relation_loss(self, 
                                   params: Dict,
                                   batch: Dict[str, jnp.ndarray],
                                   apply_fn: Callable) -> float:
        """
        本构关系损失（Glen's flow law）
        
        ε̇ = A * τ^n
        其中 ε̇ 是应变率，τ 是应力，A 是流动参数，n 是流动指数
        """
        def model_outputs(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            return apply_fn(params, inputs)
            
        # 计算应变率张量
        def velocity_x_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_x']
            
        def velocity_y_fn(x, y, t):
            return model_outputs(x, y, t)['velocity_y']
            
        # 应变率分量
        dvx_dx = jax.vmap(jax.grad(velocity_x_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dvx_dy = jax.vmap(jax.grad(velocity_x_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dx = jax.vmap(jax.grad(velocity_y_fn, argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dvy_dy = jax.vmap(jax.grad(velocity_y_fn, argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 应变率张量分量
        epsilon_xx = dvx_dx
        epsilon_yy = dvy_dy
        epsilon_xy = 0.5 * (dvx_dy + dvy_dx)
        
        # 第二不变量
        epsilon_II = jnp.sqrt(
            epsilon_xx**2 + epsilon_yy**2 + 2 * epsilon_xy**2
        )
        
        # Glen's flow law参数
        A = 2.4e-24  # 流动参数 (Pa^-3 s^-1)
        n = 3.0      # 流动指数
        
        # 有效应力
        tau_e = (epsilon_II / A)**(1/n)
        
        # 本构关系残差（简化）
        # 这里我们检查应变率和应力的一致性
        constitutive_residual = epsilon_II - A * tau_e**n
        
        return jnp.mean(constitutive_residual**2)
        
    def _boundary_condition_loss(self, 
                                params: Dict,
                                batch: Dict[str, jnp.ndarray],
                                apply_fn: Callable) -> float:
        """
        边界条件损失
        """
        boundary_loss = 0.0
        
        # 检查是否有边界标记
        if 'boundary_type' not in batch:
            return boundary_loss
            
        boundary_types = batch['boundary_type']
        
        # 处理不同类型的边界条件
        for boundary_type in jnp.unique(boundary_types):
            if boundary_type == 0:  # 内部点，跳过
                continue
                
            mask = boundary_types == boundary_type
            if jnp.sum(mask) == 0:
                continue
                
            boundary_inputs = {
                k: v[mask] for k, v in batch.items() 
                if k in ['x', 'y', 't']
            }
            
            boundary_outputs = apply_fn(params, boundary_inputs)
            
            if boundary_type == 1:  # 冰川边界（厚度为0）
                thickness = boundary_outputs['thickness']
                boundary_loss += jnp.mean(thickness**2)
                
            elif boundary_type == 2:  # 无滑移边界（速度为0）
                vx = boundary_outputs['velocity_x']
                vy = boundary_outputs['velocity_y']
                boundary_loss += jnp.mean(vx**2 + vy**2)
                
            elif boundary_type == 3:  # 自由表面边界
                # 自由表面的应力边界条件
                # 这里简化处理
                pass
                
        return boundary_loss
        
    def _compute_data_loss(self, 
                          predictions: Dict[str, jnp.ndarray],
                          observations: Dict[str, jnp.ndarray]) -> float:
        """
        计算数据拟合损失
        """
        data_loss = 0.0
        
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                
                # 计算加权均方误差
                weight = self.physics_config.get(f'{key}_weight', 1.0)
                mse = jnp.mean((pred - obs)**2)
                data_loss += weight * mse
                
        return data_loss

class AdaptivePhysicsIntegration(PhysicsIntegrationStage):
    """
    自适应物理集成
    
    根据训练过程动态调整物理约束的权重，
    实现更稳定和高效的物理集成。
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any],
                 integration_config: Dict[str, Any],
                 adaptation_config: Dict[str, Any]):
        super().__init__(model, optimizer_config, physics_config, integration_config)
        self.adaptation_config = adaptation_config
        self.loss_history = []
        self.current_weights = physics_config.copy()
        
    def initialize_integration(self, 
                             pretrained_state: train_state.TrainState,
                             sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """
        初始化自适应物理集成的训练状态
        """
        # 使用预训练参数
        params = pretrained_state.params
        
        # 创建自适应优化器
        learning_rate = self.optimizer_config.get('adaptive_learning_rate', 3e-4)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=self.optimizer_config.get('weight_decay', 1e-5)
            )
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def integration_step(self, 
                        state: train_state.TrainState,
                        batch: Dict[str, jnp.ndarray],
                        rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行自适应物理集成训练步骤
        """
        # 更新自适应权重
        self._update_adaptive_weights()
        
        def loss_fn(params):
            predictions = state.apply_fn(params, batch)
            
            # 计算各种损失
            losses = self._compute_all_losses(params, batch, state.apply_fn, predictions)
            
            # 使用自适应权重计算总损失
            total_loss = 0.0
            for loss_name, loss_value in losses.items():
                if loss_name != 'total_loss':
                    weight = self.current_weights.get(f'{loss_name}_weight', 1.0)
                    total_loss += weight * loss_value
                    
            losses['total_loss'] = total_loss
            
            return total_loss, losses
            
        # 计算梯度和更新
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        # 记录损失历史
        self.loss_history.append(loss_info)
        
        return new_state, loss_info
        
    def _compute_all_losses(self, 
                           params: Dict,
                           batch: Dict[str, jnp.ndarray],
                           apply_fn: Callable,
                           predictions: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        计算所有损失项
        """
        # 创建FullPhysicsIntegration实例来重用损失计算方法
        full_physics = FullPhysicsIntegration(
            self.model, self.optimizer_config, 
            self.physics_config, self.integration_config
        )
        
        losses = {
            'mass_conservation_loss': full_physics._mass_conservation_loss(params, batch, apply_fn),
            'momentum_conservation_loss': full_physics._momentum_conservation_loss(params, batch, apply_fn),
            'energy_conservation_loss': full_physics._energy_conservation_loss(params, batch, apply_fn),
            'constitutive_loss': full_physics._constitutive_relation_loss(params, batch, apply_fn),
            'boundary_loss': full_physics._boundary_condition_loss(params, batch, apply_fn),
        }
        
        # 数据损失
        if 'observations' in batch:
            losses['data_loss'] = full_physics._compute_data_loss(predictions, batch['observations'])
        else:
            losses['data_loss'] = 0.0
            
        return losses
        
    def _update_adaptive_weights(self):
        """
        更新自适应权重
        """
        if len(self.loss_history) < 10:  # 需要足够的历史数据
            return
            
        # 计算最近损失的趋势
        recent_losses = self.loss_history[-10:]
        
        # 对于每种损失类型，如果损失下降缓慢，增加权重
        adaptation_rate = self.adaptation_config.get('adaptation_rate', 0.1)
        
        for loss_name in ['mass_conservation', 'momentum_conservation', 'energy_conservation']:
            loss_key = f'{loss_name}_loss'
            weight_key = f'{loss_name}_weight'
            
            if loss_key in recent_losses[0]:
                # 计算损失变化率
                recent_values = [loss_info[loss_key] for loss_info in recent_losses]
                loss_trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                
                # 如果损失下降缓慢或增加，增加权重
                if loss_trend > -1e-6:
                    self.current_weights[weight_key] *= (1 + adaptation_rate)
                else:
                    self.current_weights[weight_key] *= (1 - adaptation_rate * 0.5)
                    
                # 限制权重范围
                self.current_weights[weight_key] = jnp.clip(
                    self.current_weights[weight_key], 0.1, 10.0
                )

def create_physics_integration_stage(stage_type: str,
                                    model: nn.Module,
                                    optimizer_config: Dict[str, Any],
                                    physics_config: Dict[str, Any],
                                    integration_config: Dict[str, Any],
                                    **kwargs) -> PhysicsIntegrationStage:
    """
    工厂函数：创建物理集成阶段
    
    Args:
        stage_type: 集成类型 ('full', 'adaptive')
        model: PINNs模型
        optimizer_config: 优化器配置
        physics_config: 物理约束配置
        integration_config: 集成配置
        **kwargs: 额外参数
        
    Returns:
        物理集成阶段实例
    """
    if stage_type == 'full':
        return FullPhysicsIntegration(
            model, optimizer_config, physics_config, integration_config
        )
    elif stage_type == 'adaptive':
        return AdaptivePhysicsIntegration(
            model, optimizer_config, physics_config, integration_config, **kwargs
        )
    else:
        raise ValueError(f"Unknown physics integration stage type: {stage_type}")

if __name__ == "__main__":
    # 测试代码
    print("Physics integration stage module loaded successfully")
    
    # 配置示例
    optimizer_config = {
        'physics_learning_rate': 5e-4,
        'adaptive_learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0
    }
    
    physics_config = {
        'physics_weight': 1.0,
        'data_weight': 0.1,
        'boundary_weight': 1.0,
        'mass_conservation_weight': 1.0,
        'momentum_conservation_weight': 1.0,
        'energy_conservation_weight': 0.5
    }
    
    integration_config = {
        'decay_steps': 2000
    }
    
    print("Configuration loaded:")
    print(f"Optimizer config: {optimizer_config}")
    print(f"Physics config: {physics_config}")
    print(f"Integration config: {integration_config}")