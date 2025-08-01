#!/usr/bin/env python3
"""
阶段1：预训练模块

实现PINNs模型的预训练阶段，包括基础物理约束学习、
数据拟合和模型初始化策略。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from abc import ABC, abstractmethod

class PretrainingStage(ABC):
    """
    预训练阶段基类
    
    定义预训练阶段的通用接口和基础功能，包括：
    - 模型初始化
    - 基础物理约束学习
    - 数据拟合策略
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any]):
        """
        初始化预训练阶段
        
        Args:
            model: PINNs模型
            optimizer_config: 优化器配置
            physics_config: 物理约束配置
        """
        self.model = model
        self.optimizer_config = optimizer_config
        self.physics_config = physics_config
        self.train_state = None
        
    @abstractmethod
    def initialize_training(self, 
                          sample_input: Dict[str, jnp.ndarray],
                          rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        初始化训练状态
        
        Args:
            sample_input: 样本输入
            rng_key: 随机数生成器密钥
            
        Returns:
            训练状态
        """
        pass
        
    @abstractmethod
    def pretrain_step(self, 
                     state: train_state.TrainState,
                     batch: Dict[str, jnp.ndarray],
                     rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行一步预训练
        
        Args:
            state: 当前训练状态
            batch: 训练批次
            rng_key: 随机数生成器密钥
            
        Returns:
            更新后的训练状态和损失信息
        """
        pass
        
    def run_pretraining(self, 
                       training_data: Dict[str, jnp.ndarray],
                       n_epochs: int,
                       batch_size: int,
                       rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        运行完整的预训练过程
        
        Args:
            training_data: 训练数据
            n_epochs: 训练轮数
            batch_size: 批次大小
            rng_key: 随机数生成器密钥
            
        Returns:
            训练完成的状态
        """
        # 初始化训练状态
        sample_input = {k: v[:1] for k, v in training_data.items() if k in ['x', 'y', 't']}
        self.train_state = self.initialize_training(sample_input, rng_key)
        
        # 创建数据批次
        n_samples = len(training_data['x'])
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            # 打乱数据
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.random.permutation(shuffle_key, n_samples)
            
            for batch_idx in range(n_batches):
                # 创建批次
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = perm[start_idx:end_idx]
                
                batch = {k: v[batch_indices] for k, v in training_data.items()}
                
                # 执行训练步骤
                rng_key, step_key = jax.random.split(rng_key)
                self.train_state, loss_info = self.pretrain_step(
                    self.train_state, batch, step_key
                )
                
                epoch_loss += loss_info.get('total_loss', 0.0)
                
            # 打印进度
            if epoch % 100 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Pretraining Epoch {epoch}, Average Loss: {avg_loss:.6f}")
                
        return self.train_state

class BasicPhysicsPretraining(PretrainingStage):
    """
    基础物理预训练
    
    专注于学习基本的物理约束，如质量守恒、动量守恒等，
    为后续的复杂训练阶段奠定基础。
    """
    
    def initialize_training(self, 
                          sample_input: Dict[str, jnp.ndarray],
                          rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        初始化基础物理预训练的训练状态
        """
        # 初始化模型参数
        params = self.model.init(rng_key, sample_input)
        
        # 创建优化器
        optimizer = optax.adam(
            learning_rate=self.optimizer_config.get('learning_rate', 1e-3)
        )
        
        # 创建训练状态
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def pretrain_step(self, 
                     state: train_state.TrainState,
                     batch: Dict[str, jnp.ndarray],
                     rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行基础物理预训练步骤
        """
        def loss_fn(params):
            # 前向传播
            predictions = state.apply_fn(params, batch)
            
            # 计算物理损失
            physics_loss = self._compute_physics_loss(params, batch, state.apply_fn)
            
            # 计算数据拟合损失（如果有观测数据）
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_data_loss(predictions, batch['observations'])
                
            # 总损失
            total_loss = (
                self.physics_config.get('physics_weight', 1.0) * physics_loss +
                self.physics_config.get('data_weight', 1.0) * data_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'physics_loss': physics_loss,
                'data_loss': data_loss
            }
            
        # 计算梯度
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # 更新参数
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_info
        
    def _compute_physics_loss(self, 
                             params: Dict,
                             batch: Dict[str, jnp.ndarray],
                             apply_fn: Callable) -> float:
        """
        计算基础物理损失
        
        Args:
            params: 模型参数
            batch: 输入批次
            apply_fn: 模型应用函数
            
        Returns:
            物理损失值
        """
        # 计算模型输出的梯度
        def model_fn(inputs):
            return apply_fn(params, inputs)
            
        # 计算一阶和二阶导数
        grad_fn = jax.grad(lambda inputs: jnp.sum(model_fn(inputs)))
        
        # 质量守恒约束
        mass_conservation_loss = self._mass_conservation_constraint(
            params, batch, apply_fn
        )
        
        # 动量守恒约束
        momentum_conservation_loss = self._momentum_conservation_constraint(
            params, batch, apply_fn
        )
        
        # 边界条件约束
        boundary_loss = self._boundary_condition_constraint(
            params, batch, apply_fn
        )
        
        return (
            mass_conservation_loss + 
            momentum_conservation_loss + 
            boundary_loss
        )
        
    def _mass_conservation_constraint(self, 
                                    params: Dict,
                                    batch: Dict[str, jnp.ndarray],
                                    apply_fn: Callable) -> float:
        """
        质量守恒约束
        
        ∂h/∂t + ∇·(h*v) = 0
        """
        def thickness_fn(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            outputs = apply_fn(params, inputs)
            return outputs['thickness']
            
        def velocity_fn(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            outputs = apply_fn(params, inputs)
            return outputs['velocity_x'], outputs['velocity_y']
            
        # 计算时间导数
        dh_dt = jax.vmap(jax.grad(thickness_fn, argnums=2))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 计算空间导数
        h_values = jax.vmap(thickness_fn)(batch['x'], batch['y'], batch['t'])
        vx_values, vy_values = jax.vmap(velocity_fn)(batch['x'], batch['y'], batch['t'])
        
        # 计算通量散度
        flux_x = h_values * vx_values
        flux_y = h_values * vy_values
        
        dflux_x_dx = jax.vmap(jax.grad(lambda x, y, t: 
            thickness_fn(x, y, t) * velocity_fn(x, y, t)[0], argnums=0))(
            batch['x'], batch['y'], batch['t']
        )
        dflux_y_dy = jax.vmap(jax.grad(lambda x, y, t: 
            thickness_fn(x, y, t) * velocity_fn(x, y, t)[1], argnums=1))(
            batch['x'], batch['y'], batch['t']
        )
        
        # 质量守恒残差
        mass_residual = dh_dt + dflux_x_dx + dflux_y_dy
        
        return jnp.mean(mass_residual**2)
        
    def _momentum_conservation_constraint(self, 
                                        params: Dict,
                                        batch: Dict[str, jnp.ndarray],
                                        apply_fn: Callable) -> float:
        """
        动量守恒约束（简化的浅冰近似）
        """
        def stress_fn(x, y, t):
            inputs = {'x': x, 'y': y, 't': t}
            outputs = apply_fn(params, inputs)
            return outputs.get('stress', 0.0)
            
        # 简化的动量平衡
        # τ = ρgh∇s (其中s是表面高程)
        stress_values = jax.vmap(stress_fn)(batch['x'], batch['y'], batch['t'])
        
        # 这里使用简化的约束，实际实现需要更复杂的物理模型
        momentum_residual = stress_values  # 占位符
        
        return jnp.mean(momentum_residual**2)
        
    def _boundary_condition_constraint(self, 
                                     params: Dict,
                                     batch: Dict[str, jnp.ndarray],
                                     apply_fn: Callable) -> float:
        """
        边界条件约束
        """
        # 检查是否有边界点
        if 'boundary_mask' not in batch:
            return 0.0
            
        boundary_mask = batch['boundary_mask']
        
        if jnp.sum(boundary_mask) == 0:
            return 0.0
            
        # 在边界处应用约束
        boundary_inputs = {
            k: v[boundary_mask] for k, v in batch.items() 
            if k in ['x', 'y', 't']
        }
        
        boundary_outputs = apply_fn(params, boundary_inputs)
        
        # 边界处的厚度约束（例如，边界处厚度为0）
        boundary_thickness = boundary_outputs['thickness']
        boundary_loss = jnp.mean(boundary_thickness**2)
        
        return boundary_loss
        
    def _compute_data_loss(self, 
                          predictions: Dict[str, jnp.ndarray],
                          observations: Dict[str, jnp.ndarray]) -> float:
        """
        计算数据拟合损失
        
        Args:
            predictions: 模型预测
            observations: 观测数据
            
        Returns:
            数据损失值
        """
        data_loss = 0.0
        
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                
                # 计算均方误差
                mse = jnp.mean((pred - obs)**2)
                data_loss += mse
                
        return data_loss

class DataDrivenPretraining(PretrainingStage):
    """
    数据驱动预训练
    
    主要关注拟合观测数据，学习数据中的模式和趋势，
    为物理约束的引入做准备。
    """
    
    def initialize_training(self, 
                          sample_input: Dict[str, jnp.ndarray],
                          rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        初始化数据驱动预训练的训练状态
        """
        # 初始化模型参数
        params = self.model.init(rng_key, sample_input)
        
        # 创建优化器（可能使用不同的学习率）
        optimizer = optax.adam(
            learning_rate=self.optimizer_config.get('data_learning_rate', 5e-4)
        )
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def pretrain_step(self, 
                     state: train_state.TrainState,
                     batch: Dict[str, jnp.ndarray],
                     rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行数据驱动预训练步骤
        """
        def loss_fn(params):
            # 前向传播
            predictions = state.apply_fn(params, batch)
            
            # 主要关注数据拟合损失
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_weighted_data_loss(
                    predictions, batch['observations']
                )
                
            # 添加轻微的正则化
            regularization_loss = self._compute_regularization_loss(params)
            
            total_loss = (
                data_loss + 
                self.optimizer_config.get('regularization_weight', 1e-6) * regularization_loss
            )
            
            return total_loss, {
                'total_loss': total_loss,
                'data_loss': data_loss,
                'regularization_loss': regularization_loss
            }
            
        # 计算梯度
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # 更新参数
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_info
        
    def _compute_weighted_data_loss(self, 
                                   predictions: Dict[str, jnp.ndarray],
                                   observations: Dict[str, jnp.ndarray]) -> float:
        """
        计算加权数据损失
        
        Args:
            predictions: 模型预测
            observations: 观测数据
            
        Returns:
            加权数据损失值
        """
        total_loss = 0.0
        
        # 不同变量的权重
        variable_weights = {
            'thickness': 1.0,
            'velocity_x': 0.5,
            'velocity_y': 0.5,
            'surface_elevation': 0.8
        }
        
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                weight = variable_weights.get(key, 1.0)
                
                # 计算加权均方误差
                mse = jnp.mean((pred - obs)**2)
                total_loss += weight * mse
                
        return total_loss
        
    def _compute_regularization_loss(self, params: Dict) -> float:
        """
        计算正则化损失
        
        Args:
            params: 模型参数
            
        Returns:
            正则化损失值
        """
        # L2正则化
        l2_loss = 0.0
        
        def add_l2(x):
            return jnp.sum(x**2)
            
        l2_loss = jax.tree_util.tree_reduce(
            lambda acc, x: acc + add_l2(x),
            params,
            initializer=0.0
        )
        
        return l2_loss

class HybridPretraining(PretrainingStage):
    """
    混合预训练
    
    结合数据驱动和物理约束的预训练策略，
    在学习数据模式的同时逐步引入物理约束。
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer_config: Dict[str, Any],
                 physics_config: Dict[str, Any],
                 hybrid_schedule: Dict[str, Any]):
        super().__init__(model, optimizer_config, physics_config)
        self.hybrid_schedule = hybrid_schedule
        self.current_epoch = 0
        
    def initialize_training(self, 
                          sample_input: Dict[str, jnp.ndarray],
                          rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """
        初始化混合预训练的训练状态
        """
        params = self.model.init(rng_key, sample_input)
        
        # 使用学习率调度
        schedule = optax.exponential_decay(
            init_value=self.optimizer_config.get('learning_rate', 1e-3),
            transition_steps=self.hybrid_schedule.get('decay_steps', 1000),
            decay_rate=self.hybrid_schedule.get('decay_rate', 0.95)
        )
        
        optimizer = optax.adam(learning_rate=schedule)
        
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
    def pretrain_step(self, 
                     state: train_state.TrainState,
                     batch: Dict[str, jnp.ndarray],
                     rng_key: jax.random.PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """
        执行混合预训练步骤
        """
        # 计算当前的权重调度
        data_weight, physics_weight = self._compute_weight_schedule()
        
        def loss_fn(params):
            predictions = state.apply_fn(params, batch)
            
            # 数据损失
            data_loss = 0.0
            if 'observations' in batch:
                data_loss = self._compute_data_loss(predictions, batch['observations'])
                
            # 物理损失
            physics_loss = self._compute_simplified_physics_loss(
                params, batch, state.apply_fn
            )
            
            # 混合损失
            total_loss = data_weight * data_loss + physics_weight * physics_loss
            
            return total_loss, {
                'total_loss': total_loss,
                'data_loss': data_loss,
                'physics_loss': physics_loss,
                'data_weight': data_weight,
                'physics_weight': physics_weight
            }
            
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        self.current_epoch += 1
        
        return new_state, loss_info
        
    def _compute_weight_schedule(self) -> Tuple[float, float]:
        """
        计算数据和物理损失的权重调度
        
        Returns:
            (data_weight, physics_weight)
        """
        # 开始时主要关注数据，逐渐增加物理约束的权重
        transition_epochs = self.hybrid_schedule.get('transition_epochs', 500)
        
        if self.current_epoch < transition_epochs:
            # 线性过渡
            progress = self.current_epoch / transition_epochs
            data_weight = 1.0 - 0.5 * progress
            physics_weight = 0.1 + 0.9 * progress
        else:
            # 稳定阶段
            data_weight = 0.5
            physics_weight = 1.0
            
        return data_weight, physics_weight
        
    def _compute_data_loss(self, 
                          predictions: Dict[str, jnp.ndarray],
                          observations: Dict[str, jnp.ndarray]) -> float:
        """
        计算数据损失
        """
        data_loss = 0.0
        
        for key in observations:
            if key in predictions:
                pred = predictions[key]
                obs = observations[key]
                mse = jnp.mean((pred - obs)**2)
                data_loss += mse
                
        return data_loss
        
    def _compute_simplified_physics_loss(self, 
                                        params: Dict,
                                        batch: Dict[str, jnp.ndarray],
                                        apply_fn: Callable) -> float:
        """
        计算简化的物理损失
        """
        # 简化的物理约束，避免在预训练阶段过于复杂
        
        # 厚度非负约束
        outputs = apply_fn(params, batch)
        thickness = outputs.get('thickness', 0.0)
        thickness_constraint = jnp.mean(jnp.maximum(0.0, -thickness)**2)
        
        # 速度合理性约束
        velocity_x = outputs.get('velocity_x', 0.0)
        velocity_y = outputs.get('velocity_y', 0.0)
        velocity_magnitude = jnp.sqrt(velocity_x**2 + velocity_y**2)
        
        # 限制速度不要过大（例如，不超过1000 m/year）
        max_velocity = 1000.0
        velocity_constraint = jnp.mean(
            jnp.maximum(0.0, velocity_magnitude - max_velocity)**2
        )
        
        return thickness_constraint + velocity_constraint

def create_pretraining_stage(stage_type: str,
                            model: nn.Module,
                            optimizer_config: Dict[str, Any],
                            physics_config: Dict[str, Any],
                            **kwargs) -> PretrainingStage:
    """
    工厂函数：创建预训练阶段
    
    Args:
        stage_type: 预训练类型 ('physics', 'data', 'hybrid')
        model: PINNs模型
        optimizer_config: 优化器配置
        physics_config: 物理约束配置
        **kwargs: 额外参数
        
    Returns:
        预训练阶段实例
    """
    if stage_type == 'physics':
        return BasicPhysicsPretraining(model, optimizer_config, physics_config)
    elif stage_type == 'data':
        return DataDrivenPretraining(model, optimizer_config, physics_config)
    elif stage_type == 'hybrid':
        return HybridPretraining(model, optimizer_config, physics_config, **kwargs)
    else:
        raise ValueError(f"Unknown pretraining stage type: {stage_type}")

if __name__ == "__main__":
    # 测试代码
    print("Pretraining stage module loaded successfully")
    
    # 这里可以添加简单的测试
    optimizer_config = {
        'learning_rate': 1e-3,
        'data_learning_rate': 5e-4,
        'regularization_weight': 1e-6
    }
    
    physics_config = {
        'physics_weight': 1.0,
        'data_weight': 1.0
    }
    
    print("Configuration loaded:")
    print(f"Optimizer config: {optimizer_config}")
    print(f"Physics config: {physics_config}")