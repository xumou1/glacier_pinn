#!/usr/bin/env python3
"""
Base Physics-Informed Neural Networks (PINNs)

这是PINNs的基础实现，提供了所有PINNs变体的通用接口和基础功能。
包含标准的前向传播、物理损失计算、边界条件处理等核心功能。

主要特点：
- 标准的深度神经网络架构
- 物理定律嵌入
- 自动微分计算导数
- 边界条件处理
- 损失函数组合

参考文献：
- Raissi et al., Physics-informed neural networks, Journal of Computational Physics, 2019
- Karniadakis et al., Physics-informed machine learning, Nature Reviews Physics, 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod


class AdaptiveActivation(nn.Module):
    """
    自适应激活函数
    
    可学习的激活函数，能够根据数据自动调整激活特性
    """
    
    def __init__(self, activation_type: str = 'tanh', learnable: bool = True):
        super().__init__()
        self.activation_type = activation_type
        self.learnable = learnable
        
        if learnable:
            # 可学习的缩放和偏移参数
            self.scale = nn.Parameter(torch.ones(1))
            self.shift = nn.Parameter(torch.zeros(1))
            self.slope = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            output: 激活后的张量
        """
        if self.activation_type == 'tanh':
            activated = torch.tanh(x)
        elif self.activation_type == 'relu':
            activated = F.relu(x)
        elif self.activation_type == 'gelu':
            activated = F.gelu(x)
        elif self.activation_type == 'swish':
            activated = x * torch.sigmoid(x)
        elif self.activation_type == 'sin':
            activated = torch.sin(x)
        else:
            activated = torch.tanh(x)
        
        if self.learnable:
            # 应用可学习参数
            activated = self.scale * activated + self.shift
            # 添加线性分量
            activated = activated + self.slope * x
        
        return activated


class ResidualBlock(nn.Module):
    """
    残差块
    
    带有跳跃连接的残差块，有助于训练深层网络
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 activation_type: str = 'tanh',
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.activation1 = AdaptiveActivation(activation_type, learnable=True)
        self.activation2 = AdaptiveActivation(activation_type, learnable=True)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            output: 输出张量
        """
        residual = x
        
        # 第一层
        out = self.layer_norm1(x)
        out = self.linear1(out)
        out = self.activation1(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # 第二层
        out = self.layer_norm2(out)
        out = self.linear2(out)
        out = self.activation2(out)
        
        # 残差连接
        out = out + residual
        
        return out


class PhysicsEmbedding(nn.Module):
    """
    物理嵌入层
    
    将物理先验知识嵌入到神经网络中
    """
    
    def __init__(self, 
                 input_dim: int, 
                 embedding_dim: int,
                 physics_type: str = 'glacier'):
        super().__init__()
        self.physics_type = physics_type
        self.embedding_dim = embedding_dim
        
        # 物理特征提取
        self.physics_encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            AdaptiveActivation('sin', learnable=True),
            nn.Linear(embedding_dim, embedding_dim),
            AdaptiveActivation('tanh', learnable=True)
        )
        
        # 物理约束权重
        self.physics_weights = nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (x, y, t)
            
        Returns:
            embedded: 物理嵌入特征
        """
        # 基础物理嵌入
        embedded = self.physics_encoder(x)
        
        # 应用物理权重
        embedded = embedded * self.physics_weights
        
        # 冰川特定的物理先验
        if self.physics_type == 'glacier':
            # 添加高程相关的特征
            if x.shape[1] >= 2:  # 确保有x, y坐标
                elevation_feature = torch.sin(x[:, 0:1] * 0.1) * torch.cos(x[:, 1:2] * 0.1)
                embedded = torch.cat([embedded, elevation_feature], dim=1)
        
        return embedded


class BasePINN(nn.Module):
    """
    基础Physics-Informed Neural Networks
    
    所有PINNs变体的基类，提供标准的PINN功能
    """
    
    def __init__(self, 
                 input_dim: int = 3,  # (x, y, t)
                 output_dim: int = 4,  # (h, u, v, T)
                 hidden_dims: List[int] = [64, 128, 128, 64],
                 activation_type: str = 'tanh',
                 use_residual: bool = True,
                 use_physics_embedding: bool = True,
                 dropout_rate: float = 0.0,
                 layer_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_type = activation_type
        self.use_residual = use_residual
        self.use_physics_embedding = use_physics_embedding
        
        # 物理嵌入
        if use_physics_embedding:
            self.physics_embedding = PhysicsEmbedding(input_dim, hidden_dims[0] // 2)
            effective_input_dim = input_dim + hidden_dims[0] // 2 + 1  # +1 for elevation feature
        else:
            effective_input_dim = input_dim
        
        # 输入层
        self.input_layer = nn.Linear(effective_input_dim, hidden_dims[0])
        self.input_activation = AdaptiveActivation(activation_type, learnable=True)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                # 使用残差块
                self.hidden_layers.append(
                    ResidualBlock(hidden_dims[i], activation_type, dropout_rate)
                )
            else:
                # 标准线性层
                layer = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]) if layer_norm else nn.Identity(),
                    AdaptiveActivation(activation_type, learnable=True),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
                self.hidden_layers.append(layer)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # 输出缩放（用于数值稳定性）
        self.output_scale = nn.Parameter(torch.ones(output_dim))
        self.output_bias = nn.Parameter(torch.zeros(output_dim))
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入张量 (batch_size, input_dim)
            
        Returns:
            outputs: 输出张量 (batch_size, output_dim)
        """
        x = inputs
        
        # 物理嵌入
        if self.use_physics_embedding:
            physics_features = self.physics_embedding(x)
            x = torch.cat([x, physics_features], dim=1)
        
        # 输入层
        x = self.input_layer(x)
        x = self.input_activation(x)
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 输出层
        outputs = self.output_layer(x)
        
        # 输出缩放
        outputs = outputs * self.output_scale + self.output_bias
        
        return outputs
    
    def compute_derivatives(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算输出相对于输入的导数
        
        Args:
            inputs: 输入张量，需要requires_grad=True
            
        Returns:
            derivatives: 导数字典
        """
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        
        outputs = self.forward(inputs)
        derivatives = {}
        
        # 变量名映射
        var_names = ['h', 'u', 'v', 'T']  # 冰川厚度、x方向速度、y方向速度、温度
        coord_names = ['x', 'y', 't']     # 空间坐标x, y和时间t
        
        # 计算一阶导数
        for i in range(min(self.output_dim, len(var_names))):
            for j in range(min(self.input_dim, len(coord_names))):
                grad = torch.autograd.grad(
                    outputs[:, i].sum(), inputs,
                    create_graph=True, retain_graph=True
                )[0][:, j:j+1]
                
                derivatives[f'd{var_names[i]}_d{coord_names[j]}'] = grad
        
        # 计算二阶导数（拉普拉斯算子等）
        for i in range(min(self.output_dim, len(var_names))):
            for j in range(min(self.input_dim-1, len(coord_names)-1)):  # 只对空间坐标
                # 计算二阶导数
                first_grad = derivatives[f'd{var_names[i]}_d{coord_names[j]}']
                second_grad = torch.autograd.grad(
                    first_grad.sum(), inputs,
                    create_graph=True, retain_graph=True
                )[0][:, j:j+1]
                
                derivatives[f'd2{var_names[i]}_d{coord_names[j]}2'] = second_grad
        
        return derivatives
    
    def physics_loss(self, 
                    inputs: torch.Tensor, 
                    physics_laws: List[Callable],
                    law_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        计算物理损失
        
        Args:
            inputs: 输入张量
            physics_laws: 物理定律函数列表
            law_weights: 物理定律权重
            
        Returns:
            physics_loss: 物理损失
        """
        if not physics_laws:
            return torch.tensor(0.0, device=inputs.device)
        
        # 确保输入需要梯度
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        
        # 前向传播
        outputs = self.forward(inputs)
        
        # 计算导数
        derivatives = self.compute_derivatives(inputs)
        
        # 计算物理定律残差
        total_loss = 0.0
        
        if law_weights is None:
            law_weights = [1.0] * len(physics_laws)
        
        for i, law in enumerate(physics_laws):
            try:
                residual = law(inputs, outputs, derivatives)
                loss = torch.mean(residual ** 2)
                total_loss += law_weights[i] * loss
            except Exception as e:
                print(f"Warning: Physics law {i} failed: {e}")
                continue
        
        return total_loss
    
    def boundary_loss(self, 
                     boundary_data: Dict[str, torch.Tensor],
                     boundary_conditions: List[Callable]) -> torch.Tensor:
        """
        计算边界损失
        
        Args:
            boundary_data: 边界数据字典
            boundary_conditions: 边界条件函数列表
            
        Returns:
            boundary_loss: 边界损失
        """
        if not boundary_conditions or not boundary_data:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        
        for bc in boundary_conditions:
            try:
                loss = bc(self, boundary_data)
                total_loss += loss
            except Exception as e:
                print(f"Warning: Boundary condition failed: {e}")
                continue
        
        return total_loss
    
    def data_loss(self, 
                  inputs: torch.Tensor, 
                  targets: torch.Tensor,
                  loss_type: str = 'mse') -> torch.Tensor:
        """
        计算数据损失
        
        Args:
            inputs: 输入张量
            targets: 目标张量
            loss_type: 损失类型
            
        Returns:
            data_loss: 数据损失
        """
        outputs = self.forward(inputs)
        
        if loss_type == 'mse':
            return F.mse_loss(outputs, targets)
        elif loss_type == 'mae':
            return F.l1_loss(outputs, targets)
        elif loss_type == 'huber':
            return F.smooth_l1_loss(outputs, targets)
        else:
            return F.mse_loss(outputs, targets)
    
    def total_loss(self, 
                   data_inputs: torch.Tensor,
                   data_targets: torch.Tensor,
                   physics_inputs: torch.Tensor,
                   physics_laws: List[Callable],
                   boundary_data: Optional[Dict[str, torch.Tensor]] = None,
                   boundary_conditions: Optional[List[Callable]] = None,
                   loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            data_inputs: 数据输入
            data_targets: 数据目标
            physics_inputs: 物理输入
            physics_laws: 物理定律
            boundary_data: 边界数据
            boundary_conditions: 边界条件
            loss_weights: 损失权重
            
        Returns:
            losses: 损失字典
        """
        if loss_weights is None:
            loss_weights = {'data': 1.0, 'physics': 1.0, 'boundary': 1.0}
        
        # 数据损失
        data_loss = self.data_loss(data_inputs, data_targets)
        
        # 物理损失
        physics_loss = self.physics_loss(physics_inputs, physics_laws)
        
        # 边界损失
        if boundary_data is not None and boundary_conditions is not None:
            boundary_loss = self.boundary_loss(boundary_data, boundary_conditions)
        else:
            boundary_loss = torch.tensor(0.0, device=data_loss.device)
        
        # 总损失
        total = (loss_weights.get('data', 1.0) * data_loss + 
                loss_weights.get('physics', 1.0) * physics_loss + 
                loss_weights.get('boundary', 1.0) * boundary_loss)
        
        return {
            'total': total,
            'data': data_loss,
            'physics': physics_loss,
            'boundary': boundary_loss
        }
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        预测
        
        Args:
            inputs: 输入张量
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)
    
    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """
        获取模型信息
        
        Returns:
            info: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation_type': self.activation_type,
            'use_residual': self.use_residual,
            'use_physics_embedding': self.use_physics_embedding
        }


def create_glacier_pinn(hidden_dims: List[int] = [64, 128, 128, 64],
                       activation_type: str = 'tanh',
                       use_advanced_features: bool = True) -> BasePINN:
    """
    创建冰川专用的PINN模型
    
    Args:
        hidden_dims: 隐藏层维度
        activation_type: 激活函数类型
        use_advanced_features: 是否使用高级特性
        
    Returns:
        model: 配置好的PINN模型
    """
    model = BasePINN(
        input_dim=3,  # (x, y, t)
        output_dim=4,  # (h, u, v, T)
        hidden_dims=hidden_dims,
        activation_type=activation_type,
        use_residual=use_advanced_features,
        use_physics_embedding=use_advanced_features,
        dropout_rate=0.1 if use_advanced_features else 0.0,
        layer_norm=use_advanced_features
    )
    
    return model


if __name__ == "__main__":
    # 示例使用
    print("BasePINN模型初始化完成")
    
    # 创建模型
    model = create_glacier_pinn()
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试前向传播
    test_input = torch.randn(100, 3, requires_grad=True)
    test_output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")
    
    # 测试导数计算
    derivatives = model.compute_derivatives(test_input)
    print(f"计算的导数: {list(derivatives.keys())}")
    
    # 测试物理损失（示例）
    def mass_conservation(inputs, outputs, derivatives):
        # 简化的质量守恒定律
        dh_dt = derivatives.get('dh_dt', torch.zeros_like(outputs[:, 0:1]))
        du_dx = derivatives.get('du_dx', torch.zeros_like(outputs[:, 1:2]))
        dv_dy = derivatives.get('dv_dy', torch.zeros_like(outputs[:, 2:3]))
        
        # ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
        h = outputs[:, 0:1]
        u = outputs[:, 1:2]
        v = outputs[:, 2:3]
        
        residual = dh_dt + h * du_dx + h * dv_dy
        return residual
    
    physics_laws = [mass_conservation]
    physics_loss = model.physics_loss(test_input, physics_laws)
    print(f"物理损失: {physics_loss.item():.6f}")
    
    print("BasePINN测试完成")