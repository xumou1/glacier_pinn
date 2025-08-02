#!/usr/bin/env python3
"""
PIKAN模型实现

Physics-Informed Kolmogorov-Arnold Networks (PIKAN)是一种结合了
Kolmogorov-Arnold表示定理和物理信息神经网络的先进架构。

主要特点：
- 基于Kolmogorov-Arnold表示的函数分解
- 物理定律的显式嵌入
- 高效的多变量函数逼近
- 强物理约束保证

参考文献：
- Kolmogorov-Arnold Networks (KAN)
- Physics-Informed Neural Networks (PINNs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network层
    
    实现KAN的基本构建块，使用可学习的单变量函数
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 创建网格点
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))
        
        # 可学习的样条系数
        self.spline_coeffs = nn.Parameter(
            torch.randn(input_dim, output_dim, grid_size + spline_order)
        )
        
        # 基函数权重
        self.base_weights = nn.Parameter(torch.randn(input_dim, output_dim))
        
        # 激活函数
        self.activation = nn.SiLU()
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算B样条基函数
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            basis: B样条基函数值 [batch_size, input_dim, grid_size + spline_order]
        """
        batch_size = x.shape[0]
        
        # 将输入映射到网格范围
        x_normalized = torch.clamp(x, -1, 1)
        
        # 计算B样条基函数（简化实现）
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0).expand(batch_size, self.input_dim, -1)
        x_expanded = x_normalized.unsqueeze(-1).expand(-1, -1, self.grid_size)
        
        # 使用高斯基函数近似B样条
        sigma = 2.0 / self.grid_size
        basis = torch.exp(-0.5 * ((x_expanded - grid_expanded) / sigma) ** 2)
        
        # 扩展到所需维度
        padding = torch.zeros(batch_size, self.input_dim, self.spline_order)
        basis = torch.cat([basis, padding], dim=-1)
        
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            output: 输出张量 [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # 计算B样条基函数
        basis = self.b_splines(x)  # [batch_size, input_dim, grid_size + spline_order]
        
        # 计算样条函数值
        spline_values = torch.einsum('bij,ijk->bik', basis, self.spline_coeffs)
        
        # 基函数贡献
        base_values = torch.einsum('bi,ij->bj', self.activation(x), self.base_weights)
        
        # 组合样条和基函数
        output = torch.sum(spline_values, dim=1) + base_values
        
        return output


class PhysicsEmbedding(nn.Module):
    """
    物理嵌入层
    
    将物理定律直接嵌入到网络架构中
    """
    
    def __init__(self, 
                 physics_laws: List[Callable],
                 embedding_dim: int = 64):
        super().__init__()
        self.physics_laws = physics_laws
        self.embedding_dim = embedding_dim
        
        # 物理特征提取器
        self.physics_encoder = nn.Sequential(
            nn.Linear(len(physics_laws), embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, 
                inputs: torch.Tensor, 
                outputs: torch.Tensor,
                derivatives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算物理嵌入
        
        Args:
            inputs: 输入坐标
            outputs: 网络输出
            derivatives: 导数信息
            
        Returns:
            physics_embedding: 物理嵌入向量
        """
        # 计算物理定律残差
        physics_residuals = []
        for law in self.physics_laws:
            residual = law(inputs, outputs, derivatives)
            physics_residuals.append(torch.mean(residual, dim=-1, keepdim=True))
        
        # 堆叠残差
        residual_stack = torch.cat(physics_residuals, dim=-1)
        
        # 编码物理信息
        physics_embedding = self.physics_encoder(residual_stack)
        
        return physics_embedding


class PIKANBlock(nn.Module):
    """
    PIKAN基本块
    
    结合KAN层和物理嵌入的基本构建块
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 physics_laws: List[Callable],
                 grid_size: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # KAN层
        self.kan_layer1 = KANLayer(input_dim, hidden_dim, grid_size)
        self.kan_layer2 = KANLayer(hidden_dim, output_dim, grid_size)
        
        # 物理嵌入
        self.physics_embedding = PhysicsEmbedding(physics_laws, hidden_dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, 
                inputs: torch.Tensor,
                derivatives: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入张量
            derivatives: 导数信息（用于物理嵌入）
            
        Returns:
            output: 输出张量
        """
        # KAN前向传播
        kan_hidden = self.kan_layer1(inputs)
        kan_output = self.kan_layer2(kan_hidden)
        
        # 如果有导数信息，计算物理嵌入
        if derivatives is not None:
            physics_emb = self.physics_embedding(inputs, kan_output, derivatives)
            
            # 融合KAN特征和物理嵌入
            combined_features = torch.cat([kan_hidden, physics_emb], dim=-1)
            fused_features = self.fusion_layer(combined_features)
            
            # 更新KAN隐藏状态
            kan_hidden = kan_hidden + self.residual_weight * fused_features
            
            # 重新计算输出
            kan_output = self.kan_layer2(kan_hidden)
        
        return kan_output


class PIKAN(nn.Module):
    """
    Physics-Informed Kolmogorov-Arnold Networks
    
    完整的PIKAN模型实现
    """
    
    def __init__(self, 
                 input_dim: int = 3,  # (x, y, t)
                 output_dim: int = 4,  # (h, u, v, T)
                 hidden_dims: List[int] = [64, 128, 64],
                 physics_laws: Optional[List[Callable]] = None,
                 grid_size: int = 5,
                 num_blocks: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.physics_laws = physics_laws or []
        
        # 输入标准化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 构建PIKAN块
        self.blocks = nn.ModuleList()
        
        # 第一个块
        self.blocks.append(
            PIKANBlock(input_dim, hidden_dims[0], hidden_dims[0], 
                      self.physics_laws, grid_size)
        )
        
        # 中间块
        for i in range(1, num_blocks - 1):
            self.blocks.append(
                PIKANBlock(hidden_dims[i-1], hidden_dims[i], hidden_dims[i], 
                          self.physics_laws, grid_size)
            )
        
        # 输出块
        self.blocks.append(
            PIKANBlock(hidden_dims[-2], hidden_dims[-1], output_dim, 
                      self.physics_laws, grid_size)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # 物理约束权重
        self.physics_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入张量 [batch_size, input_dim]
            
        Returns:
            outputs: 输出张量 [batch_size, output_dim]
        """
        # 输入标准化
        x = self.input_norm(inputs)
        
        # 通过PIKAN块
        for block in self.blocks:
            x = block(x)
        
        # 输出层
        outputs = self.output_layer(x)
        
        return outputs
    
    def compute_derivatives(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算导数
        
        Args:
            inputs: 输入张量
            
        Returns:
            derivatives: 导数字典
        """
        inputs.requires_grad_(True)
        outputs = self.forward(inputs)
        
        derivatives = {}
        
        # 计算一阶导数
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                grad = torch.autograd.grad(
                    outputs[:, i].sum(), inputs,
                    create_graph=True, retain_graph=True
                )[0][:, j:j+1]
                
                var_names = ['h', 'u', 'v', 'T']
                coord_names = ['x', 'y', 't']
                
                if i < len(var_names) and j < len(coord_names):
                    derivatives[f'd{var_names[i]}_d{coord_names[j]}'] = grad
        
        return derivatives
    
    def physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        计算物理损失
        
        Args:
            inputs: 输入张量
            outputs: 输出张量
            
        Returns:
            physics_loss: 物理损失
        """
        if not self.physics_laws:
            return torch.tensor(0.0, device=inputs.device)
        
        # 计算导数
        derivatives = self.compute_derivatives(inputs)
        
        # 计算物理定律残差
        total_loss = 0.0
        for law in self.physics_laws:
            residual = law(inputs, outputs, derivatives)
            total_loss += torch.mean(residual ** 2)
        
        return self.physics_weight * total_loss


def create_glacier_pikan(physics_laws: List[Callable]) -> PIKAN:
    """
    创建冰川专用的PIKAN模型
    
    Args:
        physics_laws: 物理定律列表
        
    Returns:
        model: 配置好的PIKAN模型
    """
    model = PIKAN(
        input_dim=3,  # (x, y, t)
        output_dim=4,  # (h, u, v, T)
        hidden_dims=[64, 128, 128, 64],
        physics_laws=physics_laws,
        grid_size=7,
        num_blocks=4
    )
    
    return model


class PIKANTrainer:
    """
    PIKAN训练器
    
    专门用于训练PIKAN模型的训练器
    """
    
    def __init__(self, 
                 model: PIKAN, 
                 data_weight: float = 1.0,
                 physics_weight: float = 1.0,
                 boundary_weight: float = 1.0):
        self.model = model
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.boundary_weight = boundary_weight
    
    def compute_loss(self, 
                    inputs: torch.Tensor, 
                    targets: torch.Tensor,
                    boundary_inputs: Optional[torch.Tensor] = None,
                    boundary_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            inputs: 训练输入
            targets: 训练目标
            boundary_inputs: 边界输入
            boundary_targets: 边界目标
            
        Returns:
            losses: 损失字典
        """
        # 数据损失
        outputs = self.model(inputs)
        data_loss = F.mse_loss(outputs, targets)
        
        # 物理损失
        physics_loss = self.model.physics_loss(inputs, outputs)
        
        # 边界损失
        boundary_loss = torch.tensor(0.0, device=inputs.device)
        if boundary_inputs is not None and boundary_targets is not None:
            boundary_outputs = self.model(boundary_inputs)
            boundary_loss = F.mse_loss(boundary_outputs, boundary_targets)
        
        # 总损失
        total_loss = (
            self.data_weight * data_loss +
            self.physics_weight * physics_loss +
            self.boundary_weight * boundary_loss
        )
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'boundary_loss': boundary_loss
        }


if __name__ == "__main__":
    # 示例使用
    print("PIKAN模型初始化完成")
    
    # 创建示例物理定律
    def mass_conservation(inputs, outputs, derivatives):
        # 简化的质量守恒
        return torch.zeros_like(outputs[:, 0:1])
    
    physics_laws = [mass_conservation]
    
    # 创建模型
    model = create_glacier_pikan(physics_laws)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    test_input = torch.randn(10, 3)
    test_output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")