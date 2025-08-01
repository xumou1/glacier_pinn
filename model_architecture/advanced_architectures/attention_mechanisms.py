#!/usr/bin/env python3
"""
注意力机制实现

为PINNs提供各种注意力机制，包括：
- 自注意力机制
- 空间注意力
- 时间注意力
- 物理感知注意力
- 多头注意力

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    实现标准的多头自注意力
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 bias: bool = True, batch_first: bool = True):
        """
        初始化多头注意力
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比率
            bias: 是否使用偏置
            batch_first: 是否batch维度在前
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        # 线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attn_mask: Optional[torch.Tensor] = None,
               key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, embed_dim]
            key: 键张量 [batch_size, seq_len, embed_dim]
            value: 值张量 [batch_size, seq_len, embed_dim]
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            Tuple: (注意力输出, 注意力权重)
        """
        if not self.batch_first:
            # 转换为batch_first格式
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, embed_dim = query.shape
        
        # 线性变换
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)    # [batch_size, seq_len, embed_dim]
        V = self.v_linear(value)  # [batch_size, seq_len, embed_dim]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状为 [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用掩码
        if attn_mask is not None:
            attn_scores += attn_mask
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # 输出线性变换
        attn_output = self.out_linear(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_weights.mean(dim=1)  # 平均多头权重

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    
    专门处理空间维度的注意力
    """
    
    def __init__(self, feature_dim: int, spatial_dims: int = 2,
                 attention_dim: int = 64, dropout: float = 0.1):
        """
        初始化空间注意力
        
        Args:
            feature_dim: 特征维度
            spatial_dims: 空间维度数
            attention_dim: 注意力维度
            dropout: Dropout比率
        """
        super(SpatialAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.spatial_dims = spatial_dims
        self.attention_dim = attention_dim
        
        # 空间编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dims, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # 特征编码
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # 注意力计算
        self.attention_layer = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 特征张量 [batch_size, num_points, feature_dim]
            coordinates: 空间坐标 [batch_size, num_points, spatial_dims]
            
        Returns:
            Tuple: (注意力加权特征, 注意力权重)
        """
        batch_size, num_points, _ = features.shape
        
        # 编码空间信息
        spatial_encoded = self.spatial_encoder(coordinates)
        
        # 编码特征信息
        feature_encoded = self.feature_encoder(features)
        
        # 组合空间和特征信息
        combined = torch.cat([spatial_encoded, feature_encoded], dim=-1)
        
        # 计算注意力权重
        attention_scores = self.attention_layer(combined)  # [batch_size, num_points, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 应用注意力权重
        attended_features = features * attention_weights
        
        return attended_features, attention_weights.squeeze(-1)

class TemporalAttention(nn.Module):
    """
    时间注意力机制
    
    专门处理时间序列的注意力
    """
    
    def __init__(self, feature_dim: int, temporal_dim: int = 1,
                 attention_dim: int = 64, num_heads: int = 8):
        """
        初始化时间注意力
        
        Args:
            feature_dim: 特征维度
            temporal_dim: 时间维度
            attention_dim: 注意力维度
            num_heads: 注意力头数
        """
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.temporal_dim = temporal_dim
        self.attention_dim = attention_dim
        
        # 时间编码
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # 多头注意力
        self.multihead_attn = MultiHeadAttention(
            embed_dim=feature_dim + attention_dim,
            num_heads=num_heads
        )
        
        # 输出投影
        self.output_projection = nn.Linear(feature_dim + attention_dim, feature_dim)
        
    def forward(self, features: torch.Tensor, time_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 特征张量 [batch_size, seq_len, feature_dim]
            time_coords: 时间坐标 [batch_size, seq_len, temporal_dim]
            
        Returns:
            Tuple: (时间注意力特征, 注意力权重)
        """
        # 编码时间信息
        temporal_encoded = self.temporal_encoder(time_coords)
        
        # 组合特征和时间信息
        combined_features = torch.cat([features, temporal_encoded], dim=-1)
        
        # 应用多头注意力
        attended_features, attention_weights = self.multihead_attn(
            combined_features, combined_features, combined_features
        )
        
        # 输出投影
        output_features = self.output_projection(attended_features)
        
        return output_features, attention_weights

class PhysicsAwareAttention(nn.Module):
    """
    物理感知注意力机制
    
    基于物理约束的注意力机制
    """
    
    def __init__(self, feature_dim: int, physics_dim: int,
                 attention_dim: int = 64, physics_weight: float = 1.0):
        """
        初始化物理感知注意力
        
        Args:
            feature_dim: 特征维度
            physics_dim: 物理量维度
            attention_dim: 注意力维度
            physics_weight: 物理权重
        """
        super(PhysicsAwareAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.physics_dim = physics_dim
        self.attention_dim = attention_dim
        self.physics_weight = physics_weight
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # 物理编码器
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
        
        # 物理约束注意力
        self.physics_attention = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # 特征注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, features: torch.Tensor, physics_quantities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 特征张量 [batch_size, num_points, feature_dim]
            physics_quantities: 物理量 [batch_size, num_points, physics_dim]
            
        Returns:
            Tuple: (物理感知特征, 注意力权重字典)
        """
        # 编码特征和物理量
        feature_encoded = self.feature_encoder(features)
        physics_encoded = self.physics_encoder(physics_quantities)
        
        # 计算物理约束注意力
        physics_combined = torch.cat([feature_encoded, physics_encoded], dim=-1)
        physics_attention_scores = self.physics_attention(physics_combined)
        physics_attention_weights = F.softmax(physics_attention_scores, dim=1)
        
        # 计算特征注意力
        feature_attention_scores = self.feature_attention(feature_encoded)
        feature_attention_weights = F.softmax(feature_attention_scores, dim=1)
        
        # 组合注意力权重
        combined_weights = (1 - self.physics_weight) * feature_attention_weights + \
                          self.physics_weight * physics_attention_weights
        
        # 应用注意力
        attended_features = features * combined_weights
        
        attention_weights = {
            'physics': physics_attention_weights.squeeze(-1),
            'feature': feature_attention_weights.squeeze(-1),
            'combined': combined_weights.squeeze(-1)
        }
        
        return attended_features, attention_weights

class CrossAttention(nn.Module):
    """
    交叉注意力机制
    
    处理不同模态之间的注意力
    """
    
    def __init__(self, query_dim: int, key_value_dim: int, 
                 attention_dim: int = 64, num_heads: int = 8):
        """
        初始化交叉注意力
        
        Args:
            query_dim: 查询维度
            key_value_dim: 键值维度
            attention_dim: 注意力维度
            num_heads: 注意力头数
        """
        super(CrossAttention, self).__init__()
        
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "注意力维度必须能被头数整除"
        
        # 投影层
        self.query_projection = nn.Linear(query_dim, attention_dim)
        self.key_projection = nn.Linear(key_value_dim, attention_dim)
        self.value_projection = nn.Linear(key_value_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, query_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, query_len, query_dim]
            key_value: 键值张量 [batch_size, kv_len, key_value_dim]
            
        Returns:
            Tuple: (交叉注意力输出, 注意力权重)
        """
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        
        # 投影
        Q = self.query_projection(query)
        K = self.key_projection(key_value)
        V = self.value_projection(key_value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.attention_dim
        )
        
        # 输出投影
        output = self.output_projection(attn_output)
        
        return output, attn_weights.mean(dim=1)

class AdaptiveAttention(nn.Module):
    """
    自适应注意力机制
    
    根据输入动态调整注意力模式
    """
    
    def __init__(self, feature_dim: int, num_attention_types: int = 3,
                 attention_dim: int = 64):
        """
        初始化自适应注意力
        
        Args:
            feature_dim: 特征维度
            num_attention_types: 注意力类型数量
            attention_dim: 注意力维度
        """
        super(AdaptiveAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_attention_types = num_attention_types
        self.attention_dim = attention_dim
        
        # 注意力类型选择器
        self.attention_selector = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, num_attention_types),
            nn.Softmax(dim=-1)
        )
        
        # 多种注意力机制
        self.attention_mechanisms = nn.ModuleList([
            MultiHeadAttention(feature_dim, num_heads=8),
            SpatialAttention(feature_dim),
            TemporalAttention(feature_dim)
        ])
        
    def forward(self, features: torch.Tensor, coordinates: Optional[torch.Tensor] = None,
               time_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 特征张量
            coordinates: 空间坐标（可选）
            time_coords: 时间坐标（可选）
            
        Returns:
            Tuple: (自适应注意力输出, 注意力信息字典)
        """
        # 选择注意力类型
        attention_weights = self.attention_selector(features.mean(dim=1))  # [batch_size, num_attention_types]
        
        # 应用不同的注意力机制
        attention_outputs = []
        attention_info = {}
        
        # 多头注意力
        if self.num_attention_types > 0:
            mha_output, mha_weights = self.attention_mechanisms[0](features, features, features)
            attention_outputs.append(mha_output)
            attention_info['multihead_weights'] = mha_weights
        
        # 空间注意力
        if self.num_attention_types > 1 and coordinates is not None:
            spatial_output, spatial_weights = self.attention_mechanisms[1](features, coordinates)
            attention_outputs.append(spatial_output)
            attention_info['spatial_weights'] = spatial_weights
        
        # 时间注意力
        if self.num_attention_types > 2 and time_coords is not None:
            temporal_output, temporal_weights = self.attention_mechanisms[2](features, time_coords)
            attention_outputs.append(temporal_output)
            attention_info['temporal_weights'] = temporal_weights
        
        # 加权组合
        final_output = torch.zeros_like(features)
        for i, output in enumerate(attention_outputs):
            weight = attention_weights[:, i:i+1].unsqueeze(1)  # [batch_size, 1, 1]
            final_output += weight * output
        
        attention_info['type_weights'] = attention_weights
        
        return final_output, attention_info

class GlacierAttention(nn.Module):
    """
    冰川专用注意力机制
    
    针对冰川物理过程的特化注意力
    """
    
    def __init__(self, feature_dim: int, physics_dim: int = 3,
                 spatial_dim: int = 2, temporal_dim: int = 1,
                 attention_dim: int = 64):
        """
        初始化冰川注意力
        
        Args:
            feature_dim: 特征维度
            physics_dim: 物理量维度（厚度、速度等）
            spatial_dim: 空间维度
            temporal_dim: 时间维度
            attention_dim: 注意力维度
        """
        super(GlacierAttention, self).__init__()
        
        # 物理感知注意力
        self.physics_attention = PhysicsAwareAttention(
            feature_dim, physics_dim, attention_dim
        )
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(
            feature_dim, spatial_dim, attention_dim
        )
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(
            feature_dim, temporal_dim, attention_dim
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, features: torch.Tensor, physics_quantities: torch.Tensor,
               spatial_coords: torch.Tensor, temporal_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 特征张量
            physics_quantities: 物理量
            spatial_coords: 空间坐标
            temporal_coords: 时间坐标
            
        Returns:
            Tuple: (冰川注意力输出, 注意力信息)
        """
        # 物理注意力
        physics_features, physics_weights = self.physics_attention(features, physics_quantities)
        
        # 空间注意力
        spatial_features, spatial_weights = self.spatial_attention(features, spatial_coords)
        
        # 时间注意力
        temporal_features, temporal_weights = self.temporal_attention(features, temporal_coords)
        
        # 融合多种注意力
        combined_features = torch.cat([physics_features, spatial_features, temporal_features], dim=-1)
        fused_output = self.fusion_network(combined_features)
        
        attention_info = {
            'physics': physics_weights,
            'spatial': spatial_weights,
            'temporal': temporal_weights
        }
        
        return fused_output, attention_info

if __name__ == "__main__":
    # 测试注意力机制
    batch_size, seq_len, feature_dim = 32, 100, 64
    
    # 多头注意力测试
    mha = MultiHeadAttention(feature_dim, num_heads=8)
    features = torch.randn(batch_size, seq_len, feature_dim)
    output, weights = mha(features, features, features)
    print(f"多头注意力输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 空间注意力测试
    spatial_attn = SpatialAttention(feature_dim, spatial_dims=2)
    coords = torch.randn(batch_size, seq_len, 2)
    spatial_output, spatial_weights = spatial_attn(features, coords)
    print(f"空间注意力输出形状: {spatial_output.shape}")
    
    # 冰川注意力测试
    glacier_attn = GlacierAttention(feature_dim)
    physics_quantities = torch.randn(batch_size, seq_len, 3)
    spatial_coords = torch.randn(batch_size, seq_len, 2)
    temporal_coords = torch.randn(batch_size, seq_len, 1)
    
    glacier_output, glacier_info = glacier_attn(
        features, physics_quantities, spatial_coords, temporal_coords
    )
    print(f"冰川注意力输出形状: {glacier_output.shape}")
    print(f"注意力信息键: {list(glacier_info.keys())}")