#!/usr/bin/env python3
"""
变分推断实现

实现各种变分推断方法用于不确定性量化，包括：
- 变分自编码器
- 平均场变分推断
- 标准化流
- 变分贝叶斯神经网络

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable, Union
from abc import ABC, abstractmethod
import math

class VariationalInferenceBase(nn.Module, ABC):
    """
    变分推断基类
    
    定义变分推断的抽象接口
    """
    
    def __init__(self):
        super(VariationalInferenceBase, self).__init__()
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码器：将输入映射到潜在空间的参数
        
        Args:
            x: 输入张量
            
        Returns:
            Tuple: (均值, 方差)
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码器：将潜在变量映射回数据空间
        
        Args:
            z: 潜在变量
            
        Returns:
            Tensor: 重构输出
        """
        pass
    
    @abstractmethod
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化采样
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            Tensor: 采样结果
        """
        pass
    
    @abstractmethod
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            Tensor: KL散度
        """
        pass

class VariationalAutoencoder(VariationalInferenceBase):
    """
    变分自编码器
    
    实现标准的VAE架构
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 latent_dim: int, activation: str = 'relu'):
        """
        初始化VAE
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            activation: 激活函数
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码器前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tuple: (均值, 对数方差)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码器前向传播
        
        Args:
            z: 潜在变量
            
        Returns:
            Tensor: 重构输出
        """
        return self.decoder(z)
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化采样
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            Tensor: 采样结果
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度（相对于标准正态分布）
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            Tensor: KL散度
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tuple: (重构输出, 均值, 对数方差)
        """
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        VAE损失函数
        
        Args:
            recon_x: 重构输出
            x: 原始输入
            mu: 均值
            logvar: 对数方差
            beta: KL散度权重
            
        Returns:
            Dict: 损失字典
        """
        # 重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL散度
        kl_loss = torch.sum(self.kl_divergence(mu, logvar))
        
        # 总损失
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def generate(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        生成新样本
        
        Args:
            num_samples: 样本数量
            device: 设备
            
        Returns:
            Tensor: 生成的样本
        """
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z)
        
        return samples

class MeanFieldVariationalInference(nn.Module):
    """
    平均场变分推断
    
    实现平均场近似的变分推断
    """
    
    def __init__(self, num_params: int, prior_mean: float = 0.0, 
                 prior_std: float = 1.0):
        """
        初始化平均场变分推断
        
        Args:
            num_params: 参数数量
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super(MeanFieldVariationalInference, self).__init__()
        
        self.num_params = num_params
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # 变分参数
        self.mu = nn.Parameter(torch.zeros(num_params))
        self.log_sigma = nn.Parameter(torch.zeros(num_params))
    
    def sample_parameters(self, num_samples: int = 1) -> torch.Tensor:
        """
        从变分分布中采样参数
        
        Args:
            num_samples: 采样数量
            
        Returns:
            Tensor: 采样的参数
        """
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn(num_samples, self.num_params, device=self.mu.device)
        
        if num_samples == 1:
            return self.mu + sigma * eps.squeeze(0)
        else:
            return self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算变分分布与先验分布的KL散度
        
        Returns:
            Tensor: KL散度
        """
        sigma = torch.exp(self.log_sigma)
        
        # KL(q(θ)||p(θ))
        kl = 0.5 * torch.sum(
            (self.mu - self.prior_mean)**2 / self.prior_std**2 +
            sigma**2 / self.prior_std**2 -
            1 - 2 * self.log_sigma + 2 * math.log(self.prior_std)
        )
        
        return kl
    
    def log_prob(self, params: torch.Tensor) -> torch.Tensor:
        """
        计算参数的对数概率
        
        Args:
            params: 参数张量
            
        Returns:
            Tensor: 对数概率
        """
        sigma = torch.exp(self.log_sigma)
        
        log_prob = -0.5 * torch.sum(
            (params - self.mu)**2 / sigma**2 + 2 * self.log_sigma + math.log(2 * math.pi)
        )
        
        return log_prob

class NormalizingFlow(nn.Module):
    """
    标准化流
    
    实现标准化流进行灵活的变分推断
    """
    
    def __init__(self, dim: int, num_flows: int = 4):
        """
        初始化标准化流
        
        Args:
            dim: 数据维度
            num_flows: 流的数量
        """
        super(NormalizingFlow, self).__init__()
        
        self.dim = dim
        self.num_flows = num_flows
        
        # 平面流参数
        self.u = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(num_flows)])
        self.w = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(num_flows)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_flows)])
    
    def planar_flow(self, z: torch.Tensor, u: torch.Tensor, 
                   w: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        平面流变换
        
        Args:
            z: 输入张量
            u: u参数
            w: w参数
            b: b参数
            
        Returns:
            Tuple: (变换后的z, 对数雅可比行列式)
        """
        # 确保流是可逆的
        wu = torch.sum(w * u)
        m = -1 + F.softplus(wu)
        u_hat = u + (m - wu) * w / torch.sum(w * w)
        
        # 变换
        linear = torch.sum(w * z, dim=1, keepdim=True) + b
        z_new = z + u_hat * torch.tanh(linear)
        
        # 雅可比行列式
        psi = (1 - torch.tanh(linear)**2) * w
        log_det_jacobian = torch.log(torch.abs(1 + torch.sum(psi * u_hat, dim=1)) + 1e-8)
        
        return z_new, log_det_jacobian
    
    def forward(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z0: 初始潜在变量
            
        Returns:
            Tuple: (最终潜在变量, 总对数雅可比行列式)
        """
        z = z0
        log_det_jacobian_sum = torch.zeros(z.shape[0], device=z.device)
        
        for i in range(self.num_flows):
            z, log_det_jacobian = self.planar_flow(z, self.u[i], self.w[i], self.b[i])
            log_det_jacobian_sum += log_det_jacobian
        
        return z, log_det_jacobian_sum
    
    def log_prob(self, z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """
        计算对数概率
        
        Args:
            z: 最终潜在变量
            z0: 初始潜在变量
            
        Returns:
            Tensor: 对数概率
        """
        # 基分布的对数概率（标准正态分布）
        log_prob_base = -0.5 * torch.sum(z0**2, dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        
        # 变换的雅可比行列式
        _, log_det_jacobian = self.forward(z0)
        
        return log_prob_base + log_det_jacobian

class VariationalBayesianNN(nn.Module):
    """
    变分贝叶斯神经网络
    
    实现权重不确定性的贝叶斯神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, prior_std: float = 1.0):
        """
        初始化变分贝叶斯神经网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            prior_std: 先验标准差
        """
        super(VariationalBayesianNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # 构建网络层
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = VariationalLinear(dims[i], dims[i+1], prior_std)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            num_samples: 采样数量
            
        Returns:
            Tensor: 输出张量
        """
        if num_samples == 1:
            # 单次采样
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # 最后一层不加激活
                    x = F.relu(x)
            return x
        else:
            # 多次采样
            outputs = []
            for _ in range(num_samples):
                h = x
                for i, layer in enumerate(self.layers):
                    h = layer(h)
                    if i < len(self.layers) - 1:
                        h = F.relu(h)
                outputs.append(h)
            return torch.stack(outputs, dim=0)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算所有层的KL散度
        
        Returns:
            Tensor: 总KL散度
        """
        kl_sum = 0
        for layer in self.layers:
            kl_sum += layer.kl_divergence()
        return kl_sum
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测并估计不确定性
        
        Args:
            x: 输入张量
            num_samples: 采样数量
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        with torch.no_grad():
            outputs = self.forward(x, num_samples)  # [num_samples, batch_size, output_dim]
            mean = torch.mean(outputs, dim=0)
            std = torch.std(outputs, dim=0)
        
        return mean, std

class VariationalLinear(nn.Module):
    """
    变分线性层
    
    实现权重和偏置的变分推断
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 prior_std: float = 1.0):
        """
        初始化变分线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            prior_std: 先验标准差
        """
        super(VariationalLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        # 采样权重
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_eps = torch.randn_like(weight_sigma)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        # 采样偏置
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_eps = torch.randn_like(bias_sigma)
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度
        
        Returns:
            Tensor: KL散度
        """
        # 权重KL散度
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 +
            weight_sigma**2 / self.prior_std**2 -
            1 - 2 * self.weight_log_sigma + 2 * math.log(self.prior_std)
        )
        
        # 偏置KL散度
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2 +
            bias_sigma**2 / self.prior_std**2 -
            1 - 2 * self.bias_log_sigma + 2 * math.log(self.prior_std)
        )
        
        return weight_kl + bias_kl

def create_variational_model(model_type: str = 'vae', **kwargs) -> nn.Module:
    """
    创建变分推断模型
    
    Args:
        model_type: 模型类型
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 变分推断模型实例
    """
    if model_type == 'vae':
        return VariationalAutoencoder(**kwargs)
    elif model_type == 'mean_field':
        return MeanFieldVariationalInference(**kwargs)
    elif model_type == 'normalizing_flow':
        return NormalizingFlow(**kwargs)
    elif model_type == 'bayesian_nn':
        return VariationalBayesianNN(**kwargs)
    else:
        raise ValueError(f"不支持的变分推断模型类型: {model_type}")

if __name__ == "__main__":
    # 测试变分推断方法
    
    # 1. VAE测试
    print("=== VAE测试 ===")
    vae = VariationalAutoencoder(
        input_dim=10,
        hidden_dims=[20, 15],
        latent_dim=5
    )
    
    x_test = torch.randn(32, 10)
    recon_x, mu, logvar = vae(x_test)
    loss_dict = vae.loss_function(recon_x, x_test, mu, logvar)
    
    print(f"VAE - 输入形状: {x_test.shape}")
    print(f"VAE - 重构形状: {recon_x.shape}")
    print(f"VAE - 总损失: {loss_dict['total_loss'].item():.4f}")
    
    # 生成样本
    generated = vae.generate(5)
    print(f"VAE - 生成样本形状: {generated.shape}")
    
    # 2. 变分贝叶斯神经网络测试
    print("\n=== 变分贝叶斯神经网络测试 ===")
    vbnn = VariationalBayesianNN(
        input_dim=5,
        hidden_dims=[10, 8],
        output_dim=3
    )
    
    x_test = torch.randn(20, 5)
    
    # 单次预测
    output = vbnn(x_test)
    print(f"VBNN - 单次预测形状: {output.shape}")
    
    # 不确定性预测
    mean, std = vbnn.predict_with_uncertainty(x_test, num_samples=50)
    print(f"VBNN - 预测均值形状: {mean.shape}")
    print(f"VBNN - 预测标准差形状: {std.shape}")
    print(f"VBNN - 平均不确定性: {torch.mean(std).item():.4f}")
    
    # KL散度
    kl_div = vbnn.kl_divergence()
    print(f"VBNN - KL散度: {kl_div.item():.4f}")
    
    # 3. 标准化流测试
    print("\n=== 标准化流测试 ===")
    nf = NormalizingFlow(dim=3, num_flows=4)
    
    z0 = torch.randn(10, 3)
    z, log_det_jac = nf(z0)
    
    print(f"标准化流 - 输入形状: {z0.shape}")
    print(f"标准化流 - 输出形状: {z.shape}")
    print(f"标准化流 - 对数雅可比行列式形状: {log_det_jac.shape}")
    
    # 对数概率
    log_prob = nf.log_prob(z, z0)
    print(f"标准化流 - 对数概率形状: {log_prob.shape}")
    
    print("\n变分推断方法测试完成！")