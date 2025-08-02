#!/usr/bin/env python3
"""
贝叶斯推理实现

实现不确定性量化的贝叶斯推理方法，包括：
- 变分推理
- MCMC采样
- 贝叶斯神经网络
- 不确定性传播

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from abc import ABC, abstractmethod
import math

class BayesianLayer(nn.Module):
    """
    贝叶斯神经网络层
    
    实现变分推理的贝叶斯层
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 prior_std: float = 1.0, init_std: float = 0.1):
        """
        初始化贝叶斯层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            prior_std: 先验标准差
            init_std: 初始化标准差
        """
        super(BayesianLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # 权重参数（均值和对数方差）
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * init_std)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), math.log(init_std**2)))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), math.log(init_std**2)))
        
        # 先验分布
        self.register_buffer('prior_weight_mu', torch.zeros(out_features, in_features))
        self.register_buffer('prior_weight_std', torch.full((out_features, in_features), prior_std))
        self.register_buffer('prior_bias_mu', torch.zeros(out_features))
        self.register_buffer('prior_bias_std', torch.full((out_features,), prior_std))
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样权重
            
        Returns:
            Tensor: 输出张量
        """
        if sample:
            # 采样权重和偏置
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            # 使用均值
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度
        
        Returns:
            Tensor: KL散度
        """
        # 权重KL散度
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu - self.prior_weight_mu)**2 / self.prior_weight_std**2 +
            weight_var / self.prior_weight_std**2 -
            1 - self.weight_logvar + 2 * torch.log(self.prior_weight_std)
        )
        
        # 偏置KL散度
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu - self.prior_bias_mu)**2 / self.prior_bias_std**2 +
            bias_var / self.prior_bias_std**2 -
            1 - self.bias_logvar + 2 * torch.log(self.prior_bias_std)
        )
        
        return weight_kl + bias_kl

class BayesianNeuralNetwork(nn.Module):
    """
    贝叶斯神经网络
    
    实现完整的贝叶斯神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'tanh', prior_std: float = 1.0):
        """
        初始化贝叶斯神经网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数
            prior_std: 先验标准差
        """
        super(BayesianNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # 构建网络层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(BayesianLayer(dims[i], dims[i+1], prior_std=prior_std))
        
        self.layers = nn.ModuleList(layers)
        
        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样权重
            
        Returns:
            Tensor: 输出张量
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:  # 最后一层不使用激活函数
                x = self.activation(x)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算总KL散度
        
        Returns:
            Tensor: 总KL散度
        """
        total_kl = 0
        for layer in self.layers:
            total_kl += layer.kl_divergence()
        return total_kl
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测并估计不确定性
        
        Args:
            x: 输入张量
            num_samples: 采样次数
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std

class VariationalInference:
    """
    变分推理
    
    实现变分贝叶斯推理
    """
    
    def __init__(self, model: BayesianNeuralNetwork, 
                 likelihood_std: float = 0.1,
                 kl_weight: float = 1.0):
        """
        初始化变分推理
        
        Args:
            model: 贝叶斯神经网络
            likelihood_std: 似然标准差
            kl_weight: KL散度权重
        """
        self.model = model
        self.likelihood_std = likelihood_std
        self.kl_weight = kl_weight
    
    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor, 
                  num_samples: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算ELBO损失
        
        Args:
            x: 输入数据
            y: 目标数据
            num_samples: 蒙特卡洛样本数
            
        Returns:
            Tuple: (总损失, 损失组件字典)
        """
        batch_size = x.shape[0]
        
        # 蒙特卡洛估计似然
        log_likelihood = 0
        for _ in range(num_samples):
            pred = self.model(x, sample=True)
            # 高斯似然
            log_likelihood += -0.5 * torch.sum(
                (pred - y)**2 / self.likelihood_std**2 +
                math.log(2 * math.pi * self.likelihood_std**2)
            )
        
        log_likelihood /= num_samples
        
        # KL散度
        kl_div = self.model.kl_divergence()
        
        # ELBO = log p(y|x) - KL[q(θ)||p(θ)]
        # 损失 = -ELBO = -log p(y|x) + KL[q(θ)||p(θ)]
        elbo = log_likelihood - self.kl_weight * kl_div / batch_size
        loss = -elbo
        
        loss_components = {
            'total_loss': loss,
            'negative_log_likelihood': -log_likelihood,
            'kl_divergence': kl_div,
            'elbo': elbo
        }
        
        return loss, loss_components
    
    def train_step(self, optimizer: torch.optim.Optimizer,
                   x: torch.Tensor, y: torch.Tensor,
                   num_samples: int = 1) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            optimizer: 优化器
            x: 输入数据
            y: 目标数据
            num_samples: 蒙特卡洛样本数
            
        Returns:
            Dict: 训练指标
        """
        self.model.train()
        optimizer.zero_grad()
        
        loss, loss_components = self.elbo_loss(x, y, num_samples)
        loss.backward()
        optimizer.step()
        
        # 转换为标量
        metrics = {}
        for key, value in loss_components.items():
            metrics[key] = value.item()
        
        return metrics

class MCMCSampler:
    """
    MCMC采样器
    
    实现马尔可夫链蒙特卡洛采样
    """
    
    def __init__(self, model: nn.Module, likelihood_fn: Callable,
                 prior_fn: Callable, step_size: float = 0.01):
        """
        初始化MCMC采样器
        
        Args:
            model: 神经网络模型
            likelihood_fn: 似然函数
            prior_fn: 先验函数
            step_size: 步长
        """
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn
        self.step_size = step_size
        
        # 获取模型参数
        self.param_shapes = []
        self.param_sizes = []
        for param in model.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
        
        self.total_params = sum(self.param_sizes)
    
    def params_to_vector(self) -> torch.Tensor:
        """
        将模型参数转换为向量
        
        Returns:
            Tensor: 参数向量
        """
        params = []
        for param in self.model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)
    
    def vector_to_params(self, vector: torch.Tensor):
        """
        将向量转换为模型参数
        
        Args:
            vector: 参数向量
        """
        start_idx = 0
        for param, shape, size in zip(self.model.parameters(), 
                                    self.param_shapes, self.param_sizes):
            param.data = vector[start_idx:start_idx+size].view(shape)
            start_idx += size
    
    def log_posterior(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算对数后验概率
        
        Args:
            x: 输入数据
            y: 目标数据
            
        Returns:
            Tensor: 对数后验概率
        """
        # 似然
        pred = self.model(x)
        log_likelihood = self.likelihood_fn(pred, y)
        
        # 先验
        log_prior = self.prior_fn(self.params_to_vector())
        
        return log_likelihood + log_prior
    
    def metropolis_hastings_step(self, current_params: torch.Tensor,
                                x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Metropolis-Hastings步骤
        
        Args:
            current_params: 当前参数
            x: 输入数据
            y: 目标数据
            
        Returns:
            Tuple: (新参数, 是否接受)
        """
        # 当前对数后验
        self.vector_to_params(current_params)
        current_log_posterior = self.log_posterior(x, y)
        
        # 提议新参数
        proposal = current_params + self.step_size * torch.randn_like(current_params)
        self.vector_to_params(proposal)
        proposal_log_posterior = self.log_posterior(x, y)
        
        # 接受概率
        log_alpha = proposal_log_posterior - current_log_posterior
        alpha = torch.exp(torch.clamp(log_alpha, max=0))
        
        # 接受或拒绝
        if torch.rand(1).item() < alpha.item():
            return proposal, True
        else:
            return current_params, False
    
    def sample(self, x: torch.Tensor, y: torch.Tensor,
              num_samples: int = 1000, burn_in: int = 100,
              thin: int = 1) -> List[torch.Tensor]:
        """
        MCMC采样
        
        Args:
            x: 输入数据
            y: 目标数据
            num_samples: 采样数量
            burn_in: 燃烧期
            thin: 稀疏化间隔
            
        Returns:
            List: 参数样本列表
        """
        # 初始化
        current_params = self.params_to_vector()
        samples = []
        accepted = 0
        
        total_iterations = burn_in + num_samples * thin
        
        for i in range(total_iterations):
            current_params, accept = self.metropolis_hastings_step(
                current_params, x, y
            )
            
            if accept:
                accepted += 1
            
            # 收集样本（跳过燃烧期）
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current_params.clone())
        
        acceptance_rate = accepted / total_iterations
        print(f"MCMC接受率: {acceptance_rate:.3f}")
        
        return samples

class UncertaintyPropagation:
    """
    不确定性传播
    
    实现不确定性的传播和量化
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化不确定性传播
        
        Args:
            model: 神经网络模型
        """
        self.model = model
    
    def monte_carlo_dropout(self, x: torch.Tensor, 
                          num_samples: int = 100,
                          dropout_rate: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        蒙特卡洛Dropout不确定性估计
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            dropout_rate: Dropout率
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        # 临时添加dropout
        def add_dropout(module):
            if isinstance(module, nn.Linear):
                return nn.Sequential(
                    module,
                    nn.Dropout(dropout_rate)
                )
            return module
        
        # 启用训练模式以激活dropout
        self.model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
    
    def ensemble_uncertainty(self, models: List[nn.Module], 
                           x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        集成不确定性估计
        
        Args:
            models: 模型列表
            x: 输入数据
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        predictions = []
        
        with torch.no_grad():
            for model in models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
    
    def epistemic_aleatoric_decomposition(self, x: torch.Tensor,
                                        num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        认知和偶然不确定性分解
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            
        Returns:
            Dict: 不确定性分解结果
        """
        if not isinstance(self.model, BayesianNeuralNetwork):
            raise ValueError("需要贝叶斯神经网络进行不确定性分解")
        
        # 获取预测分布
        mean, total_std = self.model.predict_with_uncertainty(x, num_samples)
        
        # 假设模型输出包含均值和方差
        if mean.shape[-1] % 2 == 0:
            output_dim = mean.shape[-1] // 2
            pred_mean = mean[..., :output_dim]
            pred_logvar = mean[..., output_dim:]
            
            # 偶然不确定性（数据噪声）
            aleatoric_std = torch.exp(0.5 * pred_logvar)
            
            # 认知不确定性（模型不确定性）
            epistemic_var = total_std[..., :output_dim]**2 - aleatoric_std**2
            epistemic_std = torch.sqrt(torch.clamp(epistemic_var, min=1e-8))
            
            return {
                'prediction_mean': pred_mean,
                'total_uncertainty': total_std[..., :output_dim],
                'epistemic_uncertainty': epistemic_std,
                'aleatoric_uncertainty': aleatoric_std
            }
        else:
            # 简化情况：只有认知不确定性
            return {
                'prediction_mean': mean,
                'total_uncertainty': total_std,
                'epistemic_uncertainty': total_std,
                'aleatoric_uncertainty': torch.zeros_like(total_std)
            }

class BayesianPINN(nn.Module):
    """
    贝叶斯物理信息神经网络
    
    结合贝叶斯推理和物理约束
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 physics_loss_weight: float = 1.0,
                 prior_std: float = 1.0):
        """
        初始化贝叶斯PINN
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            output_dim: 输出维度
            physics_loss_weight: 物理损失权重
            prior_std: 先验标准差
        """
        super(BayesianPINN, self).__init__()
        
        self.network = BayesianNeuralNetwork(
            input_dim, hidden_dims, output_dim, prior_std=prior_std
        )
        self.physics_loss_weight = physics_loss_weight
        
        # 变分推理
        self.vi = VariationalInference(self.network)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样
            
        Returns:
            Tensor: 输出张量
        """
        return self.network(x, sample=sample)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        物理损失（需要子类实现）
        
        Args:
            x: 输入张量
            
        Returns:
            Tensor: 物理损失
        """
        # 这里应该实现具体的物理约束
        # 例如：PDE残差、边界条件等
        return torch.tensor(0.0, device=x.device)
    
    def total_loss(self, x_data: torch.Tensor, y_data: torch.Tensor,
                   x_physics: torch.Tensor, num_samples: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            x_data: 数据输入
            y_data: 数据目标
            x_physics: 物理约束输入
            num_samples: 采样次数
            
        Returns:
            Tuple: (总损失, 损失组件)
        """
        # 数据损失（ELBO）
        data_loss, loss_components = self.vi.elbo_loss(x_data, y_data, num_samples)
        
        # 物理损失
        physics_loss = 0
        for _ in range(num_samples):
            physics_loss += self.physics_loss(x_physics)
        physics_loss /= num_samples
        
        # 总损失
        total_loss = data_loss + self.physics_loss_weight * physics_loss
        
        loss_components['physics_loss'] = physics_loss
        loss_components['total_loss'] = total_loss
        
        return total_loss, loss_components
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                               num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测并估计不确定性
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        return self.network.predict_with_uncertainty(x, num_samples)

def create_bayesian_model(model_type: str = 'bnn', **kwargs) -> nn.Module:
    """
    创建贝叶斯模型
    
    Args:
        model_type: 模型类型
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 贝叶斯模型实例
    """
    if model_type == 'bnn':
        return BayesianNeuralNetwork(**kwargs)
    elif model_type == 'bpinn':
        return BayesianPINN(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试贝叶斯神经网络
    bnn = BayesianNeuralNetwork(
        input_dim=3, hidden_dims=[64, 64], output_dim=1, prior_std=1.0
    )
    
    # 测试数据
    batch_size = 100
    x = torch.randn(batch_size, 3)
    y = torch.randn(batch_size, 1)
    
    # 变分推理
    vi = VariationalInference(bnn, likelihood_std=0.1)
    optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
    
    # 训练步骤
    metrics = vi.train_step(optimizer, x, y, num_samples=5)
    print("训练指标:", metrics)
    
    # 不确定性预测
    mean, std = bnn.predict_with_uncertainty(x, num_samples=50)
    print(f"预测形状: {mean.shape}, 不确定性形状: {std.shape}")
    print(f"平均不确定性: {torch.mean(std).item():.4f}")
    
    # 不确定性传播
    up = UncertaintyPropagation(bnn)
    uncertainty_decomp = up.epistemic_aleatoric_decomposition(x, num_samples=50)
    
    print("\n不确定性分解:")
    for key, value in uncertainty_decomp.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状={value.shape}, 平均值={torch.mean(value).item():.4f}")