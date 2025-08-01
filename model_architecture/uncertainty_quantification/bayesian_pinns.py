#!/usr/bin/env python3
"""
贝叶斯物理信息神经网络

实现贝叶斯PINNs用于不确定性量化，包括：
- 贝叶斯神经网络
- 变分推断
- 马尔可夫链蒙特卡罗(MCMC)
- 不确定性传播
- 预测不确定性

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Gamma
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
import math

class BayesianLinear(nn.Module):
    """
    贝叶斯线性层
    
    使用变分推断的贝叶斯线性层
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_mean: float = 0.0, prior_std: float = 1.0):
        """
        初始化贝叶斯线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # 权重参数
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2.0)
        
        # 偏置参数
        self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 2.0)
        
        # 先验分布
        self.weight_prior = Normal(prior_mean, prior_std)
        self.bias_prior = Normal(prior_mean, prior_std)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样权重
            
        Returns:
            torch.Tensor: 输出张量
        """
        if sample:
            # 采样权重和偏置
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            # 使用均值
            weight = self.weight_mean
            bias = self.bias_mean
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度
        
        Returns:
            torch.Tensor: KL散度
        """
        # 权重KL散度
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mean - self.prior_mean)**2 / self.prior_std**2 +
            weight_var / self.prior_std**2 -
            self.weight_logvar +
            torch.log(torch.tensor(self.prior_std**2)) - 1
        )
        
        # 偏置KL散度
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mean - self.prior_mean)**2 / self.prior_std**2 +
            bias_var / self.prior_std**2 -
            self.bias_logvar +
            torch.log(torch.tensor(self.prior_std**2)) - 1
        )
        
        return weight_kl + bias_kl

class BayesianNetwork(nn.Module):
    """
    贝叶斯神经网络
    
    完整的贝叶斯神经网络实现
    """
    
    def __init__(self, layers: List[int], activation: str = 'tanh',
                 prior_mean: float = 0.0, prior_std: float = 1.0):
        """
        初始化贝叶斯神经网络
        
        Args:
            layers: 层大小列表
            activation: 激活函数
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super(BayesianNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 构建贝叶斯层
        for i in range(len(layers) - 1):
            self.layers.append(
                BayesianLinear(layers[i], layers[i+1], prior_mean, prior_std)
            )
        
        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"未知的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样权重
            
        Returns:
            torch.Tensor: 输出张量
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, sample)
            if i < len(self.layers) - 1:  # 最后一层不使用激活函数
                x = self.activation(x)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算总KL散度
        
        Returns:
            torch.Tensor: 总KL散度
        """
        total_kl = torch.tensor(0.0)
        for layer in self.layers:
            total_kl += layer.kl_divergence()
        
        return total_kl
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带不确定性的预测
        
        Args:
            x: 输入张量
            num_samples: 采样次数
            
        Returns:
            Tuple: (预测均值, 预测标准差)
        """
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std

class VariationalInference:
    """
    变分推断
    
    实现变分推断算法
    """
    
    def __init__(self, model: BayesianNetwork, likelihood_std: float = 0.1):
        """
        初始化变分推断
        
        Args:
            model: 贝叶斯模型
            likelihood_std: 似然标准差
        """
        self.model = model
        self.likelihood_std = likelihood_std
    
    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor,
                  physics_loss: torch.Tensor = None,
                  num_samples: int = 1, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        计算ELBO损失
        
        Args:
            x: 输入数据
            y: 目标数据
            physics_loss: 物理损失
            num_samples: 采样次数
            beta: KL权重
            
        Returns:
            Dict: 损失字典
        """
        # 数据似然
        likelihood = torch.tensor(0.0)
        for _ in range(num_samples):
            pred = self.model(x, sample=True)
            likelihood += -0.5 * torch.sum((pred - y)**2) / (self.likelihood_std**2)
        likelihood /= num_samples
        
        # KL散度
        kl_div = self.model.kl_divergence()
        
        # 物理损失
        if physics_loss is not None:
            physics_term = -physics_loss
        else:
            physics_term = torch.tensor(0.0)
        
        # ELBO = 似然 - β * KL散度 + 物理项
        elbo = likelihood - beta * kl_div + physics_term
        
        # 返回负ELBO作为损失
        loss = -elbo
        
        return {
            'total_loss': loss,
            'likelihood': -likelihood,
            'kl_divergence': kl_div,
            'physics_loss': -physics_term if physics_loss is not None else torch.tensor(0.0),
            'elbo': elbo
        }
    
    def predictive_distribution(self, x: torch.Tensor,
                              num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        预测分布
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            
        Returns:
            Dict: 预测统计量
        """
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.model(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        var = torch.var(predictions, dim=0)
        
        # 分位数
        quantiles = torch.quantile(predictions, torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975]), dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'var': var,
            'q025': quantiles[0],
            'q25': quantiles[1],
            'median': quantiles[2],
            'q75': quantiles[3],
            'q975': quantiles[4],
            'samples': predictions
        }

class MCMCSampler:
    """
    马尔可夫链蒙特卡罗采样器
    
    实现MCMC采样算法
    """
    
    def __init__(self, model: nn.Module, log_likelihood_fn: Callable,
                 log_prior_fn: Callable = None):
        """
        初始化MCMC采样器
        
        Args:
            model: 神经网络模型
            log_likelihood_fn: 对数似然函数
            log_prior_fn: 对数先验函数
        """
        self.model = model
        self.log_likelihood_fn = log_likelihood_fn
        self.log_prior_fn = log_prior_fn or self._default_log_prior
        
        # 获取模型参数
        self.param_shapes = []
        self.param_sizes = []
        for param in model.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
        
        self.total_params = sum(self.param_sizes)
    
    def _default_log_prior(self, params: torch.Tensor) -> torch.Tensor:
        """
        默认对数先验（标准正态分布）
        
        Args:
            params: 参数向量
            
        Returns:
            torch.Tensor: 对数先验
        """
        return -0.5 * torch.sum(params**2)
    
    def _params_to_vector(self) -> torch.Tensor:
        """
        将模型参数转换为向量
        
        Returns:
            torch.Tensor: 参数向量
        """
        params = []
        for param in self.model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)
    
    def _vector_to_params(self, vector: torch.Tensor):
        """
        将向量转换为模型参数
        
        Args:
            vector: 参数向量
        """
        start_idx = 0
        for param, shape, size in zip(self.model.parameters(), self.param_shapes, self.param_sizes):
            param.data = vector[start_idx:start_idx + size].view(shape)
            start_idx += size
    
    def log_posterior(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算对数后验
        
        Args:
            params: 参数向量
            x: 输入数据
            y: 目标数据
            
        Returns:
            torch.Tensor: 对数后验
        """
        # 设置模型参数
        self._vector_to_params(params)
        
        # 计算对数似然
        log_likelihood = self.log_likelihood_fn(self.model, x, y)
        
        # 计算对数先验
        log_prior = self.log_prior_fn(params)
        
        return log_likelihood + log_prior
    
    def metropolis_hastings(self, x: torch.Tensor, y: torch.Tensor,
                          num_samples: int = 1000, step_size: float = 0.01,
                          burn_in: int = 100) -> Dict[str, torch.Tensor]:
        """
        Metropolis-Hastings采样
        
        Args:
            x: 输入数据
            y: 目标数据
            num_samples: 采样次数
            step_size: 步长
            burn_in: 预热步数
            
        Returns:
            Dict: 采样结果
        """
        # 初始化
        current_params = self._params_to_vector()
        current_log_posterior = self.log_posterior(current_params, x, y)
        
        samples = []
        log_posteriors = []
        accepted = 0
        
        for i in range(num_samples + burn_in):
            # 提议新参数
            proposal = current_params + step_size * torch.randn_like(current_params)
            proposal_log_posterior = self.log_posterior(proposal, x, y)
            
            # 接受概率
            log_alpha = proposal_log_posterior - current_log_posterior
            alpha = torch.exp(torch.clamp(log_alpha, max=0))
            
            # 接受或拒绝
            if torch.rand(1) < alpha:
                current_params = proposal
                current_log_posterior = proposal_log_posterior
                accepted += 1
            
            # 保存样本（跳过预热期）
            if i >= burn_in:
                samples.append(current_params.clone())
                log_posteriors.append(current_log_posterior.item())
        
        acceptance_rate = accepted / (num_samples + burn_in)
        
        return {
            'samples': torch.stack(samples),
            'log_posteriors': torch.tensor(log_posteriors),
            'acceptance_rate': acceptance_rate
        }
    
    def hamiltonian_monte_carlo(self, x: torch.Tensor, y: torch.Tensor,
                              num_samples: int = 1000, step_size: float = 0.01,
                              num_leapfrog: int = 10, burn_in: int = 100) -> Dict[str, torch.Tensor]:
        """
        哈密顿蒙特卡罗采样
        
        Args:
            x: 输入数据
            y: 目标数据
            num_samples: 采样次数
            step_size: 步长
            num_leapfrog: 蛙跳步数
            burn_in: 预热步数
            
        Returns:
            Dict: 采样结果
        """
        # 初始化
        current_params = self._params_to_vector()
        current_params.requires_grad_(True)
        
        samples = []
        log_posteriors = []
        accepted = 0
        
        for i in range(num_samples + burn_in):
            # 初始动量
            momentum = torch.randn_like(current_params)
            
            # 计算初始能量
            current_log_posterior = self.log_posterior(current_params, x, y)
            current_kinetic = 0.5 * torch.sum(momentum**2)
            current_energy = -current_log_posterior + current_kinetic
            
            # 蛙跳积分
            params = current_params.clone()
            p = momentum.clone()
            
            # 半步动量更新
            params.grad = None
            log_posterior = self.log_posterior(params, x, y)
            log_posterior.backward()
            p += 0.5 * step_size * params.grad
            
            # 蛙跳步
            for _ in range(num_leapfrog):
                # 位置更新
                params = params + step_size * p
                
                # 动量更新
                params.grad = None
                log_posterior = self.log_posterior(params, x, y)
                log_posterior.backward()
                p += step_size * params.grad
            
            # 最后半步动量更新
            params.grad = None
            log_posterior = self.log_posterior(params, x, y)
            log_posterior.backward()
            p += 0.5 * step_size * params.grad
            
            # 计算新能量
            new_log_posterior = log_posterior
            new_kinetic = 0.5 * torch.sum(p**2)
            new_energy = -new_log_posterior + new_kinetic
            
            # 接受概率
            log_alpha = current_energy - new_energy
            alpha = torch.exp(torch.clamp(log_alpha, max=0))
            
            # 接受或拒绝
            if torch.rand(1) < alpha:
                current_params = params.detach()
                current_log_posterior = new_log_posterior
                accepted += 1
            
            current_params.requires_grad_(True)
            
            # 保存样本（跳过预热期）
            if i >= burn_in:
                samples.append(current_params.detach().clone())
                log_posteriors.append(current_log_posterior.item())
        
        acceptance_rate = accepted / (num_samples + burn_in)
        
        return {
            'samples': torch.stack(samples),
            'log_posteriors': torch.tensor(log_posteriors),
            'acceptance_rate': acceptance_rate
        }

class UncertaintyPropagation:
    """
    不确定性传播
    
    实现不确定性传播算法
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化不确定性传播
        
        Args:
            model: 神经网络模型
        """
        self.model = model
    
    def monte_carlo_dropout(self, x: torch.Tensor, num_samples: int = 100,
                          dropout_rate: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        蒙特卡罗Dropout
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            dropout_rate: Dropout率
            
        Returns:
            Dict: 不确定性结果
        """
        # 启用训练模式以使用dropout
        self.model.train()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                # 应用dropout
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # 恢复评估模式
        self.model.eval()
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        # 认知不确定性（模型不确定性）
        epistemic_uncertainty = std
        
        return {
            'mean': mean,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictions': predictions
        }
    
    def ensemble_uncertainty(self, models: List[nn.Module], x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        集成不确定性
        
        Args:
            models: 模型列表
            x: 输入数据
            
        Returns:
            Dict: 不确定性结果
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return {
            'mean': mean,
            'ensemble_uncertainty': std,
            'predictions': predictions
        }
    
    def deep_ensemble_uncertainty(self, models: List[nn.Module], x: torch.Tensor,
                                num_mc_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        深度集成不确定性
        
        Args:
            models: 模型列表
            x: 输入数据
            num_mc_samples: 每个模型的MC采样次数
            
        Returns:
            Dict: 不确定性结果
        """
        all_predictions = []
        
        for model in models:
            model.train()  # 启用dropout
            model_predictions = []
            
            for _ in range(num_mc_samples):
                with torch.no_grad():
                    pred = model(x)
                    model_predictions.append(pred)
            
            model_predictions = torch.stack(model_predictions, dim=0)
            all_predictions.append(model_predictions)
            
            model.eval()
        
        # 合并所有预测
        all_predictions = torch.cat(all_predictions, dim=0)
        
        mean = torch.mean(all_predictions, dim=0)
        total_uncertainty = torch.std(all_predictions, dim=0)
        
        # 分解不确定性
        model_means = []
        for i, model in enumerate(models):
            start_idx = i * num_mc_samples
            end_idx = (i + 1) * num_mc_samples
            model_mean = torch.mean(all_predictions[start_idx:end_idx], dim=0)
            model_means.append(model_mean)
        
        model_means = torch.stack(model_means, dim=0)
        epistemic_uncertainty = torch.std(model_means, dim=0)
        
        # 偶然不确定性（数据不确定性）
        aleatoric_uncertainty = torch.sqrt(torch.clamp(
            total_uncertainty**2 - epistemic_uncertainty**2, min=0
        ))
        
        return {
            'mean': mean,
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'predictions': all_predictions
        }

class BayesianPINN(nn.Module):
    """
    贝叶斯物理信息神经网络
    
    集成贝叶斯方法和物理约束
    """
    
    def __init__(self, layers: List[int], physics_loss_fn: Callable = None,
                 prior_mean: float = 0.0, prior_std: float = 1.0):
        """
        初始化贝叶斯PINN
        
        Args:
            layers: 网络层大小
            physics_loss_fn: 物理损失函数
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super(BayesianPINN, self).__init__()
        
        self.network = BayesianNetwork(layers, prior_mean=prior_mean, prior_std=prior_std)
        self.physics_loss_fn = physics_loss_fn
        self.variational_inference = VariationalInference(self.network)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sample: 是否采样
            
        Returns:
            torch.Tensor: 输出张量
        """
        return self.network(x, sample)
    
    def compute_loss(self, x_data: torch.Tensor, y_data: torch.Tensor,
                    x_physics: torch.Tensor, num_samples: int = 1,
                    beta: float = 1.0, physics_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            x_data: 数据输入
            y_data: 数据目标
            x_physics: 物理点
            num_samples: 采样次数
            beta: KL权重
            physics_weight: 物理损失权重
            
        Returns:
            Dict: 损失字典
        """
        # 物理损失
        physics_loss = None
        if self.physics_loss_fn is not None and x_physics is not None:
            physics_residuals = []
            for _ in range(num_samples):
                pred_physics = self.network(x_physics, sample=True)
                residual = self.physics_loss_fn(x_physics, pred_physics)
                physics_residuals.append(residual)
            
            physics_loss = torch.mean(torch.stack(physics_residuals)) * physics_weight
        
        # ELBO损失
        loss_dict = self.variational_inference.elbo_loss(
            x_data, y_data, physics_loss, num_samples, beta
        )
        
        return loss_dict
    
    def predict_with_uncertainty(self, x: torch.Tensor,
                               num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        带不确定性的预测
        
        Args:
            x: 输入数据
            num_samples: 采样次数
            
        Returns:
            Dict: 预测结果
        """
        return self.variational_inference.predictive_distribution(x, num_samples)

if __name__ == "__main__":
    # 测试贝叶斯PINNs组件
    torch.manual_seed(42)
    
    # 网络配置
    layers = [2, 50, 50, 1]
    
    # 测试数据
    batch_size = 32
    x_data = torch.randn(batch_size, 2)
    y_data = torch.randn(batch_size, 1)
    x_physics = torch.randn(100, 2)
    
    # 物理损失函数示例
    def physics_loss_fn(x, u):
        # 简单的拉普拉斯方程: ∇²u = 0
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x[:, 0].sum(), x, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_x[:, 1].sum(), x, create_graph=True)[0][:, 1]
        laplacian = u_xx + u_yy
        return torch.mean(laplacian**2)
    
    # 测试贝叶斯线性层
    bayesian_layer = BayesianLinear(10, 5)
    x_test = torch.randn(32, 10)
    output = bayesian_layer(x_test)
    kl = bayesian_layer.kl_divergence()
    print(f"贝叶斯线性层输出形状: {output.shape}, KL散度: {kl.item():.4f}")
    
    # 测试贝叶斯网络
    bayesian_net = BayesianNetwork(layers)
    output = bayesian_net(x_data)
    mean, std = bayesian_net.predict_with_uncertainty(x_data, num_samples=50)
    print(f"贝叶斯网络输出形状: {output.shape}")
    print(f"预测不确定性 - 均值形状: {mean.shape}, 标准差形状: {std.shape}")
    
    # 测试变分推断
    vi = VariationalInference(bayesian_net)
    loss_dict = vi.elbo_loss(x_data, y_data, num_samples=5)
    print(f"ELBO损失: {loss_dict['total_loss'].item():.4f}")
    
    # 测试贝叶斯PINN
    bayesian_pinn = BayesianPINN(layers, physics_loss_fn)
    x_physics.requires_grad_(True)
    loss_dict = bayesian_pinn.compute_loss(x_data, y_data, x_physics, num_samples=3)
    print(f"贝叶斯PINN总损失: {loss_dict['total_loss'].item():.4f}")
    
    # 预测不确定性
    uncertainty_results = bayesian_pinn.predict_with_uncertainty(x_data, num_samples=50)
    print(f"预测不确定性键: {list(uncertainty_results.keys())}")
    print(f"预测均值形状: {uncertainty_results['mean'].shape}")
    print(f"预测标准差形状: {uncertainty_results['std'].shape}")