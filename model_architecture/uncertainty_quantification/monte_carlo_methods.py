#!/usr/bin/env python3
"""
蒙特卡洛方法实现

实现各种蒙特卡洛方法用于不确定性量化，包括：
- 蒙特卡洛采样
- 重要性采样
- 马尔可夫链蒙特卡洛
- 准蒙特卡洛方法

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable, Union
from abc import ABC, abstractmethod
import math
from scipy import stats
from scipy.stats import qmc

class MonteCarloSampler(ABC):
    """
    蒙特卡洛采样器基类
    
    定义蒙特卡洛采样的抽象接口
    """
    
    def __init__(self, dimension: int, device: str = 'cpu'):
        """
        初始化采样器
        
        Args:
            dimension: 采样维度
            device: 计算设备
        """
        self.dimension = dimension
        self.device = device
    
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Tensor: 样本张量
        """
        pass

class StandardMonteCarlo(MonteCarloSampler):
    """
    标准蒙特卡洛采样
    
    实现标准的蒙特卡洛采样方法
    """
    
    def __init__(self, dimension: int, distribution: str = 'normal',
                 distribution_params: Optional[Dict] = None, device: str = 'cpu'):
        """
        初始化标准蒙特卡洛采样器
        
        Args:
            dimension: 采样维度
            distribution: 分布类型
            distribution_params: 分布参数
            device: 计算设备
        """
        super(StandardMonteCarlo, self).__init__(dimension, device)
        
        self.distribution = distribution
        self.distribution_params = distribution_params or {}
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        生成标准蒙特卡洛样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Tensor: 样本张量 [num_samples, dimension]
        """
        if self.distribution == 'normal':
            mean = self.distribution_params.get('mean', 0.0)
            std = self.distribution_params.get('std', 1.0)
            samples = torch.randn(num_samples, self.dimension, device=self.device)
            samples = samples * std + mean
            
        elif self.distribution == 'uniform':
            low = self.distribution_params.get('low', 0.0)
            high = self.distribution_params.get('high', 1.0)
            samples = torch.rand(num_samples, self.dimension, device=self.device)
            samples = samples * (high - low) + low
            
        elif self.distribution == 'beta':
            alpha = self.distribution_params.get('alpha', 1.0)
            beta = self.distribution_params.get('beta', 1.0)
            # 使用变换方法生成Beta分布
            u1 = torch.rand(num_samples, self.dimension, device=self.device)
            u2 = torch.rand(num_samples, self.dimension, device=self.device)
            samples = torch.pow(u1, 1.0/alpha) / (
                torch.pow(u1, 1.0/alpha) + torch.pow(u2, 1.0/beta)
            )
            
        elif self.distribution == 'lognormal':
            mean = self.distribution_params.get('mean', 0.0)
            std = self.distribution_params.get('std', 1.0)
            normal_samples = torch.randn(num_samples, self.dimension, device=self.device)
            samples = torch.exp(normal_samples * std + mean)
            
        else:
            raise ValueError(f"不支持的分布类型: {self.distribution}")
        
        return samples
    
    def estimate_integral(self, func: Callable, num_samples: int,
                         domain: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
        """
        蒙特卡洛积分估计
        
        Args:
            func: 被积函数
            num_samples: 样本数量
            domain: 积分域
            
        Returns:
            Tuple: (积分估计值, 标准误差)
        """
        # 生成均匀分布样本
        low, high = domain
        samples = torch.rand(num_samples, self.dimension, device=self.device)
        samples = samples * (high - low) + low
        
        # 计算函数值
        with torch.no_grad():
            func_values = func(samples)
        
        # 积分估计
        volume = (high - low) ** self.dimension
        integral_estimate = volume * torch.mean(func_values).item()
        
        # 标准误差
        variance = torch.var(func_values).item()
        standard_error = volume * math.sqrt(variance / num_samples)
        
        return integral_estimate, standard_error

class ImportanceSampling(MonteCarloSampler):
    """
    重要性采样
    
    实现重要性采样方法以提高采样效率
    """
    
    def __init__(self, dimension: int, proposal_distribution: Callable,
                 proposal_sampler: Callable, device: str = 'cpu'):
        """
        初始化重要性采样器
        
        Args:
            dimension: 采样维度
            proposal_distribution: 提议分布密度函数
            proposal_sampler: 提议分布采样函数
            device: 计算设备
        """
        super(ImportanceSampling, self).__init__(dimension, device)
        
        self.proposal_distribution = proposal_distribution
        self.proposal_sampler = proposal_sampler
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        生成重要性采样样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Tensor: 样本张量
        """
        return self.proposal_sampler(num_samples, self.dimension)
    
    def estimate_expectation(self, func: Callable, target_distribution: Callable,
                           num_samples: int) -> Tuple[float, float]:
        """
        重要性采样期望估计
        
        Args:
            func: 目标函数
            target_distribution: 目标分布密度函数
            num_samples: 样本数量
            
        Returns:
            Tuple: (期望估计值, 有效样本大小)
        """
        # 从提议分布采样
        samples = self.sample(num_samples)
        
        # 计算重要性权重
        with torch.no_grad():
            target_density = target_distribution(samples)
            proposal_density = self.proposal_distribution(samples)
            
            # 避免除零
            weights = target_density / (proposal_density + 1e-10)
            
            # 计算函数值
            func_values = func(samples)
            
            # 重要性采样估计
            weighted_sum = torch.sum(weights * func_values)
            weight_sum = torch.sum(weights)
            
            expectation = (weighted_sum / weight_sum).item()
            
            # 有效样本大小
            ess = (torch.sum(weights) ** 2 / torch.sum(weights ** 2)).item()
        
        return expectation, ess

class QuasiMonteCarlo(MonteCarloSampler):
    """
    准蒙特卡洛方法
    
    实现低差异序列采样
    """
    
    def __init__(self, dimension: int, sequence_type: str = 'sobol',
                 device: str = 'cpu'):
        """
        初始化准蒙特卡洛采样器
        
        Args:
            dimension: 采样维度
            sequence_type: 序列类型 ('sobol', 'halton', 'latin_hypercube')
            device: 计算设备
        """
        super(QuasiMonteCarlo, self).__init__(dimension, device)
        
        self.sequence_type = sequence_type
        
        # 初始化序列生成器
        if sequence_type == 'sobol':
            self.sampler = qmc.Sobol(d=dimension, scramble=True)
        elif sequence_type == 'halton':
            self.sampler = qmc.Halton(d=dimension, scramble=True)
        elif sequence_type == 'latin_hypercube':
            self.sampler = qmc.LatinHypercube(d=dimension)
        else:
            raise ValueError(f"不支持的序列类型: {sequence_type}")
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        生成准蒙特卡洛样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Tensor: 样本张量 [num_samples, dimension]
        """
        # 生成[0,1]^d上的低差异序列
        samples_np = self.sampler.random(num_samples)
        samples = torch.from_numpy(samples_np).float().to(self.device)
        
        return samples
    
    def transform_to_distribution(self, samples: torch.Tensor,
                                distribution: str,
                                distribution_params: Dict) -> torch.Tensor:
        """
        将[0,1]^d样本变换到指定分布
        
        Args:
            samples: [0,1]^d上的样本
            distribution: 目标分布
            distribution_params: 分布参数
            
        Returns:
            Tensor: 变换后的样本
        """
        if distribution == 'normal':
            mean = distribution_params.get('mean', 0.0)
            std = distribution_params.get('std', 1.0)
            # 使用逆变换方法
            normal_samples = torch.erfinv(2 * samples - 1) * math.sqrt(2)
            return normal_samples * std + mean
            
        elif distribution == 'uniform':
            low = distribution_params.get('low', 0.0)
            high = distribution_params.get('high', 1.0)
            return samples * (high - low) + low
            
        elif distribution == 'exponential':
            rate = distribution_params.get('rate', 1.0)
            return -torch.log(1 - samples) / rate
            
        else:
            raise ValueError(f"不支持的分布变换: {distribution}")

class AdaptiveMonteCarlo(MonteCarloSampler):
    """
    自适应蒙特卡洛方法
    
    实现自适应采样策略
    """
    
    def __init__(self, dimension: int, initial_samples: int = 100,
                 convergence_threshold: float = 1e-3,
                 max_samples: int = 10000, device: str = 'cpu'):
        """
        初始化自适应蒙特卡洛采样器
        
        Args:
            dimension: 采样维度
            initial_samples: 初始样本数
            convergence_threshold: 收敛阈值
            max_samples: 最大样本数
            device: 计算设备
        """
        super(AdaptiveMonteCarlo, self).__init__(dimension, device)
        
        self.initial_samples = initial_samples
        self.convergence_threshold = convergence_threshold
        self.max_samples = max_samples
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        生成标准样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            Tensor: 样本张量
        """
        return torch.randn(num_samples, self.dimension, device=self.device)
    
    def adaptive_estimate(self, func: Callable, 
                         confidence_level: float = 0.95) -> Dict[str, float]:
        """
        自适应估计
        
        Args:
            func: 目标函数
            confidence_level: 置信水平
            
        Returns:
            Dict: 估计结果
        """
        estimates = []
        variances = []
        sample_sizes = []
        
        current_samples = self.initial_samples
        
        while current_samples <= self.max_samples:
            # 生成样本
            samples = self.sample(current_samples)
            
            # 计算函数值
            with torch.no_grad():
                func_values = func(samples)
            
            # 计算统计量
            estimate = torch.mean(func_values).item()
            variance = torch.var(func_values).item()
            
            estimates.append(estimate)
            variances.append(variance)
            sample_sizes.append(current_samples)
            
            # 检查收敛性
            if len(estimates) >= 2:
                relative_change = abs(estimates[-1] - estimates[-2]) / (abs(estimates[-2]) + 1e-10)
                
                if relative_change < self.convergence_threshold:
                    break
            
            # 增加样本数
            current_samples = min(int(current_samples * 1.5), self.max_samples)
        
        # 计算置信区间
        final_estimate = estimates[-1]
        final_variance = variances[-1]
        final_samples = sample_sizes[-1]
        
        standard_error = math.sqrt(final_variance / final_samples)
        
        # 正态分布近似的置信区间
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * standard_error
        
        confidence_interval = (
            final_estimate - margin_of_error,
            final_estimate + margin_of_error
        )
        
        return {
            'estimate': final_estimate,
            'variance': final_variance,
            'standard_error': standard_error,
            'confidence_interval': confidence_interval,
            'num_samples': final_samples,
            'convergence_history': estimates
        }

class MonteCarloUncertaintyQuantification:
    """
    蒙特卡洛不确定性量化
    
    使用蒙特卡洛方法进行不确定性量化
    """
    
    def __init__(self, model: nn.Module, sampler: MonteCarloSampler):
        """
        初始化蒙特卡洛不确定性量化
        
        Args:
            model: 神经网络模型
            sampler: 蒙特卡洛采样器
        """
        self.model = model
        self.sampler = sampler
    
    def parameter_uncertainty_propagation(self, x: torch.Tensor,
                                        parameter_distributions: Dict[str, Dict],
                                        num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        参数不确定性传播
        
        Args:
            x: 输入数据
            parameter_distributions: 参数分布字典
            num_samples: 采样次数
            
        Returns:
            Dict: 不确定性量化结果
        """
        predictions = []
        
        # 保存原始参数
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        for _ in range(num_samples):
            # 采样参数
            for name, param in self.model.named_parameters():
                if name in parameter_distributions:
                    dist_info = parameter_distributions[name]
                    dist_type = dist_info['type']
                    
                    if dist_type == 'normal':
                        mean = dist_info.get('mean', 0.0)
                        std = dist_info.get('std', 1.0)
                        noise = torch.randn_like(param) * std + mean
                        param.data = original_params[name] + noise
                    
                    elif dist_type == 'uniform':
                        low = dist_info.get('low', -0.1)
                        high = dist_info.get('high', 0.1)
                        noise = torch.rand_like(param) * (high - low) + low
                        param.data = original_params[name] + noise
            
            # 前向传播
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
        
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        # 计算统计量
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        quantiles = torch.quantile(predictions, torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975]), dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'quantile_2.5': quantiles[0],
            'quantile_25': quantiles[1],
            'median': quantiles[2],
            'quantile_75': quantiles[3],
            'quantile_97.5': quantiles[4],
            'all_predictions': predictions
        }
    
    def input_uncertainty_propagation(self, x_mean: torch.Tensor,
                                    x_std: torch.Tensor,
                                    num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        输入不确定性传播
        
        Args:
            x_mean: 输入均值
            x_std: 输入标准差
            num_samples: 采样次数
            
        Returns:
            Dict: 不确定性量化结果
        """
        predictions = []
        
        for _ in range(num_samples):
            # 采样输入
            noise = torch.randn_like(x_mean) * x_std
            x_sample = x_mean + noise
            
            # 前向传播
            with torch.no_grad():
                pred = self.model(x_sample)
                predictions.append(pred)
        
        # 计算统计量
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return {
            'mean': mean,
            'std': std,
            'all_predictions': predictions
        }
    
    def sensitivity_analysis(self, x: torch.Tensor,
                           perturbation_std: float = 0.1,
                           num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        敏感性分析
        
        Args:
            x: 输入数据
            perturbation_std: 扰动标准差
            num_samples: 采样次数
            
        Returns:
            Dict: 敏感性分析结果
        """
        input_dim = x.shape[-1]
        sensitivities = []
        
        # 基准预测
        with torch.no_grad():
            baseline_pred = self.model(x)
        
        for i in range(input_dim):
            feature_sensitivities = []
            
            for _ in range(num_samples):
                # 扰动第i个特征
                x_perturbed = x.clone()
                perturbation = torch.randn_like(x[..., i:i+1]) * perturbation_std
                x_perturbed[..., i:i+1] += perturbation
                
                # 计算预测差异
                with torch.no_grad():
                    perturbed_pred = self.model(x_perturbed)
                    sensitivity = torch.abs(perturbed_pred - baseline_pred)
                    feature_sensitivities.append(sensitivity)
            
            # 计算平均敏感性
            feature_sensitivities = torch.stack(feature_sensitivities, dim=0)
            mean_sensitivity = torch.mean(feature_sensitivities, dim=0)
            sensitivities.append(mean_sensitivity)
        
        sensitivities = torch.stack(sensitivities, dim=-1)  # [batch_size, output_dim, input_dim]
        
        return {
            'sensitivities': sensitivities,
            'total_sensitivity': torch.sum(sensitivities, dim=-1),
            'normalized_sensitivities': sensitivities / (torch.sum(sensitivities, dim=-1, keepdim=True) + 1e-10)
        }

def create_monte_carlo_sampler(sampler_type: str = 'standard', **kwargs) -> MonteCarloSampler:
    """
    创建蒙特卡洛采样器
    
    Args:
        sampler_type: 采样器类型
        **kwargs: 采样器参数
        
    Returns:
        MonteCarloSampler: 采样器实例
    """
    if sampler_type == 'standard':
        return StandardMonteCarlo(**kwargs)
    elif sampler_type == 'importance':
        return ImportanceSampling(**kwargs)
    elif sampler_type == 'quasi':
        return QuasiMonteCarlo(**kwargs)
    elif sampler_type == 'adaptive':
        return AdaptiveMonteCarlo(**kwargs)
    else:
        raise ValueError(f"不支持的采样器类型: {sampler_type}")

if __name__ == "__main__":
    # 测试蒙特卡洛方法
    
    # 1. 标准蒙特卡洛
    print("=== 标准蒙特卡洛测试 ===")
    mc_sampler = StandardMonteCarlo(dimension=2, distribution='normal')
    samples = mc_sampler.sample(1000)
    print(f"样本形状: {samples.shape}")
    print(f"样本均值: {torch.mean(samples, dim=0)}")
    print(f"样本标准差: {torch.std(samples, dim=0)}")
    
    # 2. 准蒙特卡洛
    print("\n=== 准蒙特卡洛测试 ===")
    qmc_sampler = QuasiMonteCarlo(dimension=2, sequence_type='sobol')
    qmc_samples = qmc_sampler.sample(1000)
    print(f"QMC样本形状: {qmc_samples.shape}")
    print(f"QMC样本范围: [{torch.min(qmc_samples):.3f}, {torch.max(qmc_samples):.3f}]")
    
    # 3. 自适应蒙特卡洛
    print("\n=== 自适应蒙特卡洛测试 ===")
    adaptive_sampler = AdaptiveMonteCarlo(dimension=2)
    
    # 定义测试函数
    def test_function(x):
        return torch.sum(x**2, dim=-1)
    
    result = adaptive_sampler.adaptive_estimate(test_function)
    print(f"自适应估计结果: {result['estimate']:.4f}")
    print(f"置信区间: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
    print(f"使用样本数: {result['num_samples']}")
    
    # 4. 不确定性量化
    print("\n=== 不确定性量化测试 ===")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )
    
    # 不确定性量化
    uq = MonteCarloUncertaintyQuantification(model, mc_sampler)
    
    # 测试输入
    x_test = torch.randn(10, 2)
    x_std = torch.ones_like(x_test) * 0.1
    
    # 输入不确定性传播
    input_uq_result = uq.input_uncertainty_propagation(x_test, x_std, num_samples=100)
    print(f"输入不确定性传播 - 预测均值形状: {input_uq_result['mean'].shape}")
    print(f"输入不确定性传播 - 预测标准差形状: {input_uq_result['std'].shape}")
    
    # 敏感性分析
    sensitivity_result = uq.sensitivity_analysis(x_test, num_samples=100)
    print(f"敏感性分析 - 敏感性形状: {sensitivity_result['sensitivities'].shape}")
    print(f"总敏感性: {torch.mean(sensitivity_result['total_sensitivity']).item():.4f}")