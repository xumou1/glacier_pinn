#!/usr/bin/env python3
"""
Monte Carlo Dropout用于不确定性量化

实现Monte Carlo Dropout方法来估计模型不确定性，包括：
- 标准MC Dropout
- 变分Dropout
- 结构化Dropout
- 自适应Dropout
- 不确定性校准

Author: Tibetan Glacier PINNs Project Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
import math
from collections import defaultdict

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout层
    
    在推理时保持Dropout激活以估计不确定性
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        初始化MC Dropout
        
        Args:
            p: Dropout概率
            inplace: 是否原地操作
        """
        super(MCDropout, self).__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        # 在训练和推理时都应用Dropout
        return F.dropout(x, self.p, training=True, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'

class VariationalDropout(nn.Module):
    """
    变分Dropout
    
    学习每个权重的Dropout概率
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 prior_log_sigma: float = -3.0):
        """
        初始化变分Dropout
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            prior_log_sigma: 先验对数标准差
        """
        super(VariationalDropout, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.prior_log_sigma = prior_log_sigma
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.full((output_size, input_size), prior_log_sigma))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.zeros(output_size))
        self.bias_log_sigma = nn.Parameter(torch.full((output_size,), prior_log_sigma))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        if self.training:
            # 训练时：采样权重
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_eps
            
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_eps
        else:
            # 推理时：使用均值
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        计算KL散度
        
        Returns:
            torch.Tensor: KL散度
        """
        # 权重KL散度
        weight_kl = 0.5 * torch.sum(
            torch.exp(2 * self.weight_log_sigma) + self.weight_mu**2 - 
            2 * self.weight_log_sigma - 1
        )
        
        # 偏置KL散度
        bias_kl = 0.5 * torch.sum(
            torch.exp(2 * self.bias_log_sigma) + self.bias_mu**2 - 
            2 * self.bias_log_sigma - 1
        )
        
        return weight_kl + bias_kl

class StructuredDropout(nn.Module):
    """
    结构化Dropout
    
    对特征组或通道进行结构化丢弃
    """
    
    def __init__(self, p: float = 0.5, dropout_type: str = 'channel'):
        """
        初始化结构化Dropout
        
        Args:
            p: Dropout概率
            dropout_type: Dropout类型 ('channel', 'spatial', 'feature_group')
        """
        super(StructuredDropout, self).__init__()
        self.p = p
        self.dropout_type = dropout_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        if not self.training and not hasattr(self, '_mc_mode'):
            return x
        
        if self.dropout_type == 'channel':
            # 通道Dropout
            if len(x.shape) == 4:  # [B, C, H, W]
                mask_shape = (x.shape[0], x.shape[1], 1, 1)
            elif len(x.shape) == 3:  # [B, C, L]
                mask_shape = (x.shape[0], x.shape[1], 1)
            else:  # [B, C]
                mask_shape = (x.shape[0], x.shape[1])
            
            mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=x.device))
            return x * mask / (1 - self.p)
        
        elif self.dropout_type == 'spatial':
            # 空间Dropout
            if len(x.shape) == 4:  # [B, C, H, W]
                mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
            elif len(x.shape) == 3:  # [B, C, L]
                mask_shape = (x.shape[0], 1, x.shape[2])
            else:
                return F.dropout(x, self.p, training=True)
            
            mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=x.device))
            return x * mask / (1 - self.p)
        
        else:
            # 标准Dropout
            return F.dropout(x, self.p, training=True)
    
    def enable_mc_mode(self):
        """启用MC模式"""
        self._mc_mode = True
    
    def disable_mc_mode(self):
        """禁用MC模式"""
        if hasattr(self, '_mc_mode'):
            delattr(self, '_mc_mode')

class AdaptiveDropout(nn.Module):
    """
    自适应Dropout
    
    根据输入特征自适应调整Dropout概率
    """
    
    def __init__(self, input_size: int, min_p: float = 0.1, max_p: float = 0.9):
        """
        初始化自适应Dropout
        
        Args:
            input_size: 输入特征维度
            min_p: 最小Dropout概率
            max_p: 最大Dropout概率
        """
        super(AdaptiveDropout, self).__init__()
        self.min_p = min_p
        self.max_p = max_p
        
        # 学习Dropout概率的网络
        self.dropout_predictor = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        # 计算自适应Dropout概率
        p_raw = self.dropout_predictor(x.mean(dim=tuple(range(1, len(x.shape)))))
        p = self.min_p + (self.max_p - self.min_p) * p_raw
        
        # 应用Dropout
        if self.training or hasattr(self, '_mc_mode'):
            # 为每个样本使用不同的Dropout概率
            output = []
            for i in range(x.shape[0]):
                sample_p = p[i].item()
                sample_output = F.dropout(x[i:i+1], sample_p, training=True)
                output.append(sample_output)
            return torch.cat(output, dim=0)
        else:
            return x
    
    def enable_mc_mode(self):
        """启用MC模式"""
        self._mc_mode = True
    
    def disable_mc_mode(self):
        """禁用MC模式"""
        if hasattr(self, '_mc_mode'):
            delattr(self, '_mc_mode')

class MCDropoutNetwork(nn.Module):
    """
    带有MC Dropout的神经网络
    
    集成多种Dropout方法的网络架构
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 dropout_type: str = 'standard', dropout_p: float = 0.1):
        """
        初始化MC Dropout网络
        
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层维度列表
            output_size: 输出维度
            dropout_type: Dropout类型
            dropout_p: Dropout概率
        """
        super(MCDropoutNetwork, self).__init__()
        
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            
            # 添加Dropout层
            if dropout_type == 'standard':
                layers.append(MCDropout(dropout_p))
            elif dropout_type == 'variational':
                # 注意：变分Dropout需要特殊处理
                layers.append(MCDropout(dropout_p))
            elif dropout_type == 'structured':
                layers.append(StructuredDropout(dropout_p, 'channel'))
            elif dropout_type == 'adaptive':
                layers.append(AdaptiveDropout(hidden_size))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        return self.network(x)
    
    def enable_mc_dropout(self):
        """启用MC Dropout模式"""
        for module in self.modules():
            if hasattr(module, 'enable_mc_mode'):
                module.enable_mc_mode()
    
    def disable_mc_dropout(self):
        """禁用MC Dropout模式"""
        for module in self.modules():
            if hasattr(module, 'disable_mc_mode'):
                module.disable_mc_mode()
    
    def mc_predict(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo预测
        
        Args:
            x: 输入数据
            num_samples: MC采样次数
            
        Returns:
            Dict: 预测结果和不确定性
        """
        self.enable_mc_dropout()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # 统计量
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        var = torch.var(predictions, dim=0)
        
        # 分位数
        quantiles = torch.quantile(predictions, torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975]), dim=0)
        
        self.disable_mc_dropout()
        
        return {
            'mean': mean,
            'std': std,
            'var': var,
            'predictions': predictions,
            'q025': quantiles[0],
            'q25': quantiles[1],
            'median': quantiles[2],
            'q75': quantiles[3],
            'q975': quantiles[4]
        }

class UncertaintyCalibration:
    """
    不确定性校准
    
    校准MC Dropout的不确定性估计
    """
    
    def __init__(self):
        """
        初始化不确定性校准
        """
        self.calibration_params = {}
    
    def temperature_scaling(self, logits: torch.Tensor, labels: torch.Tensor,
                          max_iter: int = 50, lr: float = 0.01) -> float:
        """
        温度缩放校准
        
        Args:
            logits: 模型输出logits
            labels: 真实标签
            max_iter: 最大迭代次数
            lr: 学习率
            
        Returns:
            float: 最优温度参数
        """
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return temperature.item()
    
    def platt_scaling(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """
        Platt缩放校准
        
        Args:
            scores: 模型置信度分数
            labels: 真实标签
            
        Returns:
            Tuple: (A, B) 参数
        """
        # 使用逻辑回归拟合
        A = nn.Parameter(torch.ones(1))
        B = nn.Parameter(torch.zeros(1))
        
        optimizer = torch.optim.LBFGS([A, B], lr=0.01, max_iter=100)
        
        def eval_loss():
            calibrated_scores = torch.sigmoid(A * scores + B)
            loss = F.binary_cross_entropy(calibrated_scores, labels.float())
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return A.item(), B.item()
    
    def isotonic_regression(self, scores: torch.Tensor, labels: torch.Tensor) -> Callable:
        """
        等渗回归校准
        
        Args:
            scores: 模型置信度分数
            labels: 真实标签
            
        Returns:
            Callable: 校准函数
        """
        # 简化的等渗回归实现
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 排序
        sorted_indices = np.argsort(scores_np)
        sorted_scores = scores_np[sorted_indices]
        sorted_labels = labels_np[sorted_indices]
        
        # 计算累积平均
        cumsum_labels = np.cumsum(sorted_labels)
        cumsum_counts = np.arange(1, len(sorted_labels) + 1)
        calibrated_probs = cumsum_labels / cumsum_counts
        
        # 创建插值函数
        def calibration_function(new_scores):
            return np.interp(new_scores, sorted_scores, calibrated_probs)
        
        return calibration_function
    
    def reliability_diagram(self, predictions: torch.Tensor, labels: torch.Tensor,
                          num_bins: int = 10) -> Dict[str, np.ndarray]:
        """
        可靠性图
        
        Args:
            predictions: 预测概率
            labels: 真实标签
            num_bins: 分箱数量
            
        Returns:
            Dict: 可靠性图数据
        """
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions_np > bin_lower) & (predictions_np <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels_np[in_bin].mean()
                avg_confidence_in_bin = predictions_np[in_bin].mean()
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                counts.append(in_bin.sum())
            else:
                accuracies.append(0)
                confidences.append(0)
                counts.append(0)
        
        return {
            'accuracies': np.array(accuracies),
            'confidences': np.array(confidences),
            'counts': np.array(counts),
            'bin_boundaries': bin_boundaries
        }
    
    def expected_calibration_error(self, predictions: torch.Tensor, labels: torch.Tensor,
                                 num_bins: int = 10) -> float:
        """
        期望校准误差(ECE)
        
        Args:
            predictions: 预测概率
            labels: 真实标签
            num_bins: 分箱数量
            
        Returns:
            float: ECE值
        """
        reliability_data = self.reliability_diagram(predictions, labels, num_bins)
        
        accuracies = reliability_data['accuracies']
        confidences = reliability_data['confidences']
        counts = reliability_data['counts']
        
        total_samples = counts.sum()
        ece = 0
        
        for acc, conf, count in zip(accuracies, confidences, counts):
            if count > 0:
                ece += (count / total_samples) * abs(acc - conf)
        
        return ece

class MCDropoutPINN(nn.Module):
    """
    带有MC Dropout的物理信息神经网络
    
    结合物理约束和不确定性量化
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 dropout_p: float = 0.1, physics_weight: float = 1.0):
        """
        初始化MC Dropout PINN
        
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层维度列表
            output_size: 输出维度
            dropout_p: Dropout概率
            physics_weight: 物理损失权重
        """
        super(MCDropoutPINN, self).__init__()
        
        self.physics_weight = physics_weight
        
        # 主网络
        self.network = MCDropoutNetwork(input_size, hidden_sizes, output_size, 
                                      'standard', dropout_p)
        
        # 不确定性校准
        self.calibration = UncertaintyCalibration()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        return self.network(x)
    
    def physics_loss(self, x: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        物理损失
        
        Args:
            x: 输入坐标
            predictions: 预测值
            
        Returns:
            torch.Tensor: 物理损失
        """
        # TODO: 实现具体的物理约束
        # 这里是一个示例，实际应根据具体物理方程实现
        
        # 计算梯度
        x.requires_grad_(True)
        u = self.forward(x)
        
        # 一阶导数
        du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
        # 二阶导数
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
        
        # 示例物理方程：拉普拉斯方程
        physics_residual = d2u_dx2.sum(dim=1, keepdim=True)
        
        return torch.mean(physics_residual**2)
    
    def total_loss(self, x_data: torch.Tensor, y_data: torch.Tensor,
                  x_physics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        总损失
        
        Args:
            x_data: 数据点坐标
            y_data: 数据点值
            x_physics: 物理点坐标
            
        Returns:
            Dict: 损失字典
        """
        # 数据损失
        pred_data = self.forward(x_data)
        data_loss = F.mse_loss(pred_data, y_data)
        
        # 物理损失
        pred_physics = self.forward(x_physics)
        physics_loss = self.physics_loss(x_physics, pred_physics)
        
        # 总损失
        total_loss = data_loss + self.physics_weight * physics_loss
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        带不确定性的预测
        
        Args:
            x: 输入数据
            num_samples: MC采样次数
            
        Returns:
            Dict: 预测结果和不确定性
        """
        return self.network.mc_predict(x, num_samples)

if __name__ == "__main__":
    # 测试MC Dropout方法
    torch.manual_seed(42)
    
    # 测试数据
    x = torch.randn(100, 2)
    y = torch.randn(100, 1)
    x_test = torch.randn(20, 2)
    
    # 测试标准MC Dropout
    print("测试标准MC Dropout...")
    mc_network = MCDropoutNetwork(2, [50, 50], 1, 'standard', 0.1)
    
    # 训练几步
    optimizer = torch.optim.Adam(mc_network.parameters(), lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        pred = mc_network(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
    
    # MC预测
    mc_results = mc_network.mc_predict(x_test, num_samples=50)
    print(f"MC预测均值形状: {mc_results['mean'].shape}")
    print(f"MC预测标准差形状: {mc_results['std'].shape}")
    
    # 测试变分Dropout
    print("\n测试变分Dropout...")
    var_dropout = VariationalDropout(10, 5)
    x_var = torch.randn(32, 10)
    output = var_dropout(x_var)
    kl_div = var_dropout.kl_divergence()
    print(f"变分Dropout输出形状: {output.shape}")
    print(f"KL散度: {kl_div.item():.4f}")
    
    # 测试结构化Dropout
    print("\n测试结构化Dropout...")
    struct_dropout = StructuredDropout(0.2, 'channel')
    struct_dropout.enable_mc_mode()
    x_struct = torch.randn(8, 16, 32, 32)
    output_struct = struct_dropout(x_struct)
    print(f"结构化Dropout输出形状: {output_struct.shape}")
    
    # 测试自适应Dropout
    print("\n测试自适应Dropout...")
    adaptive_dropout = AdaptiveDropout(64)
    adaptive_dropout.enable_mc_mode()
    x_adaptive = torch.randn(16, 64)
    output_adaptive = adaptive_dropout(x_adaptive)
    print(f"自适应Dropout输出形状: {output_adaptive.shape}")
    
    # 测试不确定性校准
    print("\n测试不确定性校准...")
    calibration = UncertaintyCalibration()
    
    # 模拟预测和标签
    predictions = torch.rand(100)
    labels = torch.randint(0, 2, (100,))
    
    # 计算ECE
    ece = calibration.expected_calibration_error(predictions, labels)
    print(f"期望校准误差(ECE): {ece:.4f}")
    
    # 可靠性图
    reliability_data = calibration.reliability_diagram(predictions, labels)
    print(f"可靠性图分箱数: {len(reliability_data['accuracies'])}")
    
    # 测试MC Dropout PINN
    print("\n测试MC Dropout PINN...")
    mc_pinn = MCDropoutPINN(2, [50, 50], 1, dropout_p=0.1)
    
    # 模拟训练数据和物理点
    x_data = torch.randn(50, 2)
    y_data = torch.randn(50, 1)
    x_physics = torch.randn(100, 2)
    
    # 计算损失
    loss_dict = mc_pinn.total_loss(x_data, y_data, x_physics)
    print(f"PINN总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"数据损失: {loss_dict['data_loss'].item():.4f}")
    print(f"物理损失: {loss_dict['physics_loss'].item():.4f}")
    
    # 不确定性预测
    uncertainty_results = mc_pinn.predict_with_uncertainty(x_test, num_samples=30)
    print(f"PINN不确定性预测均值形状: {uncertainty_results['mean'].shape}")
    print(f"PINN不确定性预测标准差形状: {uncertainty_results['std'].shape}")