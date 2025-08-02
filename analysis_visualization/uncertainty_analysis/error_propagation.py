#!/usr/bin/env python3
"""
误差传播分析模块

该模块提供冰川PINNs模型的误差传播分析功能，包括：
- 线性误差传播
- 蒙特卡洛误差传播
- 敏感性分析
- 不确定性量化
- 误差源识别

作者：冰川PINNs项目组
日期：2024年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.linalg import cholesky
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

warnings.filterwarnings('ignore')


class PropagationMethod(Enum):
    """误差传播方法枚举"""
    LINEAR = "linear"
    MONTE_CARLO = "monte_carlo"
    POLYNOMIAL_CHAOS = "polynomial_chaos"
    UNSCENTED_TRANSFORM = "unscented_transform"
    TAYLOR_SERIES = "taylor_series"
    DELTA_METHOD = "delta_method"


class UncertaintyType(Enum):
    """不确定性类型枚举"""
    ALEATORY = "aleatory"  # 随机不确定性
    EPISTEMIC = "epistemic"  # 认知不确定性
    MIXED = "mixed"  # 混合不确定性


class DistributionType(Enum):
    """概率分布类型枚举"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    TRIANGULAR = "triangular"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"


class SensitivityMethod(Enum):
    """敏感性分析方法枚举"""
    SOBOL = "sobol"
    MORRIS = "morris"
    FAST = "fast"
    DERIVATIVE_BASED = "derivative_based"
    CORRELATION_BASED = "correlation_based"
    REGRESSION_BASED = "regression_based"


@dataclass
class UncertaintyParameter:
    """不确定性参数定义"""
    name: str
    distribution: DistributionType
    parameters: Dict[str, float]  # 分布参数
    uncertainty_type: UncertaintyType = UncertaintyType.ALEATORY
    correlation_matrix: Optional[np.ndarray] = None
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""


@dataclass
class ErrorPropagationConfig:
    """误差传播配置"""
    method: PropagationMethod = PropagationMethod.MONTE_CARLO
    n_samples: int = 10000
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    sensitivity_method: SensitivityMethod = SensitivityMethod.SOBOL
    n_bootstrap: int = 1000
    parallel: bool = True
    n_jobs: int = -1
    random_seed: int = 42
    convergence_threshold: float = 1e-4
    max_iterations: int = 100
    taylor_order: int = 2
    polynomial_order: int = 3
    alpha_cut_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class PropagationResult:
    """误差传播结果"""
    output_name: str
    mean: float
    std: float
    variance: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    percentiles: Dict[float, float]
    samples: Optional[np.ndarray] = None
    sensitivity_indices: Optional[Dict[str, float]] = None
    sobol_indices: Optional[Dict[str, Dict[str, float]]] = None
    correlation_matrix: Optional[np.ndarray] = None
    convergence_info: Optional[Dict[str, Any]] = None
    computation_time: float = 0.0


class DistributionSampler:
    """概率分布采样器"""
    
    @staticmethod
    def sample(param: UncertaintyParameter, n_samples: int, random_state: int = 42) -> np.ndarray:
        """从指定分布采样"""
        np.random.seed(random_state)
        
        if param.distribution == DistributionType.NORMAL:
            return np.random.normal(
                param.parameters['mean'], 
                param.parameters['std'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.UNIFORM:
            return np.random.uniform(
                param.parameters['low'], 
                param.parameters['high'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.LOGNORMAL:
            return np.random.lognormal(
                param.parameters['mean'], 
                param.parameters['sigma'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.BETA:
            return np.random.beta(
                param.parameters['alpha'], 
                param.parameters['beta'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.GAMMA:
            return np.random.gamma(
                param.parameters['shape'], 
                param.parameters['scale'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.TRIANGULAR:
            return np.random.triangular(
                param.parameters['left'], 
                param.parameters['mode'], 
                param.parameters['right'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.EXPONENTIAL:
            return np.random.exponential(
                param.parameters['scale'], 
                n_samples
            )
        
        elif param.distribution == DistributionType.WEIBULL:
            return np.random.weibull(
                param.parameters['a'], 
                n_samples
            ) * param.parameters.get('scale', 1.0)
        
        else:
            raise ValueError(f"不支持的分布类型: {param.distribution}")
    
    @staticmethod
    def sample_correlated(params: List[UncertaintyParameter], 
                         correlation_matrix: np.ndarray,
                         n_samples: int, 
                         random_state: int = 42) -> np.ndarray:
        """采样相关的随机变量"""
        np.random.seed(random_state)
        
        # 首先生成独立的标准正态分布样本
        n_params = len(params)
        independent_samples = np.random.standard_normal((n_samples, n_params))
        
        # 使用Cholesky分解引入相关性
        try:
            L = cholesky(correlation_matrix, lower=True)
            correlated_normal = independent_samples @ L.T
        except np.linalg.LinAlgError:
            # 如果相关矩阵不是正定的，使用特征值分解
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # 确保正定
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
            correlated_normal = independent_samples @ L.T
        
        # 转换为目标分布
        correlated_samples = np.zeros_like(correlated_normal)
        
        for i, param in enumerate(params):
            # 使用正态分布的CDF转换为均匀分布
            uniform_samples = stats.norm.cdf(correlated_normal[:, i])
            
            # 转换为目标分布
            if param.distribution == DistributionType.NORMAL:
                correlated_samples[:, i] = stats.norm.ppf(
                    uniform_samples, 
                    param.parameters['mean'], 
                    param.parameters['std']
                )
            elif param.distribution == DistributionType.UNIFORM:
                correlated_samples[:, i] = stats.uniform.ppf(
                    uniform_samples,
                    param.parameters['low'],
                    param.parameters['high'] - param.parameters['low']
                )
            elif param.distribution == DistributionType.LOGNORMAL:
                correlated_samples[:, i] = stats.lognorm.ppf(
                    uniform_samples,
                    param.parameters['sigma'],
                    scale=np.exp(param.parameters['mean'])
                )
            # 可以添加更多分布类型的转换
            else:
                # 对于不支持的分布，使用独立采样
                correlated_samples[:, i] = DistributionSampler.sample(param, n_samples, random_state + i)
        
        return correlated_samples


class LinearPropagator:
    """线性误差传播器"""
    
    def __init__(self, config: ErrorPropagationConfig):
        self.config = config
    
    def propagate(self, 
                 model_function: Callable,
                 parameters: List[UncertaintyParameter],
                 nominal_values: np.ndarray) -> PropagationResult:
        """线性误差传播"""
        start_time = time.time()
        
        # 计算雅可比矩阵（数值微分）
        jacobian = self._compute_jacobian(model_function, nominal_values)
        
        # 构建协方差矩阵
        covariance_matrix = self._build_covariance_matrix(parameters)
        
        # 线性传播公式：Var(Y) = J * Cov(X) * J^T
        output_variance = jacobian @ covariance_matrix @ jacobian.T
        
        # 计算输出统计量
        nominal_output = model_function(nominal_values)
        output_std = np.sqrt(np.diag(output_variance))
        
        # 假设输出为正态分布，计算置信区间
        confidence_intervals = {}
        for level in self.config.confidence_levels:
            alpha = 1 - level
            z_score = stats.norm.ppf(1 - alpha/2)
            lower = nominal_output - z_score * output_std
            upper = nominal_output + z_score * output_std
            confidence_intervals[level] = (lower[0] if len(lower) == 1 else lower, 
                                         upper[0] if len(upper) == 1 else upper)
        
        # 计算百分位数
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            z_score = stats.norm.ppf(p/100)
            value = nominal_output + z_score * output_std
            percentiles[p] = value[0] if len(value) == 1 else value
        
        computation_time = time.time() - start_time
        
        return PropagationResult(
            output_name="model_output",
            mean=nominal_output[0] if len(nominal_output) == 1 else nominal_output,
            std=output_std[0] if len(output_std) == 1 else output_std,
            variance=output_variance[0, 0] if output_variance.shape == (1, 1) else np.diag(output_variance),
            confidence_intervals=confidence_intervals,
            percentiles=percentiles,
            computation_time=computation_time
        )
    
    def _compute_jacobian(self, model_function: Callable, nominal_values: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        eps = 1e-8
        n_params = len(nominal_values)
        
        # 计算名义输出
        nominal_output = model_function(nominal_values)
        n_outputs = len(nominal_output) if hasattr(nominal_output, '__len__') else 1
        
        jacobian = np.zeros((n_outputs, n_params))
        
        for i in range(n_params):
            # 前向差分
            perturbed_values = nominal_values.copy()
            perturbed_values[i] += eps
            perturbed_output = model_function(perturbed_values)
            
            if n_outputs == 1:
                jacobian[0, i] = (perturbed_output - nominal_output) / eps
            else:
                jacobian[:, i] = (perturbed_output - nominal_output) / eps
        
        return jacobian
    
    def _build_covariance_matrix(self, parameters: List[UncertaintyParameter]) -> np.ndarray:
        """构建协方差矩阵"""
        n_params = len(parameters)
        covariance_matrix = np.zeros((n_params, n_params))
        
        for i, param in enumerate(parameters):
            # 对角元素（方差）
            if param.distribution == DistributionType.NORMAL:
                variance = param.parameters['std'] ** 2
            elif param.distribution == DistributionType.UNIFORM:
                a, b = param.parameters['low'], param.parameters['high']
                variance = (b - a) ** 2 / 12
            elif param.distribution == DistributionType.LOGNORMAL:
                sigma = param.parameters['sigma']
                mu = param.parameters['mean']
                variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
            else:
                # 对于其他分布，使用数值方法估计方差
                samples = DistributionSampler.sample(param, 10000, self.config.random_seed)
                variance = np.var(samples)
            
            covariance_matrix[i, i] = variance
        
        # 如果有相关性信息，更新协方差矩阵
        if hasattr(parameters[0], 'correlation_matrix') and parameters[0].correlation_matrix is not None:
            # 转换相关矩阵为协方差矩阵
            std_vector = np.sqrt(np.diag(covariance_matrix))
            correlation_matrix = parameters[0].correlation_matrix
            covariance_matrix = np.outer(std_vector, std_vector) * correlation_matrix
        
        return covariance_matrix


class MonteCarloPropagator:
    """蒙特卡洛误差传播器"""
    
    def __init__(self, config: ErrorPropagationConfig):
        self.config = config
    
    def propagate(self, 
                 model_function: Callable,
                 parameters: List[UncertaintyParameter],
                 correlation_matrix: Optional[np.ndarray] = None) -> PropagationResult:
        """蒙特卡洛误差传播"""
        start_time = time.time()
        
        # 生成输入样本
        if correlation_matrix is not None:
            input_samples = DistributionSampler.sample_correlated(
                parameters, correlation_matrix, self.config.n_samples, self.config.random_seed
            )
        else:
            input_samples = np.column_stack([
                DistributionSampler.sample(param, self.config.n_samples, self.config.random_seed + i)
                for i, param in enumerate(parameters)
            ])
        
        # 并行计算输出样本
        if self.config.parallel and self.config.n_jobs != 1:
            output_samples = self._parallel_evaluation(model_function, input_samples)
        else:
            output_samples = self._sequential_evaluation(model_function, input_samples)
        
        # 计算统计量
        mean = np.mean(output_samples)
        std = np.std(output_samples, ddof=1)
        variance = np.var(output_samples, ddof=1)
        
        # 计算置信区间
        confidence_intervals = {}
        for level in self.config.confidence_levels:
            alpha = 1 - level
            lower = np.percentile(output_samples, 100 * alpha/2)
            upper = np.percentile(output_samples, 100 * (1 - alpha/2))
            confidence_intervals[level] = (lower, upper)
        
        # 计算百分位数
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = np.percentile(output_samples, p)
        
        # 收敛性分析
        convergence_info = self._analyze_convergence(output_samples)
        
        computation_time = time.time() - start_time
        
        return PropagationResult(
            output_name="model_output",
            mean=mean,
            std=std,
            variance=variance,
            confidence_intervals=confidence_intervals,
            percentiles=percentiles,
            samples=output_samples,
            convergence_info=convergence_info,
            computation_time=computation_time
        )
    
    def _parallel_evaluation(self, model_function: Callable, input_samples: np.ndarray) -> np.ndarray:
        """并行评估模型"""
        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else mp.cpu_count()
        chunk_size = max(1, len(input_samples) // n_jobs)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 分块处理
            chunks = [input_samples[i:i+chunk_size] for i in range(0, len(input_samples), chunk_size)]
            
            # 提交任务
            futures = [executor.submit(self._evaluate_chunk, model_function, chunk) for chunk in chunks]
            
            # 收集结果
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
        
        return np.array(results)
    
    def _sequential_evaluation(self, model_function: Callable, input_samples: np.ndarray) -> np.ndarray:
        """顺序评估模型"""
        output_samples = []
        for sample in input_samples:
            try:
                output = model_function(sample)
                output_samples.append(output)
            except Exception as e:
                # 处理模型评估失败的情况
                output_samples.append(np.nan)
        
        return np.array(output_samples)
    
    @staticmethod
    def _evaluate_chunk(model_function: Callable, chunk: np.ndarray) -> List[float]:
        """评估一个数据块"""
        results = []
        for sample in chunk:
            try:
                output = model_function(sample)
                results.append(output)
            except Exception:
                results.append(np.nan)
        return results
    
    def _analyze_convergence(self, output_samples: np.ndarray) -> Dict[str, Any]:
        """分析收敛性"""
        n_samples = len(output_samples)
        sample_sizes = np.logspace(2, np.log10(n_samples), 20, dtype=int)
        sample_sizes = sample_sizes[sample_sizes <= n_samples]
        
        means = []
        stds = []
        
        for size in sample_sizes:
            subset = output_samples[:size]
            means.append(np.mean(subset))
            stds.append(np.std(subset, ddof=1))
        
        # 检查收敛性
        final_mean = means[-1]
        final_std = stds[-1]
        
        mean_converged = False
        std_converged = False
        
        if len(means) > 5:
            # 检查最后几个值的相对变化
            recent_means = means[-5:]
            recent_stds = stds[-5:]
            
            mean_rel_change = np.abs((recent_means[-1] - recent_means[0]) / recent_means[-1])
            std_rel_change = np.abs((recent_stds[-1] - recent_stds[0]) / recent_stds[-1])
            
            mean_converged = mean_rel_change < self.config.convergence_threshold
            std_converged = std_rel_change < self.config.convergence_threshold
        
        return {
            "sample_sizes": sample_sizes.tolist(),
            "means": means,
            "stds": stds,
            "mean_converged": mean_converged,
            "std_converged": std_converged,
            "final_mean": final_mean,
            "final_std": final_std
        }


class SensitivityAnalyzer:
    """敏感性分析器"""
    
    def __init__(self, config: ErrorPropagationConfig):
        self.config = config
    
    def analyze(self, 
               model_function: Callable,
               parameters: List[UncertaintyParameter],
               output_samples: np.ndarray,
               input_samples: np.ndarray) -> Dict[str, Any]:
        """执行敏感性分析"""
        if self.config.sensitivity_method == SensitivityMethod.SOBOL:
            return self._sobol_analysis(model_function, parameters)
        elif self.config.sensitivity_method == SensitivityMethod.CORRELATION_BASED:
            return self._correlation_analysis(input_samples, output_samples, parameters)
        elif self.config.sensitivity_method == SensitivityMethod.REGRESSION_BASED:
            return self._regression_analysis(input_samples, output_samples, parameters)
        else:
            return self._correlation_analysis(input_samples, output_samples, parameters)
    
    def _sobol_analysis(self, model_function: Callable, parameters: List[UncertaintyParameter]) -> Dict[str, Any]:
        """Sobol敏感性分析"""
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
        except ImportError:
            # 如果没有SALib，使用简化的方法
            return self._simplified_sobol(model_function, parameters)
        
        # 定义问题
        problem = {
            'num_vars': len(parameters),
            'names': [param.name for param in parameters],
            'bounds': []
        }
        
        # 设置参数边界
        for param in parameters:
            if param.distribution == DistributionType.NORMAL:
                mean = param.parameters['mean']
                std = param.parameters['std']
                bounds = [mean - 3*std, mean + 3*std]
            elif param.distribution == DistributionType.UNIFORM:
                bounds = [param.parameters['low'], param.parameters['high']]
            else:
                # 对于其他分布，使用样本的范围
                samples = DistributionSampler.sample(param, 1000, self.config.random_seed)
                bounds = [np.min(samples), np.max(samples)]
            
            problem['bounds'].append(bounds)
        
        # 生成Saltelli样本
        param_values = saltelli.sample(problem, self.config.n_samples // (2 * len(parameters) + 2))
        
        # 评估模型
        Y = np.array([model_function(X) for X in param_values])
        
        # Sobol分析
        Si = sobol.analyze(problem, Y)
        
        return {
            'first_order': dict(zip(problem['names'], Si['S1'])),
            'total_order': dict(zip(problem['names'], Si['ST'])),
            'second_order': Si['S2'] if 'S2' in Si else None,
            'confidence_intervals': {
                'first_order': dict(zip(problem['names'], Si['S1_conf'])),
                'total_order': dict(zip(problem['names'], Si['ST_conf']))
            }
        }
    
    def _simplified_sobol(self, model_function: Callable, parameters: List[UncertaintyParameter]) -> Dict[str, Any]:
        """简化的Sobol分析"""
        n_params = len(parameters)
        first_order = {}
        total_order = {}
        
        # 基准样本
        base_samples = np.column_stack([
            DistributionSampler.sample(param, self.config.n_samples, self.config.random_seed + i)
            for i, param in enumerate(parameters)
        ])
        
        base_outputs = np.array([model_function(sample) for sample in base_samples])
        total_variance = np.var(base_outputs)
        
        for i, param in enumerate(parameters):
            # 一阶敏感性指数
            # 固定其他参数，只变化当前参数
            fixed_samples = base_samples.copy()
            varied_samples = DistributionSampler.sample(param, self.config.n_samples, self.config.random_seed + 100 + i)
            fixed_samples[:, i] = varied_samples
            
            varied_outputs = np.array([model_function(sample) for sample in fixed_samples])
            
            # 计算条件方差
            conditional_variance = np.var(varied_outputs)
            first_order[param.name] = conditional_variance / total_variance if total_variance > 0 else 0
            
            # 总敏感性指数（简化计算）
            # 固定当前参数，变化其他参数
            total_samples = base_samples.copy()
            for j in range(n_params):
                if j != i:
                    new_samples = DistributionSampler.sample(parameters[j], self.config.n_samples, self.config.random_seed + 200 + j)
                    total_samples[:, j] = new_samples
            
            total_outputs = np.array([model_function(sample) for sample in total_samples])
            remaining_variance = np.var(total_outputs)
            total_order[param.name] = 1 - (remaining_variance / total_variance) if total_variance > 0 else 0
        
        return {
            'first_order': first_order,
            'total_order': total_order,
            'second_order': None,
            'confidence_intervals': None
        }
    
    def _correlation_analysis(self, input_samples: np.ndarray, output_samples: np.ndarray, parameters: List[UncertaintyParameter]) -> Dict[str, Any]:
        """基于相关性的敏感性分析"""
        correlations = {}
        rank_correlations = {}
        
        for i, param in enumerate(parameters):
            # Pearson相关系数
            corr, p_value = stats.pearsonr(input_samples[:, i], output_samples)
            correlations[param.name] = {'correlation': corr, 'p_value': p_value}
            
            # Spearman秩相关系数
            rank_corr, rank_p_value = stats.spearmanr(input_samples[:, i], output_samples)
            rank_correlations[param.name] = {'correlation': rank_corr, 'p_value': rank_p_value}
        
        return {
            'pearson_correlations': correlations,
            'spearman_correlations': rank_correlations
        }
    
    def _regression_analysis(self, input_samples: np.ndarray, output_samples: np.ndarray, parameters: List[UncertaintyParameter]) -> Dict[str, Any]:
        """基于回归的敏感性分析"""
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        # 线性回归
        linear_model = LinearRegression()
        linear_model.fit(input_samples, output_samples)
        linear_r2 = r2_score(output_samples, linear_model.predict(input_samples))
        
        # 随机森林回归
        rf_model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_seed)
        rf_model.fit(input_samples, output_samples)
        rf_r2 = r2_score(output_samples, rf_model.predict(input_samples))
        
        # 特征重要性
        linear_importance = {param.name: abs(coef) for param, coef in zip(parameters, linear_model.coef_)}
        rf_importance = {param.name: importance for param, importance in zip(parameters, rf_model.feature_importances_)}
        
        return {
            'linear_regression': {
                'r2_score': linear_r2,
                'coefficients': {param.name: coef for param, coef in zip(parameters, linear_model.coef_)},
                'importance': linear_importance
            },
            'random_forest': {
                'r2_score': rf_r2,
                'importance': rf_importance
            }
        }


class ErrorPropagationVisualizer:
    """误差传播可视化器"""
    
    def __init__(self, config: ErrorPropagationConfig):
        self.config = config
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_uncertainty_distribution(self, result: PropagationResult, title: str = "不确定性分布") -> plt.Figure:
        """绘制不确定性分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if result.samples is not None:
            # 直方图
            ax1.hist(result.samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(result.mean, color='red', linestyle='--', linewidth=2, label=f'均值: {result.mean:.3f}')
            ax1.axvline(result.mean - result.std, color='orange', linestyle='--', alpha=0.7, label=f'±1σ')
            ax1.axvline(result.mean + result.std, color='orange', linestyle='--', alpha=0.7)
            ax1.set_xlabel('输出值')
            ax1.set_ylabel('概率密度')
            ax1.set_title('概率分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Q-Q图
            stats.probplot(result.samples, dist="norm", plot=ax2)
            ax2.set_title('正态性检验 (Q-Q图)')
            ax2.grid(True, alpha=0.3)
        else:
            # 如果没有样本，绘制理论分布
            x = np.linspace(result.mean - 4*result.std, result.mean + 4*result.std, 1000)
            y = stats.norm.pdf(x, result.mean, result.std)
            
            ax1.plot(x, y, 'b-', linewidth=2, label='理论分布')
            ax1.axvline(result.mean, color='red', linestyle='--', linewidth=2, label=f'均值: {result.mean:.3f}')
            ax1.fill_between(x, 0, y, alpha=0.3)
            ax1.set_xlabel('输出值')
            ax1.set_ylabel('概率密度')
            ax1.set_title('理论正态分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.text(0.5, 0.5, '无样本数据\n无法绘制Q-Q图', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Q-Q图')
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_intervals(self, result: PropagationResult, title: str = "置信区间") -> plt.Figure:
        """绘制置信区间图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        levels = sorted(result.confidence_intervals.keys())
        lower_bounds = [result.confidence_intervals[level][0] for level in levels]
        upper_bounds = [result.confidence_intervals[level][1] for level in levels]
        widths = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]
        
        # 绘制置信区间
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(levels)))
        
        for i, (level, lower, upper, width) in enumerate(zip(levels, lower_bounds, upper_bounds, widths)):
            ax.barh(i, width, left=lower, height=0.6, color=colors[i], 
                   alpha=0.7, label=f'{level*100:.0f}% CI')
            
            # 添加数值标签
            ax.text(lower, i, f'{lower:.3f}', ha='right', va='center', fontweight='bold')
            ax.text(upper, i, f'{upper:.3f}', ha='left', va='center', fontweight='bold')
        
        # 添加均值线
        ax.axvline(result.mean, color='red', linestyle='--', linewidth=2, label=f'均值: {result.mean:.3f}')
        
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([f'{level*100:.0f}%' for level in levels])
        ax.set_xlabel('输出值')
        ax.set_ylabel('置信水平')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, Any], title: str = "敏感性分析") -> plt.Figure:
        """绘制敏感性分析图"""
        if 'first_order' in sensitivity_results and 'total_order' in sensitivity_results:
            return self._plot_sobol_indices(sensitivity_results, title)
        elif 'pearson_correlations' in sensitivity_results:
            return self._plot_correlation_analysis(sensitivity_results, title)
        elif 'linear_regression' in sensitivity_results:
            return self._plot_regression_analysis(sensitivity_results, title)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '无可用的敏感性分析结果', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
    
    def _plot_sobol_indices(self, sensitivity_results: Dict[str, Any], title: str) -> plt.Figure:
        """绘制Sobol指数图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        first_order = sensitivity_results['first_order']
        total_order = sensitivity_results['total_order']
        
        params = list(first_order.keys())
        first_values = list(first_order.values())
        total_values = list(total_order.values())
        
        x = np.arange(len(params))
        width = 0.35
        
        # 一阶和总敏感性指数对比
        ax1.bar(x - width/2, first_values, width, label='一阶敏感性', alpha=0.8)
        ax1.bar(x + width/2, total_values, width, label='总敏感性', alpha=0.8)
        
        ax1.set_xlabel('参数')
        ax1.set_ylabel('敏感性指数')
        ax1.set_title('Sobol敏感性指数')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 交互效应（总敏感性 - 一阶敏感性）
        interaction_effects = [total - first for total, first in zip(total_values, first_values)]
        
        ax2.bar(params, interaction_effects, alpha=0.8, color='orange')
        ax2.set_xlabel('参数')
        ax2.set_ylabel('交互效应')
        ax2.set_title('参数交互效应')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_correlation_analysis(self, sensitivity_results: Dict[str, Any], title: str) -> plt.Figure:
        """绘制相关性分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        pearson_corr = sensitivity_results['pearson_correlations']
        spearman_corr = sensitivity_results['spearman_correlations']
        
        params = list(pearson_corr.keys())
        pearson_values = [abs(pearson_corr[param]['correlation']) for param in params]
        spearman_values = [abs(spearman_corr[param]['correlation']) for param in params]
        
        # Pearson相关系数
        ax1.barh(params, pearson_values, alpha=0.8, color='skyblue')
        ax1.set_xlabel('|Pearson相关系数|')
        ax1.set_title('Pearson线性相关性')
        ax1.grid(True, alpha=0.3)
        
        # Spearman相关系数
        ax2.barh(params, spearman_values, alpha=0.8, color='lightcoral')
        ax2.set_xlabel('|Spearman相关系数|')
        ax2.set_title('Spearman秩相关性')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_regression_analysis(self, sensitivity_results: Dict[str, Any], title: str) -> plt.Figure:
        """绘制回归分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        linear_importance = sensitivity_results['linear_regression']['importance']
        rf_importance = sensitivity_results['random_forest']['importance']
        
        params = list(linear_importance.keys())
        linear_values = list(linear_importance.values())
        rf_values = list(rf_importance.values())
        
        # 线性回归重要性
        ax1.barh(params, linear_values, alpha=0.8, color='lightgreen')
        ax1.set_xlabel('线性回归重要性')
        ax1.set_title(f"线性回归 (R² = {sensitivity_results['linear_regression']['r2_score']:.3f})")
        ax1.grid(True, alpha=0.3)
        
        # 随机森林重要性
        ax2.barh(params, rf_values, alpha=0.8, color='gold')
        ax2.set_xlabel('随机森林重要性')
        ax2.set_title(f"随机森林 (R² = {sensitivity_results['random_forest']['r2_score']:.3f})")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, convergence_info: Dict[str, Any], title: str = "收敛性分析") -> plt.Figure:
        """绘制收敛性分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        sample_sizes = convergence_info['sample_sizes']
        means = convergence_info['means']
        stds = convergence_info['stds']
        
        # 均值收敛
        ax1.semilogx(sample_sizes, means, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(convergence_info['final_mean'], color='red', linestyle='--', 
                   label=f"最终均值: {convergence_info['final_mean']:.3f}")
        ax1.set_xlabel('样本数量')
        ax1.set_ylabel('均值')
        ax1.set_title(f"均值收敛 ({'已收敛' if convergence_info['mean_converged'] else '未收敛'})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标准差收敛
        ax2.semilogx(sample_sizes, stds, 'g-o', linewidth=2, markersize=4)
        ax2.axhline(convergence_info['final_std'], color='red', linestyle='--', 
                   label=f"最终标准差: {convergence_info['final_std']:.3f}")
        ax2.set_xlabel('样本数量')
        ax2.set_ylabel('标准差')
        ax2.set_title(f"标准差收敛 ({'已收敛' if convergence_info['std_converged'] else '未收敛'})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_uncertainty_plot(self, result: PropagationResult, title: str = "交互式不确定性分析") -> go.Figure:
        """创建交互式不确定性分析图"""
        if result.samples is None:
            # 如果没有样本，创建理论分布
            x = np.linspace(result.mean - 4*result.std, result.mean + 4*result.std, 1000)
            y = stats.norm.pdf(x, result.mean, result.std)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='理论分布', fill='tonexty'))
            fig.add_vline(x=result.mean, line_dash="dash", line_color="red", 
                         annotation_text=f"均值: {result.mean:.3f}")
            
            fig.update_layout(
                title=title,
                xaxis_title="输出值",
                yaxis_title="概率密度",
                hovermode='x unified'
            )
            
            return fig
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('概率分布', '累积分布', '箱线图', '置信区间'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 概率分布直方图
        fig.add_trace(
            go.Histogram(x=result.samples, nbinsx=50, name='样本分布', 
                        histnorm='probability density', opacity=0.7),
            row=1, col=1
        )
        
        # 累积分布
        sorted_samples = np.sort(result.samples)
        cumulative_prob = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        
        fig.add_trace(
            go.Scatter(x=sorted_samples, y=cumulative_prob, mode='lines', 
                      name='经验累积分布', line=dict(width=2)),
            row=1, col=2
        )
        
        # 箱线图
        fig.add_trace(
            go.Box(y=result.samples, name='输出分布', boxpoints='outliers'),
            row=2, col=1
        )
        
        # 置信区间
        levels = sorted(result.confidence_intervals.keys())
        lower_bounds = [result.confidence_intervals[level][0] for level in levels]
        upper_bounds = [result.confidence_intervals[level][1] for level in levels]
        
        fig.add_trace(
            go.Scatter(x=lower_bounds, y=[f'{level*100:.0f}%' for level in levels],
                      mode='markers', name='下界', marker=dict(symbol='triangle-left', size=10)),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=upper_bounds, y=[f'{level*100:.0f}%' for level in levels],
                      mode='markers', name='上界', marker=dict(symbol='triangle-right', size=10)),
            row=2, col=2
        )
        
        # 添加均值线
        for row, col in [(1, 1), (1, 2)]:
            fig.add_vline(x=result.mean, line_dash="dash", line_color="red", 
                         annotation_text=f"均值: {result.mean:.3f}", row=row, col=col)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig


class ErrorPropagationAnalyzer:
    """主要的误差传播分析器"""
    
    def __init__(self, config: ErrorPropagationConfig):
        self.config = config
        self.linear_propagator = LinearPropagator(config)
        self.mc_propagator = MonteCarloPropagator(config)
        self.sensitivity_analyzer = SensitivityAnalyzer(config)
        self.visualizer = ErrorPropagationVisualizer(config)
    
    def analyze(self, 
               model_function: Callable,
               parameters: List[UncertaintyParameter],
               nominal_values: Optional[np.ndarray] = None,
               correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """执行完整的误差传播分析"""
        results = {}
        
        # 如果没有提供名义值，使用分布的均值
        if nominal_values is None:
            nominal_values = np.array([self._get_distribution_mean(param) for param in parameters])
        
        # 线性误差传播
        if self.config.method in [PropagationMethod.LINEAR, PropagationMethod.DELTA_METHOD]:
            print("执行线性误差传播...")
            linear_result = self.linear_propagator.propagate(model_function, parameters, nominal_values)
            results['linear'] = linear_result
        
        # 蒙特卡洛误差传播
        if self.config.method == PropagationMethod.MONTE_CARLO:
            print("执行蒙特卡洛误差传播...")
            mc_result = self.mc_propagator.propagate(model_function, parameters, correlation_matrix)
            results['monte_carlo'] = mc_result
            
            # 敏感性分析
            if mc_result.samples is not None:
                print("执行敏感性分析...")
                # 重新生成输入样本用于敏感性分析
                if correlation_matrix is not None:
                    input_samples = DistributionSampler.sample_correlated(
                        parameters, correlation_matrix, self.config.n_samples, self.config.random_seed
                    )
                else:
                    input_samples = np.column_stack([
                        DistributionSampler.sample(param, self.config.n_samples, self.config.random_seed + i)
                        for i, param in enumerate(parameters)
                    ])
                
                sensitivity_results = self.sensitivity_analyzer.analyze(
                    model_function, parameters, mc_result.samples, input_samples
                )
                results['sensitivity'] = sensitivity_results
        
        return results
    
    def _get_distribution_mean(self, param: UncertaintyParameter) -> float:
        """获取分布的均值"""
        if param.distribution == DistributionType.NORMAL:
            return param.parameters['mean']
        elif param.distribution == DistributionType.UNIFORM:
            return (param.parameters['low'] + param.parameters['high']) / 2
        elif param.distribution == DistributionType.LOGNORMAL:
            mu = param.parameters['mean']
            sigma = param.parameters['sigma']
            return np.exp(mu + sigma**2 / 2)
        elif param.distribution == DistributionType.BETA:
            alpha = param.parameters['alpha']
            beta = param.parameters['beta']
            return alpha / (alpha + beta)
        elif param.distribution == DistributionType.GAMMA:
            return param.parameters['shape'] * param.parameters['scale']
        elif param.distribution == DistributionType.TRIANGULAR:
            a, b, c = param.parameters['left'], param.parameters['mode'], param.parameters['right']
            return (a + b + c) / 3
        elif param.distribution == DistributionType.EXPONENTIAL:
            return param.parameters['scale']
        else:
            # 对于不支持的分布，使用数值方法
            samples = DistributionSampler.sample(param, 10000, self.config.random_seed)
            return np.mean(samples)
    
    def generate_report(self, results: Dict[str, Any], parameters: List[UncertaintyParameter]) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            "analysis_summary": {
                "n_parameters": len(parameters),
                "parameter_names": [param.name for param in parameters],
                "methods_used": list(results.keys()),
                "configuration": {
                    "method": self.config.method.value,
                    "n_samples": self.config.n_samples,
                    "confidence_levels": self.config.confidence_levels
                }
            },
            "parameter_summary": []
        }
        
        # 参数摘要
        for param in parameters:
            param_info = {
                "name": param.name,
                "distribution": param.distribution.value,
                "parameters": param.parameters,
                "uncertainty_type": param.uncertainty_type.value,
                "description": param.description
            }
            report["parameter_summary"].append(param_info)
        
        # 结果摘要
        for method, result in results.items():
            if method in ['linear', 'monte_carlo']:
                method_summary = {
                    "mean": result.mean,
                    "std": result.std,
                    "variance": result.variance,
                    "coefficient_of_variation": result.std / abs(result.mean) if result.mean != 0 else float('inf'),
                    "confidence_intervals": result.confidence_intervals,
                    "computation_time": result.computation_time
                }
                
                if hasattr(result, 'convergence_info') and result.convergence_info:
                    method_summary["convergence"] = {
                        "mean_converged": result.convergence_info['mean_converged'],
                        "std_converged": result.convergence_info['std_converged']
                    }
                
                report[f"{method}_results"] = method_summary
            
            elif method == 'sensitivity':
                report["sensitivity_analysis"] = result
        
        return report
    
    def create_visualizations(self, results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """创建可视化图表"""
        figures = {}
        
        # 不确定性分布图
        for method in ['linear', 'monte_carlo']:
            if method in results:
                result = results[method]
                fig = self.visualizer.plot_uncertainty_distribution(
                    result, title=f"{method.replace('_', ' ').title()} 不确定性分布"
                )
                figures[f"{method}_distribution"] = fig
                
                # 置信区间图
                fig_ci = self.visualizer.plot_confidence_intervals(
                    result, title=f"{method.replace('_', ' ').title()} 置信区间"
                )
                figures[f"{method}_confidence_intervals"] = fig_ci
        
        # 敏感性分析图
        if 'sensitivity' in results:
            fig_sens = self.visualizer.plot_sensitivity_analysis(
                results['sensitivity'], title="敏感性分析"
            )
            figures["sensitivity_analysis"] = fig_sens
        
        # 收敛性分析图
        if 'monte_carlo' in results and hasattr(results['monte_carlo'], 'convergence_info'):
            if results['monte_carlo'].convergence_info:
                fig_conv = self.visualizer.plot_convergence(
                    results['monte_carlo'].convergence_info, title="蒙特卡洛收敛性分析"
                )
                figures["convergence_analysis"] = fig_conv
        
        return figures
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    report: Dict[str, Any],
                    output_dir: str = "error_propagation_results"):
        """保存分析结果"""
        import os
        import json
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存报告
        with open(os.path.join(output_dir, "error_propagation_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细结果（使用pickle保存完整对象）
        with open(os.path.join(output_dir, "error_propagation_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"误差传播分析结果已保存到: {output_dir}")


def create_error_propagation_analyzer(config: Optional[ErrorPropagationConfig] = None) -> ErrorPropagationAnalyzer:
    """创建误差传播分析器的工厂函数"""
    if config is None:
        config = ErrorPropagationConfig()
    return ErrorPropagationAnalyzer(config)


if __name__ == "__main__":
    # 示例用法
    import math
    
    # 定义一个简单的冰川模型函数
    def glacier_model(params):
        """简单的冰川质量平衡模型
        
        参数:
        - params[0]: 温度 (°C)
        - params[1]: 降水量 (mm)
        - params[2]: 辐射 (W/m²)
        - params[3]: 风速 (m/s)
        
        返回: 质量平衡 (mm w.e.)
        """
        temperature, precipitation, radiation, wind_speed = params
        
        # 简化的质量平衡计算
        # 积累项（降水）
        accumulation = precipitation * (1 if temperature < 0 else max(0, 1 - temperature/10))
        
        # 消融项（温度和辐射）
        ablation = max(0, temperature * 50 + radiation * 0.1 + wind_speed * 10)
        
        # 质量平衡 = 积累 - 消融
        mass_balance = accumulation - ablation
        
        return mass_balance
    
    # 定义不确定性参数
    parameters = [
        UncertaintyParameter(
            name="temperature",
            distribution=DistributionType.NORMAL,
            parameters={"mean": -5.0, "std": 2.0},
            uncertainty_type=UncertaintyType.ALEATORY,
            description="平均温度 (°C)"
        ),
        UncertaintyParameter(
            name="precipitation",
            distribution=DistributionType.LOGNORMAL,
            parameters={"mean": 4.0, "sigma": 0.5},
            uncertainty_type=UncertaintyType.ALEATORY,
            description="年降水量 (mm)"
        ),
        UncertaintyParameter(
            name="radiation",
            distribution=DistributionType.UNIFORM,
            parameters={"low": 150.0, "high": 250.0},
            uncertainty_type=UncertaintyType.EPISTEMIC,
            description="太阳辐射 (W/m²)"
        ),
        UncertaintyParameter(
            name="wind_speed",
            distribution=DistributionType.GAMMA,
            parameters={"shape": 2.0, "scale": 1.5},
            uncertainty_type=UncertaintyType.ALEATORY,
            description="风速 (m/s)"
        )
    ]
    
    # 创建配置
    config = ErrorPropagationConfig(
        method=PropagationMethod.MONTE_CARLO,
        n_samples=5000,
        confidence_levels=[0.68, 0.95, 0.99],
        sensitivity_method=SensitivityMethod.CORRELATION_BASED,
        parallel=True,
        random_seed=42
    )
    
    # 创建分析器
    analyzer = create_error_propagation_analyzer(config)
    
    print("开始误差传播分析...")
    print(f"模型函数: 冰川质量平衡模型")
    print(f"参数数量: {len(parameters)}")
    print(f"分析方法: {config.method.value}")
    print(f"样本数量: {config.n_samples}")
    print("-" * 50)
    
    # 执行分析
    results = analyzer.analyze(glacier_model, parameters)
    
    # 生成报告
    report = analyzer.generate_report(results, parameters)
    
    # 打印结果摘要
    print("\n=== 误差传播分析结果 ===")
    
    if 'monte_carlo' in results:
        mc_result = results['monte_carlo']
        print(f"\n蒙特卡洛分析结果:")
        print(f"  均值: {mc_result.mean:.3f} mm w.e.")
        print(f"  标准差: {mc_result.std:.3f} mm w.e.")
        print(f"  变异系数: {mc_result.std/abs(mc_result.mean)*100:.1f}%")
        
        print(f"\n置信区间:")
        for level, (lower, upper) in mc_result.confidence_intervals.items():
            print(f"  {level*100:.0f}%: [{lower:.3f}, {upper:.3f}] mm w.e.")
        
        print(f"\n百分位数:")
        for p, value in mc_result.percentiles.items():
            print(f"  P{p}: {value:.3f} mm w.e.")
        
        if mc_result.convergence_info:
            conv_info = mc_result.convergence_info
            print(f"\n收敛性:")
            print(f"  均值收敛: {'是' if conv_info['mean_converged'] else '否'}")
            print(f"  标准差收敛: {'是' if conv_info['std_converged'] else '否'}")
        
        print(f"\n计算时间: {mc_result.computation_time:.2f} 秒")
    
    if 'sensitivity' in results:
        sens_result = results['sensitivity']
        print(f"\n敏感性分析结果:")
        
        if 'pearson_correlations' in sens_result:
            print(f"  Pearson相关系数:")
            for param, corr_info in sens_result['pearson_correlations'].items():
                print(f"    {param}: {corr_info['correlation']:.3f} (p={corr_info['p_value']:.3e})")
        
        if 'first_order' in sens_result:
            print(f"  Sobol一阶敏感性指数:")
            for param, index in sens_result['first_order'].items():
                print(f"    {param}: {index:.3f}")
    
    # 创建可视化
    print("\n创建可视化图表...")
    figures = analyzer.create_visualizations(results)
    
    # 保存结果
    print("\n保存分析结果...")
    analyzer.save_results(results, report, "error_propagation_example")
    
    # 保存图表
    import os
    output_dir = "error_propagation_example"
    for name, fig in figures.items():
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n分析完成！结果已保存到: {output_dir}")
    print(f"生成的图表: {list(figures.keys())}")
    
    # 创建交互式图表示例
    if 'monte_carlo' in results:
        print("\n创建交互式图表...")
        interactive_fig = analyzer.visualizer.create_interactive_uncertainty_plot(
            results['monte_carlo'], "交互式冰川质量平衡不确定性分析"
        )
        interactive_fig.write_html(os.path.join(output_dir, "interactive_uncertainty.html"))
        print("交互式图表已保存为: interactive_uncertainty.html")
    
    print("\n示例运行完成！")