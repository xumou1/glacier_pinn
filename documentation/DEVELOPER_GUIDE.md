# 开发指南 - Developer Guide

## 项目架构 - Project Architecture

### 整体设计原则

本项目采用模块化、可扩展的架构设计，遵循以下核心原则：

1. **模块化设计**: 每个功能模块独立开发和测试
2. **接口标准化**: 统一的API接口设计
3. **可配置性**: 通过配置文件控制模型行为
4. **可扩展性**: 支持新的模型架构和物理约束
5. **可重现性**: 确保实验结果的可重现性

### 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    main_experiment.py                      │
│                     (主入口点)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            training_framework/                      │   │
│  │         (训练框架 - 核心协调层)                        │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                       │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │            model_architecture/                      │   │
│  │            (模型架构层)                              │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                       │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │            data_management/                         │   │
│  │            (数据管理层)                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         validation_testing/                         │   │
│  │         (验证测试层)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       analysis_visualization/                       │   │
│  │       (分析可视化层)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       deployment_application/                       │   │
│  │       (部署应用层)                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 开发环境设置 - Development Environment Setup

### 1. 代码风格和工具

```bash
# 安装开发工具
pip install black flake8 mypy pre-commit pytest pytest-cov

# 设置pre-commit钩子
pre-commit install

# 配置IDE (以VSCode为例)
# 在.vscode/settings.json中添加:
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true
}
```

### 2. Git工作流

```bash
# 创建功能分支
git checkout -b feature/new-physics-constraint

# 提交代码
git add .
git commit -m "feat: add new ice flow physics constraint"

# 推送分支
git push origin feature/new-physics-constraint

# 创建Pull Request
# 在GitHub上创建PR，等待代码审查
```

### 3. 测试驱动开发

```python
# 先写测试
# tests/unit_tests/test_new_feature.py
import pytest
from model_architecture.core_pinns import NewPhysicsConstraint

def test_new_physics_constraint():
    constraint = NewPhysicsConstraint()
    result = constraint.compute_residual(test_data)
    assert result.shape == expected_shape
    assert torch.allclose(result, expected_result, atol=1e-6)

# 再实现功能
# model_architecture/core_pinns/physics_laws.py
class NewPhysicsConstraint:
    def compute_residual(self, data):
        # 实现物理约束计算
        return residual
```

## 核心模块开发 - Core Module Development

### 1. 数据管理模块开发

#### 添加新的数据处理器

```python
# data_management/preprocessing/new_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import xarray as xr
import geopandas as gpd

class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
    
    @abstractmethod
    def load_data(self) -> xr.Dataset:
        """加载原始数据"""
        pass
    
    @abstractmethod
    def preprocess(self) -> xr.Dataset:
        """预处理数据"""
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, bool]:
        """验证数据质量"""
        pass
    
    def _setup_logger(self):
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        return logger

class NewDataProcessor(BaseDataProcessor):
    """新数据源处理器"""
    
    def load_data(self) -> xr.Dataset:
        # 实现数据加载逻辑
        data_path = self.config['data_path']
        dataset = xr.open_dataset(data_path)
        self.logger.info(f"加载数据: {data_path}")
        return dataset
    
    def preprocess(self) -> xr.Dataset:
        # 实现预处理逻辑
        data = self.load_data()
        # 坐标转换、重采样、滤波等
        processed_data = self._apply_transformations(data)
        return processed_data
    
    def validate(self) -> Dict[str, bool]:
        # 实现数据验证逻辑
        validation_results = {
            'completeness': self._check_completeness(),
            'consistency': self._check_consistency(),
            'physical_bounds': self._check_physical_bounds()
        }
        return validation_results
```

#### 数据质量控制

```python
# data_management/preprocessing/quality_control.py
class QualityController:
    """数据质量控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('quality_thresholds', {})
    
    def run_quality_checks(self, data: xr.Dataset) -> Dict[str, Any]:
        """运行所有质量检查"""
        results = {
            'completeness': self.check_completeness(data),
            'outliers': self.detect_outliers(data),
            'consistency': self.check_temporal_consistency(data),
            'physical_bounds': self.validate_physical_bounds(data)
        }
        return results
    
    def check_completeness(self, data: xr.Dataset) -> Dict[str, float]:
        """检查数据完整性"""
        completeness = {}
        for var in data.data_vars:
            total_points = data[var].size
            valid_points = data[var].count().item()
            completeness[var] = valid_points / total_points
        return completeness
    
    def detect_outliers(self, data: xr.Dataset, method: str = 'iqr') -> xr.Dataset:
        """检测异常值"""
        outlier_masks = {}
        
        for var in data.data_vars:
            if method == 'iqr':
                q1 = data[var].quantile(0.25)
                q3 = data[var].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_masks[var] = (data[var] < lower_bound) | (data[var] > upper_bound)
            
            elif method == 'zscore':
                z_scores = (data[var] - data[var].mean()) / data[var].std()
                outlier_masks[var] = abs(z_scores) > 3
        
        return xr.Dataset(outlier_masks)
```

### 2. 模型架构模块开发

#### 添加新的PINNs架构

```python
# model_architecture/advanced_architectures/new_architecture.py
import torch
import torch.nn as nn
from typing import List, Optional
from ..core_pinns.base_pinn import BasePINN

class NewPINNArchitecture(BasePINN):
    """新的PINNs架构实现"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int],
                 special_param: float = 1.0,
                 **kwargs):
        super().__init__(input_dim, output_dim, hidden_layers, **kwargs)
        
        self.special_param = special_param
        self.special_layers = self._build_special_layers()
    
    def _build_special_layers(self) -> nn.ModuleList:
        """构建特殊层结构"""
        layers = nn.ModuleList()
        # 实现特殊的网络结构
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 实现特殊的前向传播逻辑
        output = self.special_forward(x)
        return output
    
    def special_forward(self, x: torch.Tensor) -> torch.Tensor:
        """特殊的前向传播方法"""
        # 实现架构特有的计算逻辑
        pass
    
    def compute_physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """计算物理损失（可重写以适应新架构）"""
        # 可以重写父类的物理损失计算
        base_physics_loss = super().compute_physics_loss(x, y_pred)
        
        # 添加架构特有的物理约束
        special_constraint = self._compute_special_constraint(x, y_pred)
        
        return base_physics_loss + self.special_param * special_constraint
    
    def _compute_special_constraint(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """计算架构特有的约束"""
        # 实现特殊约束的计算
        pass
```

#### 物理定律实现

```python
# model_architecture/glacier_physics/new_physics_law.py
import torch
from typing import Dict, Any
from ..core_pinns.physics_laws import PhysicsLaws

class NewPhysicsLaw(PhysicsLaws):
    """新物理定律实现"""
    
    @staticmethod
    def compute_residual(fields: Dict[str, torch.Tensor], 
                        coords: torch.Tensor,
                        params: Dict[str, Any]) -> torch.Tensor:
        """计算物理定律残差
        
        Args:
            fields: 物理场字典 (velocity, thickness, temperature等)
            coords: 坐标张量 (x, y, z, t)
            params: 物理参数
            
        Returns:
            residual: 物理定律残差
        """
        # 提取坐标
        x, y, z, t = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        
        # 提取物理场
        velocity = fields['velocity']
        thickness = fields['thickness']
        temperature = fields.get('temperature', None)
        
        # 计算梯度
        velocity_grad = torch.autograd.grad(
            velocity, coords, 
            grad_outputs=torch.ones_like(velocity),
            create_graph=True, retain_graph=True
        )[0]
        
        # 实现具体的物理定律
        # 例如：新的冰流定律
        residual = NewPhysicsLaw._ice_flow_law(
            velocity, velocity_grad, temperature, params
        )
        
        return residual
    
    @staticmethod
    def _ice_flow_law(velocity: torch.Tensor,
                     velocity_grad: torch.Tensor,
                     temperature: torch.Tensor,
                     params: Dict[str, Any]) -> torch.Tensor:
        """新的冰流定律实现"""
        # 实现具体的物理方程
        A = params.get('flow_parameter', 1e-16)  # 流动参数
        n = params.get('flow_exponent', 3)       # 流动指数
        
        # 计算应变率
        strain_rate = NewPhysicsLaw._compute_strain_rate(velocity_grad)
        
        # 计算应力
        stress = NewPhysicsLaw._compute_stress(strain_rate, A, n, temperature)
        
        # 返回本构关系残差
        residual = stress - NewPhysicsLaw._glen_flow_law(strain_rate, A, n)
        
        return residual
```

### 3. 训练框架模块开发

#### 自定义训练策略

```python
# training_framework/training_stages/custom_trainer.py
from typing import Dict, Any, Optional
import torch
from .progressive_trainer import ProgressiveTrainer

class CustomTrainer(ProgressiveTrainer):
    """自定义训练器"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.custom_params = config.get('custom_training', {})
    
    def custom_training_stage(self) -> Dict[str, float]:
        """自定义训练阶段"""
        self.logger.info("开始自定义训练阶段")
        
        # 设置特殊的学习率
        custom_lr = self.custom_params.get('learning_rate', 1e-4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=custom_lr)
        
        # 自定义损失权重
        loss_weights = self.custom_params.get('loss_weights', {})
        
        epoch_losses = []
        for epoch in range(self.custom_params.get('epochs', 1000)):
            epoch_loss = self._custom_training_step(optimizer, loss_weights)
            epoch_losses.append(epoch_loss)
            
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        return {
            'final_loss': epoch_losses[-1],
            'loss_history': epoch_losses,
            'converged': self._check_convergence(epoch_losses)
        }
    
    def _custom_training_step(self, optimizer, loss_weights) -> float:
        """自定义训练步骤"""
        optimizer.zero_grad()
        
        # 采样训练点
        physics_points = self.sampler.sample_physics_points(self.batch_size)
        boundary_points = self.sampler.sample_boundary_points(self.batch_size // 4)
        data_points = self.sampler.sample_data_points(self.batch_size // 2)
        
        # 计算各项损失
        physics_loss = self._compute_physics_loss(physics_points)
        boundary_loss = self._compute_boundary_loss(boundary_points)
        data_loss = self._compute_data_loss(data_points)
        
        # 自定义损失组合
        total_loss = (
            loss_weights.get('physics', 1.0) * physics_loss +
            loss_weights.get('boundary', 1.0) * boundary_loss +
            loss_weights.get('data', 1.0) * data_loss
        )
        
        # 添加自定义正则化项
        if 'regularization' in loss_weights:
            reg_loss = self._compute_regularization_loss()
            total_loss += loss_weights['regularization'] * reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
```

#### 自适应采样策略

```python
# training_framework/sampling_strategies/custom_sampling.py
import torch
import numpy as np
from typing import Tuple, Optional
from .adaptive_sampling import AdaptiveSampling

class CustomAdaptiveSampling(AdaptiveSampling):
    """自定义自适应采样策略"""
    
    def __init__(self, domain_bounds, config: Dict[str, Any]):
        super().__init__(domain_bounds)
        self.config = config
        self.refinement_history = []
    
    def physics_guided_sampling(self, model, n_points: int) -> torch.Tensor:
        """物理导向采样"""
        # 基于物理残差的自适应采样
        candidate_points = self._generate_candidate_points(n_points * 5)
        
        # 计算物理残差
        with torch.no_grad():
            residuals = self._compute_physics_residuals(model, candidate_points)
        
        # 基于残差大小选择采样点
        residual_magnitudes = torch.norm(residuals, dim=1)
        
        # 使用概率采样，残差大的区域采样概率高
        probabilities = torch.softmax(residual_magnitudes / self.config.get('temperature', 1.0), dim=0)
        
        # 采样
        indices = torch.multinomial(probabilities, n_points, replacement=False)
        selected_points = candidate_points[indices]
        
        return selected_points
    
    def uncertainty_guided_sampling(self, model, n_points: int) -> torch.Tensor:
        """不确定性导向采样"""
        if not hasattr(model, 'predict_with_uncertainty'):
            # 如果模型不支持不确定性预测，回退到标准采样
            return self.sample_physics_points(n_points)
        
        candidate_points = self._generate_candidate_points(n_points * 3)
        
        # 计算预测不确定性
        with torch.no_grad():
            _, uncertainties = model.predict_with_uncertainty(candidate_points)
        
        # 选择不确定性最高的点
        uncertainty_scores = torch.sum(uncertainties, dim=1)
        _, indices = torch.topk(uncertainty_scores, n_points)
        
        selected_points = candidate_points[indices]
        
        return selected_points
    
    def multi_scale_sampling(self, n_points: int, scales: List[float]) -> torch.Tensor:
        """多尺度采样"""
        all_points = []
        points_per_scale = n_points // len(scales)
        
        for scale in scales:
            # 为每个尺度生成采样点
            scale_points = self._generate_scale_specific_points(points_per_scale, scale)
            all_points.append(scale_points)
        
        # 合并所有尺度的采样点
        combined_points = torch.cat(all_points, dim=0)
        
        return combined_points
```

### 4. 验证测试模块开发

#### 自定义验证器

```python
# validation_testing/custom_validation/domain_specific_validator.py
import torch
import numpy as np
from typing import Dict, Any, List
from ..physics_validation.conservation_laws import ConservationValidator

class GlacierSpecificValidator(ConservationValidator):
    """冰川特定验证器"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.glacier_params = config.get('glacier_parameters', {})
    
    def validate_glacier_physics(self) -> Dict[str, Any]:
        """验证冰川特定物理约束"""
        results = {}
        
        # 验证冰川厚度的物理合理性
        results['thickness_validation'] = self._validate_thickness_bounds()
        
        # 验证冰川流速的物理合理性
        results['velocity_validation'] = self._validate_velocity_patterns()
        
        # 验证质量平衡
        results['mass_balance_validation'] = self._validate_mass_balance()
        
        # 验证高程依赖性
        results['elevation_dependency'] = self._validate_elevation_effects()
        
        return results
    
    def _validate_thickness_bounds(self) -> Dict[str, float]:
        """验证厚度边界条件"""
        # 生成测试点
        test_points = self._generate_test_grid()
        
        with torch.no_grad():
            predictions = self.model(test_points)
            thickness = predictions[:, 0]  # 假设厚度是第一个输出
        
        # 检查物理约束
        min_thickness = torch.min(thickness).item()
        max_thickness = torch.max(thickness).item()
        negative_thickness_ratio = (thickness < 0).float().mean().item()
        
        # 检查厚度梯度的合理性
        thickness_grad = torch.autograd.grad(
            thickness.sum(), test_points,
            create_graph=False, retain_graph=False
        )[0]
        
        max_gradient = torch.max(torch.norm(thickness_grad[:, :2], dim=1)).item()
        
        return {
            'min_thickness': min_thickness,
            'max_thickness': max_thickness,
            'negative_thickness_ratio': negative_thickness_ratio,
            'max_thickness_gradient': max_gradient,
            'thickness_bounds_valid': min_thickness >= 0 and max_thickness < 1000
        }
    
    def _validate_velocity_patterns(self) -> Dict[str, float]:
        """验证速度模式的物理合理性"""
        test_points = self._generate_test_grid()
        
        with torch.no_grad():
            predictions = self.model(test_points)
            velocity_x = predictions[:, 1]
            velocity_y = predictions[:, 2]
        
        # 计算速度大小
        velocity_magnitude = torch.sqrt(velocity_x**2 + velocity_y**2)
        
        # 检查速度的物理合理性
        max_velocity = torch.max(velocity_magnitude).item()
        mean_velocity = torch.mean(velocity_magnitude).item()
        
        # 检查速度是否遵循地形约束
        # (这里需要地形数据，简化处理)
        velocity_terrain_consistency = self._check_velocity_terrain_consistency(
            test_points, velocity_x, velocity_y
        )
        
        return {
            'max_velocity': max_velocity,
            'mean_velocity': mean_velocity,
            'velocity_terrain_consistency': velocity_terrain_consistency,
            'velocity_bounds_valid': max_velocity < 1000  # m/year
        }
```

### 5. 可视化模块开发

#### 自定义可视化组件

```python
# analysis_visualization/custom_plots/glacier_evolution_plots.py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Optional, Tuple
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class GlacierEvolutionPlotter:
    """冰川演化可视化器"""
    
    def __init__(self, model, data: xr.Dataset, config: Dict[str, Any]):
        self.model = model
        self.data = data
        self.config = config
        self.projection = ccrs.PlateCarree()
    
    def create_evolution_animation(self, 
                                 time_points: List[float],
                                 variable: str = 'thickness',
                                 save_path: Optional[str] = None) -> None:
        """创建演化动画"""
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': self.projection})
        
        def animate(frame):
            ax.clear()
            time_point = time_points[frame]
            
            # 生成该时间点的预测
            prediction = self._generate_prediction_at_time(time_point, variable)
            
            # 绘制地图
            im = ax.contourf(
                prediction.longitude, prediction.latitude, prediction.values,
                levels=20, cmap='viridis', transform=self.projection
            )
            
            # 添加地理要素
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.RIVERS)
            
            # 设置标题
            ax.set_title(f'{variable.title()} at {time_point:.0f}', fontsize=14)
            
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(time_points), 
                           interval=500, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
        
        plt.show()
    
    def create_comparison_plot(self, 
                             observation_data: xr.Dataset,
                             prediction_time: float,
                             variables: List[str]) -> plt.Figure:
        """创建观测与预测对比图"""
        n_vars = len(variables)
        fig, axes = plt.subplots(2, n_vars, figsize=(5*n_vars, 10))
        
        if n_vars == 1:
            axes = axes.reshape(2, 1)
        
        for i, var in enumerate(variables):
            # 观测数据
            obs_data = observation_data[var]
            im1 = axes[0, i].contourf(
                obs_data.longitude, obs_data.latitude, obs_data.values,
                levels=20, cmap='viridis'
            )
            axes[0, i].set_title(f'Observed {var.title()}')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 预测数据
            pred_data = self._generate_prediction_at_time(prediction_time, var)
            im2 = axes[1, i].contourf(
                pred_data.longitude, pred_data.latitude, pred_data.values,
                levels=20, cmap='viridis'
            )
            axes[1, i].set_title(f'Predicted {var.title()}')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.tight_layout()
        return fig
    
    def create_uncertainty_plot(self, 
                              prediction_time: float,
                              variable: str = 'thickness') -> plt.Figure:
        """创建不确定性可视化"""
        if not hasattr(self.model, 'predict_with_uncertainty'):
            raise ValueError("模型不支持不确定性预测")
        
        # 生成预测网格
        grid_points = self._generate_prediction_grid()
        
        # 添加时间维度
        time_grid = torch.full((grid_points.shape[0], 1), prediction_time)
        full_grid = torch.cat([grid_points, time_grid], dim=1)
        
        # 预测均值和不确定性
        with torch.no_grad():
            mean_pred, uncertainty = self.model.predict_with_uncertainty(full_grid)
        
        # 重塑为网格形状
        grid_shape = self._get_grid_shape()
        mean_reshaped = mean_pred[:, 0].reshape(grid_shape)  # 假设thickness是第一个输出
        uncertainty_reshaped = uncertainty[:, 0].reshape(grid_shape)
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 均值预测
        im1 = axes[0].contourf(mean_reshaped, levels=20, cmap='viridis')
        axes[0].set_title('Mean Prediction')
        plt.colorbar(im1, ax=axes[0])
        
        # 不确定性
        im2 = axes[1].contourf(uncertainty_reshaped, levels=20, cmap='Reds')
        axes[1].set_title('Prediction Uncertainty')
        plt.colorbar(im2, ax=axes[1])
        
        # 变异系数
        cv = uncertainty_reshaped / (mean_reshaped + 1e-8)
        im3 = axes[2].contourf(cv, levels=20, cmap='plasma')
        axes[2].set_title('Coefficient of Variation')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig
```

## 测试策略 - Testing Strategy

### 1. 单元测试

```python
# tests/unit_tests/test_physics_laws.py
import pytest
import torch
from model_architecture.glacier_physics.mass_conservation import MassConservation

class TestMassConservation:
    """质量守恒定律测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mass_conservation = MassConservation()
        self.test_coords = torch.randn(100, 4, requires_grad=True)
        self.test_fields = {
            'velocity': torch.randn(100, 2, requires_grad=True),
            'thickness': torch.randn(100, 1, requires_grad=True)
        }
    
    def test_residual_computation(self):
        """测试残差计算"""
        residual = self.mass_conservation.compute_residual(
            self.test_fields, self.test_coords, {}
        )
        
        assert residual.shape == (100,)
        assert not torch.isnan(residual).any()
        assert residual.requires_grad
    
    def test_conservation_property(self):
        """测试守恒性质"""
        # 创建满足守恒的测试场
        conserved_fields = self._create_conserved_fields()
        
        residual = self.mass_conservation.compute_residual(
            conserved_fields, self.test_coords, {}
        )
        
        # 守恒场的残差应该接近零
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
    
    def test_gradient_flow(self):
        """测试梯度流"""
        residual = self.mass_conservation.compute_residual(
            self.test_fields, self.test_coords, {}
        )
        
        loss = residual.sum()
        loss.backward()
        
        # 检查梯度是否正确计算
        assert self.test_coords.grad is not None
        assert not torch.isnan(self.test_coords.grad).any()
    
    def _create_conserved_fields(self):
        """创建满足守恒的测试场"""
        # 实现创建满足质量守恒的速度和厚度场
        pass
```

### 2. 集成测试

```python
# tests/integration_tests/test_training_pipeline.py
import pytest
import torch
from model_architecture.advanced_architectures import PIKANModel
from training_framework.training_stages import ProgressiveTrainer
from data_management.preprocessing import SyntheticDataGenerator

class TestTrainingPipeline:
    """训练流水线集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = {
            'model': {
                'input_dim': 4,
                'output_dim': 3,
                'kan_layers': [32, 64, 32]
            },
            'training': {
                'batch_size': 1000,
                'learning_rate': 1e-3,
                'epochs': {'stage1': 100, 'stage2': 50, 'stage3': 50}
            }
        }
        
        self.model = PIKANModel(**self.config['model'])
        self.data_generator = SyntheticDataGenerator()
    
    def test_full_training_pipeline(self):
        """测试完整训练流水线"""
        # 生成合成数据
        synthetic_data = self.data_generator.generate_glacier_data()
        
        # 创建训练器
        trainer = ProgressiveTrainer(self.model, self.config)
        
        # 运行训练
        results = trainer.run_three_stage_training()
        
        # 验证结果
        assert 'stage1' in results
        assert 'stage2' in results
        assert 'stage3' in results
        
        # 检查损失是否下降
        for stage in ['stage1', 'stage2', 'stage3']:
            loss_history = results[stage]['loss_history']
            assert len(loss_history) > 0
            assert loss_history[-1] < loss_history[0]  # 损失应该下降
    
    def test_model_convergence(self):
        """测试模型收敛性"""
        trainer = ProgressiveTrainer(self.model, self.config)
        
        # 运行较长时间的训练
        extended_config = self.config.copy()
        extended_config['training']['epochs'] = {
            'stage1': 500, 'stage2': 300, 'stage3': 200
        }
        
        trainer.config = extended_config
        results = trainer.run_three_stage_training()
        
        # 检查是否收敛
        final_loss = results['stage3']['final_loss']
        assert final_loss < 1e-3  # 期望的收敛阈值
```

### 3. 性能测试

```python
# tests/performance_tests/test_model_performance.py
import pytest
import torch
import time
from model_architecture.advanced_architectures import PIKANModel, PIANNModel

class TestModelPerformance:
    """模型性能测试"""
    
    @pytest.mark.parametrize("model_class", [PIKANModel, PIANNModel])
    @pytest.mark.parametrize("batch_size", [1000, 5000, 10000])
    def test_forward_pass_performance(self, model_class, batch_size):
        """测试前向传播性能"""
        model = model_class(input_dim=4, output_dim=3, hidden_layers=[64, 128, 64])
        input_data = torch.randn(batch_size, 4)
        
        # 预热
        for _ in range(10):
            _ = model(input_data)
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            output = model(input_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size / avg_time
        
        print(f"{model_class.__name__} - Batch size: {batch_size}, "
              f"Throughput: {throughput:.0f} samples/sec")
        
        # 性能断言
        assert throughput > 1000  # 最低吞吐量要求
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大模型
        model = PIKANModel(input_dim=4, output_dim=3, kan_layers=[256, 512, 256])
        large_input = torch.randn(50000, 4)
        
        # 前向传播
        output = model(large_input)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # 内存使用断言
        assert memory_increase < 2000  # 最大内存增长限制
```

## 代码质量保证 - Code Quality Assurance

### 1. 代码风格检查

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile, black]
```

### 2. 文档生成

```python
# docs/generate_docs.py
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_api_docs():
    """生成API文档"""
    import sphinx
    from sphinx.cmd.build import build_main
    
    # Sphinx配置
    source_dir = "docs/source"
    build_dir = "docs/build"
    
    # 构建文档
    build_main(['-b', 'html', source_dir, build_dir])

def generate_coverage_report():
    """生成测试覆盖率报告"""
    os.system("pytest --cov=. --cov-report=html tests/")

if __name__ == "__main__":
    generate_api_docs()
    generate_coverage_report()
```

### 3. 持续集成

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
    
    - name: Run linting
      run: |
        flake8 .
        black --check .
        mypy .
```

## 部署和发布 - Deployment and Release

### 1. 版本管理

```python
# scripts/release.py
import subprocess
import sys
from pathlib import Path

def bump_version(version_type: str):
    """更新版本号"""
    valid_types = ['patch', 'minor', 'major']
    if version_type not in valid_types:
        raise ValueError(f"版本类型必须是: {valid_types}")
    
    # 使用bumpversion工具
    subprocess.run(["bumpversion", version_type], check=True)

def create_release():
    """创建发布"""
    # 运行测试
    subprocess.run(["pytest", "tests/"], check=True)
    
    # 构建包
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], check=True)
    
    # 上传到PyPI
    subprocess.run(["twine", "upload", "dist/*"], check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python release.py [patch|minor|major]")
        sys.exit(1)
    
    version_type = sys.argv[1]
    bump_version(version_type)
    create_release()
```

### 2. Docker部署

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 安装项目
RUN pip3 install -e .

# 暴露端口
EXPOSE 8000 8501

# 设置入口点
CMD ["python3", "main_experiment.py"]
```

---

这份开发指南提供了项目开发的完整框架和最佳实践。开发者可以根据具体需求扩展和定制各个模块。