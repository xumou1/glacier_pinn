# API参考文档 - API Reference

## 概述 - Overview

本文档提供了青藏高原冰川PINNs建模项目的完整API参考。项目采用模块化设计，主要包含以下核心模块：

- **数据管理模块** (`data_management`)
- **模型架构模块** (`model_architecture`) 
- **训练框架模块** (`training_framework`)
- **验证测试模块** (`validation_testing`)
- **分析可视化模块** (`analysis_visualization`)
- **部署应用模块** (`deployment_application`)

## 数据管理模块 - Data Management Module

### data_management.preprocessing

#### RGIProcessor

```python
class RGIProcessor:
    """RGI 6.0冰川轮廓数据处理器"""
    
    def __init__(self, data_path: str, output_path: str):
        """初始化RGI处理器
        
        Args:
            data_path: RGI数据路径
            output_path: 输出路径
        """
    
    def load_rgi_data(self, region_ids: List[int] = None) -> gpd.GeoDataFrame:
        """加载RGI数据
        
        Args:
            region_ids: 区域ID列表，None表示加载所有区域
            
        Returns:
            GeoDataFrame: RGI数据
        """
    
    def filter_by_area(self, min_area: float = 0.01) -> gpd.GeoDataFrame:
        """按面积过滤冰川
        
        Args:
            min_area: 最小面积阈值(km²)
            
        Returns:
            GeoDataFrame: 过滤后的数据
        """
    
    def reproject_to_utm(self) -> gpd.GeoDataFrame:
        """重投影到UTM坐标系
        
        Returns:
            GeoDataFrame: 重投影后的数据
        """
```

#### FarinottiProcessor

```python
class FarinottiProcessor:
    """Farinotti 2019冰川厚度数据处理器"""
    
    def __init__(self, data_path: str, output_path: str):
        """初始化Farinotti处理器"""
    
    def load_thickness_data(self, glacier_ids: List[str] = None) -> xr.Dataset:
        """加载厚度数据
        
        Args:
            glacier_ids: 冰川ID列表
            
        Returns:
            Dataset: 厚度数据
        """
    
    def interpolate_missing_values(self, method: str = 'linear') -> xr.Dataset:
        """插值缺失值
        
        Args:
            method: 插值方法
            
        Returns:
            Dataset: 插值后的数据
        """
```

#### SpatialAlignment

```python
class SpatialAlignment:
    """空间对齐处理器"""
    
    def __init__(self, target_crs: str = 'EPSG:4326', target_resolution: float = 0.001):
        """初始化空间对齐器
        
        Args:
            target_crs: 目标坐标系
            target_resolution: 目标分辨率
        """
    
    def align_datasets(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        """对齐多个数据集
        
        Args:
            datasets: 数据集列表
            
        Returns:
            List[Dataset]: 对齐后的数据集
        """
    
    def create_common_grid(self, bounds: Tuple[float, float, float, float]) -> xr.Dataset:
        """创建公共网格
        
        Args:
            bounds: 边界(minx, miny, maxx, maxy)
            
        Returns:
            Dataset: 公共网格
        """
```

### data_management.validation_data

#### QualityControl

```python
class QualityControl:
    """数据质量控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化质量控制器"""
    
    def check_data_completeness(self, data: xr.DataArray) -> Dict[str, float]:
        """检查数据完整性
        
        Args:
            data: 输入数据
            
        Returns:
            Dict: 完整性统计
        """
    
    def detect_outliers(self, data: xr.DataArray, method: str = 'iqr') -> xr.DataArray:
        """检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            DataArray: 异常值掩码
        """
    
    def validate_physical_constraints(self, data: xr.Dataset) -> Dict[str, bool]:
        """验证物理约束
        
        Args:
            data: 输入数据集
            
        Returns:
            Dict: 验证结果
        """
```

## 模型架构模块 - Model Architecture Module

### model_architecture.core_pinns

#### BasePINN

```python
class BasePINN(torch.nn.Module):
    """PINNs基础类"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 hidden_layers: List[int],
                 activation: str = 'tanh'):
        """初始化PINNs模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_layers: 隐藏层配置
            activation: 激活函数
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Tensor: 输出张量
        """
    
    def physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """计算物理损失
        
        Args:
            x: 输入坐标
            y_pred: 预测值
            
        Returns:
            Tensor: 物理损失
        """
    
    def boundary_loss(self, x_boundary: torch.Tensor, y_boundary: torch.Tensor) -> torch.Tensor:
        """计算边界损失
        
        Args:
            x_boundary: 边界坐标
            y_boundary: 边界值
            
        Returns:
            Tensor: 边界损失
        """
```

#### PhysicsLaws

```python
class PhysicsLaws:
    """物理定律实现"""
    
    @staticmethod
    def mass_conservation(velocity: torch.Tensor, 
                         thickness: torch.Tensor,
                         coords: torch.Tensor) -> torch.Tensor:
        """质量守恒定律
        
        Args:
            velocity: 速度场
            thickness: 厚度场
            coords: 坐标
            
        Returns:
            Tensor: 质量守恒残差
        """
    
    @staticmethod
    def momentum_balance(velocity: torch.Tensor,
                        pressure: torch.Tensor, 
                        coords: torch.Tensor) -> torch.Tensor:
        """动量平衡方程
        
        Args:
            velocity: 速度场
            pressure: 压力场
            coords: 坐标
            
        Returns:
            Tensor: 动量平衡残差
        """
    
    @staticmethod
    def ice_flow_law(velocity: torch.Tensor,
                    stress: torch.Tensor,
                    temperature: torch.Tensor) -> torch.Tensor:
        """冰流定律
        
        Args:
            velocity: 速度场
            stress: 应力场
            temperature: 温度场
            
        Returns:
            Tensor: 冰流定律残差
        """
```

### model_architecture.advanced_architectures

#### PIKANModel

```python
class PIKANModel(BasePINN):
    """Physics-Informed Kolmogorov-Arnold Networks"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 kan_layers: List[int],
                 spline_order: int = 3):
        """初始化PIKAN模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            kan_layers: KAN层配置
            spline_order: 样条阶数
        """
    
    def kan_forward(self, x: torch.Tensor) -> torch.Tensor:
        """KAN前向传播"""
    
    def adaptive_spline_update(self, x: torch.Tensor) -> None:
        """自适应样条更新"""
```

#### PIANNModel

```python
class PIANNModel(BasePINN):
    """Physics-Informed Attention Neural Networks"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attention_heads: int = 8,
                 attention_layers: int = 4):
        """初始化PIANN模型"""
    
    def multi_head_attention(self, x: torch.Tensor) -> torch.Tensor:
        """多头注意力机制"""
    
    def physics_aware_attention(self, x: torch.Tensor) -> torch.Tensor:
        """物理感知注意力"""
```

## 训练框架模块 - Training Framework Module

### training_framework.training_stages

#### ProgressiveTrainer

```python
class ProgressiveTrainer:
    """渐进式训练器"""
    
    def __init__(self, model: BasePINN, config: Dict[str, Any]):
        """初始化训练器
        
        Args:
            model: PINNs模型
            config: 训练配置
        """
    
    def stage1_longterm_trends(self) -> Dict[str, float]:
        """阶段1：长期趋势训练
        
        Returns:
            Dict: 训练指标
        """
    
    def stage2_shortterm_dynamics(self) -> Dict[str, float]:
        """阶段2：短期动态训练
        
        Returns:
            Dict: 训练指标
        """
    
    def stage3_coupled_optimization(self) -> Dict[str, float]:
        """阶段3：耦合优化训练
        
        Returns:
            Dict: 训练指标
        """
    
    def run_three_stage_training(self) -> Dict[str, Any]:
        """运行三阶段训练
        
        Returns:
            Dict: 完整训练结果
        """
```

### training_framework.sampling_strategies

#### AdaptiveSampling

```python
class AdaptiveSampling:
    """自适应采样策略"""
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]]):
        """初始化采样器
        
        Args:
            domain_bounds: 域边界
        """
    
    def sample_physics_points(self, n_points: int) -> torch.Tensor:
        """采样物理点
        
        Args:
            n_points: 采样点数
            
        Returns:
            Tensor: 采样点坐标
        """
    
    def sample_boundary_points(self, n_points: int) -> torch.Tensor:
        """采样边界点
        
        Args:
            n_points: 采样点数
            
        Returns:
            Tensor: 边界点坐标
        """
    
    def adaptive_refinement(self, model: BasePINN, threshold: float = 0.1) -> torch.Tensor:
        """自适应细化
        
        Args:
            model: PINNs模型
            threshold: 细化阈值
            
        Returns:
            Tensor: 细化后的采样点
        """
```

### training_framework.optimization

#### AdaptiveOptimizers

```python
class AdaptiveOptimizers:
    """自适应优化器"""
    
    def __init__(self, model_parameters: Iterator[torch.Tensor]):
        """初始化优化器"""
    
    def create_lbfgs_optimizer(self, **kwargs) -> torch.optim.LBFGS:
        """创建L-BFGS优化器"""
    
    def create_adam_optimizer(self, **kwargs) -> torch.optim.Adam:
        """创建Adam优化器"""
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, 
                        scheduler_type: str) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
```

## 验证测试模块 - Validation Testing Module

### validation_testing.physics_validation

#### PhysicsValidator

```python
class PhysicsValidator:
    """物理验证器"""
    
    def __init__(self, model: BasePINN, config: Dict[str, Any]):
        """初始化验证器"""
    
    def validate_conservation_laws(self) -> Dict[str, float]:
        """验证守恒定律
        
        Returns:
            Dict: 验证结果
        """
    
    def validate_energy_balance(self) -> Dict[str, float]:
        """验证能量平衡
        
        Returns:
            Dict: 验证结果
        """
    
    def validate_causality(self) -> Dict[str, bool]:
        """验证因果性
        
        Returns:
            Dict: 验证结果
        """
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行综合验证
        
        Returns:
            Dict: 完整验证结果
        """
```

### validation_testing.performance_metrics

#### AccuracyMetrics

```python
class AccuracyMetrics:
    """精度指标计算器"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算RMSE"""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算MAE"""
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
    
    @staticmethod
    def physics_consistency_score(model: BasePINN, test_points: torch.Tensor) -> float:
        """计算物理一致性分数"""
```

## 分析可视化模块 - Analysis Visualization Module

### analysis_visualization.spatial_analysis

#### SpatialAnalyzer

```python
class SpatialAnalyzer:
    """空间分析器"""
    
    def __init__(self, model: BasePINN, data: xr.Dataset):
        """初始化分析器"""
    
    def generate_evolution_maps(self, time_points: List[float]) -> List[plt.Figure]:
        """生成演化地图
        
        Args:
            time_points: 时间点列表
            
        Returns:
            List[Figure]: 地图列表
        """
    
    def analyze_regional_differences(self) -> Dict[str, Any]:
        """分析区域差异
        
        Returns:
            Dict: 分析结果
        """
    
    def elevation_zone_analysis(self, elevation_bands: List[Tuple[float, float]]) -> pd.DataFrame:
        """高程带分析
        
        Args:
            elevation_bands: 高程带定义
            
        Returns:
            DataFrame: 分析结果
        """
```

### analysis_visualization.interactive_visualization

#### WebDashboard

```python
class WebDashboard:
    """Web仪表板"""
    
    def __init__(self, model: BasePINN, data: xr.Dataset):
        """初始化仪表板"""
    
    def create_streamlit_app(self) -> None:
        """创建Streamlit应用"""
    
    def create_plotly_dashboard(self) -> plotly.graph_objects.Figure:
        """创建Plotly仪表板"""
    
    def run_dashboard(self, port: int = 8501) -> None:
        """运行仪表板"""
```

## 部署应用模块 - Deployment Application Module

### deployment_application.model_deployment

#### InferenceEngine

```python
class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """初始化推理引擎
        
        Args:
            model_path: 模型路径
            device: 计算设备
        """
    
    def load_model(self) -> BasePINN:
        """加载模型
        
        Returns:
            BasePINN: 加载的模型
        """
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """进行预测
        
        Args:
            input_data: 输入数据
            
        Returns:
            ndarray: 预测结果
        """
    
    def batch_predict(self, input_batch: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """批量预测
        
        Args:
            input_batch: 输入批次
            batch_size: 批次大小
            
        Returns:
            ndarray: 预测结果
        """
```

#### APIService

```python
from fastapi import FastAPI

class APIService:
    """API服务"""
    
    def __init__(self, inference_engine: InferenceEngine):
        """初始化API服务"""
        self.app = FastAPI()
        self.engine = inference_engine
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置路由"""
    
    async def predict_endpoint(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """预测端点"""
    
    async def health_check(self) -> Dict[str, str]:
        """健康检查端点"""
    
    def run(self, host: str = '0.0.0.0', port: int = 8000) -> None:
        """运行API服务"""
```

## 使用示例 - Usage Examples

### 基本使用流程

```python
# 1. 数据预处理
from data_management.preprocessing import RGIProcessor, SpatialAlignment

rgi_processor = RGIProcessor('data/rgi_6.0', 'data/processed')
rgi_data = rgi_processor.load_rgi_data(region_ids=[13, 14, 15])
rgi_data = rgi_processor.filter_by_area(min_area=0.1)

# 2. 模型创建
from model_architecture.core_pinns import BasePINN
from model_architecture.advanced_architectures import PIKANModel

model = PIKANModel(
    input_dim=4,  # x, y, z, t
    output_dim=3,  # thickness, velocity_x, velocity_y
    kan_layers=[64, 128, 128, 64]
)

# 3. 训练
from training_framework.training_stages import ProgressiveTrainer

trainer = ProgressiveTrainer(model, config)
results = trainer.run_three_stage_training()

# 4. 验证
from validation_testing.physics_validation import PhysicsValidator

validator = PhysicsValidator(model, config)
validation_results = validator.run_comprehensive_validation()

# 5. 分析
from analysis_visualization.spatial_analysis import SpatialAnalyzer

analyzer = SpatialAnalyzer(model, data)
maps = analyzer.generate_evolution_maps([2000, 2010, 2020])

# 6. 部署
from deployment_application.model_deployment import InferenceEngine, APIService

engine = InferenceEngine('models/trained_model.pth')
api = APIService(engine)
api.run()
```

### 高级配置示例

```python
# 多GPU训练配置
config = {
    'model': {
        'type': 'PIKAN',
        'input_dim': 4,
        'output_dim': 3,
        'kan_layers': [64, 128, 256, 128, 64]
    },
    'training': {
        'use_multi_gpu': True,
        'gpu_ids': [0, 1, 2, 3],
        'batch_size': 10000,
        'learning_rate': 1e-3,
        'epochs': {
            'stage1': 5000,
            'stage2': 3000,
            'stage3': 2000
        }
    },
    'physics': {
        'mass_conservation_weight': 1.0,
        'momentum_balance_weight': 1.0,
        'ice_flow_law_weight': 0.5
    }
}
```

## 错误处理 - Error Handling

### 常见异常类型

```python
class GlacierPINNsError(Exception):
    """项目基础异常类"""
    pass

class DataProcessingError(GlacierPINNsError):
    """数据处理异常"""
    pass

class ModelTrainingError(GlacierPINNsError):
    """模型训练异常"""
    pass

class ValidationError(GlacierPINNsError):
    """验证异常"""
    pass

class PhysicsConstraintError(GlacierPINNsError):
    """物理约束异常"""
    pass
```

### 异常处理示例

```python
try:
    model = PIKANModel(input_dim=4, output_dim=3)
    trainer = ProgressiveTrainer(model, config)
    results = trainer.run_three_stage_training()
except ModelTrainingError as e:
    logging.error(f"模型训练失败: {e}")
    # 处理训练失败的情况
except PhysicsConstraintError as e:
    logging.error(f"物理约束违反: {e}")
    # 调整物理约束权重
except Exception as e:
    logging.error(f"未知错误: {e}")
    # 通用错误处理
```

---

更多详细信息请参考各模块的源代码和单元测试。