# 用户指南 - User Guide

## 快速开始 - Quick Start

### 1. 环境准备

确保您已经按照 [安装指南](INSTALLATION.md) 完成了环境配置。

```bash
# 激活环境
conda activate tibetan_glacier_pinns

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import jax; print(f'JAX版本: {jax.__version__}')"
```

### 2. 数据准备

#### 2.1 下载数据集

```bash
# 创建数据目录
mkdir -p data_management/raw_data

# 下载RGI 6.0数据
wget -P data_management/raw_data/rgi_6.0/ https://www.glims.org/RGI/rgi60_files/00_rgi60.zip

# 下载Farinotti 2019厚度数据
wget -P data_management/raw_data/farinotti_2019/ https://www.research-collection.ethz.ch/handle/20.500.11850/315707

# 下载其他数据集...
```

#### 2.2 数据预处理

```python
from data_management.preprocessing import (
    RGIProcessor, FarinottiProcessor, MillanProcessor,
    HugonnetProcessor, DussaillantProcessor, SpatialAlignment
)

# 处理RGI数据
rgi_processor = RGIProcessor(
    data_path='data_management/raw_data/rgi_6.0',
    output_path='data_management/processed_data'
)
rgi_data = rgi_processor.load_rgi_data(region_ids=[13, 14, 15])  # 青藏高原区域
rgi_filtered = rgi_processor.filter_by_area(min_area=0.01)  # 过滤小冰川

# 处理厚度数据
far_processor = FarinottiProcessor(
    data_path='data_management/raw_data/farinotti_2019',
    output_path='data_management/processed_data'
)
thickness_data = far_processor.load_thickness_data()

# 空间对齐
aligner = SpatialAlignment(target_crs='EPSG:4326', target_resolution=0.001)
aligned_datasets = aligner.align_datasets([rgi_data, thickness_data])
```

### 3. 模型配置

#### 3.1 创建配置文件

```yaml
# experiments/experiment_configs/my_experiment.yml
experiment_name: "tibetan_glacier_evolution_v1"
random_seed: 42

# 模型配置
model:
  type: "PIKAN"  # 可选: PIKAN, PIANN, BPINN
  input_dim: 4   # x, y, z, t
  output_dim: 3  # thickness, velocity_x, velocity_y
  hidden_layers: [64, 128, 256, 128, 64]
  activation: "tanh"
  
# 训练配置
training:
  batch_size: 10000
  learning_rate: 1e-3
  epochs:
    stage1: 5000  # 长期趋势
    stage2: 3000  # 短期动态
    stage3: 2000  # 耦合优化
  
# 物理约束权重
physics:
  mass_conservation_weight: 1.0
  momentum_balance_weight: 1.0
  ice_flow_law_weight: 0.5
  boundary_condition_weight: 2.0

# 数据配置
data:
  time_range: [1980, 2020]
  spatial_bounds: [70, 25, 105, 40]  # 青藏高原边界
  validation_split: 0.2
```

#### 3.2 模型创建

```python
from model_architecture.advanced_architectures import PIKANModel, PIANNModel, BPINNModel
from model_architecture.core_pinns import BasePINN

# 创建PIKAN模型
model = PIKANModel(
    input_dim=4,
    output_dim=3,
    kan_layers=[64, 128, 256, 128, 64],
    spline_order=3
)

# 或创建PIANN模型
# model = PIANNModel(
#     input_dim=4,
#     output_dim=3,
#     attention_heads=8,
#     attention_layers=4
# )
```

### 4. 训练模型

#### 4.1 三阶段训练

```python
from training_framework.training_stages import ProgressiveTrainer
from training_framework.sampling_strategies import AdaptiveSampling
from training_framework.optimization import AdaptiveOptimizers

# 创建训练器
trainer = ProgressiveTrainer(model, config)

# 运行三阶段训练
print("开始三阶段训练...")
results = trainer.run_three_stage_training()

print(f"阶段1损失: {results['stage1']['final_loss']:.6f}")
print(f"阶段2损失: {results['stage2']['final_loss']:.6f}")
print(f"阶段3损失: {results['stage3']['final_loss']:.6f}")
```

#### 4.2 监控训练过程

```python
import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(results['stage1']['loss_history'])
plt.title('Stage 1: Long-term Trends')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(results['stage2']['loss_history'])
plt.title('Stage 2: Short-term Dynamics')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(results['stage3']['loss_history'])
plt.title('Stage 3: Coupled Optimization')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```

### 5. 模型验证

#### 5.1 物理验证

```python
from validation_testing.physics_validation import PhysicsValidator
from validation_testing.performance_metrics import AccuracyMetrics

# 创建验证器
validator = PhysicsValidator(model, config)

# 运行物理验证
physics_results = validator.run_comprehensive_validation()

print("物理验证结果:")
print(f"质量守恒误差: {physics_results['mass_conservation_error']:.6f}")
print(f"动量平衡误差: {physics_results['momentum_balance_error']:.6f}")
print(f"能量平衡误差: {physics_results['energy_balance_error']:.6f}")
```

#### 5.2 交叉验证

```python
from validation_testing.cross_validation import (
    TemporalHoldout, SpatialHoldout, GlacierWiseValidation
)

# 时间维度交叉验证
temporal_validator = TemporalHoldout(test_years=[2015, 2016, 2017, 2018, 2019])
temporal_results = temporal_validator.validate(model, data)

# 空间维度交叉验证
spatial_validator = SpatialHoldout(test_regions=['region_13_subset'])
spatial_results = spatial_validator.validate(model, data)

print(f"时间验证RMSE: {temporal_results['rmse']:.4f}")
print(f"空间验证RMSE: {spatial_results['rmse']:.4f}")
```

### 6. 结果分析

#### 6.1 空间分析

```python
from analysis_visualization.spatial_analysis import SpatialAnalyzer
import numpy as np

# 创建分析器
analyzer = SpatialAnalyzer(model, data)

# 生成演化地图
time_points = np.arange(1980, 2021, 5)
evolution_maps = analyzer.generate_evolution_maps(time_points)

# 区域对比分析
regional_analysis = analyzer.analyze_regional_differences()
print("区域分析结果:")
for region, stats in regional_analysis.items():
    print(f"{region}: 平均变化率 {stats['mean_change_rate']:.4f} m/year")

# 高程带分析
elevation_bands = [(3000, 4000), (4000, 5000), (5000, 6000), (6000, 7000)]
elevation_analysis = analyzer.elevation_zone_analysis(elevation_bands)
print(elevation_analysis)
```

#### 6.2 时间序列分析

```python
from analysis_visualization.temporal_analysis import (
    TrendAnalysis, SeasonalPatterns, InterannualVariability
)

# 趋势分析
trend_analyzer = TrendAnalysis(model, data)
trend_results = trend_analyzer.analyze_long_term_trends()

# 季节模式分析
seasonal_analyzer = SeasonalPatterns(model, data)
seasonal_results = seasonal_analyzer.extract_seasonal_cycles()

# 年际变异分析
variability_analyzer = InterannualVariability(model, data)
variability_results = variability_analyzer.analyze_interannual_patterns()
```

#### 6.3 不确定性分析

```python
from analysis_visualization.uncertainty_analysis import (
    PredictionIntervals, SensitivityAnalysis, ErrorPropagation
)

# 预测区间
interval_analyzer = PredictionIntervals(model, data)
confidence_intervals = interval_analyzer.compute_prediction_intervals(confidence=0.95)

# 敏感性分析
sensitivity_analyzer = SensitivityAnalysis(model, data)
sensitivity_results = sensitivity_analyzer.analyze_parameter_sensitivity()

# 误差传播
error_analyzer = ErrorPropagation(model, data)
error_propagation = error_analyzer.propagate_input_uncertainties()
```

### 7. 交互式可视化

#### 7.1 启动Web仪表板

```python
from analysis_visualization.interactive_visualization import WebDashboard

# 创建仪表板
dashboard = WebDashboard(model, data)

# 启动Streamlit应用
dashboard.run_dashboard(port=8501)
```

访问 http://localhost:8501 查看交互式仪表板。

#### 7.2 3D可视化

```python
from analysis_visualization.interactive_visualization import GlacierViewer3D

# 创建3D查看器
viewer = GlacierViewer3D(model, data)

# 生成3D可视化
fig = viewer.create_3d_glacier_evolution()
fig.show()
```

### 8. 模型部署

#### 8.1 模型导出

```python
# 保存训练好的模型
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'training_results': results
}, 'experiments/checkpoints/trained_model.pth')
```

#### 8.2 推理服务

```python
from deployment_application.model_deployment import InferenceEngine, APIService

# 创建推理引擎
engine = InferenceEngine(
    model_path='experiments/checkpoints/trained_model.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 创建API服务
api_service = APIService(engine)

# 启动API服务
api_service.run(host='0.0.0.0', port=8000)
```

#### 8.3 批量预测

```python
# 准备预测数据
predict_coords = np.array([
    [90.0, 30.0, 4000.0, 2025.0],  # 经度, 纬度, 高程, 时间
    [91.0, 31.0, 4500.0, 2025.0],
    # ... 更多坐标点
])

# 进行预测
predictions = engine.batch_predict(predict_coords)

# 解析结果
thickness_pred = predictions[:, 0]
velocity_x_pred = predictions[:, 1]
velocity_y_pred = predictions[:, 2]

print(f"预测厚度范围: {thickness_pred.min():.2f} - {thickness_pred.max():.2f} m")
print(f"预测速度范围: {np.sqrt(velocity_x_pred**2 + velocity_y_pred**2).max():.2f} m/year")
```

## 高级用法 - Advanced Usage

### 1. 自定义物理约束

```python
from model_architecture.core_pinns import PhysicsLaws

class CustomPhysicsLaws(PhysicsLaws):
    @staticmethod
    def custom_constraint(velocity, thickness, temperature, coords):
        """自定义物理约束"""
        # 实现自定义物理定律
        return constraint_residual

# 在模型中使用自定义约束
model.add_physics_constraint('custom', CustomPhysicsLaws.custom_constraint, weight=0.5)
```

### 2. 多GPU训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式训练
dist.init_process_group(backend='nccl')

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 使用分布式采样器
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
```

### 3. 混合精度训练

```python
from torch.cuda.amp import GradScaler, autocast

# 创建梯度缩放器
scaler = GradScaler()

# 在训练循环中使用混合精度
with autocast():
    predictions = model(input_data)
    loss = compute_loss(predictions, targets)

# 缩放损失并反向传播
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 实验跟踪

```python
import wandb

# 初始化Weights & Biases
wandb.init(
    project="tibetan-glacier-pinns",
    config=config,
    name=experiment_name
)

# 记录训练指标
wandb.log({
    'epoch': epoch,
    'loss': loss.item(),
    'physics_loss': physics_loss.item(),
    'data_loss': data_loss.item()
})

# 记录模型图
wandb.watch(model)
```

### 5. 超参数优化

```python
import optuna

def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [5000, 10000, 20000])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)
    
    # 创建模型和训练器
    model = PIKANModel(input_dim=4, output_dim=3, kan_layers=[hidden_dim]*4)
    trainer = ProgressiveTrainer(model, config)
    
    # 训练并返回验证损失
    results = trainer.run_three_stage_training()
    return results['validation_loss']

# 运行超参数优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## 故障排除 - Troubleshooting

### 常见问题及解决方案

#### 1. 训练不收敛

**问题**: 损失函数不下降或震荡

**解决方案**:
```python
# 调整学习率
config['training']['learning_rate'] = 1e-4  # 降低学习率

# 调整物理约束权重
config['physics']['mass_conservation_weight'] = 0.1  # 降低物理约束权重

# 使用学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)
```

#### 2. 内存不足

**问题**: CUDA out of memory

**解决方案**:
```python
# 减小批次大小
config['training']['batch_size'] = 5000

# 启用梯度检查点
config['training']['gradient_checkpointing'] = True

# 使用混合精度
config['training']['use_amp'] = True
```

#### 3. 物理约束违反

**问题**: 模型预测不满足物理定律

**解决方案**:
```python
# 增加物理约束权重
config['physics']['mass_conservation_weight'] = 2.0

# 使用更多的物理采样点
config['sampling']['physics_points'] = 50000

# 添加额外的物理约束
model.add_physics_constraint('energy_balance', energy_balance_law, weight=1.0)
```

#### 4. 数据质量问题

**问题**: 数据中存在异常值或缺失值

**解决方案**:
```python
from data_management.preprocessing import QualityControl

# 运行质量控制
qc = QualityControl(config)
outliers = qc.detect_outliers(data, method='isolation_forest')
data_cleaned = qc.remove_outliers(data, outliers)

# 插值缺失值
data_filled = qc.interpolate_missing_values(data_cleaned, method='kriging')
```

### 性能优化建议

1. **数据预处理优化**:
   - 使用Dask进行大数据处理
   - 预计算常用的派生变量
   - 使用HDF5格式存储处理后的数据

2. **模型训练优化**:
   - 使用自适应采样减少计算量
   - 实施早停机制避免过拟合
   - 使用模型检查点保存训练进度

3. **推理优化**:
   - 使用TorchScript或ONNX优化模型
   - 实施模型量化减少内存使用
   - 使用批量推理提高吞吐量

## 最佳实践 - Best Practices

### 1. 实验管理

- 为每个实验创建唯一的配置文件
- 使用版本控制跟踪代码变更
- 记录详细的实验日志和结果
- 定期备份重要的模型检查点

### 2. 代码质量

- 遵循PEP 8代码风格规范
- 编写全面的单元测试
- 使用类型注解提高代码可读性
- 定期进行代码审查

### 3. 数据管理

- 保持原始数据的完整性
- 记录所有数据处理步骤
- 实施数据版本控制
- 定期验证数据质量

### 4. 模型开发

- 从简单模型开始逐步增加复杂性
- 定期验证物理约束的满足情况
- 使用多种验证方法评估模型性能
- 保持模型的可解释性

---

更多详细信息请参考 [API参考文档](API_REFERENCE.md) 和 [开发指南](DEVELOPER_GUIDE.md)。