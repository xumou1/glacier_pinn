# 安装指南 - Installation Guide

## 系统要求 - System Requirements

### 最低配置 - Minimum Requirements
- **CPU**: 16核 Intel/AMD 处理器
- **内存**: 64GB RAM
- **GPU**: NVIDIA RTX 3080 或同等性能
- **存储**: 2TB NVMe SSD
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+

### 推荐配置 - Recommended Configuration
- **CPU**: 32核 Intel Xeon / AMD EPYC
- **内存**: 128GB RAM
- **GPU**: NVIDIA A100 / RTX 4090 (多卡)
- **存储**: 4TB NVMe SSD
- **网络**: 高速网络连接（用于数据下载）

## 安装方式 - Installation Methods

### 方式1: Conda环境安装 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/glacier-research/tibetan-glacier-pinns.git
cd tibetan-glacier-pinns

# 2. 创建Conda环境
conda env create -f environment.yml
conda activate tibetan_glacier_pinns

# 3. 安装项目
pip install -e .

# 4. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import jax; print(f'JAX版本: {jax.__version__}')"
```

### 方式2: Docker安装

```bash
# 1. 克隆项目
git clone https://github.com/glacier-research/tibetan-glacier-pinns.git
cd tibetan-glacier-pinns

# 2. 构建Docker镜像
docker-compose build

# 3. 启动服务
docker-compose up -d

# 4. 验证安装
docker-compose exec glacier-pinns-training python -c "import torch; print('安装成功')"
```

### 方式3: 手动安装

```bash
# 1. 创建Python虚拟环境
python -m venv glacier_env
source glacier_env/bin/activate  # Linux/macOS
# 或 glacier_env\Scripts\activate  # Windows

# 2. 升级pip
pip install --upgrade pip

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装项目
pip install -e .
```

## GPU支持配置 - GPU Support Configuration

### NVIDIA GPU配置

```bash
# 1. 检查CUDA版本
nvidia-smi

# 2. 安装对应的PyTorch版本
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 验证GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

### JAX GPU配置

```bash
# 安装JAX GPU版本
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 验证JAX GPU
python -c "import jax; print(f'JAX设备: {jax.devices()}')"
```

## 数据准备 - Data Preparation

### 创建数据目录结构

```bash
# 创建必要的数据目录
mkdir -p data_management/raw_data/{rgi_6.0,farinotti_2019,millan_2022,hugonnet_2021,dussaillant_2025,auxiliary_data}
mkdir -p data_management/processed_data/{aligned_grids,temporal_series,quality_controlled,training_ready}
mkdir -p data_management/validation_data/{field_observations,independent_satellite,cross_validation}
```

### 数据下载脚本

```bash
# 运行数据下载脚本（需要实现）
python data_management/preprocessing/download_datasets.py
```

## 配置文件设置 - Configuration Setup

### 创建实验配置

```bash
# 复制默认配置
cp experiments/experiment_configs/default_config.yml experiments/experiment_configs/my_experiment.yml

# 编辑配置文件
vim experiments/experiment_configs/my_experiment.yml
```

### 环境变量设置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export GLACIER_PROJECT_ROOT=/path/to/tibetan-glacier-pinns
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=your_wandb_key  # 可选：用于实验跟踪
```

## 验证安装 - Installation Verification

### 运行测试套件

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit_tests/ -v
pytest tests/integration_tests/ -v
```

### 快速功能测试

```bash
# 测试数据处理
python -m data_management.preprocessing.quality_control --test

# 测试模型架构
python -m model_architecture.core_pinns.base_pinn --test

# 测试训练框架
python -m training_framework.training_stages.progressive_trainer --test
```

## 常见问题 - Troubleshooting

### 1. CUDA相关问题

```bash
# 问题：CUDA版本不匹配
# 解决：重新安装对应版本的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足问题

```bash
# 问题：训练时内存不足
# 解决：调整批次大小和数据加载参数
# 在配置文件中设置：
# batch_size: 32  # 减小批次大小
# num_workers: 4  # 减少数据加载进程
```

### 3. 依赖冲突问题

```bash
# 问题：包版本冲突
# 解决：创建新的干净环境
conda deactivate
conda env remove -n tibetan_glacier_pinns
conda env create -f environment.yml
```

### 4. 数据下载问题

```bash
# 问题：数据下载失败
# 解决：检查网络连接，使用代理或手动下载
export https_proxy=http://proxy.example.com:8080
export http_proxy=http://proxy.example.com:8080
```

## 性能优化 - Performance Optimization

### 1. 多GPU配置

```python
# 在配置文件中设置
gpu_config:
  use_multi_gpu: true
  gpu_ids: [0, 1, 2, 3]
  strategy: "ddp"  # 分布式数据并行
```

### 2. 内存优化

```python
# 启用混合精度训练
training_config:
  use_amp: true  # 自动混合精度
  gradient_checkpointing: true  # 梯度检查点
```

### 3. 数据加载优化

```python
# 优化数据加载
data_config:
  num_workers: 8  # 数据加载进程数
  pin_memory: true  # 固定内存
  prefetch_factor: 2  # 预取因子
```

## 开发环境设置 - Development Environment

### 代码格式化

```bash
# 安装开发工具
pip install black flake8 mypy pre-commit

# 设置pre-commit钩子
pre-commit install

# 格式化代码
black .
flake8 .
mypy .
```

### Jupyter环境

```bash
# 安装Jupyter内核
python -m ipykernel install --user --name=tibetan_glacier_pinns

# 启动Jupyter Lab
jupyter lab
```

## 更新和维护 - Updates and Maintenance

### 更新项目

```bash
# 拉取最新代码
git pull origin main

# 更新依赖
conda env update -f environment.yml
pip install -e . --upgrade
```

### 清理缓存

```bash
# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 清理实验缓存
rm -rf experiments/logs/temp_*
rm -rf experiments/checkpoints/temp_*
```

## 支持和帮助 - Support and Help

- **文档**: 查看 `documentation/` 目录下的其他文档
- **问题报告**: 在GitHub Issues中提交问题
- **讨论**: 参与GitHub Discussions
- **邮件**: glacier.research@example.com

---

安装完成后，请参考 [用户指南](USER_GUIDE.md) 开始使用项目。