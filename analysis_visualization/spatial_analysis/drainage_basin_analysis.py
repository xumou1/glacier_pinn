#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流域分析模块

该模块实现了冰川流域分析功能，包括流域划分、流域特征分析、
流域间对比、水文分析等。

主要功能:
- 流域自动划分
- 流域特征提取
- 流域间对比分析
- 水文特征分析
- 流域网络分析
- 流域可视化

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats, ndimage
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter, label, binary_fill_holes
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasinDelineationMethod(Enum):
    """流域划分方法"""
    WATERSHED = "watershed"                    # 分水岭算法
    FLOW_ACCUMULATION = "flow_accumulation"    # 流量累积
    TOPOGRAPHIC = "topographic"                # 地形分析
    CLUSTERING = "clustering"                  # 聚类方法
    MANUAL = "manual"                          # 手动划分
    HYBRID = "hybrid"                          # 混合方法

class BasinCharacteristic(Enum):
    """流域特征"""
    AREA = "area"                              # 面积
    PERIMETER = "perimeter"                    # 周长
    LENGTH = "length"                          # 长度
    WIDTH = "width"                            # 宽度
    ELEVATION_RANGE = "elevation_range"        # 高程范围
    MEAN_ELEVATION = "mean_elevation"          # 平均高程
    SLOPE = "slope"                            # 平均坡度
    ASPECT = "aspect"                          # 主要坡向
    COMPACTNESS = "compactness"                # 紧凑度
    ELONGATION = "elongation"                  # 伸长率
    CIRCULARITY = "circularity"                # 圆形度
    DRAINAGE_DENSITY = "drainage_density"      # 河网密度

class HydrologicalMetric(Enum):
    """水文指标"""
    FLOW_ACCUMULATION = "flow_accumulation"    # 流量累积
    FLOW_DIRECTION = "flow_direction"          # 流向
    STREAM_ORDER = "stream_order"              # 河流等级
    CHANNEL_LENGTH = "channel_length"          # 河道长度
    CHANNEL_GRADIENT = "channel_gradient"      # 河道坡度
    CONFLUENCE_COUNT = "confluence_count"      # 汇流点数量
    OUTLET_ELEVATION = "outlet_elevation"      # 出口高程
    RELIEF_RATIO = "relief_ratio"              # 地势比

class AnalysisType(Enum):
    """分析类型"""
    MORPHOMETRIC = "morphometric"              # 形态计量
    HYDROLOGICAL = "hydrological"              # 水文分析
    COMPARATIVE = "comparative"                # 对比分析
    NETWORK = "network"                        # 网络分析
    TEMPORAL = "temporal"                      # 时间分析
    STATISTICAL = "statistical"                # 统计分析

class VisualizationType(Enum):
    """可视化类型"""
    BASIN_MAP = "basin_map"                    # 流域地图
    DRAINAGE_NETWORK = "drainage_network"      # 河网图
    ELEVATION_PROFILE = "elevation_profile"    # 高程剖面
    HYPSOMETRIC_CURVE = "hypsometric_curve"    # 高程曲线
    SCATTER_MATRIX = "scatter_matrix"          # 散点矩阵
    COMPARISON_CHART = "comparison_chart"      # 对比图表
    NETWORK_GRAPH = "network_graph"            # 网络图
    FLOW_MAP = "flow_map"                      # 流向图

@dataclass
class DrainageBasin:
    """流域定义"""
    basin_id: int                              # 流域编号
    name: str                                  # 流域名称
    boundary: np.ndarray                       # 边界坐标
    outlet: Tuple[float, float]                # 出口坐标
    area: float = 0.0                          # 面积(km²)
    characteristics: Dict[str, float] = field(default_factory=dict)  # 特征值
    hydrological_metrics: Dict[str, float] = field(default_factory=dict)  # 水文指标
    sub_basins: List['DrainageBasin'] = field(default_factory=list)  # 子流域
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

@dataclass
class DrainageConfig:
    """流域分析配置"""
    # 划分设置
    delineation_method: BasinDelineationMethod = BasinDelineationMethod.WATERSHED
    min_basin_area: float = 1.0                # 最小流域面积(km²)
    flow_threshold: float = 100.0              # 流量阈值
    smoothing_factor: float = 1.0              # 平滑因子
    
    # 特征分析
    characteristics: List[BasinCharacteristic] = field(
        default_factory=lambda: [
            BasinCharacteristic.AREA,
            BasinCharacteristic.MEAN_ELEVATION,
            BasinCharacteristic.SLOPE
        ]
    )
    hydrological_metrics: List[HydrologicalMetric] = field(
        default_factory=lambda: [
            HydrologicalMetric.FLOW_ACCUMULATION,
            HydrologicalMetric.STREAM_ORDER
        ]
    )
    
    # 分析设置
    analysis_types: List[AnalysisType] = field(
        default_factory=lambda: [AnalysisType.MORPHOMETRIC, AnalysisType.HYDROLOGICAL]
    )
    include_sub_basins: bool = True            # 包含子流域
    network_analysis: bool = True              # 网络分析
    
    # 空间设置
    spatial_resolution: float = 100.0          # 空间分辨率(米)
    buffer_distance: float = 1000.0            # 缓冲距离
    coordinate_system: str = "EPSG:4326"
    
    # 统计设置
    confidence_level: float = 0.95
    outlier_threshold: float = 3.0
    correlation_method: str = 'pearson'
    
    # 可视化设置
    visualization_types: List[VisualizationType] = field(
        default_factory=lambda: [VisualizationType.BASIN_MAP, VisualizationType.DRAINAGE_NETWORK]
    )
    color_scheme: str = 'Set3'
    figure_size: Tuple[float, float] = (12, 8)
    
    # 输出设置
    save_results: bool = True
    output_format: str = 'png'
    output_dir: str = './drainage_analysis'
    
    # 其他设置
    parallel_processing: bool = False
    memory_efficient: bool = True

@dataclass
class DrainageData:
    """流域数据"""
    # 空间坐标
    x: np.ndarray                              # x坐标
    y: np.ndarray                              # y坐标
    elevation: np.ndarray                      # 高程数据
    
    # 水文数据
    flow_direction: Optional[np.ndarray] = None # 流向
    flow_accumulation: Optional[np.ndarray] = None # 流量累积
    stream_network: Optional[np.ndarray] = None # 河网
    
    # 时间信息
    time: Optional[np.ndarray] = None          # 时间序列
    
    # 其他数据
    data: Dict[str, np.ndarray] = field(default_factory=dict)  # 其他数据
    
    # 元数据
    coordinate_system: str = "EPSG:4326"
    units: str = "meters"
    
    def __post_init__(self):
        # 检查数据维度一致性
        if self.elevation.shape != (len(self.y), len(self.x)):
            raise ValueError("高程数据维度与坐标不匹配")

class BasinDelineator:
    """流域划分器"""
    
    def __init__(self, config: DrainageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def delineate_basins(self, drainage_data: DrainageData) -> List[DrainageBasin]:
        """划分流域"""
        try:
            if self.config.delineation_method == BasinDelineationMethod.WATERSHED:
                return self._watershed_delineation(drainage_data)
            
            elif self.config.delineation_method == BasinDelineationMethod.FLOW_ACCUMULATION:
                return self._flow_accumulation_delineation(drainage_data)
            
            elif self.config.delineation_method == BasinDelineationMethod.TOPOGRAPHIC:
                return self._topographic_delineation(drainage_data)
            
            elif self.config.delineation_method == BasinDelineationMethod.CLUSTERING:
                return self._clustering_delineation(drainage_data)
            
            else:
                return self._watershed_delineation(drainage_data)
        
        except Exception as e:
            self.logger.error(f"流域划分失败: {e}")
            return []
    
    def _watershed_delineation(self, drainage_data: DrainageData) -> List[DrainageBasin]:
        """分水岭算法划分"""
        try:
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_maxima
            
            elevation = drainage_data.elevation
            
            # 计算梯度
            grad_y, grad_x = np.gradient(elevation)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 寻找局部最大值作为种子点
            local_maxima = peak_local_maxima(
                elevation, 
                min_distance=int(self.config.buffer_distance / self.config.spatial_resolution),
                threshold_abs=np.std(elevation) * 0.5
            )
            
            # 创建标记图像
            markers = np.zeros_like(elevation, dtype=int)
            for i, (y, x) in enumerate(zip(local_maxima[0], local_maxima[1])):
                markers[y, x] = i + 1
            
            # 执行分水岭算法
            labels = watershed(-elevation, markers, mask=~np.isnan(elevation))
            
            # 转换为流域对象
            basins = self._labels_to_basins(labels, drainage_data)
            
            return basins
        
        except Exception as e:
            self.logger.warning(f"分水岭算法失败: {e}")
            return []
    
    def _flow_accumulation_delineation(self, drainage_data: DrainageData) -> List[DrainageBasin]:
        """基于流量累积的划分"""
        try:
            # 计算流向和流量累积
            if drainage_data.flow_direction is None or drainage_data.flow_accumulation is None:
                flow_dir, flow_acc = self._calculate_flow_properties(drainage_data.elevation)
            else:
                flow_dir = drainage_data.flow_direction
                flow_acc = drainage_data.flow_accumulation
            
            # 识别河道
            stream_threshold = np.percentile(flow_acc[flow_acc > 0], 95)
            streams = flow_acc > stream_threshold
            
            # 寻找汇流点
            confluences = self._find_confluences(flow_dir, streams)
            
            # 基于汇流点划分流域
            basins = self._delineate_from_confluences(confluences, flow_dir, drainage_data)
            
            return basins
        
        except Exception as e:
            self.logger.warning(f"流量累积划分失败: {e}")
            return []
    
    def _topographic_delineation(self, drainage_data: DrainageData) -> List[DrainageBasin]:
        """基于地形的划分"""
        try:
            elevation = drainage_data.elevation
            
            # 计算地形特征
            grad_y, grad_x = np.gradient(elevation)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            curvature = self._calculate_curvature(elevation)
            
            # 识别山脊线和谷线
            ridges = self._identify_ridges(elevation, curvature)
            valleys = self._identify_valleys(elevation, curvature)
            
            # 基于山脊线划分流域
            basins = self._delineate_from_ridges(ridges, valleys, drainage_data)
            
            return basins
        
        except Exception as e:
            self.logger.warning(f"地形划分失败: {e}")
            return []
    
    def _clustering_delineation(self, drainage_data: DrainageData) -> List[DrainageBasin]:
        """基于聚类的划分"""
        try:
            elevation = drainage_data.elevation
            
            # 准备特征数据
            features = self._prepare_clustering_features(drainage_data)
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 确定最优聚类数
            n_clusters = self._determine_optimal_clusters(features_scaled)
            
            # 执行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            
            # 重塑标签到空间网格
            labels_grid = labels.reshape(elevation.shape)
            
            # 转换为流域对象
            basins = self._labels_to_basins(labels_grid, drainage_data)
            
            return basins
        
        except Exception as e:
            self.logger.warning(f"聚类划分失败: {e}")
            return []
    
    def _calculate_flow_properties(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算流向和流量累积"""
        try:
            # D8流向算法
            flow_direction = np.zeros_like(elevation, dtype=int)
            flow_accumulation = np.ones_like(elevation)
            
            # 流向编码 (1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE)
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            direction_codes = [1, 2, 4, 8, 16, 32, 64, 128]
            
            rows, cols = elevation.shape
            
            for i in range(rows):
                for j in range(cols):
                    if np.isnan(elevation[i, j]):
                        continue
                    
                    max_slope = -np.inf
                    flow_dir = 0
                    
                    for k, (di, dj) in enumerate(directions):
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(elevation[ni, nj]):
                            slope = (elevation[i, j] - elevation[ni, nj]) / np.sqrt(di**2 + dj**2)
                            
                            if slope > max_slope:
                                max_slope = slope
                                flow_dir = direction_codes[k]
                    
                    flow_direction[i, j] = flow_dir
            
            # 简化的流量累积计算
            # 实际应用中应使用更复杂的算法
            for i in range(rows):
                for j in range(cols):
                    if np.isnan(elevation[i, j]):
                        continue
                    
                    # 计算上游贡献
                    upstream_count = 0
                    for k, (di, dj) in enumerate(directions):
                        ni, nj = i - di, j - dj  # 上游方向
                        
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            not np.isnan(elevation[ni, nj]) and
                            flow_direction[ni, nj] == direction_codes[k]):
                            upstream_count += 1
                    
                    flow_accumulation[i, j] = upstream_count + 1
            
            return flow_direction, flow_accumulation
        
        except Exception as e:
            self.logger.warning(f"流向计算失败: {e}")
            return np.zeros_like(elevation), np.ones_like(elevation)
    
    def _find_confluences(self, flow_direction: np.ndarray, streams: np.ndarray) -> List[Tuple[int, int]]:
        """寻找汇流点"""
        confluences = []
        
        try:
            rows, cols = flow_direction.shape
            
            for i in range(rows):
                for j in range(cols):
                    if not streams[i, j]:
                        continue
                    
                    # 计算流入该点的河道数量
                    inflow_count = 0
                    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
                    direction_codes = [1, 2, 4, 8, 16, 32, 64, 128]
                    
                    for k, (di, dj) in enumerate(directions):
                        ni, nj = i - di, j - dj  # 上游方向
                        
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            streams[ni, nj] and
                            flow_direction[ni, nj] == direction_codes[k]):
                            inflow_count += 1
                    
                    # 如果有多条河道汇入，则为汇流点
                    if inflow_count >= 2:
                        confluences.append((i, j))
            
            return confluences
        
        except Exception as e:
            self.logger.warning(f"汇流点识别失败: {e}")
            return []
    
    def _calculate_curvature(self, elevation: np.ndarray) -> np.ndarray:
        """计算曲率"""
        try:
            # 计算二阶导数
            grad_y, grad_x = np.gradient(elevation)
            grad_yy, grad_yx = np.gradient(grad_y)
            grad_xy, grad_xx = np.gradient(grad_x)
            
            # 计算平均曲率
            curvature = (grad_xx + grad_yy) / 2
            
            return curvature
        
        except Exception as e:
            self.logger.warning(f"曲率计算失败: {e}")
            return np.zeros_like(elevation)
    
    def _identify_ridges(self, elevation: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """识别山脊线"""
        try:
            # 山脊线特征：高程局部最大值且曲率为负
            from skimage.morphology import local_maxima
            
            local_max = local_maxima(elevation)
            negative_curvature = curvature < -np.std(curvature) * 0.5
            
            ridges = local_max & negative_curvature
            
            return ridges
        
        except Exception as e:
            self.logger.warning(f"山脊线识别失败: {e}")
            return np.zeros_like(elevation, dtype=bool)
    
    def _identify_valleys(self, elevation: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """识别谷线"""
        try:
            # 谷线特征：高程局部最小值且曲率为正
            from skimage.morphology import local_minima
            
            local_min = local_minima(elevation)
            positive_curvature = curvature > np.std(curvature) * 0.5
            
            valleys = local_min & positive_curvature
            
            return valleys
        
        except Exception as e:
            self.logger.warning(f"谷线识别失败: {e}")
            return np.zeros_like(elevation, dtype=bool)
    
    def _prepare_clustering_features(self, drainage_data: DrainageData) -> np.ndarray:
        """准备聚类特征"""
        try:
            elevation = drainage_data.elevation
            
            # 计算地形特征
            grad_y, grad_x = np.gradient(elevation)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            aspect = np.arctan2(grad_y, grad_x)
            curvature = self._calculate_curvature(elevation)
            
            # 计算流向和流量累积
            if drainage_data.flow_accumulation is None:
                _, flow_acc = self._calculate_flow_properties(elevation)
            else:
                flow_acc = drainage_data.flow_accumulation
            
            # 组合特征
            features = []
            valid_mask = ~np.isnan(elevation)
            
            features.append(elevation[valid_mask])
            features.append(slope[valid_mask])
            features.append(np.cos(aspect[valid_mask]))  # 转换为x分量
            features.append(np.sin(aspect[valid_mask]))  # 转换为y分量
            features.append(curvature[valid_mask])
            features.append(np.log1p(flow_acc[valid_mask]))  # 对数变换
            
            return np.column_stack(features)
        
        except Exception as e:
            self.logger.warning(f"特征准备失败: {e}")
            return np.array([[]])
    
    def _determine_optimal_clusters(self, features: np.ndarray) -> int:
        """确定最优聚类数"""
        try:
            if len(features) < 100:
                return min(3, len(features) // 10)
            
            # 使用轮廓系数确定最优聚类数
            silhouette_scores = []
            k_range = range(2, min(11, len(features) // 10))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            return optimal_k
        
        except Exception as e:
            self.logger.warning(f"最优聚类数确定失败: {e}")
            return 5
    
    def _labels_to_basins(self, labels: np.ndarray, drainage_data: DrainageData) -> List[DrainageBasin]:
        """将标签转换为流域对象"""
        basins = []
        
        try:
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]  # 排除背景
            
            for basin_id in unique_labels:
                mask = labels == basin_id
                
                # 计算面积
                pixel_count = np.sum(mask)
                dx = np.abs(drainage_data.x[1] - drainage_data.x[0]) if len(drainage_data.x) > 1 else self.config.spatial_resolution
                dy = np.abs(drainage_data.y[1] - drainage_data.y[0]) if len(drainage_data.y) > 1 else self.config.spatial_resolution
                area_km2 = pixel_count * dx * dy / 1e6
                
                # 过滤小流域
                if area_km2 < self.config.min_basin_area:
                    continue
                
                # 提取边界
                boundary = self._extract_boundary(mask, drainage_data.x, drainage_data.y)
                
                # 寻找出口点（最低点）
                basin_elevation = drainage_data.elevation[mask]
                min_elev_idx = np.argmin(basin_elevation)
                y_indices, x_indices = np.where(mask)
                outlet_y = drainage_data.y[y_indices[min_elev_idx]]
                outlet_x = drainage_data.x[x_indices[min_elev_idx]]
                
                basin = DrainageBasin(
                    basin_id=int(basin_id),
                    name=f"Basin_{int(basin_id)}",
                    boundary=boundary,
                    outlet=(outlet_x, outlet_y),
                    area=area_km2
                )
                
                basins.append(basin)
            
            return basins
        
        except Exception as e:
            self.logger.warning(f"流域对象转换失败: {e}")
            return []
    
    def _extract_boundary(self, mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """提取流域边界"""
        try:
            from skimage.measure import find_contours
            
            # 寻找轮廓
            contours = find_contours(mask.astype(float), 0.5)
            
            if contours:
                # 选择最长的轮廓作为边界
                longest_contour = max(contours, key=len)
                
                # 转换为地理坐标
                boundary_coords = []
                for point in longest_contour:
                    row, col = point
                    if 0 <= int(row) < len(y) and 0 <= int(col) < len(x):
                        boundary_coords.append([x[int(col)], y[int(row)]])
                
                return np.array(boundary_coords)
            
            return np.array([])
        
        except Exception as e:
            self.logger.warning(f"边界提取失败: {e}")
            return np.array([])
    
    def _delineate_from_confluences(self, confluences: List[Tuple[int, int]], 
                                   flow_direction: np.ndarray, 
                                   drainage_data: DrainageData) -> List[DrainageBasin]:
        """基于汇流点划分流域"""
        # 简化实现，实际应用中需要更复杂的算法
        try:
            labels = np.zeros_like(flow_direction)
            
            for i, (conf_y, conf_x) in enumerate(confluences):
                # 简单的流域标记
                basin_id = i + 1
                self._trace_upstream(conf_y, conf_x, flow_direction, labels, basin_id)
            
            return self._labels_to_basins(labels, drainage_data)
        
        except Exception as e:
            self.logger.warning(f"基于汇流点的划分失败: {e}")
            return []
    
    def _trace_upstream(self, start_y: int, start_x: int, 
                       flow_direction: np.ndarray, labels: np.ndarray, basin_id: int):
        """追踪上游"""
        # 简化的上游追踪实现
        try:
            rows, cols = flow_direction.shape
            visited = set()
            stack = [(start_y, start_x)]
            
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            direction_codes = [1, 2, 4, 8, 16, 32, 64, 128]
            
            while stack:
                y, x = stack.pop()
                
                if (y, x) in visited or y < 0 or y >= rows or x < 0 or x >= cols:
                    continue
                
                visited.add((y, x))
                labels[y, x] = basin_id
                
                # 寻找流向该点的上游点
                for k, (di, dj) in enumerate(directions):
                    ny, nx = y - di, x - dj
                    
                    if (0 <= ny < rows and 0 <= nx < cols and 
                        flow_direction[ny, nx] == direction_codes[k]):
                        stack.append((ny, nx))
        
        except Exception as e:
            self.logger.warning(f"上游追踪失败: {e}")
    
    def _delineate_from_ridges(self, ridges: np.ndarray, valleys: np.ndarray, 
                              drainage_data: DrainageData) -> List[DrainageBasin]:
        """基于山脊线划分流域"""
        # 简化实现
        try:
            # 使用山脊线作为分水岭
            labels, n_labels = ndimage.label(~ridges)
            
            return self._labels_to_basins(labels, drainage_data)
        
        except Exception as e:
            self.logger.warning(f"基于山脊线的划分失败: {e}")
            return []

class BasinCharacteristicAnalyzer:
    """流域特征分析器"""
    
    def __init__(self, config: DrainageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_characteristics(self, basins: List[DrainageBasin], 
                              drainage_data: DrainageData) -> List[DrainageBasin]:
        """分析流域特征"""
        try:
            for basin in basins:
                # 计算形态特征
                if BasinCharacteristic.AREA in self.config.characteristics:
                    basin.characteristics['area'] = basin.area
                
                if BasinCharacteristic.PERIMETER in self.config.characteristics:
                    basin.characteristics['perimeter'] = self._calculate_perimeter(basin.boundary)
                
                if BasinCharacteristic.LENGTH in self.config.characteristics:
                    basin.characteristics['length'] = self._calculate_length(basin.boundary)
                
                if BasinCharacteristic.WIDTH in self.config.characteristics:
                    basin.characteristics['width'] = self._calculate_width(basin.boundary)
                
                # 计算高程特征
                elevation_stats = self._calculate_elevation_stats(basin, drainage_data)
                for key, value in elevation_stats.items():
                    if key in [char.value for char in self.config.characteristics]:
                        basin.characteristics[key] = value
                
                # 计算地形特征
                terrain_stats = self._calculate_terrain_stats(basin, drainage_data)
                for key, value in terrain_stats.items():
                    if key in [char.value for char in self.config.characteristics]:
                        basin.characteristics[key] = value
                
                # 计算形状指数
                shape_indices = self._calculate_shape_indices(basin)
                for key, value in shape_indices.items():
                    if key in [char.value for char in self.config.characteristics]:
                        basin.characteristics[key] = value
            
            return basins
        
        except Exception as e:
            self.logger.error(f"特征分析失败: {e}")
            return basins
    
    def _calculate_perimeter(self, boundary: np.ndarray) -> float:
        """计算周长"""
        try:
            if len(boundary) < 2:
                return 0.0
            
            perimeter = 0.0
            for i in range(len(boundary)):
                p1 = boundary[i]
                p2 = boundary[(i + 1) % len(boundary)]
                perimeter += np.sqrt(np.sum((p2 - p1)**2))
            
            return perimeter / 1000.0  # 转换为km
        
        except Exception as e:
            self.logger.warning(f"周长计算失败: {e}")
            return 0.0
    
    def _calculate_length(self, boundary: np.ndarray) -> float:
        """计算流域长度"""
        try:
            if len(boundary) < 2:
                return 0.0
            
            # 计算边界的主轴长度
            center = np.mean(boundary, axis=0)
            centered_boundary = boundary - center
            
            # 计算协方差矩阵
            cov_matrix = np.cov(centered_boundary.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # 主轴长度为最大特征值对应的标准差的2倍
            max_eigenvalue = np.max(eigenvalues)
            length = 2 * np.sqrt(max_eigenvalue)
            
            return length / 1000.0  # 转换为km
        
        except Exception as e:
            self.logger.warning(f"长度计算失败: {e}")
            return 0.0
    
    def _calculate_width(self, boundary: np.ndarray) -> float:
        """计算流域宽度"""
        try:
            if len(boundary) < 2:
                return 0.0
            
            # 计算边界的次轴长度
            center = np.mean(boundary, axis=0)
            centered_boundary = boundary - center
            
            # 计算协方差矩阵
            cov_matrix = np.cov(centered_boundary.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # 次轴长度为最小特征值对应的标准差的2倍
            min_eigenvalue = np.min(eigenvalues)
            width = 2 * np.sqrt(min_eigenvalue)
            
            return width / 1000.0  # 转换为km
        
        except Exception as e:
            self.logger.warning(f"宽度计算失败: {e}")
            return 0.0
    
    def _calculate_elevation_stats(self, basin: DrainageBasin, 
                                  drainage_data: DrainageData) -> Dict[str, float]:
        """计算高程统计"""
        try:
            # 获取流域内的高程数据
            basin_mask = self._create_basin_mask(basin, drainage_data)
            basin_elevation = drainage_data.elevation[basin_mask]
            basin_elevation = basin_elevation[~np.isnan(basin_elevation)]
            
            if len(basin_elevation) == 0:
                return {}
            
            stats = {
                'mean_elevation': np.mean(basin_elevation),
                'min_elevation': np.min(basin_elevation),
                'max_elevation': np.max(basin_elevation),
                'elevation_range': np.max(basin_elevation) - np.min(basin_elevation),
                'elevation_std': np.std(basin_elevation)
            }
            
            return stats
        
        except Exception as e:
            self.logger.warning(f"高程统计计算失败: {e}")
            return {}
    
    def _calculate_terrain_stats(self, basin: DrainageBasin, 
                                drainage_data: DrainageData) -> Dict[str, float]:
        """计算地形统计"""
        try:
            # 获取流域内的高程数据
            basin_mask = self._create_basin_mask(basin, drainage_data)
            basin_elevation = drainage_data.elevation[basin_mask]
            
            if np.sum(~np.isnan(basin_elevation)) < 4:
                return {}
            
            # 计算坡度和坡向
            grad_y, grad_x = np.gradient(drainage_data.elevation)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            aspect = np.arctan2(grad_y, grad_x)
            
            basin_slope = slope[basin_mask]
            basin_aspect = aspect[basin_mask]
            
            # 过滤有效值
            valid_slope = basin_slope[~np.isnan(basin_slope)]
            valid_aspect = basin_aspect[~np.isnan(basin_aspect)]
            
            stats = {}
            
            if len(valid_slope) > 0:
                stats['slope'] = np.mean(valid_slope)
                stats['slope_std'] = np.std(valid_slope)
            
            if len(valid_aspect) > 0:
                # 计算主要坡向（圆形统计）
                mean_aspect = np.arctan2(np.mean(np.sin(valid_aspect)), 
                                       np.mean(np.cos(valid_aspect)))
                stats['aspect'] = np.degrees(mean_aspect) % 360
            
            return stats
        
        except Exception as e:
            self.logger.warning(f"地形统计计算失败: {e}")
            return {}
    
    def _calculate_shape_indices(self, basin: DrainageBasin) -> Dict[str, float]:
        """计算形状指数"""
        try:
            area = basin.area  # km²
            perimeter = basin.characteristics.get('perimeter', 0)  # km
            length = basin.characteristics.get('length', 0)  # km
            width = basin.characteristics.get('width', 0)  # km
            
            indices = {}
            
            # 紧凑度 (Compactness)
            if perimeter > 0:
                indices['compactness'] = area / (perimeter**2)
            
            # 圆形度 (Circularity)
            if perimeter > 0:
                indices['circularity'] = 4 * np.pi * area / (perimeter**2)
            
            # 伸长率 (Elongation)
            if length > 0 and width > 0:
                indices['elongation'] = length / width
            
            # 形状因子 (Form Factor)
            if length > 0:
                indices['form_factor'] = area / (length**2)
            
            return indices
        
        except Exception as e:
            self.logger.warning(f"形状指数计算失败: {e}")
            return {}
    
    def _create_basin_mask(self, basin: DrainageBasin, 
                          drainage_data: DrainageData) -> np.ndarray:
        """创建流域掩膜"""
        try:
            from matplotlib.path import Path
            
            if len(basin.boundary) < 3:
                return np.zeros_like(drainage_data.elevation, dtype=bool)
            
            # 创建网格坐标
            X, Y = np.meshgrid(drainage_data.x, drainage_data.y)
            points = np.column_stack([X.ravel(), Y.ravel()])
            
            # 创建路径对象
            path = Path(basin.boundary)
            
            # 检查点是否在流域内
            mask = path.contains_points(points)
            mask = mask.reshape(X.shape)
            
            return mask
        
        except Exception as e:
            self.logger.warning(f"流域掩膜创建失败: {e}")
            return np.zeros_like(drainage_data.elevation, dtype=bool)

class HydrologicalAnalyzer:
    """水文分析器"""
    
    def __init__(self, config: DrainageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_hydrology(self, basins: List[DrainageBasin], 
                         drainage_data: DrainageData) -> List[DrainageBasin]:
        """分析水文特征"""
        try:
            # 计算流向和流量累积
            if drainage_data.flow_direction is None or drainage_data.flow_accumulation is None:
                flow_dir, flow_acc = self._calculate_flow_properties(drainage_data.elevation)
            else:
                flow_dir = drainage_data.flow_direction
                flow_acc = drainage_data.flow_accumulation
            
            for basin in basins:
                # 计算水文指标
                if HydrologicalMetric.FLOW_ACCUMULATION in self.config.hydrological_metrics:
                    basin.hydrological_metrics['flow_accumulation'] = self._calculate_basin_flow_accumulation(
                        basin, flow_acc, drainage_data
                    )
                
                if HydrologicalMetric.STREAM_ORDER in self.config.hydrological_metrics:
                    basin.hydrological_metrics['stream_order'] = self._calculate_stream_order(
                        basin, flow_dir, flow_acc, drainage_data
                    )
                
                if HydrologicalMetric.DRAINAGE_DENSITY in self.config.hydrological_metrics:
                    basin.hydrological_metrics['drainage_density'] = self._calculate_drainage_density(
                        basin, flow_acc, drainage_data
                    )
                
                if HydrologicalMetric.CHANNEL_LENGTH in self.config.hydrological_metrics:
                    basin.hydrological_metrics['channel_length'] = self._calculate_channel_length(
                        basin, flow_acc, drainage_data
                    )
                
                if HydrologicalMetric.RELIEF_RATIO in self.config.hydrological_metrics:
                    basin.hydrological_metrics['relief_ratio'] = self._calculate_relief_ratio(
                        basin, drainage_data
                    )
            
            return basins
        
        except Exception as e:
            self.logger.error(f"水文分析失败: {e}")
            return basins
    
    def _calculate_flow_properties(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算流向和流量累积"""
        # 重用BasinDelineator中的方法
        try:
            delineator = BasinDelineator(self.config)
            return delineator._calculate_flow_properties(elevation)
        except Exception as e:
            self.logger.warning(f"流向计算失败: {e}")
            return np.zeros_like(elevation), np.ones_like(elevation)
    
    def _calculate_basin_flow_accumulation(self, basin: DrainageBasin, 
                                          flow_acc: np.ndarray, 
                                          drainage_data: DrainageData) -> float:
        """计算流域流量累积"""
        try:
            analyzer = BasinCharacteristicAnalyzer(self.config)
            basin_mask = analyzer._create_basin_mask(basin, drainage_data)
            basin_flow_acc = flow_acc[basin_mask]
            
            return np.mean(basin_flow_acc[~np.isnan(basin_flow_acc)])
        
        except Exception as e:
            self.logger.warning(f"流域流量累积计算失败: {e}")
            return 0.0
    
    def _calculate_stream_order(self, basin: DrainageBasin, 
                               flow_dir: np.ndarray, 
                               flow_acc: np.ndarray, 
                               drainage_data: DrainageData) -> int:
        """计算河流等级"""
        try:
            # 简化的Strahler河流等级计算
            analyzer = BasinCharacteristicAnalyzer(self.config)
            basin_mask = analyzer._create_basin_mask(basin, drainage_data)
            
            # 识别河道
            stream_threshold = np.percentile(flow_acc[flow_acc > 0], 90)
            streams = (flow_acc > stream_threshold) & basin_mask
            
            if not np.any(streams):
                return 1
            
            # 简化计算：基于流量累积的最大值
            max_flow_acc = np.max(flow_acc[streams])
            stream_order = int(np.log2(max_flow_acc / stream_threshold)) + 1
            
            return max(1, min(stream_order, 7))  # 限制在1-7级
        
        except Exception as e:
            self.logger.warning(f"河流等级计算失败: {e}")
            return 1
    
    def _calculate_drainage_density(self, basin: DrainageBasin, 
                                   flow_acc: np.ndarray, 
                                   drainage_data: DrainageData) -> float:
        """计算河网密度"""
        try:
            analyzer = BasinCharacteristicAnalyzer(self.config)
            basin_mask = analyzer._create_basin_mask(basin, drainage_data)
            
            # 识别河道
            stream_threshold = np.percentile(flow_acc[flow_acc > 0], 85)
            streams = (flow_acc > stream_threshold) & basin_mask
            
            # 计算河道长度
            stream_length = np.sum(streams) * self.config.spatial_resolution / 1000.0  # km
            
            # 河网密度 = 河道总长度 / 流域面积
            if basin.area > 0:
                return stream_length / basin.area
            else:
                return 0.0
        
        except Exception as e:
            self.logger.warning(f"河网密度计算失败: {e}")
            return 0.0
    
    def _calculate_channel_length(self, basin: DrainageBasin, 
                                 flow_acc: np.ndarray, 
                                 drainage_data: DrainageData) -> float:
        """计算主河道长度"""
        try:
            analyzer = BasinCharacteristicAnalyzer(self.config)
            basin_mask = analyzer._create_basin_mask(basin, drainage_data)
            
            # 寻找主河道（流量累积最大的路径）
            basin_flow_acc = flow_acc.copy()
            basin_flow_acc[~basin_mask] = 0
            
            # 寻找流量累积最大的点作为起点
            max_idx = np.unravel_index(np.argmax(basin_flow_acc), basin_flow_acc.shape)
            
            # 简化计算：从最高流量累积点到出口的直线距离
            start_x = drainage_data.x[max_idx[1]]
            start_y = drainage_data.y[max_idx[0]]
            
            outlet_x, outlet_y = basin.outlet
            
            channel_length = np.sqrt((outlet_x - start_x)**2 + (outlet_y - start_y)**2) / 1000.0
            
            return channel_length
        
        except Exception as e:
            self.logger.warning(f"主河道长度计算失败: {e}")
            return 0.0
    
    def _calculate_relief_ratio(self, basin: DrainageBasin, 
                               drainage_data: DrainageData) -> float:
        """计算地势比"""
        try:
            elevation_range = basin.characteristics.get('elevation_range', 0)
            channel_length = basin.hydrological_metrics.get('channel_length', 0)
            
            if channel_length > 0:
                return elevation_range / (channel_length * 1000.0)  # 转换为m/m
            else:
                return 0.0
        
        except Exception as e:
            self.logger.warning(f"地势比计算失败: {e}")
            return 0.0

class DrainageVisualizer:
    """流域可视化器"""
    
    def __init__(self, config: DrainageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_visualizations(self, basins: List[DrainageBasin], 
                             drainage_data: DrainageData, 
                             output_dir: str) -> Dict[str, str]:
        """创建可视化"""
        plots = {}
        
        try:
            for viz_type in self.config.visualization_types:
                if viz_type == VisualizationType.BASIN_MAP:
                    plots['basin_map'] = self._create_basin_map(basins, drainage_data, output_dir)
                
                elif viz_type == VisualizationType.DRAINAGE_NETWORK:
                    plots['drainage_network'] = self._create_drainage_network(basins, drainage_data, output_dir)
                
                elif viz_type == VisualizationType.ELEVATION_PROFILE:
                    plots['elevation_profile'] = self._create_elevation_profile(basins, drainage_data, output_dir)
                
                elif viz_type == VisualizationType.HYPSOMETRIC_CURVE:
                    plots['hypsometric_curve'] = self._create_hypsometric_curve(basins, drainage_data, output_dir)
                
                elif viz_type == VisualizationType.SCATTER_MATRIX:
                    plots['scatter_matrix'] = self._create_scatter_matrix(basins, output_dir)
                
                elif viz_type == VisualizationType.COMPARISON_CHART:
                    plots['comparison_chart'] = self._create_comparison_chart(basins, output_dir)
                
                elif viz_type == VisualizationType.NETWORK_GRAPH:
                    plots['network_graph'] = self._create_network_graph(basins, drainage_data, output_dir)
                
                elif viz_type == VisualizationType.FLOW_MAP:
                    plots['flow_map'] = self._create_flow_map(basins, drainage_data, output_dir)
            
            return plots
        
        except Exception as e:
            self.logger.error(f"可视化创建失败: {e}")
            return {}
    
    def _create_basin_map(self, basins: List[DrainageBasin], 
                         drainage_data: DrainageData, output_dir: str) -> str:
        """创建流域地图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size, 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 绘制高程背景
            im = ax.contourf(drainage_data.x, drainage_data.y, drainage_data.elevation,
                           levels=20, cmap='terrain', alpha=0.7, transform=ccrs.PlateCarree())
            
            # 绘制流域边界
            colors = plt.cm.get_cmap(self.config.color_scheme)(np.linspace(0, 1, len(basins)))
            
            for i, basin in enumerate(basins):
                if len(basin.boundary) > 0:
                    boundary = basin.boundary
                    ax.plot(boundary[:, 0], boundary[:, 1], 
                           color=colors[i], linewidth=2, 
                           label=f'{basin.name} ({basin.area:.1f} km²)',
                           transform=ccrs.PlateCarree())
                    
                    # 标记出口
                    ax.plot(basin.outlet[0], basin.outlet[1], 
                           'ro', markersize=8, transform=ccrs.PlateCarree())
            
            # 添加地图要素
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.gridlines(draw_labels=True)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='高程 (m)', shrink=0.8)
            
            # 添加图例
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.title('流域分布图', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图片
            filename = f'basin_map.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"流域地图创建失败: {e}")
            return ""
    
    def _create_drainage_network(self, basins: List[DrainageBasin], 
                                drainage_data: DrainageData, output_dir: str) -> str:
        """创建河网图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 计算流量累积
            if drainage_data.flow_accumulation is None:
                delineator = BasinDelineator(self.config)
                _, flow_acc = delineator._calculate_flow_properties(drainage_data.elevation)
            else:
                flow_acc = drainage_data.flow_accumulation
            
            # 绘制高程背景
            im = ax.contourf(drainage_data.x, drainage_data.y, drainage_data.elevation,
                           levels=20, cmap='terrain', alpha=0.5)
            
            # 绘制河网
            stream_threshold = np.percentile(flow_acc[flow_acc > 0], 85)
            streams = flow_acc > stream_threshold
            
            # 根据流量累积绘制不同粗细的河道
            stream_levels = [stream_threshold, 
                           np.percentile(flow_acc[flow_acc > 0], 95),
                           np.percentile(flow_acc[flow_acc > 0], 99)]
            
            for i, level in enumerate(stream_levels):
                stream_mask = flow_acc > level
                if np.any(stream_mask):
                    ax.contour(drainage_data.x, drainage_data.y, stream_mask.astype(int),
                             levels=[0.5], colors='blue', linewidths=i+1)
            
            # 绘制流域边界
            for basin in basins:
                if len(basin.boundary) > 0:
                    boundary = basin.boundary
                    ax.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2)
                    
                    # 标记出口
                    ax.plot(basin.outlet[0], basin.outlet[1], 'ro', markersize=8)
                    
                    # 添加流域标签
                    center_x = np.mean(boundary[:, 0])
                    center_y = np.mean(boundary[:, 1])
                    ax.text(center_x, center_y, basin.name, 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.colorbar(im, ax=ax, label='高程 (m)')
            plt.title('河网分布图', fontsize=14, fontweight='bold')
            plt.xlabel('经度')
            plt.ylabel('纬度')
            plt.tight_layout()
            
            # 保存图片
            filename = f'drainage_network.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"河网图创建失败: {e}")
            return ""
    
    def _create_elevation_profile(self, basins: List[DrainageBasin], 
                                 drainage_data: DrainageData, output_dir: str) -> str:
        """创建高程剖面图"""
        try:
            fig, axes = plt.subplots(len(basins), 1, figsize=(12, 4*len(basins)))
            if len(basins) == 1:
                axes = [axes]
            
            for i, basin in enumerate(basins):
                ax = axes[i]
                
                # 获取流域内高程数据
                analyzer = BasinCharacteristicAnalyzer(self.config)
                basin_mask = analyzer._create_basin_mask(basin, drainage_data)
                basin_elevation = drainage_data.elevation[basin_mask]
                basin_elevation = basin_elevation[~np.isnan(basin_elevation)]
                
                if len(basin_elevation) > 0:
                    # 创建高程剖面
                    sorted_elevation = np.sort(basin_elevation)
                    cumulative_area = np.linspace(0, 100, len(sorted_elevation))
                    
                    ax.plot(cumulative_area, sorted_elevation, 'b-', linewidth=2)
                    ax.fill_between(cumulative_area, sorted_elevation, alpha=0.3)
                    
                    ax.set_xlabel('累积面积百分比 (%)')
                    ax.set_ylabel('高程 (m)')
                    ax.set_title(f'{basin.name} 高程剖面')
                    ax.grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_elev = np.mean(basin_elevation)
                    median_elev = np.median(basin_elevation)
                    ax.axhline(mean_elev, color='r', linestyle='--', label=f'平均高程: {mean_elev:.0f}m')
                    ax.axhline(median_elev, color='g', linestyle='--', label=f'中位高程: {median_elev:.0f}m')
                    ax.legend()
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'elevation_profile.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"高程剖面图创建失败: {e}")
            return ""
    
    def _create_hypsometric_curve(self, basins: List[DrainageBasin], 
                                 drainage_data: DrainageData, output_dir: str) -> str:
        """创建高程曲线"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            colors = plt.cm.get_cmap(self.config.color_scheme)(np.linspace(0, 1, len(basins)))
            
            for i, basin in enumerate(basins):
                # 获取流域内高程数据
                analyzer = BasinCharacteristicAnalyzer(self.config)
                basin_mask = analyzer._create_basin_mask(basin, drainage_data)
                basin_elevation = drainage_data.elevation[basin_mask]
                basin_elevation = basin_elevation[~np.isnan(basin_elevation)]
                
                if len(basin_elevation) > 0:
                    # 计算高程曲线
                    min_elev = np.min(basin_elevation)
                    max_elev = np.max(basin_elevation)
                    
                    # 归一化高程
                    normalized_elevation = (basin_elevation - min_elev) / (max_elev - min_elev)
                    
                    # 计算累积面积
                    elevation_bins = np.linspace(0, 1, 101)
                    cumulative_area = []
                    
                    for elev_threshold in elevation_bins:
                        area_above = np.sum(normalized_elevation >= elev_threshold) / len(normalized_elevation)
                        cumulative_area.append(area_above)
                    
                    ax.plot(cumulative_area, elevation_bins, 
                           color=colors[i], linewidth=2, label=f'{basin.name}')
            
            ax.set_xlabel('累积面积比例')
            ax.set_ylabel('相对高程')
            ax.set_title('高程曲线', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'hypsometric_curve.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"高程曲线创建失败: {e}")
            return ""
    
    def _create_scatter_matrix(self, basins: List[DrainageBasin], output_dir: str) -> str:
        """创建散点矩阵"""
        try:
            # 收集所有特征数据
            data = []
            for basin in basins:
                row = {'basin_name': basin.name}
                row.update(basin.characteristics)
                row.update(basin.hydrological_metrics)
                data.append(row)
            
            if not data:
                return ""
            
            df = pd.DataFrame(data)
            
            # 选择数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return ""
            
            # 创建散点矩阵
            fig = plt.figure(figsize=(15, 15))
            
            n_vars = len(numeric_cols)
            for i in range(n_vars):
                for j in range(n_vars):
                    ax = plt.subplot(n_vars, n_vars, i * n_vars + j + 1)
                    
                    if i == j:
                        # 对角线绘制直方图
                        ax.hist(df[numeric_cols[i]], bins=20, alpha=0.7, color='skyblue')
                        ax.set_ylabel('频数')
                    else:
                        # 非对角线绘制散点图
                        ax.scatter(df[numeric_cols[j]], df[numeric_cols[i]], 
                                 alpha=0.7, color='steelblue')
                        
                        # 计算相关系数
                        corr = df[numeric_cols[i]].corr(df[numeric_cols[j]])
                        ax.text(0.05, 0.95, f'r={corr:.2f}', 
                               transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # 设置标签
                    if i == n_vars - 1:
                        ax.set_xlabel(numeric_cols[j])
                    if j == 0:
                        ax.set_ylabel(numeric_cols[i])
                    
                    # 旋转x轴标签
                    plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.suptitle('流域特征散点矩阵', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图片
            filename = f'scatter_matrix.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"散点矩阵创建失败: {e}")
            return ""
    
    def _create_comparison_chart(self, basins: List[DrainageBasin], output_dir: str) -> str:
        """创建对比图表"""
        try:
            # 收集数据
            data = []
            for basin in basins:
                row = {'basin_name': basin.name, 'area': basin.area}
                row.update(basin.characteristics)
                row.update(basin.hydrological_metrics)
                data.append(row)
            
            if not data:
                return ""
            
            df = pd.DataFrame(data)
            
            # 选择主要特征进行对比
            comparison_features = ['area', 'mean_elevation', 'slope', 'drainage_density']
            available_features = [f for f in comparison_features if f in df.columns]
            
            if not available_features:
                return ""
            
            # 创建子图
            n_features = len(available_features)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(available_features[:4]):
                ax = axes[i]
                
                # 柱状图
                bars = ax.bar(df['basin_name'], df[feature], 
                             color=plt.cm.Set3(np.linspace(0, 1, len(df))))
                
                ax.set_title(f'{feature}对比', fontsize=12, fontweight='bold')
                ax.set_ylabel(feature)
                
                # 旋转x轴标签
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # 添加数值标签
                for bar, value in zip(bars, df[feature]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            # 隐藏多余的子图
            for i in range(len(available_features), 4):
                axes[i].set_visible(False)
            
            plt.suptitle('流域特征对比', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图片
            filename = f'comparison_chart.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"对比图表创建失败: {e}")
            return ""
    
    def _create_network_graph(self, basins: List[DrainageBasin], 
                             drainage_data: DrainageData, output_dir: str) -> str:
        """创建网络图"""
        try:
            # 创建网络图
            G = nx.Graph()
            
            # 添加节点（流域）
            for basin in basins:
                G.add_node(basin.basin_id, 
                          name=basin.name,
                          area=basin.area,
                          outlet=basin.outlet)
            
            # 添加边（基于空间邻接关系）
            for i, basin1 in enumerate(basins):
                for j, basin2 in enumerate(basins[i+1:], i+1):
                    # 简化的邻接判断：基于出口距离
                    dist = np.sqrt((basin1.outlet[0] - basin2.outlet[0])**2 + 
                                 (basin1.outlet[1] - basin2.outlet[1])**2)
                    
                    # 如果距离小于阈值，认为是邻接的
                    if dist < self.config.buffer_distance * 2:
                        G.add_edge(basin1.basin_id, basin2.basin_id, weight=1/dist)
            
            if len(G.nodes()) == 0:
                return ""
            
            # 绘制网络图
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 计算布局
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 绘制节点
            node_sizes = [basin.area * 100 for basin in basins if basin.basin_id in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.7, ax=ax)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
            
            # 绘制标签
            labels = {basin.basin_id: basin.name for basin in basins if basin.basin_id in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
            
            ax.set_title('流域网络图', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            filename = f'network_graph.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"网络图创建失败: {e}")
            return ""
    
    def _create_flow_map(self, basins: List[DrainageBasin], 
                        drainage_data: DrainageData, output_dir: str) -> str:
        """创建流向图"""
        try:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # 计算流向
            if drainage_data.flow_direction is None:
                delineator = BasinDelineator(self.config)
                flow_dir, _ = delineator._calculate_flow_properties(drainage_data.elevation)
            else:
                flow_dir = drainage_data.flow_direction
            
            # 绘制高程背景
            im = ax.contourf(drainage_data.x, drainage_data.y, drainage_data.elevation,
                           levels=20, cmap='terrain', alpha=0.5)
            
            # 绘制流向箭头（采样显示）
            step = max(1, len(drainage_data.x) // 50)  # 采样步长
            X, Y = np.meshgrid(drainage_data.x[::step], drainage_data.y[::step])
            
            # 流向编码转换为角度
            direction_angles = {
                1: 0,      # E
                2: 45,     # SE
                4: 90,     # S
                8: 135,    # SW
                16: 180,   # W
                32: 225,   # NW
                64: 270,   # N
                128: 315   # NE
            }
            
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    flow_code = flow_dir[i*step, j*step]
                    if flow_code in direction_angles:
                        angle = np.radians(direction_angles[flow_code])
                        U[i, j] = np.cos(angle)
                        V[i, j] = np.sin(angle)
            
            # 绘制箭头
            ax.quiver(X, Y, U, V, alpha=0.7, scale=20, width=0.003, color='blue')
            
            # 绘制流域边界
            for basin in basins:
                if len(basin.boundary) > 0:
                    boundary = basin.boundary
                    ax.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2)
                    
                    # 标记出口
                    ax.plot(basin.outlet[0], basin.outlet[1], 'ro', markersize=10)
            
            plt.colorbar(im, ax=ax, label='高程 (m)')
            plt.title('流向分布图', fontsize=14, fontweight='bold')
            plt.xlabel('经度')
            plt.ylabel('纬度')
            plt.tight_layout()
            
            # 保存图片
            filename = f'flow_map.{self.config.output_format}'
            filepath = Path(output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.warning(f"流向图创建失败: {e}")
            return ""

class DrainageBasinAnalyzer:
    """流域分析器主类"""
    
    def __init__(self, config: DrainageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.delineator = BasinDelineator(config)
        self.characteristic_analyzer = BasinCharacteristicAnalyzer(config)
        self.hydrological_analyzer = HydrologicalAnalyzer(config)
        self.visualizer = DrainageVisualizer(config)
    
    def analyze(self, drainage_data: DrainageData) -> Dict[str, Any]:
        """执行完整的流域分析"""
        try:
            self.logger.info("开始流域分析...")
            
            # 1. 流域划分
            self.logger.info("执行流域划分...")
            basins = self.delineator.delineate_basins(drainage_data)
            
            if not basins:
                self.logger.warning("未识别到有效流域")
                return {'basins': [], 'plots': {}, 'summary': {}}
            
            self.logger.info(f"识别到 {len(basins)} 个流域")
            
            # 2. 特征分析
            if AnalysisType.MORPHOMETRIC in self.config.analysis_types:
                self.logger.info("执行形态特征分析...")
                basins = self.characteristic_analyzer.analyze_characteristics(basins, drainage_data)
            
            # 3. 水文分析
            if AnalysisType.HYDROLOGICAL in self.config.analysis_types:
                self.logger.info("执行水文特征分析...")
                basins = self.hydrological_analyzer.analyze_hydrology(basins, drainage_data)
            
            # 4. 对比分析
            if AnalysisType.COMPARATIVE in self.config.analysis_types:
                self.logger.info("执行对比分析...")
                comparison_results = self._perform_comparative_analysis(basins)
            else:
                comparison_results = {}
            
            # 5. 统计分析
            if AnalysisType.STATISTICAL in self.config.analysis_types:
                self.logger.info("执行统计分析...")
                statistical_results = self._perform_statistical_analysis(basins)
            else:
                statistical_results = {}
            
            # 6. 创建可视化
            plots = {}
            if self.config.save_results:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info("创建可视化...")
                plots = self.visualizer.create_visualizations(basins, drainage_data, str(output_dir))
            
            # 7. 生成摘要
            summary = self._generate_summary(basins, comparison_results, statistical_results)
            
            # 8. 保存结果
            if self.config.save_results:
                self._save_results(basins, summary, plots)
            
            self.logger.info("流域分析完成")
            
            return {
                'basins': basins,
                'plots': plots,
                'summary': summary,
                'comparison': comparison_results,
                'statistics': statistical_results
            }
        
        except Exception as e:
            self.logger.error(f"流域分析失败: {e}")
            return {'basins': [], 'plots': {}, 'summary': {}}
    
    def _perform_comparative_analysis(self, basins: List[DrainageBasin]) -> Dict[str, Any]:
        """执行对比分析"""
        try:
            if len(basins) < 2:
                return {}
            
            # 收集特征数据
            features = []
            for basin in basins:
                feature_dict = {'basin_id': basin.basin_id, 'area': basin.area}
                feature_dict.update(basin.characteristics)
                feature_dict.update(basin.hydrological_metrics)
                features.append(feature_dict)
            
            df = pd.DataFrame(features)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            results = {
                'correlation_matrix': {},
                'ranking': {},
                'clustering': {}
            }
            
            # 相关性分析
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                results['correlation_matrix'] = corr_matrix.to_dict()
            
            # 排名分析
            for col in numeric_cols:
                if col != 'basin_id':
                    ranked = df.nlargest(len(df), col)[['basin_id', col]]
                    results['ranking'][col] = ranked.to_dict('records')
            
            # 聚类分析
            if len(numeric_cols) > 2 and len(basins) > 3:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    # 标准化数据
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[numeric_cols])
                    
                    # K-means聚类
                    n_clusters = min(3, len(basins) // 2)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    results['clustering'] = {
                        'labels': clusters.tolist(),
                        'centers': kmeans.cluster_centers_.tolist()
                    }
                
                except Exception as e:
                    self.logger.warning(f"聚类分析失败: {e}")
            
            return results
        
        except Exception as e:
            self.logger.warning(f"对比分析失败: {e}")
            return {}
    
    def _perform_statistical_analysis(self, basins: List[DrainageBasin]) -> Dict[str, Any]:
        """执行统计分析"""
        try:
            # 收集所有数值特征
            all_features = {}
            
            for basin in basins:
                for key, value in basin.characteristics.items():
                    if isinstance(value, (int, float)):
                        if key not in all_features:
                            all_features[key] = []
                        all_features[key].append(value)
                
                for key, value in basin.hydrological_metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in all_features:
                            all_features[key] = []
                        all_features[key].append(value)
            
            # 计算统计指标
            statistics = {}
            
            for feature, values in all_features.items():
                if len(values) > 0:
                    statistics[feature] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75)
                    }
                    
                    # 正态性检验
                    if len(values) >= 3:
                        try:
                            _, p_value = stats.normaltest(values)
                            statistics[feature]['normality_p'] = p_value
                        except:
                            pass
            
            return statistics
        
        except Exception as e:
            self.logger.warning(f"统计分析失败: {e}")
            return {}
    
    def _generate_summary(self, basins: List[DrainageBasin], 
                         comparison_results: Dict[str, Any],
                         statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析摘要"""
        try:
            summary = {
                'total_basins': len(basins),
                'total_area': sum(basin.area for basin in basins),
                'basin_info': [],
                'key_statistics': {},
                'analysis_config': {
                    'delineation_method': self.config.delineation_method.value,
                    'min_basin_area': self.config.min_basin_area,
                    'analysis_types': [at.value for at in self.config.analysis_types]
                }
            }
            
            # 流域信息
            for basin in basins:
                basin_info = {
                    'id': basin.basin_id,
                    'name': basin.name,
                    'area': basin.area,
                    'outlet': basin.outlet,
                    'characteristics_count': len(basin.characteristics),
                    'hydrological_metrics_count': len(basin.hydrological_metrics)
                }
                summary['basin_info'].append(basin_info)
            
            # 关键统计
            if basins:
                areas = [basin.area for basin in basins]
                summary['key_statistics'] = {
                    'area_stats': {
                        'mean': np.mean(areas),
                        'std': np.std(areas),
                        'min': np.min(areas),
                        'max': np.max(areas)
                    }
                }
                
                # 添加其他统计信息
                if statistical_results:
                    summary['key_statistics']['feature_stats'] = statistical_results
            
            return summary
        
        except Exception as e:
            self.logger.warning(f"摘要生成失败: {e}")
            return {}
    
    def _save_results(self, basins: List[DrainageBasin], 
                     summary: Dict[str, Any], plots: Dict[str, str]):
        """保存分析结果"""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存流域数据
            basin_data = []
            for basin in basins:
                basin_dict = {
                    'basin_id': basin.basin_id,
                    'name': basin.name,
                    'area': basin.area,
                    'outlet': basin.outlet,
                    'boundary': basin.boundary.tolist() if len(basin.boundary) > 0 else [],
                    'characteristics': basin.characteristics,
                    'hydrological_metrics': basin.hydrological_metrics
                }
                basin_data.append(basin_dict)
            
            # 保存为CSV
            df_basins = pd.DataFrame([
                {
                    'basin_id': basin.basin_id,
                    'name': basin.name,
                    'area': basin.area,
                    'outlet_x': basin.outlet[0],
                    'outlet_y': basin.outlet[1],
                    **basin.characteristics,
                    **basin.hydrological_metrics
                }
                for basin in basins
            ])
            
            df_basins.to_csv(output_dir / 'basin_analysis_results.csv', index=False)
            
            # 保存摘要
            import json
            with open(output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果已保存到: {output_dir}")
        
        except Exception as e:
            self.logger.warning(f"结果保存失败: {e}")

def create_drainage_basin_analyzer(config: Optional[DrainageConfig] = None) -> DrainageBasinAnalyzer:
    """创建流域分析器"""
    if config is None:
        config = DrainageConfig()
    
    return DrainageBasinAnalyzer(config)

if __name__ == "__main__":
    # 测试代码
    print("流域分析模块测试")
    
    # 创建测试数据
    np.random.seed(42)
    
    # 创建模拟地形数据
    x = np.linspace(0, 10000, 100)  # 10km范围
    y = np.linspace(0, 10000, 100)
    X, Y = np.meshgrid(x, y)
    
    # 模拟山地地形
    elevation = (1000 + 
                500 * np.sin(X / 2000) * np.cos(Y / 2000) +
                300 * np.sin(X / 1000) +
                200 * np.cos(Y / 1500) +
                100 * np.random.random(X.shape))
    
    # 创建流域数据
    drainage_data = DrainageData(
        x=x,
        y=y,
        elevation=elevation
    )
    
    # 创建配置
    config = DrainageConfig(
        delineation_method=BasinDelineationMethod.WATERSHED,
        min_basin_area=5.0,
        characteristics=[
            BasinCharacteristic.AREA,
            BasinCharacteristic.MEAN_ELEVATION,
            BasinCharacteristic.SLOPE,
            BasinCharacteristic.COMPACTNESS
        ],
        hydrological_metrics=[
            HydrologicalMetric.FLOW_ACCUMULATION,
            HydrologicalMetric.STREAM_ORDER,
            HydrologicalMetric.DRAINAGE_DENSITY
        ],
        analysis_types=[
            AnalysisType.MORPHOMETRIC,
            AnalysisType.HYDROLOGICAL,
            AnalysisType.COMPARATIVE,
            AnalysisType.STATISTICAL
        ],
        visualization_types=[
            VisualizationType.BASIN_MAP,
            VisualizationType.DRAINAGE_NETWORK,
            VisualizationType.COMPARISON_CHART
        ],
        save_results=True,
        output_dir='./test_drainage_analysis'
    )
    
    # 创建分析器
    analyzer = create_drainage_basin_analyzer(config)
    
    # 执行分析
    results = analyzer.analyze(drainage_data)
    
    # 打印结果
    print(f"\n分析完成!")
    print(f"识别流域数量: {len(results['basins'])}")
    print(f"生成图表数量: {len(results['plots'])}")
    
    if results['basins']:
        print("\n流域信息:")
        for basin in results['basins']:
            print(f"  {basin.name}: {basin.area:.2f} km²")
    
    if results['summary']:
        print(f"\n总面积: {results['summary'].get('total_area', 0):.2f} km²")
    
    print("\n流域分析测试完成!")