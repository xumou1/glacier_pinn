#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICESat数据验证模块

该模块实现了与ICESat激光测高数据的验证功能，用于验证模型在冰面高程和厚度变化预测方面的准确性。
支持ICESat-1和ICESat-2数据、轨道数据处理、高程变化分析和不确定性评估。

主要功能:
- ICESat数据处理和质量控制
- 轨道交叉点分析
- 高程变化检测
- 时间序列分析
- 空间插值和对比
- 不确定性量化

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy import stats, spatial
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICESatMission(Enum):
    """ICESat任务类型"""
    ICESAT1 = "ICESat-1"  # ICESat-1 (2003-2009)
    ICESAT2 = "ICESat-2"  # ICESat-2 (2018-present)

class DataProduct(Enum):
    """数据产品类型"""
    # ICESat-1产品
    GLAS12 = "GLAS12"  # Antarctic and Greenland Ice Sheet Altimetry
    GLAS13 = "GLAS13"  # Sea Ice Altimetry
    GLAS14 = "GLAS14"  # Global Land Surface Altimetry
    
    # ICESat-2产品
    ATL03 = "ATL03"    # Global Geolocated Photon Data
    ATL06 = "ATL06"    # Land Ice Height
    ATL07 = "ATL07"    # Sea Ice Height
    ATL08 = "ATL08"    # Land and Vegetation Height
    ATL10 = "ATL10"    # Sea Ice Freeboard
    ATL11 = "ATL11"    # Annual Land Ice Height
    ATL15 = "ATL15"    # Antarctic and Greenland Ice Sheet Height Change

class ProcessingLevel(Enum):
    """数据处理级别"""
    L1A = "L1A"  # 原始数据
    L1B = "L1B"  # 地理定位数据
    L2A = "L2A"  # 地球物理参数
    L3A = "L3A"  # 网格化产品
    L3B = "L3B"  # 时间序列产品

class QualityFlag(Enum):
    """数据质量标志"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"

class SurfaceType(Enum):
    """地表类型"""
    ICE_SHEET = "ice_sheet"      # 冰盖
    GLACIER = "glacier"          # 冰川
    SEA_ICE = "sea_ice"          # 海冰
    LAND = "land"                # 陆地
    WATER = "water"              # 水体
    CLOUD = "cloud"              # 云层

class BeamType(Enum):
    """激光束类型"""
    # ICESat-2特有
    STRONG = "strong"  # 强束
    WEAK = "weak"      # 弱束
    # ICESat-1
    SINGLE = "single"  # 单束

class ValidationMode(Enum):
    """验证模式"""
    POINT_TO_POINT = "point_to_point"      # 点对点
    CROSSOVER = "crossover"                # 轨道交叉点
    REPEAT_TRACK = "repeat_track"          # 重复轨道
    GRIDDED = "gridded"                    # 网格化
    TIME_SERIES = "time_series"            # 时间序列

@dataclass
class ICESatObservation:
    """ICESat观测数据"""
    observation_id: str
    mission: ICESatMission
    data_product: DataProduct
    processing_level: ProcessingLevel
    acquisition_time: datetime
    
    # 地理信息
    latitude: np.ndarray
    longitude: np.ndarray
    elevation: np.ndarray  # 高程数据
    
    # 轨道信息
    track_id: Optional[str] = None
    beam_type: Optional[BeamType] = None
    cycle_number: Optional[int] = None
    
    # 质量信息
    quality_flag: np.ndarray = None  # 每个点的质量标志
    uncertainty: np.ndarray = None   # 高程不确定性
    surface_type: np.ndarray = None  # 地表类型
    
    # 其他属性
    units: str = "m"
    datum: str = "WGS84"
    processing_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ICESatValidationConfig:
    """ICESat验证配置"""
    # 基本设置
    temporal_tolerance: float = 30.0  # 时间容差(天)
    spatial_tolerance: float = 1000.0  # 空间容差(米)
    elevation_tolerance: float = 100.0  # 高程容差(米)
    
    # 质量控制
    enable_quality_filter: bool = True
    min_quality: QualityFlag = QualityFlag.FAIR
    outlier_threshold: float = 3.0  # 异常值阈值(标准差倍数)
    max_slope: float = 45.0  # 最大坡度(度)
    
    # 验证模式
    validation_modes: List[ValidationMode] = field(default_factory=lambda: [ValidationMode.POINT_TO_POINT, ValidationMode.GRIDDED])
    
    # 交叉点分析
    crossover_search_radius: float = 500.0  # 交叉点搜索半径(米)
    min_time_separation: float = 30.0  # 最小时间间隔(天)
    
    # 网格化设置
    grid_resolution: float = 1000.0  # 网格分辨率(米)
    interpolation_method: str = "linear"  # 插值方法
    max_interpolation_distance: float = 5000.0  # 最大插值距离(米)
    
    # 时间序列设置
    enable_trend_analysis: bool = True
    enable_seasonal_analysis: bool = True
    detrend_method: str = "linear"
    
    # 表面类型过滤
    allowed_surface_types: List[SurfaceType] = field(default_factory=lambda: [SurfaceType.ICE_SHEET, SurfaceType.GLACIER])
    
    # 绘图设置
    enable_plotting: bool = True
    save_plots: bool = False
    plot_dir: str = "./icesat_validation_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 其他设置
    verbose: bool = True
    random_seed: int = 42

class GeospatialProcessor:
    """地理空间处理器"""
    
    def __init__(self, config: ICESatValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_distance(self, 
                          lat1: np.ndarray, lon1: np.ndarray,
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """计算地理距离(米)"""
        # 使用Haversine公式
        R = 6371000  # 地球半径(米)
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def find_nearest_points(self, 
                           query_lat: np.ndarray, query_lon: np.ndarray,
                           ref_lat: np.ndarray, ref_lon: np.ndarray,
                           max_distance: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """找到最近的参考点"""
        try:
            if max_distance is None:
                max_distance = self.config.spatial_tolerance
            
            # 构建KD树用于快速搜索
            ref_points = np.column_stack([ref_lat.flatten(), ref_lon.flatten()])
            query_points = np.column_stack([query_lat.flatten(), query_lon.flatten()])
            
            tree = spatial.cKDTree(ref_points)
            distances, indices = tree.query(query_points)
            
            # 转换为地理距离
            geo_distances = self.calculate_distance(
                query_lat.flatten(), query_lon.flatten(),
                ref_lat.flatten()[indices], ref_lon.flatten()[indices]
            )
            
            # 过滤超出距离阈值的点
            valid_mask = geo_distances <= max_distance
            
            return indices, valid_mask
        
        except Exception as e:
            self.logger.error(f"最近点搜索失败: {e}")
            return np.array([]), np.array([])
    
    def detect_crossovers(self, 
                         observations: List[ICESatObservation]) -> List[Dict[str, Any]]:
        """检测轨道交叉点"""
        try:
            crossovers = []
            
            for i, obs1 in enumerate(observations):
                for j, obs2 in enumerate(observations[i+1:], i+1):
                    # 检查时间间隔
                    time_diff = abs((obs1.acquisition_time - obs2.acquisition_time).total_seconds() / (24*3600))
                    if time_diff < self.config.min_time_separation:
                        continue
                    
                    # 寻找空间上接近的点
                    for k, (lat1, lon1, elev1) in enumerate(zip(obs1.latitude, obs1.longitude, obs1.elevation)):
                        distances = self.calculate_distance(
                            lat1, lon1, obs2.latitude, obs2.longitude
                        )
                        
                        close_indices = np.where(distances <= self.config.crossover_search_radius)[0]
                        
                        for idx in close_indices:
                            crossover = {
                                'obs1_id': obs1.observation_id,
                                'obs2_id': obs2.observation_id,
                                'obs1_point': k,
                                'obs2_point': idx,
                                'latitude': (lat1 + obs2.latitude[idx]) / 2,
                                'longitude': (lon1 + obs2.longitude[idx]) / 2,
                                'elevation_diff': elev1 - obs2.elevation[idx],
                                'time_diff': time_diff,
                                'distance': distances[idx],
                                'obs1_time': obs1.acquisition_time,
                                'obs2_time': obs2.acquisition_time
                            }
                            crossovers.append(crossover)
            
            return crossovers
        
        except Exception as e:
            self.logger.error(f"交叉点检测失败: {e}")
            return []
    
    def create_grid(self, 
                   observations: List[ICESatObservation],
                   bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """创建规则网格"""
        try:
            if bounds is None:
                # 自动计算边界
                all_lats = np.concatenate([obs.latitude for obs in observations])
                all_lons = np.concatenate([obs.longitude for obs in observations])
                
                min_lat, max_lat = np.min(all_lats), np.max(all_lats)
                min_lon, max_lon = np.min(all_lons), np.max(all_lons)
            else:
                min_lat, max_lat, min_lon, max_lon = bounds
            
            # 计算网格点数
            # 假设1度约等于111km
            lat_resolution = self.config.grid_resolution / 111000
            lon_resolution = self.config.grid_resolution / (111000 * np.cos(np.radians((min_lat + max_lat) / 2)))
            
            lat_grid = np.arange(min_lat, max_lat + lat_resolution, lat_resolution)
            lon_grid = np.arange(min_lon, max_lon + lon_resolution, lon_resolution)
            
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            return lat_mesh, lon_mesh
        
        except Exception as e:
            self.logger.error(f"网格创建失败: {e}")
            return np.array([]), np.array([])

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: ICESatValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geo_processor = GeospatialProcessor(config)
    
    def filter_observations(self, observations: List[ICESatObservation]) -> List[ICESatObservation]:
        """过滤观测数据"""
        filtered = []
        
        for obs in observations:
            try:
                # 质量过滤
                if self.config.enable_quality_filter and obs.quality_flag is not None:
                    quality_levels = {
                        QualityFlag.EXCELLENT: 5,
                        QualityFlag.GOOD: 4,
                        QualityFlag.FAIR: 3,
                        QualityFlag.POOR: 2,
                        QualityFlag.BAD: 1
                    }
                    
                    min_quality_level = quality_levels.get(self.config.min_quality, 0)
                    
                    # 为每个点检查质量
                    if isinstance(obs.quality_flag, np.ndarray):
                        valid_mask = np.array([quality_levels.get(qf, 0) >= min_quality_level 
                                             for qf in obs.quality_flag])
                    else:
                        valid_mask = np.ones(len(obs.latitude), dtype=bool)
                else:
                    valid_mask = np.ones(len(obs.latitude), dtype=bool)
                
                # 表面类型过滤
                if obs.surface_type is not None and self.config.allowed_surface_types:
                    surface_mask = np.isin(obs.surface_type, self.config.allowed_surface_types)
                    valid_mask = valid_mask & surface_mask
                
                # 异常值检测
                if self.config.outlier_threshold > 0:
                    elev_mean = np.nanmean(obs.elevation)
                    elev_std = np.nanstd(obs.elevation)
                    
                    if elev_std > 0:
                        outlier_mask = np.abs(obs.elevation - elev_mean) <= self.config.outlier_threshold * elev_std
                        valid_mask = valid_mask & outlier_mask
                
                # 应用过滤
                if np.any(valid_mask):
                    filtered_obs = ICESatObservation(
                        observation_id=obs.observation_id,
                        mission=obs.mission,
                        data_product=obs.data_product,
                        processing_level=obs.processing_level,
                        acquisition_time=obs.acquisition_time,
                        latitude=obs.latitude[valid_mask],
                        longitude=obs.longitude[valid_mask],
                        elevation=obs.elevation[valid_mask],
                        track_id=obs.track_id,
                        beam_type=obs.beam_type,
                        cycle_number=obs.cycle_number,
                        quality_flag=obs.quality_flag[valid_mask] if obs.quality_flag is not None else None,
                        uncertainty=obs.uncertainty[valid_mask] if obs.uncertainty is not None else None,
                        surface_type=obs.surface_type[valid_mask] if obs.surface_type is not None else None,
                        units=obs.units,
                        datum=obs.datum,
                        processing_info=obs.processing_info,
                        metadata=obs.metadata
                    )
                    filtered.append(filtered_obs)
            
            except Exception as e:
                self.logger.warning(f"观测数据过滤失败 {obs.observation_id}: {e}")
        
        return filtered
    
    def interpolate_to_grid(self, 
                           observations: List[ICESatObservation],
                           grid_lat: np.ndarray, 
                           grid_lon: np.ndarray) -> np.ndarray:
        """将观测数据插值到网格"""
        try:
            # 收集所有观测点
            all_lats = []
            all_lons = []
            all_elevs = []
            
            for obs in observations:
                all_lats.extend(obs.latitude)
                all_lons.extend(obs.longitude)
                all_elevs.extend(obs.elevation)
            
            if len(all_lats) == 0:
                return np.full(grid_lat.shape, np.nan)
            
            # 转换为数组
            points = np.column_stack([np.array(all_lats), np.array(all_lons)])
            values = np.array(all_elevs)
            
            # 目标网格点
            grid_points = np.column_stack([grid_lat.flatten(), grid_lon.flatten()])
            
            # 插值
            if self.config.interpolation_method == "linear":
                interpolated = griddata(points, values, grid_points, method='linear', fill_value=np.nan)
            elif self.config.interpolation_method == "cubic":
                interpolated = griddata(points, values, grid_points, method='cubic', fill_value=np.nan)
            elif self.config.interpolation_method == "rbf":
                # 使用径向基函数插值
                rbf = RBFInterpolator(points, values, kernel='linear')
                interpolated = rbf(grid_points)
            else:
                # 默认使用最近邻
                interpolated = griddata(points, values, grid_points, method='nearest', fill_value=np.nan)
            
            return interpolated.reshape(grid_lat.shape)
        
        except Exception as e:
            self.logger.error(f"网格插值失败: {e}")
            return np.full(grid_lat.shape, np.nan)
    
    def compute_elevation_changes(self, 
                                observations: List[ICESatObservation]) -> Dict[str, Any]:
        """计算高程变化"""
        try:
            # 按时间排序
            sorted_obs = sorted(observations, key=lambda x: x.acquisition_time)
            
            if len(sorted_obs) < 2:
                return {'error': '需要至少两个时间点的数据'}
            
            # 计算时间间隔和高程变化
            changes = []
            
            for i in range(len(sorted_obs) - 1):
                obs1 = sorted_obs[i]
                obs2 = sorted_obs[i + 1]
                
                # 找到空间上匹配的点
                indices, valid_mask = self.geo_processor.find_nearest_points(
                    obs2.latitude, obs2.longitude,
                    obs1.latitude, obs1.longitude
                )
                
                if np.any(valid_mask):
                    valid_indices = indices[valid_mask]
                    valid_obs2_indices = np.where(valid_mask)[0]
                    
                    time_diff = (obs2.acquisition_time - obs1.acquisition_time).total_seconds() / (365.25 * 24 * 3600)  # 年
                    
                    for j, (obs2_idx, obs1_idx) in enumerate(zip(valid_obs2_indices, valid_indices)):
                        elev_change = obs2.elevation[obs2_idx] - obs1.elevation[obs1_idx]
                        rate = elev_change / time_diff if time_diff > 0 else 0
                        
                        changes.append({
                            'latitude': obs2.latitude[obs2_idx],
                            'longitude': obs2.longitude[obs2_idx],
                            'elevation_change': elev_change,
                            'time_diff': time_diff,
                            'change_rate': rate,
                            'obs1_time': obs1.acquisition_time,
                            'obs2_time': obs2.acquisition_time
                        })
            
            if not changes:
                return {'error': '未找到匹配的观测点'}
            
            # 统计分析
            change_rates = [c['change_rate'] for c in changes]
            
            return {
                'changes': changes,
                'mean_rate': np.mean(change_rates),
                'std_rate': np.std(change_rates),
                'min_rate': np.min(change_rates),
                'max_rate': np.max(change_rates),
                'total_points': len(changes)
            }
        
        except Exception as e:
            self.logger.error(f"高程变化计算失败: {e}")
            return {'error': str(e)}

class ICESatValidator:
    """ICESat数据验证器"""
    
    def __init__(self, config: ICESatValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_processor = DataProcessor(config)
        self.geo_processor = GeospatialProcessor(config)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def validate(self, 
                model: nn.Module,
                icesat_observations: List[ICESatObservation]) -> Dict[str, Any]:
        """执行ICESat数据验证"""
        try:
            self.logger.info("开始ICESat数据验证...")
            
            # 过滤观测数据
            filtered_observations = self.data_processor.filter_observations(icesat_observations)
            
            if not filtered_observations:
                self.logger.warning("没有有效的ICESat观测数据")
                return {'error': '没有有效的观测数据'}
            
            # 生成模型预测
            model_predictions = self._generate_model_predictions(model, filtered_observations)
            
            # 执行不同模式的验证
            validation_results = {}
            
            # 点对点验证
            if ValidationMode.POINT_TO_POINT in self.config.validation_modes:
                validation_results['point_to_point'] = self._validate_point_to_point(
                    model_predictions, filtered_observations
                )
            
            # 交叉点验证
            if ValidationMode.CROSSOVER in self.config.validation_modes:
                validation_results['crossover'] = self._validate_crossovers(
                    model_predictions, filtered_observations
                )
            
            # 网格化验证
            if ValidationMode.GRIDDED in self.config.validation_modes:
                validation_results['gridded'] = self._validate_gridded(
                    model_predictions, filtered_observations
                )
            
            # 时间序列验证
            if ValidationMode.TIME_SERIES in self.config.validation_modes:
                validation_results['time_series'] = self._validate_time_series(
                    model_predictions, filtered_observations
                )
            
            # 高程变化分析
            validation_results['elevation_changes'] = self._analyze_elevation_changes(
                model_predictions, filtered_observations
            )
            
            # 不确定性分析
            validation_results['uncertainty_analysis'] = self._analyze_uncertainty(
                model_predictions, filtered_observations
            )
            
            # 绘制结果
            if self.config.enable_plotting:
                self._plot_results(validation_results, filtered_observations)
            
            # 汇总结果
            summary = {
                'total_observations': len(icesat_observations),
                'filtered_observations': len(filtered_observations),
                'validation_results': validation_results,
                'config': self.config
            }
            
            self.logger.info("ICESat数据验证完成")
            return summary
        
        except Exception as e:
            self.logger.error(f"ICESat验证失败: {e}")
            return {'error': str(e)}
    
    def _generate_model_predictions(self, 
                                  model: nn.Module,
                                  observations: List[ICESatObservation]) -> Dict[str, Any]:
        """生成模型预测"""
        predictions = {}
        
        model.eval()
        with torch.no_grad():
            for obs in observations:
                try:
                    # 准备输入数据
                    inputs = []
                    
                    for lat, lon in zip(obs.latitude, obs.longitude):
                        # 创建时间特征
                        time_features = self._create_time_features(obs.acquisition_time)
                        
                        # 组合输入特征 [lat, lon, time_features...]
                        input_vec = [lat, lon] + time_features
                        inputs.append(input_vec)
                    
                    if inputs:
                        inputs_tensor = torch.FloatTensor(inputs)
                        pred_elevations = model(inputs_tensor).numpy().flatten()
                        
                        predictions[obs.observation_id] = {
                            'predictions': pred_elevations,
                            'latitudes': obs.latitude,
                            'longitudes': obs.longitude,
                            'observations': obs.elevation,
                            'time': obs.acquisition_time,
                            'observation': obs
                        }
                
                except Exception as e:
                    self.logger.warning(f"模型预测失败 {obs.observation_id}: {e}")
        
        return predictions
    
    def _create_time_features(self, time: datetime) -> List[float]:
        """创建时间特征"""
        # 年份归一化
        year_norm = (time.year - 2000) / 50.0
        
        # 月份的周期性特征
        month_sin = np.sin(2 * np.pi * time.month / 12)
        month_cos = np.cos(2 * np.pi * time.month / 12)
        
        # 日期的周期性特征
        day_sin = np.sin(2 * np.pi * time.timetuple().tm_yday / 365)
        day_cos = np.cos(2 * np.pi * time.timetuple().tm_yday / 365)
        
        return [year_norm, month_sin, month_cos, day_sin, day_cos]
    
    def _validate_point_to_point(self, 
                               predictions: Dict[str, Any],
                               observations: List[ICESatObservation]) -> Dict[str, Any]:
        """点对点验证"""
        try:
            all_predictions = []
            all_observations = []
            all_uncertainties = []
            
            for obs in observations:
                if obs.observation_id in predictions:
                    pred_data = predictions[obs.observation_id]
                    
                    all_predictions.extend(pred_data['predictions'])
                    all_observations.extend(pred_data['observations'])
                    
                    if obs.uncertainty is not None:
                        all_uncertainties.extend(obs.uncertainty)
            
            if len(all_predictions) == 0:
                return {'error': '没有有效的预测数据'}
            
            # 转换为数组
            pred_array = np.array(all_predictions)
            obs_array = np.array(all_observations)
            
            # 移除无效值
            valid_mask = ~(np.isnan(pred_array) | np.isnan(obs_array))
            pred_valid = pred_array[valid_mask]
            obs_valid = obs_array[valid_mask]
            
            if len(pred_valid) == 0:
                return {'error': '没有有效的数据点'}
            
            # 计算统计指标
            correlation = np.corrcoef(pred_valid, obs_valid)[0, 1]
            rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))
            mae = mean_absolute_error(obs_valid, pred_valid)
            bias = np.mean(pred_valid - obs_valid)
            r_squared = r2_score(obs_valid, pred_valid)
            
            # 计算相对误差
            relative_errors = np.abs(pred_valid - obs_valid) / np.abs(obs_valid)
            mean_relative_error = np.mean(relative_errors[np.isfinite(relative_errors)])
            
            return {
                'correlation': correlation,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'r_squared': r_squared,
                'mean_relative_error': mean_relative_error,
                'total_points': len(pred_valid),
                'predictions': pred_valid,
                'observations': obs_valid
            }
        
        except Exception as e:
            self.logger.error(f"点对点验证失败: {e}")
            return {'error': str(e)}
    
    def _validate_crossovers(self, 
                           predictions: Dict[str, Any],
                           observations: List[ICESatObservation]) -> Dict[str, Any]:
        """交叉点验证"""
        try:
            # 检测交叉点
            crossovers = self.geo_processor.detect_crossovers(observations)
            
            if not crossovers:
                return {'error': '未找到交叉点'}
            
            # 分析交叉点处的模型性能
            crossover_results = []
            
            for crossover in crossovers:
                obs1_id = crossover['obs1_id']
                obs2_id = crossover['obs2_id']
                
                if obs1_id in predictions and obs2_id in predictions:
                    pred1 = predictions[obs1_id]['predictions'][crossover['obs1_point']]
                    pred2 = predictions[obs2_id]['predictions'][crossover['obs2_point']]
                    
                    # 模型预测的高程差
                    model_elev_diff = pred1 - pred2
                    
                    # 观测的高程差
                    obs_elev_diff = crossover['elevation_diff']
                    
                    crossover_results.append({
                        'model_diff': model_elev_diff,
                        'observed_diff': obs_elev_diff,
                        'error': model_elev_diff - obs_elev_diff,
                        'time_diff': crossover['time_diff'],
                        'distance': crossover['distance']
                    })
            
            if not crossover_results:
                return {'error': '交叉点处无有效预测'}
            
            # 统计分析
            errors = [r['error'] for r in crossover_results]
            model_diffs = [r['model_diff'] for r in crossover_results]
            obs_diffs = [r['observed_diff'] for r in crossover_results]
            
            return {
                'crossover_count': len(crossover_results),
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'rmse': np.sqrt(np.mean(np.array(errors)**2)),
                'correlation': np.corrcoef(model_diffs, obs_diffs)[0, 1],
                'crossover_results': crossover_results
            }
        
        except Exception as e:
            self.logger.error(f"交叉点验证失败: {e}")
            return {'error': str(e)}
    
    def _validate_gridded(self, 
                         predictions: Dict[str, Any],
                         observations: List[ICESatObservation]) -> Dict[str, Any]:
        """网格化验证"""
        try:
            # 创建网格
            grid_lat, grid_lon = self.geo_processor.create_grid(observations)
            
            if grid_lat.size == 0:
                return {'error': '网格创建失败'}
            
            # 将观测数据插值到网格
            obs_grid = self.data_processor.interpolate_to_grid(observations, grid_lat, grid_lon)
            
            # 生成网格上的模型预测
            model_grid = self._predict_on_grid(predictions, grid_lat, grid_lon)
            
            # 计算网格化指标
            valid_mask = ~(np.isnan(obs_grid) | np.isnan(model_grid))
            
            if np.sum(valid_mask) == 0:
                return {'error': '网格上无有效数据'}
            
            obs_valid = obs_grid[valid_mask]
            model_valid = model_grid[valid_mask]
            
            correlation = np.corrcoef(obs_valid, model_valid)[0, 1]
            rmse = np.sqrt(mean_squared_error(obs_valid, model_valid))
            mae = mean_absolute_error(obs_valid, model_valid)
            bias = np.mean(model_valid - obs_valid)
            
            return {
                'correlation': correlation,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'valid_grid_points': np.sum(valid_mask),
                'total_grid_points': grid_lat.size,
                'grid_coverage': np.sum(valid_mask) / grid_lat.size,
                'obs_grid': obs_grid,
                'model_grid': model_grid,
                'grid_lat': grid_lat,
                'grid_lon': grid_lon
            }
        
        except Exception as e:
            self.logger.error(f"网格化验证失败: {e}")
            return {'error': str(e)}
    
    def _predict_on_grid(self, 
                        predictions: Dict[str, Any],
                        grid_lat: np.ndarray, 
                        grid_lon: np.ndarray) -> np.ndarray:
        """在网格上进行预测"""
        try:
            # 收集所有预测点
            all_lats = []
            all_lons = []
            all_preds = []
            
            for pred_data in predictions.values():
                all_lats.extend(pred_data['latitudes'])
                all_lons.extend(pred_data['longitudes'])
                all_preds.extend(pred_data['predictions'])
            
            if len(all_lats) == 0:
                return np.full(grid_lat.shape, np.nan)
            
            # 插值到网格
            points = np.column_stack([np.array(all_lats), np.array(all_lons)])
            values = np.array(all_preds)
            grid_points = np.column_stack([grid_lat.flatten(), grid_lon.flatten()])
            
            interpolated = griddata(points, values, grid_points, method='linear', fill_value=np.nan)
            
            return interpolated.reshape(grid_lat.shape)
        
        except Exception as e:
            self.logger.error(f"网格预测失败: {e}")
            return np.full(grid_lat.shape, np.nan)
    
    def _validate_time_series(self, 
                            predictions: Dict[str, Any],
                            observations: List[ICESatObservation]) -> Dict[str, Any]:
        """时间序列验证"""
        try:
            # 按时间排序
            sorted_obs = sorted(observations, key=lambda x: x.acquisition_time)
            
            if len(sorted_obs) < 2:
                return {'error': '时间序列数据不足'}
            
            # 计算时间序列的平均高程
            times = []
            model_means = []
            obs_means = []
            
            for obs in sorted_obs:
                if obs.observation_id in predictions:
                    pred_data = predictions[obs.observation_id]
                    
                    model_mean = np.nanmean(pred_data['predictions'])
                    obs_mean = np.nanmean(pred_data['observations'])
                    
                    if not (np.isnan(model_mean) or np.isnan(obs_mean)):
                        times.append(obs.acquisition_time)
                        model_means.append(model_mean)
                        obs_means.append(obs_mean)
            
            if len(times) < 2:
                return {'error': '有效时间序列数据不足'}
            
            # 计算时间序列指标
            correlation = np.corrcoef(model_means, obs_means)[0, 1]
            rmse = np.sqrt(mean_squared_error(obs_means, model_means))
            
            # 趋势分析
            if self.config.enable_trend_analysis:
                time_numeric = [(t - times[0]).total_seconds() / (365.25 * 24 * 3600) for t in times]
                
                model_slope, _, model_r, _, _ = stats.linregress(time_numeric, model_means)
                obs_slope, _, obs_r, _, _ = stats.linregress(time_numeric, obs_means)
                
                trend_analysis = {
                    'model_trend': model_slope,
                    'observed_trend': obs_slope,
                    'trend_ratio': model_slope / obs_slope if obs_slope != 0 else np.inf,
                    'model_trend_r': model_r,
                    'observed_trend_r': obs_r
                }
            else:
                trend_analysis = {}
            
            return {
                'correlation': correlation,
                'rmse': rmse,
                'time_points': len(times),
                'time_span': (min(times), max(times)),
                'model_series': model_means,
                'observed_series': obs_means,
                'times': times,
                'trend_analysis': trend_analysis
            }
        
        except Exception as e:
            self.logger.error(f"时间序列验证失败: {e}")
            return {'error': str(e)}
    
    def _analyze_elevation_changes(self, 
                                 predictions: Dict[str, Any],
                                 observations: List[ICESatObservation]) -> Dict[str, Any]:
        """分析高程变化"""
        try:
            # 计算观测的高程变化
            obs_changes = self.data_processor.compute_elevation_changes(observations)
            
            if 'error' in obs_changes:
                return obs_changes
            
            # 计算模型预测的高程变化
            # 这里简化处理，实际应该匹配相同位置的不同时间点
            model_changes = []
            
            # 按时间排序的观测
            sorted_obs = sorted(observations, key=lambda x: x.acquisition_time)
            
            for i in range(len(sorted_obs) - 1):
                obs1 = sorted_obs[i]
                obs2 = sorted_obs[i + 1]
                
                if obs1.observation_id in predictions and obs2.observation_id in predictions:
                    pred1 = predictions[obs1.observation_id]
                    pred2 = predictions[obs2.observation_id]
                    
                    # 简化：使用平均值
                    model_mean1 = np.nanmean(pred1['predictions'])
                    model_mean2 = np.nanmean(pred2['predictions'])
                    
                    time_diff = (obs2.acquisition_time - obs1.acquisition_time).total_seconds() / (365.25 * 24 * 3600)
                    
                    if time_diff > 0 and not (np.isnan(model_mean1) or np.isnan(model_mean2)):
                        change_rate = (model_mean2 - model_mean1) / time_diff
                        model_changes.append(change_rate)
            
            # 比较模型和观测的变化率
            obs_rates = [c['change_rate'] for c in obs_changes['changes']]
            
            if model_changes and obs_rates:
                # 简化比较：使用平均值
                model_mean_rate = np.mean(model_changes)
                obs_mean_rate = obs_changes['mean_rate']
                
                return {
                    'model_mean_rate': model_mean_rate,
                    'observed_mean_rate': obs_mean_rate,
                    'rate_difference': model_mean_rate - obs_mean_rate,
                    'rate_ratio': model_mean_rate / obs_mean_rate if obs_mean_rate != 0 else np.inf,
                    'observed_changes': obs_changes
                }
            else:
                return {'error': '无法计算高程变化'}
        
        except Exception as e:
            self.logger.error(f"高程变化分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_uncertainty(self, 
                           predictions: Dict[str, Any],
                           observations: List[ICESatObservation]) -> Dict[str, Any]:
        """不确定性分析"""
        try:
            uncertainties = []
            prediction_errors = []
            
            for obs in observations:
                if obs.observation_id in predictions and obs.uncertainty is not None:
                    pred_data = predictions[obs.observation_id]
                    
                    # 计算预测误差
                    errors = np.abs(pred_data['predictions'] - pred_data['observations'])
                    
                    uncertainties.extend(obs.uncertainty)
                    prediction_errors.extend(errors)
            
            if not uncertainties:
                return {'error': '无不确定性数据'}
            
            uncertainties = np.array(uncertainties)
            prediction_errors = np.array(prediction_errors)
            
            # 移除无效值
            valid_mask = ~(np.isnan(uncertainties) | np.isnan(prediction_errors))
            unc_valid = uncertainties[valid_mask]
            err_valid = prediction_errors[valid_mask]
            
            if len(unc_valid) == 0:
                return {'error': '无有效不确定性数据'}
            
            # 分析不确定性与误差的关系
            correlation = np.corrcoef(unc_valid, err_valid)[0, 1]
            
            # 计算不确定性统计
            return {
                'uncertainty_mean': np.mean(unc_valid),
                'uncertainty_std': np.std(unc_valid),
                'error_mean': np.mean(err_valid),
                'error_std': np.std(err_valid),
                'uncertainty_error_correlation': correlation,
                'data_points': len(unc_valid)
            }
        
        except Exception as e:
            self.logger.error(f"不确定性分析失败: {e}")
            return {'error': str(e)}
    
    def _plot_results(self, 
                     results: Dict[str, Any],
                     observations: List[ICESatObservation]):
        """绘制验证结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ICESat数据验证结果', fontsize=16, fontweight='bold')
            
            # 点对点验证散点图
            if 'point_to_point' in results and 'predictions' in results['point_to_point']:
                ax = axes[0, 0]
                ptp_data = results['point_to_point']
                
                ax.scatter(ptp_data['observations'], ptp_data['predictions'], alpha=0.6)
                
                # 添加1:1线
                min_val = min(np.min(ptp_data['observations']), np.min(ptp_data['predictions']))
                max_val = max(np.max(ptp_data['observations']), np.max(ptp_data['predictions']))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax.set_xlabel('ICESat观测高程 (m)')
                ax.set_ylabel('模型预测高程 (m)')
                ax.set_title(f"点对点验证 (R² = {ptp_data.get('r_squared', 0):.3f})")
                ax.grid(True, alpha=0.3)
            
            # 时间序列对比
            if 'time_series' in results and 'times' in results['time_series']:
                ax = axes[0, 1]
                ts_data = results['time_series']
                
                ax.plot(ts_data['times'], ts_data['model_series'], 'b-', label='模型预测', linewidth=2)
                ax.plot(ts_data['times'], ts_data['observed_series'], 'r-', label='ICESat观测', linewidth=2)
                ax.set_xlabel('时间')
                ax.set_ylabel('平均高程 (m)')
                ax.set_title(f"时间序列对比 (相关系数: {ts_data.get('correlation', 0):.3f})")
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 网格化验证
            if 'gridded' in results and 'obs_grid' in results['gridded']:
                ax = axes[1, 0]
                grid_data = results['gridded']
                
                # 显示观测网格
                im = ax.imshow(grid_data['obs_grid'], cmap='viridis', aspect='auto')
                ax.set_title('观测数据网格')
                plt.colorbar(im, ax=ax, label='高程 (m)')
            
            # 高程变化分析
            if 'elevation_changes' in results and 'model_mean_rate' in results['elevation_changes']:
                ax = axes[1, 1]
                change_data = results['elevation_changes']
                
                categories = ['模型', 'ICESat']
                rates = [change_data['model_mean_rate'], change_data['observed_mean_rate']]
                
                bars = ax.bar(categories, rates, color=['blue', 'red'], alpha=0.7)
                ax.set_ylabel('高程变化率 (m/年)')
                ax.set_title('高程变化率对比')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{rate:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图片
            if self.config.save_plots:
                import os
                os.makedirs(self.config.plot_dir, exist_ok=True)
                plt.savefig(f"{self.config.plot_dir}/icesat_validation_results.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            
            if self.config.enable_plotting:
                plt.show()
        
        except Exception as e:
            self.logger.warning(f"结果绘制失败: {e}")
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       observations: List[ICESatObservation]) -> str:
        """生成验证报告"""
        report_lines = [
            "="*80,
            "ICESat数据验证报告",
            "="*80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. 数据概览",
            "-"*40,
            f"总观测数量: {results.get('total_observations', 0)}",
            f"过滤后观测数量: {results.get('filtered_observations', 0)}",
        ]
        
        # 点对点验证结果
        if 'validation_results' in results and 'point_to_point' in results['validation_results']:
            ptp_results = results['validation_results']['point_to_point']
            
            report_lines.extend([
                "",
                "2. 点对点验证",
                "-"*40,
                f"相关系数: {ptp_results.get('correlation', 'N/A'):.4f}",
                f"RMSE: {ptp_results.get('rmse', 'N/A'):.4f} m",
                f"MAE: {ptp_results.get('mae', 'N/A'):.4f} m",
                f"偏差: {ptp_results.get('bias', 'N/A'):.4f} m",
                f"R²: {ptp_results.get('r_squared', 'N/A'):.4f}",
                f"数据点数: {ptp_results.get('total_points', 'N/A')}"
            ])
        
        # 交叉点验证结果
        if 'validation_results' in results and 'crossover' in results['validation_results']:
            crossover_results = results['validation_results']['crossover']
            
            if 'crossover_count' in crossover_results:
                report_lines.extend([
                    "",
                    "3. 交叉点验证",
                    "-"*40,
                    f"交叉点数量: {crossover_results.get('crossover_count', 'N/A')}",
                    f"平均误差: {crossover_results.get('mean_error', 'N/A'):.4f} m",
                    f"RMSE: {crossover_results.get('rmse', 'N/A'):.4f} m",
                    f"相关系数: {crossover_results.get('correlation', 'N/A'):.4f}"
                ])
        
        # 网格化验证结果
        if 'validation_results' in results and 'gridded' in results['validation_results']:
            gridded_results = results['validation_results']['gridded']
            
            if 'correlation' in gridded_results:
                report_lines.extend([
                    "",
                    "4. 网格化验证",
                    "-"*40,
                    f"相关系数: {gridded_results.get('correlation', 'N/A'):.4f}",
                    f"RMSE: {gridded_results.get('rmse', 'N/A'):.4f} m",
                    f"偏差: {gridded_results.get('bias', 'N/A'):.4f} m",
                    f"网格覆盖率: {gridded_results.get('grid_coverage', 'N/A'):.2%}"
                ])
        
        # 高程变化分析
        if 'validation_results' in results and 'elevation_changes' in results['validation_results']:
            change_results = results['validation_results']['elevation_changes']
            
            if 'model_mean_rate' in change_results:
                report_lines.extend([
                    "",
                    "5. 高程变化分析",
                    "-"*40,
                    f"模型变化率: {change_results.get('model_mean_rate', 'N/A'):.4f} m/年",
                    f"观测变化率: {change_results.get('observed_mean_rate', 'N/A'):.4f} m/年",
                    f"变化率比值: {change_results.get('rate_ratio', 'N/A'):.4f}"
                ])
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)

def create_icesat_validator(config: Optional[ICESatValidationConfig] = None) -> ICESatValidator:
    """创建ICESat验证器"""
    if config is None:
        config = ICESatValidationConfig()
    
    return ICESatValidator(config)

if __name__ == "__main__":
    # 测试代码
    import torch.nn as nn
    
    # 创建简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(7, 64),  # lat, lon, year, month_sin, month_cos, day_sin, day_cos
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 创建测试数据
    test_observations = [
        ICESatObservation(
            observation_id="icesat_001",
            mission=ICESatMission.ICESAT2,
            data_product=DataProduct.ATL06,
            processing_level=ProcessingLevel.L3A,
            acquisition_time=datetime(2023, 6, 15),
            latitude=np.array([30.0, 30.1, 30.2]),
            longitude=np.array([90.0, 90.1, 90.2]),
            elevation=np.array([4500.0, 4520.0, 4510.0]),
            track_id="track_001",
            beam_type=BeamType.STRONG,
            quality_flag=np.array([QualityFlag.GOOD, QualityFlag.EXCELLENT, QualityFlag.GOOD]),
            uncertainty=np.array([0.5, 0.3, 0.4]),
            surface_type=np.array([SurfaceType.GLACIER, SurfaceType.GLACIER, SurfaceType.GLACIER])
        ),
        ICESatObservation(
            observation_id="icesat_002",
            mission=ICESatMission.ICESAT2,
            data_product=DataProduct.ATL06,
            processing_level=ProcessingLevel.L3A,
            acquisition_time=datetime(2023, 7, 15),
            latitude=np.array([30.05, 30.15, 30.25]),
            longitude=np.array([90.05, 90.15, 90.25]),
            elevation=np.array([4495.0, 4515.0, 4505.0]),
            track_id="track_002",
            beam_type=BeamType.STRONG,
            quality_flag=np.array([QualityFlag.GOOD, QualityFlag.FAIR, QualityFlag.GOOD]),
            uncertainty=np.array([0.6, 0.4, 0.5]),
            surface_type=np.array([SurfaceType.GLACIER, SurfaceType.GLACIER, SurfaceType.GLACIER])
        )
    ]
    
    # 创建配置
    config = ICESatValidationConfig(
        temporal_tolerance=30.0,
        spatial_tolerance=1000.0,
        enable_plotting=True,
        verbose=True
    )
    
    # 创建验证器
    validator = create_icesat_validator(config)
    
    # 创建测试模型
    model = TestModel()
    
    # 执行验证
    print("开始ICESat数据验证测试...")
    results = validator.validate(model, test_observations)
    
    # 打印结果
    if 'error' not in results:
        print("\n验证完成！")
        print(f"总观测数量: {results['total_observations']}")
        print(f"过滤后观测数量: {results['filtered_observations']}")
        
        # 生成报告
        report = validator.generate_report(results, test_observations)
        print("\n" + report)
    else:
        print(f"验证失败: {results['error']}")
    
    print("\nICESat验证测试完成！")