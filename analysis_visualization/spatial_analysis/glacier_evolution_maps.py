#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冰川演化地图模块

该模块实现了冰川时空演化的地图可视化功能，包括冰川边界变化、
厚度演化、速度场变化等的地图展示。

主要功能:
- 冰川边界演化地图
- 冰川厚度变化地图
- 冰川速度场演化
- 多时期对比地图
- 动态演化动画
- 交互式地图展示

作者: Assistant
日期: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapType(Enum):
    """地图类型"""
    BOUNDARY_EVOLUTION = "boundary_evolution"          # 边界演化
    THICKNESS_CHANGE = "thickness_change"              # 厚度变化
    VELOCITY_FIELD = "velocity_field"                  # 速度场
    ELEVATION_CHANGE = "elevation_change"              # 高程变化
    MASS_BALANCE = "mass_balance"                      # 物质平衡
    TEMPERATURE_FIELD = "temperature_field"            # 温度场
    STRESS_FIELD = "stress_field"                      # 应力场

class VisualizationStyle(Enum):
    """可视化风格"""
    SCIENTIFIC = "scientific"                          # 科学风格
    PRESENTATION = "presentation"                      # 演示风格
    PUBLICATION = "publication"                        # 出版风格
    INTERACTIVE = "interactive"                        # 交互风格

class ColorScheme(Enum):
    """颜色方案"""
    VIRIDIS = "viridis"                                # Viridis
    PLASMA = "plasma"                                  # Plasma
    COOLWARM = "coolwarm"                              # 冷暖色
    SEISMIC = "seismic"                                # 地震色
    TERRAIN = "terrain"                                # 地形色
    GLACIER = "glacier"                                # 冰川专用色
    CUSTOM = "custom"                                  # 自定义

class AnimationType(Enum):
    """动画类型"""
    TIME_SERIES = "time_series"                        # 时间序列
    COMPARISON = "comparison"                          # 对比动画
    MORPHING = "morphing"                              # 形变动画
    FLOW = "flow"                                      # 流动动画

@dataclass
class MapConfig:
    """地图配置"""
    # 基本设置
    map_type: MapType = MapType.BOUNDARY_EVOLUTION
    style: VisualizationStyle = VisualizationStyle.SCIENTIFIC
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    
    # 地图范围
    extent: Optional[Tuple[float, float, float, float]] = None  # (lon_min, lon_max, lat_min, lat_max)
    projection: str = "PlateCarree"  # 投影方式
    
    # 分辨率设置
    resolution: float = 100.0  # 米
    grid_size: Tuple[int, int] = (500, 500)
    
    # 颜色设置
    color_levels: int = 20
    color_range: Optional[Tuple[float, float]] = None
    transparency: float = 0.8
    
    # 标注设置
    show_contours: bool = True
    contour_levels: int = 10
    show_labels: bool = True
    show_colorbar: bool = True
    
    # 地理要素
    show_coastlines: bool = True
    show_borders: bool = True
    show_rivers: bool = False
    show_lakes: bool = False
    show_topography: bool = True
    
    # 动画设置
    animation_type: AnimationType = AnimationType.TIME_SERIES
    frame_duration: float = 500  # 毫秒
    loop: bool = True
    
    # 输出设置
    figure_size: Tuple[float, float] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    output_dir: str = "./glacier_maps"
    
    # 交互设置
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    
    # 其他设置
    title: str = "冰川演化地图"
    subtitle: str = ""
    show_legend: bool = True
    font_size: int = 12

@dataclass
class GlacierData:
    """冰川数据"""
    # 空间坐标
    x: np.ndarray  # 经度或x坐标
    y: np.ndarray  # 纬度或y坐标
    
    # 时间信息
    time: np.ndarray  # 时间戳
    
    # 冰川属性
    thickness: Optional[np.ndarray] = None  # 厚度 [time, y, x]
    velocity_x: Optional[np.ndarray] = None  # x方向速度
    velocity_y: Optional[np.ndarray] = None  # y方向速度
    elevation: Optional[np.ndarray] = None  # 表面高程
    temperature: Optional[np.ndarray] = None  # 温度
    mass_balance: Optional[np.ndarray] = None  # 物质平衡
    
    # 边界信息
    boundaries: Optional[List[Polygon]] = None  # 各时期边界
    
    # 元数据
    glacier_id: str = "unknown"
    glacier_name: str = "Unknown Glacier"
    region: str = "Unknown Region"
    coordinate_system: str = "EPSG:4326"

class ColorMapGenerator:
    """颜色映射生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_colormap(self, scheme: ColorScheme, n_colors: int = 256) -> LinearSegmentedColormap:
        """获取颜色映射"""
        try:
            if scheme == ColorScheme.GLACIER:
                # 冰川专用颜色：深蓝到白到红
                colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', 
                         '#ffffff', '#fee0d2', '#fc9272', '#de2d26', '#a50f15']
                return LinearSegmentedColormap.from_list('glacier', colors, N=n_colors)
            
            elif scheme == ColorScheme.TERRAIN:
                # 地形颜色：绿到棕到白
                colors = ['#1a5490', '#2d8659', '#7cb342', '#c0ca33',
                         '#f9a825', '#ff8f00', '#d84315', '#ffffff']
                return LinearSegmentedColormap.from_list('terrain', colors, N=n_colors)
            
            elif scheme == ColorScheme.CUSTOM:
                # 自定义颜色方案
                colors = ['#440154', '#31688e', '#35b779', '#fde725']
                return LinearSegmentedColormap.from_list('custom', colors, N=n_colors)
            
            else:
                # 使用matplotlib内置颜色方案
                return plt.cm.get_cmap(scheme.value, n_colors)
        
        except Exception as e:
            self.logger.warning(f"颜色映射生成失败: {e}，使用默认viridis")
            return plt.cm.viridis

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def interpolate_to_grid(self, 
                           x: np.ndarray, 
                           y: np.ndarray, 
                           values: np.ndarray,
                           grid_x: np.ndarray, 
                           grid_y: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """插值到规则网格"""
        try:
            # 移除NaN值
            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                return np.full((len(grid_y), len(grid_x)), np.nan)
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            values_valid = values[valid_mask]
            
            # 创建网格
            grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
            
            # 插值
            grid_values = griddata(
                (x_valid, y_valid), values_valid, 
                (grid_X, grid_Y), method=method, fill_value=np.nan
            )
            
            return grid_values
        
        except Exception as e:
            self.logger.error(f"网格插值失败: {e}")
            return np.full((len(grid_y), len(grid_x)), np.nan)
    
    def smooth_data(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """数据平滑"""
        try:
            # 处理NaN值
            mask = ~np.isnan(data)
            if not np.any(mask):
                return data
            
            # 高斯滤波
            smoothed = gaussian_filter(data, sigma=sigma)
            
            # 保持原始NaN位置
            smoothed[~mask] = np.nan
            
            return smoothed
        
        except Exception as e:
            self.logger.warning(f"数据平滑失败: {e}")
            return data
    
    def calculate_gradients(self, data: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        try:
            grad_y, grad_x = np.gradient(data, dy, dx)
            return grad_x, grad_y
        
        except Exception as e:
            self.logger.warning(f"梯度计算失败: {e}")
            return np.zeros_like(data), np.zeros_like(data)
    
    def detect_boundaries(self, thickness: np.ndarray, threshold: float = 1.0) -> List[LineString]:
        """检测冰川边界"""
        try:
            from skimage import measure
            
            # 二值化
            binary = thickness > threshold
            
            # 查找轮廓
            contours = measure.find_contours(binary, 0.5)
            
            # 转换为LineString
            boundaries = []
            for contour in contours:
                if len(contour) > 2:
                    boundaries.append(LineString(contour))
            
            return boundaries
        
        except Exception as e:
            self.logger.warning(f"边界检测失败: {e}")
            return []

class StaticMapGenerator:
    """静态地图生成器"""
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.color_generator = ColorMapGenerator()
        self.data_processor = DataProcessor()
    
    def create_boundary_evolution_map(self, glacier_data: GlacierData) -> plt.Figure:
        """创建边界演化地图"""
        try:
            fig, ax = plt.subplots(
                figsize=self.config.figure_size,
                subplot_kw={'projection': getattr(ccrs, self.config.projection)()}
            )
            
            # 设置地图范围
            if self.config.extent:
                ax.set_extent(self.config.extent, crs=ccrs.PlateCarree())
            
            # 添加地理要素
            self._add_geographic_features(ax)
            
            # 绘制边界演化
            if glacier_data.boundaries:
                colors = plt.cm.viridis(np.linspace(0, 1, len(glacier_data.boundaries)))
                
                for i, boundary in enumerate(glacier_data.boundaries):
                    if hasattr(boundary, 'exterior'):
                        coords = list(boundary.exterior.coords)
                        lons, lats = zip(*coords)
                        
                        ax.plot(lons, lats, 
                               color=colors[i], 
                               linewidth=2, 
                               label=f'时期 {i+1}',
                               transform=ccrs.PlateCarree())
            
            # 设置标题和标签
            ax.set_title(f"{self.config.title} - 边界演化", fontsize=self.config.font_size + 2)
            
            if self.config.show_legend:
                ax.legend(loc='upper right')
            
            # 添加网格
            self._add_gridlines(ax)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"边界演化地图创建失败: {e}")
            return plt.figure()
    
    def create_thickness_change_map(self, glacier_data: GlacierData, time_index: int = -1) -> plt.Figure:
        """创建厚度变化地图"""
        try:
            fig, ax = plt.subplots(
                figsize=self.config.figure_size,
                subplot_kw={'projection': getattr(ccrs, self.config.projection)()}
            )
            
            # 设置地图范围
            if self.config.extent:
                ax.set_extent(self.config.extent, crs=ccrs.PlateCarree())
            
            # 添加地理要素
            self._add_geographic_features(ax)
            
            # 绘制厚度数据
            if glacier_data.thickness is not None:
                thickness = glacier_data.thickness[time_index]
                
                # 获取颜色映射
                cmap = self.color_generator.get_colormap(self.config.color_scheme)
                
                # 设置颜色范围
                if self.config.color_range:
                    vmin, vmax = self.config.color_range
                else:
                    vmin, vmax = np.nanmin(thickness), np.nanmax(thickness)
                
                # 绘制厚度场
                im = ax.contourf(
                    glacier_data.x, glacier_data.y, thickness,
                    levels=self.config.color_levels,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=self.config.transparency,
                    transform=ccrs.PlateCarree()
                )
                
                # 添加等高线
                if self.config.show_contours:
                    cs = ax.contour(
                        glacier_data.x, glacier_data.y, thickness,
                        levels=self.config.contour_levels,
                        colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree()
                    )
                    
                    if self.config.show_labels:
                        ax.clabel(cs, inline=True, fontsize=8)
                
                # 添加颜色条
                if self.config.show_colorbar:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
                    cbar.set_label('厚度 (m)', fontsize=self.config.font_size)
            
            # 设置标题
            time_str = f"时间: {glacier_data.time[time_index]:.1f}" if len(glacier_data.time) > time_index else ""
            ax.set_title(f"{self.config.title} - 厚度分布 {time_str}", fontsize=self.config.font_size + 2)
            
            # 添加网格
            self._add_gridlines(ax)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"厚度变化地图创建失败: {e}")
            return plt.figure()
    
    def create_velocity_field_map(self, glacier_data: GlacierData, time_index: int = -1) -> plt.Figure:
        """创建速度场地图"""
        try:
            fig, ax = plt.subplots(
                figsize=self.config.figure_size,
                subplot_kw={'projection': getattr(ccrs, self.config.projection)()}
            )
            
            # 设置地图范围
            if self.config.extent:
                ax.set_extent(self.config.extent, crs=ccrs.PlateCarree())
            
            # 添加地理要素
            self._add_geographic_features(ax)
            
            # 绘制速度场
            if glacier_data.velocity_x is not None and glacier_data.velocity_y is not None:
                vx = glacier_data.velocity_x[time_index]
                vy = glacier_data.velocity_y[time_index]
                
                # 计算速度大小
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                
                # 获取颜色映射
                cmap = self.color_generator.get_colormap(self.config.color_scheme)
                
                # 绘制速度大小
                im = ax.contourf(
                    glacier_data.x, glacier_data.y, velocity_magnitude,
                    levels=self.config.color_levels,
                    cmap=cmap,
                    alpha=self.config.transparency,
                    transform=ccrs.PlateCarree()
                )
                
                # 添加速度矢量
                skip = max(1, len(glacier_data.x) // 20)  # 稀疏化矢量
                ax.quiver(
                    glacier_data.x[::skip], glacier_data.y[::skip],
                    vx[::skip, ::skip], vy[::skip, ::skip],
                    scale=None, scale_units='xy',
                    color='white', alpha=0.8,
                    transform=ccrs.PlateCarree()
                )
                
                # 添加颜色条
                if self.config.show_colorbar:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
                    cbar.set_label('速度 (m/year)', fontsize=self.config.font_size)
            
            # 设置标题
            time_str = f"时间: {glacier_data.time[time_index]:.1f}" if len(glacier_data.time) > time_index else ""
            ax.set_title(f"{self.config.title} - 速度场 {time_str}", fontsize=self.config.font_size + 2)
            
            # 添加网格
            self._add_gridlines(ax)
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            self.logger.error(f"速度场地图创建失败: {e}")
            return plt.figure()
    
    def _add_geographic_features(self, ax):
        """添加地理要素"""
        try:
            if self.config.show_coastlines:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            if self.config.show_borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            
            if self.config.show_rivers:
                ax.add_feature(cfeature.RIVERS, linewidth=0.3)
            
            if self.config.show_lakes:
                ax.add_feature(cfeature.LAKES, alpha=0.5)
            
            if self.config.show_topography:
                ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        
        except Exception as e:
            self.logger.warning(f"地理要素添加失败: {e}")
    
    def _add_gridlines(self, ax):
        """添加网格线"""
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        except Exception as e:
            self.logger.warning(f"网格线添加失败: {e}")

class AnimationGenerator:
    """动画生成器"""
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.static_generator = StaticMapGenerator(config)
    
    def create_time_series_animation(self, glacier_data: GlacierData) -> animation.FuncAnimation:
        """创建时间序列动画"""
        try:
            fig, ax = plt.subplots(
                figsize=self.config.figure_size,
                subplot_kw={'projection': getattr(ccrs, self.config.projection)()}
            )
            
            # 设置地图范围
            if self.config.extent:
                ax.set_extent(self.config.extent, crs=ccrs.PlateCarree())
            
            # 添加地理要素
            self.static_generator._add_geographic_features(ax)
            
            # 初始化绘图元素
            im = None
            title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                               ha='center', fontsize=self.config.font_size + 2)
            
            def animate(frame):
                nonlocal im
                
                # 清除之前的图像
                if im is not None:
                    for coll in im.collections:
                        coll.remove()
                
                # 绘制当前时间步的数据
                if glacier_data.thickness is not None:
                    thickness = glacier_data.thickness[frame]
                    
                    # 获取颜色映射
                    cmap = self.static_generator.color_generator.get_colormap(self.config.color_scheme)
                    
                    # 绘制厚度场
                    im = ax.contourf(
                        glacier_data.x, glacier_data.y, thickness,
                        levels=self.config.color_levels,
                        cmap=cmap,
                        alpha=self.config.transparency,
                        transform=ccrs.PlateCarree()
                    )
                
                # 更新标题
                time_str = f"时间: {glacier_data.time[frame]:.1f}" if frame < len(glacier_data.time) else ""
                title_text.set_text(f"{self.config.title} - {time_str}")
                
                return [title_text]
            
            # 创建动画
            anim = animation.FuncAnimation(
                fig, animate, frames=len(glacier_data.time),
                interval=self.config.frame_duration,
                repeat=self.config.loop,
                blit=False
            )
            
            return anim
        
        except Exception as e:
            self.logger.error(f"时间序列动画创建失败: {e}")
            return None
    
    def save_animation(self, anim: animation.FuncAnimation, filename: str):
        """保存动画"""
        try:
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            filepath = os.path.join(self.config.output_dir, filename)
            
            if filename.endswith('.gif'):
                anim.save(filepath, writer='pillow', fps=1000/self.config.frame_duration)
            elif filename.endswith('.mp4'):
                anim.save(filepath, writer='ffmpeg', fps=1000/self.config.frame_duration)
            else:
                anim.save(f"{filepath}.gif", writer='pillow', fps=1000/self.config.frame_duration)
            
            self.logger.info(f"动画已保存到: {filepath}")
        
        except Exception as e:
            self.logger.error(f"动画保存失败: {e}")

class InteractiveMapGenerator:
    """交互式地图生成器"""
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_folium_map(self, glacier_data: GlacierData) -> folium.Map:
        """创建Folium交互式地图"""
        try:
            # 计算地图中心
            center_lat = np.mean([np.min(glacier_data.y), np.max(glacier_data.y)])
            center_lon = np.mean([np.min(glacier_data.x), np.max(glacier_data.x)])
            
            # 创建地图
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # 添加厚度数据
            if glacier_data.thickness is not None:
                # 使用最新时间步的数据
                thickness = glacier_data.thickness[-1]
                
                # 创建热力图
                heat_data = []
                for i in range(len(glacier_data.y)):
                    for j in range(len(glacier_data.x)):
                        if not np.isnan(thickness[i, j]) and thickness[i, j] > 0:
                            heat_data.append([
                                glacier_data.y[i], 
                                glacier_data.x[j], 
                                float(thickness[i, j])
                            ])
                
                if heat_data:
                    plugins.HeatMap(heat_data).add_to(m)
            
            # 添加边界
            if glacier_data.boundaries:
                for i, boundary in enumerate(glacier_data.boundaries):
                    if hasattr(boundary, 'exterior'):
                        coords = [[lat, lon] for lon, lat in boundary.exterior.coords]
                        
                        folium.Polygon(
                            locations=coords,
                            color=f'C{i}',
                            weight=2,
                            fillOpacity=0.1,
                            popup=f'时期 {i+1}'
                        ).add_to(m)
            
            return m
        
        except Exception as e:
            self.logger.error(f"Folium地图创建失败: {e}")
            return folium.Map()
    
    def create_plotly_map(self, glacier_data: GlacierData) -> go.Figure:
        """创建Plotly交互式地图"""
        try:
            fig = go.Figure()
            
            # 添加厚度数据
            if glacier_data.thickness is not None:
                thickness = glacier_data.thickness[-1]
                
                fig.add_trace(go.Heatmap(
                    x=glacier_data.x,
                    y=glacier_data.y,
                    z=thickness,
                    colorscale='Viridis',
                    name='厚度',
                    hovertemplate='经度: %{x}<br>纬度: %{y}<br>厚度: %{z:.1f}m<extra></extra>'
                ))
            
            # 设置布局
            fig.update_layout(
                title=self.config.title,
                xaxis_title='经度',
                yaxis_title='纬度',
                width=800,
                height=600
            )
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Plotly地图创建失败: {e}")
            return go.Figure()

class GlacierEvolutionMapper:
    """冰川演化地图生成器"""
    
    def __init__(self, config: Optional[MapConfig] = None):
        self.config = config or MapConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化生成器
        self.static_generator = StaticMapGenerator(self.config)
        self.animation_generator = AnimationGenerator(self.config)
        self.interactive_generator = InteractiveMapGenerator(self.config)
    
    def generate_evolution_maps(self, glacier_data: GlacierData) -> Dict[str, Any]:
        """生成冰川演化地图集"""
        results = {}
        
        try:
            # 静态地图
            if self.config.map_type == MapType.BOUNDARY_EVOLUTION:
                results['boundary_map'] = self.static_generator.create_boundary_evolution_map(glacier_data)
            
            elif self.config.map_type == MapType.THICKNESS_CHANGE:
                results['thickness_maps'] = []
                for i in range(len(glacier_data.time)):
                    fig = self.static_generator.create_thickness_change_map(glacier_data, i)
                    results['thickness_maps'].append(fig)
            
            elif self.config.map_type == MapType.VELOCITY_FIELD:
                results['velocity_maps'] = []
                for i in range(len(glacier_data.time)):
                    fig = self.static_generator.create_velocity_field_map(glacier_data, i)
                    results['velocity_maps'].append(fig)
            
            # 动画
            if self.config.animation_type == AnimationType.TIME_SERIES:
                results['animation'] = self.animation_generator.create_time_series_animation(glacier_data)
            
            # 交互式地图
            results['folium_map'] = self.interactive_generator.create_folium_map(glacier_data)
            results['plotly_map'] = self.interactive_generator.create_plotly_map(glacier_data)
            
            return results
        
        except Exception as e:
            self.logger.error(f"演化地图生成失败: {e}")
            return {}
    
    def save_maps(self, results: Dict[str, Any]):
        """保存地图"""
        try:
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 保存静态地图
            for key, value in results.items():
                if isinstance(value, plt.Figure):
                    filepath = os.path.join(self.config.output_dir, f"{key}.{self.config.save_format}")
                    value.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    self.logger.info(f"地图已保存: {filepath}")
                
                elif isinstance(value, list) and value and isinstance(value[0], plt.Figure):
                    for i, fig in enumerate(value):
                        filepath = os.path.join(self.config.output_dir, f"{key}_{i:03d}.{self.config.save_format}")
                        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    self.logger.info(f"地图序列已保存: {key}")
            
            # 保存动画
            if 'animation' in results and results['animation'] is not None:
                self.animation_generator.save_animation(results['animation'], 'glacier_evolution.gif')
            
            # 保存交互式地图
            if 'folium_map' in results:
                filepath = os.path.join(self.config.output_dir, 'interactive_map.html')
                results['folium_map'].save(filepath)
                self.logger.info(f"交互式地图已保存: {filepath}")
        
        except Exception as e:
            self.logger.error(f"地图保存失败: {e}")

def create_glacier_evolution_mapper(config: Optional[MapConfig] = None) -> GlacierEvolutionMapper:
    """创建冰川演化地图生成器"""
    return GlacierEvolutionMapper(config)

if __name__ == "__main__":
    # 测试代码
    print("开始冰川演化地图测试...")
    
    # 生成测试数据
    np.random.seed(42)
    
    # 空间网格
    x = np.linspace(85.0, 86.0, 50)  # 经度
    y = np.linspace(28.0, 29.0, 50)  # 纬度
    X, Y = np.meshgrid(x, y)
    
    # 时间序列
    time = np.arange(2000, 2021, 1)
    n_times = len(time)
    
    # 生成模拟冰川数据
    thickness = np.zeros((n_times, len(y), len(x)))
    velocity_x = np.zeros((n_times, len(y), len(x)))
    velocity_y = np.zeros((n_times, len(y), len(x)))
    
    for t in range(n_times):
        # 模拟厚度变化（逐渐减薄）
        center_x, center_y = len(x)//2, len(y)//2
        radius = 15 - t * 0.3  # 逐渐缩小
        
        for i in range(len(y)):
            for j in range(len(x)):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < radius:
                    thickness[t, i, j] = max(0, 100 - dist * 3 - t * 2)
                    # 模拟速度场
                    velocity_x[t, i, j] = (j - center_x) * 0.1
                    velocity_y[t, i, j] = (i - center_y) * 0.1
    
    # 创建冰川数据对象
    glacier_data = GlacierData(
        x=x, y=y, time=time,
        thickness=thickness,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        glacier_name="测试冰川",
        region="喜马拉雅"
    )
    
    # 创建地图配置
    config = MapConfig(
        map_type=MapType.THICKNESS_CHANGE,
        style=VisualizationStyle.SCIENTIFIC,
        color_scheme=ColorScheme.GLACIER,
        extent=(84.5, 86.5, 27.5, 29.5),
        show_contours=True,
        show_colorbar=True,
        animation_type=AnimationType.TIME_SERIES,
        title="喜马拉雅冰川演化"
    )
    
    # 创建地图生成器
    mapper = create_glacier_evolution_mapper(config)
    
    # 生成地图
    results = mapper.generate_evolution_maps(glacier_data)
    
    print(f"\n生成了 {len(results)} 个地图产品")
    
    # 显示结果
    if 'thickness_maps' in results:
        print(f"厚度地图数量: {len(results['thickness_maps'])}")
        # 显示第一个和最后一个
        if results['thickness_maps']:
            results['thickness_maps'][0].show()
            results['thickness_maps'][-1].show()
    
    if 'plotly_map' in results:
        print("显示交互式地图...")
        results['plotly_map'].show()
    
    # 保存地图
    mapper.save_maps(results)
    
    print("\n冰川演化地图测试完成！")