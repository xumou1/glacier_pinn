"""API Service Module.

This module provides REST API endpoints for serving PINNs models
through HTTP requests, including prediction, health checks, and model management.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .inference_engine import InferenceEngine, InferenceConfig
from .model_packaging import ModelVersionManager


# Pydantic models for API requests/responses
class CoordinateInput(BaseModel):
    """Input coordinates for prediction."""
    x: float = Field(..., description="X coordinate (meters)")
    y: float = Field(..., description="Y coordinate (meters)")
    t: float = Field(..., description="Time (years)")
    
    @validator('x', 'y')
    def validate_coordinates(cls, v):
        if not -1e6 <= v <= 1e6:
            raise ValueError('Coordinates must be within reasonable bounds')
        return v
    
    @validator('t')
    def validate_time(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Time must be between 0 and 100 years')
        return v


class BatchCoordinateInput(BaseModel):
    """Batch input coordinates for prediction."""
    coordinates: List[CoordinateInput] = Field(..., description="List of coordinates")
    
    @validator('coordinates')
    def validate_batch_size(cls, v):
        if len(v) > 10000:
            raise ValueError('Batch size cannot exceed 10000')
        return v


class GridInput(BaseModel):
    """Grid input for spatial predictions."""
    x_min: float = Field(..., description="Minimum X coordinate")
    x_max: float = Field(..., description="Maximum X coordinate")
    y_min: float = Field(..., description="Minimum Y coordinate")
    y_max: float = Field(..., description="Maximum Y coordinate")
    t: float = Field(..., description="Time value")
    resolution_x: int = Field(100, description="X resolution", ge=10, le=500)
    resolution_y: int = Field(100, description="Y resolution", ge=10, le=500)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    thickness: float = Field(..., description="Ice thickness (meters)")
    velocity_x: float = Field(..., description="X velocity (m/year)")
    velocity_y: float = Field(..., description="Y velocity (m/year)")
    coordinates: CoordinateInput = Field(..., description="Input coordinates")
    prediction_time: float = Field(..., description="Prediction time (seconds)")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_predictions: int = Field(..., description="Total number of predictions")
    total_time: float = Field(..., description="Total processing time (seconds)")
    average_time: float = Field(..., description="Average prediction time (seconds)")


class GridPredictionResponse(BaseModel):
    """Response model for grid predictions."""
    thickness: List[List[float]] = Field(..., description="Thickness grid")
    velocity_x: List[List[float]] = Field(..., description="X velocity grid")
    velocity_y: List[List[float]] = Field(..., description="Y velocity grid")
    x_coordinates: List[float] = Field(..., description="X coordinate array")
    y_coordinates: List[float] = Field(..., description="Y coordinate array")
    t: float = Field(..., description="Time value")
    resolution: List[int] = Field(..., description="Grid resolution [nx, ny]")
    prediction_time: float = Field(..., description="Prediction time (seconds)")


class ModelInfo(BaseModel):
    """Model information response."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    created_at: str = Field(..., description="Model creation timestamp")
    framework: str = Field(..., description="Framework used")
    model_type: str = Field(..., description="Model type")
    status: str = Field(..., description="Model status")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    inference_engine_ready: bool = Field(..., description="Whether inference engine is ready")
    uptime: float = Field(..., description="Service uptime (seconds)")
    performance_stats: Dict[str, Any] = Field(..., description="Performance statistics")


class APIService:
    """REST API service for PINNs model deployment."""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        model_info: Dict[str, Any],
        version_manager: Optional[ModelVersionManager] = None
    ):
        """
        Initialize API service.
        
        Args:
            inference_engine: Inference engine instance
            model_info: Model metadata
            version_manager: Optional version manager
        """
        self.inference_engine = inference_engine
        self.model_info = model_info
        self.version_manager = version_manager
        self.start_time = datetime.now()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Glacier Dynamics PINNs API",
            description="REST API for glacier dynamics predictions using Physics-Informed Neural Networks",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Glacier Dynamics PINNs API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            performance_stats = self.inference_engine.get_performance_stats()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=True,
                inference_engine_ready=True,
                uptime=uptime,
                performance_stats=performance_stats
            )
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def get_model_info():
            """Get model information."""
            return ModelInfo(
                name=self.model_info.get('name', 'Unknown'),
                version=self.model_info.get('version', '1.0.0'),
                created_at=self.model_info.get('created_at', ''),
                framework=self.model_info.get('framework', 'JAX/Flax'),
                model_type=self.model_info.get('model_type', 'PINNs'),
                status="active"
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(coordinates: CoordinateInput):
            """Make a single prediction."""
            try:
                start_time = datetime.now()
                
                # Convert to JAX array
                coords_array = jnp.array([[coordinates.x, coordinates.y, coordinates.t]])
                
                # Make prediction
                predictions = self.inference_engine.predict(coords_array)
                
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                return PredictionResponse(
                    thickness=float(predictions['thickness'][0]),
                    velocity_x=float(predictions['velocity_x'][0]),
                    velocity_y=float(predictions['velocity_y'][0]),
                    coordinates=coordinates,
                    prediction_time=prediction_time
                )
            
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(batch_input: BatchCoordinateInput):
            """Make batch predictions."""
            try:
                start_time = datetime.now()
                
                # Convert to JAX array
                coords_list = [
                    [coord.x, coord.y, coord.t] for coord in batch_input.coordinates
                ]
                coords_array = jnp.array(coords_list)
                
                # Make predictions
                predictions = self.inference_engine.predict(coords_array)
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                # Format response
                prediction_responses = []
                for i, coord in enumerate(batch_input.coordinates):
                    prediction_responses.append(PredictionResponse(
                        thickness=float(predictions['thickness'][i]),
                        velocity_x=float(predictions['velocity_x'][i]),
                        velocity_y=float(predictions['velocity_y'][i]),
                        coordinates=coord,
                        prediction_time=total_time / len(batch_input.coordinates)
                    ))
                
                return BatchPredictionResponse(
                    predictions=prediction_responses,
                    total_predictions=len(prediction_responses),
                    total_time=total_time,
                    average_time=total_time / len(prediction_responses)
                )
            
            except Exception as e:
                self.logger.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @self.app.post("/predict/grid", response_model=GridPredictionResponse)
        async def predict_grid(grid_input: GridInput):
            """Make grid predictions."""
            try:
                start_time = datetime.now()
                
                # Make grid predictions
                grid_predictions = self.inference_engine.predict_on_grid(
                    x_range=(grid_input.x_min, grid_input.x_max),
                    y_range=(grid_input.y_min, grid_input.y_max),
                    t_value=grid_input.t,
                    resolution=(grid_input.resolution_x, grid_input.resolution_y)
                )
                
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                # Convert to lists for JSON serialization
                return GridPredictionResponse(
                    thickness=grid_predictions['thickness'].tolist(),
                    velocity_x=grid_predictions['velocity_x'].tolist(),
                    velocity_y=grid_predictions['velocity_y'].tolist(),
                    x_coordinates=grid_predictions['X'][0, :].tolist(),
                    y_coordinates=grid_predictions['Y'][:, 0].tolist(),
                    t=grid_input.t,
                    resolution=[grid_input.resolution_x, grid_input.resolution_y],
                    prediction_time=prediction_time
                )
            
            except Exception as e:
                self.logger.error(f"Grid prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Grid prediction failed: {str(e)}")
        
        @self.app.get("/stats", response_model=Dict[str, Any])
        async def get_performance_stats():
            """Get performance statistics."""
            return self.inference_engine.get_performance_stats()
        
        @self.app.post("/cache/clear")
        async def clear_cache():
            """Clear inference cache."""
            self.inference_engine.clear_cache()
            return {"message": "Cache cleared successfully"}
        
        @self.app.post("/stats/reset")
        async def reset_stats():
            """Reset performance statistics."""
            self.inference_engine.reset_stats()
            return {"message": "Statistics reset successfully"}
        
        if self.version_manager:
            @self.app.get("/models", response_model=Dict[str, Any])
            async def list_models():
                """List available models."""
                # This would require implementing list_all_models in version_manager
                return {"message": "Model listing not implemented"}
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ):
        """Run the API service."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level
        )


class AsyncAPIService(APIService):
    """Asynchronous version of API service for better performance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_async_routes()
    
    def _setup_async_routes(self):
        """Setup asynchronous routes."""
        
        @self.app.post("/predict/async", response_model=Dict[str, str])
        async def predict_async(
            coordinates: CoordinateInput,
            background_tasks: BackgroundTasks
        ):
            """Submit asynchronous prediction request."""
            task_id = f"task_{datetime.now().timestamp()}"
            
            # Add background task
            background_tasks.add_task(
                self._process_async_prediction,
                task_id,
                coordinates
            )
            
            return {
                "task_id": task_id,
                "status": "submitted",
                "message": "Prediction task submitted"
            }
        
        @self.app.get("/predict/async/{task_id}")
        async def get_async_result(task_id: str):
            """Get asynchronous prediction result."""
            # This would require implementing a task queue/storage system
            return {
                "task_id": task_id,
                "status": "not_implemented",
                "message": "Async result retrieval not implemented"
            }
    
    async def _process_async_prediction(self, task_id: str, coordinates: CoordinateInput):
        """Process asynchronous prediction."""
        # This would store results in a database or cache
        # For now, just log the task
        self.logger.info(f"Processing async task {task_id} for coordinates {coordinates}")


def create_api_service(
    inference_engine: InferenceEngine,
    model_info: Dict[str, Any],
    version_manager: Optional[ModelVersionManager] = None,
    async_mode: bool = False
) -> Union[APIService, AsyncAPIService]:
    """
    Create an API service instance.
    
    Args:
        inference_engine: Inference engine
        model_info: Model metadata
        version_manager: Optional version manager
        async_mode: Whether to use async service
        
    Returns:
        API service instance
    """
    if async_mode:
        return AsyncAPIService(inference_engine, model_info, version_manager)
    else:
        return APIService(inference_engine, model_info, version_manager)


if __name__ == "__main__":
    # Example usage
    import tempfile
    from .inference_engine import create_inference_engine
    
    # Mock model function
    def mock_model_apply(params, inputs):
        x, y, t = inputs[0, 0], inputs[0, 1], inputs[0, 2]
        thickness = 100.0 + 10.0 * jnp.sin(x / 1000.0) * jnp.cos(y / 1000.0)
        velocity_x = 10.0 + 2.0 * jnp.sin(t)
        velocity_y = 5.0 + 1.0 * jnp.cos(t)
        return jnp.array([thickness, velocity_x, velocity_y])
    
    # Mock parameters and model info
    mock_params = {'dummy': jnp.array([1.0])}
    mock_model_info = {
        'name': 'glacier_dynamics_model',
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'framework': 'JAX/Flax',
        'model_type': 'PINNs'
    }
    
    # Create inference engine
    config = InferenceConfig(batch_size=1000, enable_jit=True, enable_caching=True)
    engine = create_inference_engine(mock_model_apply, mock_params, config)
    
    # Create and run API service
    api_service = create_api_service(engine, mock_model_info)
    
    print("Starting API service...")
    print("API documentation available at: http://localhost:8000/docs")
    api_service.run(host="localhost", port=8000)