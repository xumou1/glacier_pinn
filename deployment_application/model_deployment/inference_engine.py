"""Inference Engine Module.

This module provides high-performance inference capabilities for deployed
PINNs models, including batch processing, caching, and optimization.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze, unfreeze


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    batch_size: int = 1000
    max_workers: int = 4
    enable_jit: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    precision: str = "float32"  # "float32" or "float64"
    device: str = "cpu"  # "cpu" or "gpu"


class InferenceEngine:
    """High-performance inference engine for PINNs models."""
    
    def __init__(
        self,
        model_apply_fn: Callable,
        model_params: Dict[str, Any],
        config: InferenceConfig = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_apply_fn: Model application function
            model_params: Trained model parameters
            config: Inference configuration
        """
        self.model_apply_fn = model_apply_fn
        self.model_params = freeze(model_params)
        self.config = config or InferenceConfig()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize JIT compilation if enabled
        if self.config.enable_jit:
            self._setup_jit_functions()
        
        # Initialize cache if enabled
        if self.config.enable_caching:
            self._setup_cache()
        
        # Performance metrics
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _setup_jit_functions(self):
        """Setup JIT-compiled functions for faster inference."""
        @jax.jit
        def jit_predict(params, inputs):
            return self.model_apply_fn(params, inputs)
        
        @jax.jit
        def jit_batch_predict(params, inputs):
            return jax.vmap(
                lambda x: self.model_apply_fn(params, x.reshape(1, -1))
            )(inputs)
        
        self.jit_predict = jit_predict
        self.jit_batch_predict = jit_batch_predict
    
    def _setup_cache(self):
        """Setup inference cache."""
        self.cache = {}
        self.cache_keys = []
        self.max_cache_size = self.config.cache_size
    
    def _get_cache_key(self, inputs: jnp.ndarray) -> str:
        """Generate cache key for inputs."""
        # Simple hash-based cache key
        return str(hash(inputs.tobytes()))
    
    def _add_to_cache(self, key: str, result: jnp.ndarray):
        """Add result to cache."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = result
        self.cache_keys.append(key)
    
    def predict(
        self,
        coordinates: jnp.ndarray,
        return_gradients: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Make predictions for given coordinates.
        
        Args:
            coordinates: Input coordinates (x, y, t)
            return_gradients: Whether to return gradients
            
        Returns:
            Dictionary containing predictions
        """
        start_time = time.time()
        
        # Ensure proper shape
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        
        # Check cache if enabled
        if self.config.enable_caching:
            cache_key = self._get_cache_key(coordinates)
            if cache_key in self.cache:
                self.inference_stats['cache_hits'] += 1
                return self.cache[cache_key]
            else:
                self.inference_stats['cache_misses'] += 1
        
        # Make prediction
        if self.config.enable_jit:
            if coordinates.shape[0] == 1:
                predictions = self.jit_predict(self.model_params, coordinates)
            else:
                predictions = self.jit_batch_predict(self.model_params, coordinates)
        else:
            if coordinates.shape[0] == 1:
                predictions = self.model_apply_fn(self.model_params, coordinates)
            else:
                predictions = jax.vmap(
                    lambda x: self.model_apply_fn(self.model_params, x.reshape(1, -1))
                )(coordinates)
        
        # Process predictions
        result = self._process_predictions(predictions, coordinates, return_gradients)
        
        # Add to cache if enabled
        if self.config.enable_caching:
            self._add_to_cache(cache_key, result)
        
        # Update stats
        inference_time = time.time() - start_time
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_time'] += inference_time
        
        return result
    
    def _process_predictions(
        self,
        predictions: jnp.ndarray,
        coordinates: jnp.ndarray,
        return_gradients: bool
    ) -> Dict[str, jnp.ndarray]:
        """Process raw predictions into structured output."""
        # Assuming predictions are [thickness, velocity_x, velocity_y]
        result = {
            'thickness': predictions[..., 0],
            'velocity_x': predictions[..., 1],
            'velocity_y': predictions[..., 2]
        }
        
        if return_gradients:
            # Compute gradients
            gradients = self._compute_gradients(coordinates)
            result.update(gradients)
        
        return result
    
    def _compute_gradients(self, coordinates: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute gradients of predictions with respect to inputs."""
        def predict_fn(coords):
            return self.model_apply_fn(self.model_params, coords.reshape(1, -1))
        
        # Compute Jacobian
        jacobian_fn = jax.jacfwd(predict_fn)
        
        if coordinates.shape[0] == 1:
            jacobian = jacobian_fn(coordinates.flatten())
        else:
            jacobian = jax.vmap(jacobian_fn)(coordinates)
        
        # Extract specific gradients
        gradients = {
            'thickness_grad_x': jacobian[..., 0, 0],
            'thickness_grad_y': jacobian[..., 0, 1],
            'velocity_x_grad_x': jacobian[..., 1, 0],
            'velocity_x_grad_y': jacobian[..., 1, 1],
            'velocity_y_grad_x': jacobian[..., 2, 0],
            'velocity_y_grad_y': jacobian[..., 2, 1]
        }
        
        return gradients
    
    def batch_predict(
        self,
        coordinates_list: List[jnp.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, jnp.ndarray]]:
        """
        Make predictions for multiple coordinate sets in batches.
        
        Args:
            coordinates_list: List of coordinate arrays
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(coordinates_list), batch_size):
            batch = coordinates_list[i:i + batch_size]
            batch_results = []
            
            for coordinates in batch:
                result = self.predict(coordinates)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def parallel_predict(
        self,
        coordinates_list: List[jnp.ndarray],
        max_workers: Optional[int] = None
    ) -> List[Dict[str, jnp.ndarray]]:
        """
        Make predictions in parallel using multiple workers.
        
        Args:
            coordinates_list: List of coordinate arrays
            max_workers: Maximum number of worker threads
            
        Returns:
            List of prediction dictionaries
        """
        max_workers = max_workers or self.config.max_workers
        results = [None] * len(coordinates_list)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.predict, coords): i
                for i, coords in enumerate(coordinates_list)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    self.logger.error(f"Prediction {index} generated an exception: {exc}")
                    results[index] = None
        
        return results
    
    def predict_on_grid(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        t_value: float,
        resolution: Tuple[int, int] = (100, 100)
    ) -> Dict[str, jnp.ndarray]:
        """
        Make predictions on a regular grid.
        
        Args:
            x_range: (min_x, max_x)
            y_range: (min_y, max_y)
            t_value: Time value
            resolution: Grid resolution (nx, ny)
            
        Returns:
            Dictionary containing gridded predictions
        """
        # Create grid
        x = jnp.linspace(x_range[0], x_range[1], resolution[0])
        y = jnp.linspace(y_range[0], y_range[1], resolution[1])
        X, Y = jnp.meshgrid(x, y)
        
        # Flatten grid and add time dimension
        coordinates = jnp.stack([
            X.flatten(),
            Y.flatten(),
            jnp.full(X.size, t_value)
        ], axis=1)
        
        # Make predictions
        predictions = self.predict(coordinates)
        
        # Reshape back to grid
        grid_predictions = {}
        for key, values in predictions.items():
            grid_predictions[key] = values.reshape(resolution[1], resolution[0])
        
        # Add coordinate grids
        grid_predictions['X'] = X
        grid_predictions['Y'] = Y
        grid_predictions['t'] = t_value
        
        return grid_predictions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        stats = self.inference_stats.copy()
        
        if stats['total_inferences'] > 0:
            stats['average_inference_time'] = stats['total_time'] / stats['total_inferences']
            stats['inferences_per_second'] = stats['total_inferences'] / stats['total_time']
        else:
            stats['average_inference_time'] = 0.0
            stats['inferences_per_second'] = 0.0
        
        if self.config.enable_caching:
            total_cache_requests = stats['cache_hits'] + stats['cache_misses']
            if total_cache_requests > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
            else:
                stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear inference cache."""
        if self.config.enable_caching:
            self.cache.clear()
            self.cache_keys.clear()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }


class BatchInferenceEngine:
    """Specialized engine for large-scale batch inference."""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        chunk_size: int = 10000
    ):
        """
        Initialize batch inference engine.
        
        Args:
            inference_engine: Base inference engine
            chunk_size: Size of chunks for processing large datasets
        """
        self.inference_engine = inference_engine
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    def process_large_dataset(
        self,
        coordinates: jnp.ndarray,
        output_file: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Process a large dataset in chunks.
        
        Args:
            coordinates: Large coordinate array
            output_file: Optional file to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing all predictions
        """
        total_samples = coordinates.shape[0]
        num_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        self.logger.info(f"Processing {total_samples} samples in {num_chunks} chunks")
        
        all_predictions = {}
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_samples)
            
            chunk_coords = coordinates[start_idx:end_idx]
            chunk_predictions = self.inference_engine.predict(chunk_coords)
            
            # Accumulate results
            if chunk_idx == 0:
                # Initialize result arrays
                for key, values in chunk_predictions.items():
                    all_predictions[key] = jnp.zeros(
                        (total_samples,) + values.shape[1:],
                        dtype=values.dtype
                    )
            
            # Store chunk results
            for key, values in chunk_predictions.items():
                all_predictions[key] = all_predictions[key].at[start_idx:end_idx].set(values)
            
            # Progress callback
            if progress_callback:
                progress = (chunk_idx + 1) / num_chunks
                progress_callback(progress, chunk_idx + 1, num_chunks)
            
            self.logger.info(f"Processed chunk {chunk_idx + 1}/{num_chunks}")
        
        # Save results if requested
        if output_file:
            self._save_results(all_predictions, coordinates, output_file)
        
        return all_predictions
    
    def _save_results(
        self,
        predictions: Dict[str, jnp.ndarray],
        coordinates: jnp.ndarray,
        output_file: str
    ):
        """Save results to file."""
        import h5py
        
        with h5py.File(output_file, 'w') as f:
            # Save coordinates
            f.create_dataset('coordinates', data=np.array(coordinates))
            
            # Save predictions
            for key, values in predictions.items():
                f.create_dataset(f'predictions/{key}', data=np.array(values))
            
            # Save metadata
            f.attrs['total_samples'] = coordinates.shape[0]
            f.attrs['coordinate_dims'] = coordinates.shape[1]
            f.attrs['created_at'] = time.time()
        
        self.logger.info(f"Results saved to {output_file}")


def create_inference_engine(
    model_apply_fn: Callable,
    model_params: Dict[str, Any],
    config: Optional[InferenceConfig] = None
) -> InferenceEngine:
    """
    Convenience function to create an inference engine.
    
    Args:
        model_apply_fn: Model application function
        model_params: Trained model parameters
        config: Inference configuration
        
    Returns:
        Configured inference engine
    """
    return InferenceEngine(model_apply_fn, model_params, config)


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Mock model function
    def mock_model_apply(params, inputs):
        # Simple mock model
        x, y, t = inputs[0, 0], inputs[0, 1], inputs[0, 2]
        thickness = 100.0 + 10.0 * jnp.sin(x / 1000.0) * jnp.cos(y / 1000.0)
        velocity_x = 10.0 + 2.0 * jnp.sin(t)
        velocity_y = 5.0 + 1.0 * jnp.cos(t)
        return jnp.array([thickness, velocity_x, velocity_y])
    
    # Mock parameters
    mock_params = {'dummy': jnp.array([1.0])}
    
    # Create inference engine
    config = InferenceConfig(batch_size=100, enable_jit=True, enable_caching=True)
    engine = create_inference_engine(mock_model_apply, mock_params, config)
    
    # Test single prediction
    test_coords = jnp.array([[1000.0, 2000.0, 5.0]])
    predictions = engine.predict(test_coords)
    print("Single prediction:", predictions)
    
    # Test grid prediction
    grid_predictions = engine.predict_on_grid(
        x_range=(0, 5000),
        y_range=(0, 3000),
        t_value=10.0,
        resolution=(50, 30)
    )
    print("Grid prediction shape:", grid_predictions['thickness'].shape)
    
    # Test performance stats
    stats = engine.get_performance_stats()
    print("Performance stats:", stats)