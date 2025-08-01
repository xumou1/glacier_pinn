"""Batch Processing Module.

This module provides functionality for large-scale batch processing
of glacier dynamics predictions, including distributed processing,
data pipeline management, and result aggregation.
"""

import os
import json
import time
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import jax.numpy as jnp
import pandas as pd
import h5py
from tqdm import tqdm

from .inference_engine import InferenceEngine, BatchInferenceEngine


@dataclass
class BatchJobConfig:
    """Configuration for batch processing jobs."""
    job_name: str
    input_file: str
    output_file: str
    chunk_size: int = 10000
    max_workers: int = None
    use_multiprocessing: bool = True
    save_intermediate: bool = True
    intermediate_dir: str = None
    progress_callback: bool = True
    compression: str = "gzip"  # For HDF5 output
    precision: str = "float32"


class BatchProcessor:
    """High-performance batch processor for large-scale predictions."""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        config: BatchJobConfig
    ):
        """
        Initialize batch processor.
        
        Args:
            inference_engine: Inference engine instance
            config: Batch job configuration
        """
        self.inference_engine = inference_engine
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup multiprocessing
        if config.max_workers is None:
            self.config.max_workers = mp.cpu_count()
        
        # Setup intermediate directory
        if config.intermediate_dir is None:
            self.config.intermediate_dir = f"./batch_temp_{config.job_name}"
        
        Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        
        # Job statistics
        self.job_stats = {
            'start_time': None,
            'end_time': None,
            'total_samples': 0,
            'processed_samples': 0,
            'failed_samples': 0,
            'processing_rate': 0.0,
            'estimated_completion': None
        }
    
    def process_csv_file(
        self,
        coordinate_columns: List[str] = ['x', 'y', 't'],
        output_columns: List[str] = ['thickness', 'velocity_x', 'velocity_y']
    ) -> str:
        """
        Process a CSV file containing coordinates.
        
        Args:
            coordinate_columns: Names of coordinate columns
            output_columns: Names of output columns
            
        Returns:
            Path to output file
        """
        self.logger.info(f"Starting batch processing of {self.config.input_file}")
        self.job_stats['start_time'] = datetime.now()
        
        # Read input data
        df = pd.read_csv(self.config.input_file)
        coordinates = df[coordinate_columns].values
        self.job_stats['total_samples'] = len(coordinates)
        
        self.logger.info(f"Loaded {len(coordinates)} coordinate samples")
        
        # Process in chunks
        if self.config.use_multiprocessing:
            results = self._process_multiprocessing(coordinates)
        else:
            results = self._process_sequential(coordinates)
        
        # Save results
        output_df = df.copy()
        for i, col in enumerate(output_columns):
            output_df[col] = results[:, i]
        
        output_df.to_csv(self.config.output_file, index=False)
        
        self.job_stats['end_time'] = datetime.now()
        self._log_completion_stats()
        
        return self.config.output_file
    
    def process_hdf5_file(
        self,
        coordinate_dataset: str = 'coordinates',
        output_group: str = 'predictions'
    ) -> str:
        """
        Process an HDF5 file containing coordinates.
        
        Args:
            coordinate_dataset: Name of coordinate dataset
            output_group: Name of output group
            
        Returns:
            Path to output file
        """
        self.logger.info(f"Starting HDF5 batch processing of {self.config.input_file}")
        self.job_stats['start_time'] = datetime.now()
        
        # Read input data
        with h5py.File(self.config.input_file, 'r') as f:
            coordinates = f[coordinate_dataset][:]
            self.job_stats['total_samples'] = len(coordinates)
        
        self.logger.info(f"Loaded {len(coordinates)} coordinate samples")
        
        # Process data
        if self.config.use_multiprocessing:
            results = self._process_multiprocessing(coordinates)
        else:
            results = self._process_sequential(coordinates)
        
        # Save results to HDF5
        self._save_hdf5_results(coordinates, results, output_group)
        
        self.job_stats['end_time'] = datetime.now()
        self._log_completion_stats()
        
        return self.config.output_file
    
    def process_numpy_array(
        self,
        coordinates: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Process a numpy array of coordinates.
        
        Args:
            coordinates: Coordinate array
            metadata: Optional metadata
            
        Returns:
            Dictionary containing predictions
        """
        self.logger.info(f"Starting numpy array batch processing")
        self.job_stats['start_time'] = datetime.now()
        self.job_stats['total_samples'] = len(coordinates)
        
        # Process data
        if self.config.use_multiprocessing:
            results = self._process_multiprocessing(coordinates)
        else:
            results = self._process_sequential(coordinates)
        
        # Format results
        predictions = {
            'thickness': results[:, 0],
            'velocity_x': results[:, 1],
            'velocity_y': results[:, 2]
        }
        
        # Save to file if output file specified
        if self.config.output_file:
            self._save_numpy_results(coordinates, predictions, metadata)
        
        self.job_stats['end_time'] = datetime.now()
        self._log_completion_stats()
        
        return predictions
    
    def _process_sequential(self, coordinates: np.ndarray) -> np.ndarray:
        """Process coordinates sequentially."""
        batch_engine = BatchInferenceEngine(
            self.inference_engine,
            chunk_size=self.config.chunk_size
        )
        
        def progress_callback(progress, chunk_idx, total_chunks):
            self.job_stats['processed_samples'] = int(progress * self.job_stats['total_samples'])
            if self.config.progress_callback:
                self.logger.info(f"Progress: {progress:.1%} ({chunk_idx}/{total_chunks} chunks)")
        
        predictions = batch_engine.process_large_dataset(
            jnp.array(coordinates),
            progress_callback=progress_callback
        )
        
        # Stack results
        return np.column_stack([
            predictions['thickness'],
            predictions['velocity_x'],
            predictions['velocity_y']
        ])
    
    def _process_multiprocessing(self, coordinates: np.ndarray) -> np.ndarray:
        """Process coordinates using multiprocessing."""
        # Split coordinates into chunks for parallel processing
        chunks = self._split_into_chunks(coordinates)
        
        self.logger.info(f"Processing {len(chunks)} chunks with {self.config.max_workers} workers")
        
        results = []
        completed_chunks = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, i): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    results.append((chunk_idx, chunk_result))
                    completed_chunks += 1
                    
                    # Update progress
                    progress = completed_chunks / len(chunks)
                    self.job_stats['processed_samples'] = int(progress * self.job_stats['total_samples'])
                    
                    if self.config.progress_callback:
                        self.logger.info(f"Progress: {progress:.1%} ({completed_chunks}/{len(chunks)} chunks)")
                    
                except Exception as exc:
                    self.logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")
                    self.job_stats['failed_samples'] += len(chunks[chunk_idx])
        
        # Sort results by chunk index and concatenate
        results.sort(key=lambda x: x[0])
        return np.vstack([result[1] for result in results])
    
    def _split_into_chunks(self, coordinates: np.ndarray) -> List[np.ndarray]:
        """Split coordinates into chunks for parallel processing."""
        chunks = []
        for i in range(0, len(coordinates), self.config.chunk_size):
            chunk = coordinates[i:i + self.config.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _process_chunk(self, chunk: np.ndarray, chunk_idx: int) -> np.ndarray:
        """Process a single chunk of coordinates."""
        try:
            # Convert to JAX array
            coords_jax = jnp.array(chunk)
            
            # Make predictions
            predictions = self.inference_engine.predict(coords_jax)
            
            # Stack results
            result = np.column_stack([
                predictions['thickness'],
                predictions['velocity_x'],
                predictions['velocity_y']
            ])
            
            # Save intermediate result if enabled
            if self.config.save_intermediate:
                intermediate_file = Path(self.config.intermediate_dir) / f"chunk_{chunk_idx}.npy"
                np.save(intermediate_file, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
            # Return zeros for failed chunk
            return np.zeros((len(chunk), 3))
    
    def _save_hdf5_results(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        output_group: str
    ):
        """Save results to HDF5 file."""
        with h5py.File(self.config.output_file, 'w') as f:
            # Save coordinates
            f.create_dataset(
                'coordinates',
                data=coordinates,
                compression=self.config.compression
            )
            
            # Create predictions group
            pred_group = f.create_group(output_group)
            
            # Save predictions
            pred_group.create_dataset(
                'thickness',
                data=predictions[:, 0],
                compression=self.config.compression
            )
            pred_group.create_dataset(
                'velocity_x',
                data=predictions[:, 1],
                compression=self.config.compression
            )
            pred_group.create_dataset(
                'velocity_y',
                data=predictions[:, 2],
                compression=self.config.compression
            )
            
            # Save metadata
            f.attrs['job_name'] = self.config.job_name
            f.attrs['total_samples'] = self.job_stats['total_samples']
            f.attrs['processed_samples'] = self.job_stats['processed_samples']
            f.attrs['failed_samples'] = self.job_stats['failed_samples']
            f.attrs['processing_time'] = str(self.job_stats['end_time'] - self.job_stats['start_time'])
            f.attrs['created_at'] = datetime.now().isoformat()
    
    def _save_numpy_results(
        self,
        coordinates: np.ndarray,
        predictions: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]]
    ):
        """Save results to numpy file."""
        results = {
            'coordinates': coordinates,
            'predictions': predictions,
            'metadata': metadata or {},
            'job_stats': self.job_stats
        }
        
        np.savez_compressed(self.config.output_file, **results)
    
    def _log_completion_stats(self):
        """Log job completion statistics."""
        duration = self.job_stats['end_time'] - self.job_stats['start_time']
        self.job_stats['processing_rate'] = self.job_stats['processed_samples'] / duration.total_seconds()
        
        self.logger.info(f"Batch processing completed:")
        self.logger.info(f"  Total samples: {self.job_stats['total_samples']}")
        self.logger.info(f"  Processed samples: {self.job_stats['processed_samples']}")
        self.logger.info(f"  Failed samples: {self.job_stats['failed_samples']}")
        self.logger.info(f"  Processing time: {duration}")
        self.logger.info(f"  Processing rate: {self.job_stats['processing_rate']:.2f} samples/sec")
    
    def get_job_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        return self.job_stats.copy()
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files."""
        import shutil
        if Path(self.config.intermediate_dir).exists():
            shutil.rmtree(self.config.intermediate_dir)
            self.logger.info(f"Cleaned up intermediate directory: {self.config.intermediate_dir}")


class DistributedBatchProcessor:
    """Distributed batch processor for very large datasets."""
    
    def __init__(
        self,
        inference_engines: List[InferenceEngine],
        config: BatchJobConfig
    ):
        """
        Initialize distributed batch processor.
        
        Args:
            inference_engines: List of inference engines (one per worker)
            config: Batch job configuration
        """
        self.inference_engines = inference_engines
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.num_workers = len(inference_engines)
        self.logger.info(f"Initialized distributed processor with {self.num_workers} workers")
    
    def process_large_dataset(
        self,
        coordinates: np.ndarray,
        output_file: str
    ) -> str:
        """
        Process a very large dataset using distributed processing.
        
        Args:
            coordinates: Large coordinate array
            output_file: Output file path
            
        Returns:
            Path to output file
        """
        self.logger.info(f"Starting distributed processing of {len(coordinates)} samples")
        
        # Split data among workers
        worker_chunks = np.array_split(coordinates, self.num_workers)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (engine, chunk) in enumerate(zip(self.inference_engines, worker_chunks)):
                future = executor.submit(self._process_worker_chunk, engine, chunk, i)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                worker_result = future.result()
                results.append(worker_result)
        
        # Combine results
        combined_results = np.vstack(results)
        
        # Save combined results
        self._save_distributed_results(coordinates, combined_results, output_file)
        
        return output_file
    
    def _process_worker_chunk(
        self,
        engine: InferenceEngine,
        chunk: np.ndarray,
        worker_id: int
    ) -> np.ndarray:
        """Process a chunk on a specific worker."""
        self.logger.info(f"Worker {worker_id} processing {len(chunk)} samples")
        
        coords_jax = jnp.array(chunk)
        predictions = engine.predict(coords_jax)
        
        result = np.column_stack([
            predictions['thickness'],
            predictions['velocity_x'],
            predictions['velocity_y']
        ])
        
        self.logger.info(f"Worker {worker_id} completed processing")
        return result
    
    def _save_distributed_results(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        output_file: str
    ):
        """Save distributed processing results."""
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('coordinates', data=coordinates, compression='gzip')
            
            pred_group = f.create_group('predictions')
            pred_group.create_dataset('thickness', data=predictions[:, 0], compression='gzip')
            pred_group.create_dataset('velocity_x', data=predictions[:, 1], compression='gzip')
            pred_group.create_dataset('velocity_y', data=predictions[:, 2], compression='gzip')
            
            f.attrs['num_workers'] = self.num_workers
            f.attrs['total_samples'] = len(coordinates)
            f.attrs['created_at'] = datetime.now().isoformat()


class BatchJobManager:
    """Manager for batch processing jobs."""
    
    def __init__(self, jobs_directory: str = "./batch_jobs"):
        """
        Initialize batch job manager.
        
        Args:
            jobs_directory: Directory to store job information
        """
        self.jobs_directory = Path(jobs_directory)
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.active_jobs = {}
    
    def submit_job(
        self,
        inference_engine: InferenceEngine,
        config: BatchJobConfig
    ) -> str:
        """Submit a batch processing job."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.job_name}"
        
        # Create job directory
        job_dir = self.jobs_directory / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Save job configuration
        config_file = job_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Create processor
        processor = BatchProcessor(inference_engine, config)
        self.active_jobs[job_id] = processor
        
        self.logger.info(f"Submitted batch job: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job."""
        if job_id in self.active_jobs:
            processor = self.active_jobs[job_id]
            return processor.get_job_stats()
        else:
            return {'status': 'not_found'}
    
    def list_jobs(self) -> List[str]:
        """List all jobs."""
        return list(self.active_jobs.keys())


def create_batch_processor(
    inference_engine: InferenceEngine,
    job_name: str,
    input_file: str,
    output_file: str,
    **kwargs
) -> BatchProcessor:
    """
    Convenience function to create a batch processor.
    
    Args:
        inference_engine: Inference engine
        job_name: Name of the job
        input_file: Input file path
        output_file: Output file path
        **kwargs: Additional configuration options
        
    Returns:
        Configured batch processor
    """
    config = BatchJobConfig(
        job_name=job_name,
        input_file=input_file,
        output_file=output_file,
        **kwargs
    )
    
    return BatchProcessor(inference_engine, config)


if __name__ == "__main__":
    # Example usage
    import tempfile
    from .inference_engine import create_inference_engine, InferenceConfig
    
    # Mock model function
    def mock_model_apply(params, inputs):
        x, y, t = inputs[0, 0], inputs[0, 1], inputs[0, 2]
        thickness = 100.0 + 10.0 * jnp.sin(x / 1000.0) * jnp.cos(y / 1000.0)
        velocity_x = 10.0 + 2.0 * jnp.sin(t)
        velocity_y = 5.0 + 1.0 * jnp.cos(t)
        return jnp.array([thickness, velocity_x, velocity_y])
    
    # Create inference engine
    mock_params = {'dummy': jnp.array([1.0])}
    config = InferenceConfig(batch_size=1000, enable_jit=True)
    engine = create_inference_engine(mock_model_apply, mock_params, config)
    
    # Create test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate test coordinates
        test_coords = np.random.rand(5000, 3) * [5000, 3000, 10]
        
        # Create batch processor
        batch_config = BatchJobConfig(
            job_name="test_batch",
            input_file="",  # Will use numpy array directly
            output_file=os.path.join(temp_dir, "batch_results.npz"),
            chunk_size=1000,
            max_workers=2
        )
        
        processor = BatchProcessor(engine, batch_config)
        
        # Process data
        results = processor.process_numpy_array(test_coords)
        
        print(f"Processed {len(test_coords)} samples")
        print(f"Results shape: {results['thickness'].shape}")
        print(f"Job stats: {processor.get_job_stats()}")