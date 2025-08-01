"""Model Packaging Module.

This module provides functionality for packaging trained PINNs models
for deployment, including serialization, compression, and metadata management.
"""

import os
import json
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization


class ModelPackager:
    """Package trained PINNs models for deployment."""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        """
        Initialize model packager.
        
        Args:
            model_name: Name of the model
            version: Model version
        """
        self.model_name = model_name
        self.version = version
        self.package_metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'framework': 'JAX/Flax',
            'model_type': 'PINNs'
        }
    
    def package_model(
        self,
        model_params: Dict[str, Any],
        model_config: Dict[str, Any],
        training_metadata: Dict[str, Any],
        output_path: str,
        include_preprocessing: bool = True
    ) -> str:
        """
        Package a complete model for deployment.
        
        Args:
            model_params: Trained model parameters
            model_config: Model configuration
            training_metadata: Training metadata and metrics
            output_path: Output directory path
            include_preprocessing: Whether to include preprocessing components
            
        Returns:
            Path to the packaged model file
        """
        package_dir = Path(output_path) / f"{self.model_name}_v{self.version}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        params_path = package_dir / "model_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)
        
        # Save model configuration
        config_path = package_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save training metadata
        metadata_path = package_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        # Save package metadata
        package_metadata_path = package_dir / "package_metadata.json"
        self.package_metadata.update({
            'model_config': model_config,
            'training_metadata': training_metadata
        })
        with open(package_metadata_path, 'w') as f:
            json.dump(self.package_metadata, f, indent=2)
        
        # Create deployment script
        self._create_deployment_script(package_dir, model_config)
        
        # Create requirements file
        self._create_requirements_file(package_dir)
        
        # Create compressed package
        zip_path = f"{package_dir}.zip"
        self._create_zip_package(package_dir, zip_path)
        
        return zip_path
    
    def _create_deployment_script(self, package_dir: Path, model_config: Dict[str, Any]):
        """Create deployment script for the packaged model."""
        script_content = f'''#!/usr/bin/env python3
"""Deployment script for {self.model_name} v{self.version}."""

import os
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization


class DeployedModel:
    """Deployed model wrapper."""
    
    def __init__(self, package_path: str):
        """Initialize deployed model."""
        self.package_path = Path(package_path)
        self.load_model()
    
    def load_model(self):
        """Load model from package."""
        # Load model parameters
        with open(self.package_path / "model_params.pkl", 'rb') as f:
            self.model_params = pickle.load(f)
        
        # Load model configuration
        with open(self.package_path / "model_config.json", 'r') as f:
            self.model_config = json.load(f)
        
        # Load package metadata
        with open(self.package_path / "package_metadata.json", 'r') as f:
            self.package_metadata = json.load(f)
        
        print(f"Loaded model: {{self.package_metadata['model_name']}} v{{self.package_metadata['version']}}")
    
    def predict(self, coordinates: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Make predictions using the deployed model."""
        # This is a placeholder - actual implementation would depend on model architecture
        batch_size = coordinates.shape[0]
        
        # Mock predictions based on model configuration
        predictions = {{
            'thickness': jnp.ones(batch_size) * 100.0,
            'velocity_x': jnp.ones(batch_size) * 10.0,
            'velocity_y': jnp.ones(batch_size) * 5.0
        }}
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {{
            'name': self.package_metadata['model_name'],
            'version': self.package_metadata['version'],
            'created_at': self.package_metadata['created_at'],
            'framework': self.package_metadata['framework'],
            'model_type': self.package_metadata['model_type']
        }}


if __name__ == "__main__":
    # Example usage
    model = DeployedModel(".")
    
    # Test prediction
    test_coords = jnp.array([[100.0, 200.0, 5.0], [300.0, 400.0, 5.0]])
    predictions = model.predict(test_coords)
    
    print("Model Info:", model.get_model_info())
    print("Test Predictions:", predictions)
'''
        
        script_path = package_dir / "deploy.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _create_requirements_file(self, package_dir: Path):
        """Create requirements file for deployment."""
        requirements = [
            "jax>=0.4.0",
            "flax>=0.7.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "optax>=0.1.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0"
        ]
        
        requirements_path = package_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_zip_package(self, package_dir: Path, zip_path: str):
        """Create compressed zip package."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zipf.write(file_path, arcname)
    
    def validate_package(self, package_path: str) -> Dict[str, bool]:
        """
        Validate a packaged model.
        
        Args:
            package_path: Path to the package directory or zip file
            
        Returns:
            Validation results
        """
        if package_path.endswith('.zip'):
            # Extract zip for validation
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(package_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                package_dir = Path(temp_dir) / f"{self.model_name}_v{self.version}"
                return self._validate_package_directory(package_dir)
        else:
            return self._validate_package_directory(Path(package_path))
    
    def _validate_package_directory(self, package_dir: Path) -> Dict[str, bool]:
        """Validate package directory contents."""
        required_files = [
            "model_params.pkl",
            "model_config.json",
            "training_metadata.json",
            "package_metadata.json",
            "deploy.py",
            "requirements.txt"
        ]
        
        validation_results = {}
        
        for file_name in required_files:
            file_path = package_dir / file_name
            validation_results[f"has_{file_name}"] = file_path.exists()
        
        # Validate JSON files
        json_files = ["model_config.json", "training_metadata.json", "package_metadata.json"]
        for json_file in json_files:
            try:
                with open(package_dir / json_file, 'r') as f:
                    json.load(f)
                validation_results[f"valid_{json_file}"] = True
            except (json.JSONDecodeError, FileNotFoundError):
                validation_results[f"valid_{json_file}"] = False
        
        # Validate pickle file
        try:
            with open(package_dir / "model_params.pkl", 'rb') as f:
                pickle.load(f)
            validation_results["valid_model_params.pkl"] = True
        except (pickle.PickleError, FileNotFoundError):
            validation_results["valid_model_params.pkl"] = False
        
        return validation_results


class ModelVersionManager:
    """Manage model versions and deployments."""
    
    def __init__(self, models_registry_path: str):
        """
        Initialize version manager.
        
        Args:
            models_registry_path: Path to models registry directory
        """
        self.registry_path = Path(models_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_path / "models_registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load models registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
    
    def save_registry(self):
        """Save models registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        version: str,
        package_path: str,
        metadata: Dict[str, Any]
    ):
        """Register a new model version."""
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {'versions': {}}
        
        self.registry['models'][model_name]['versions'][version] = {
            'package_path': package_path,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        self.save_registry()
    
    def get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions of a model."""
        if model_name in self.registry['models']:
            return list(self.registry['models'][model_name]['versions'].keys())
        return []
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        versions = self.get_model_versions(model_name)
        if versions:
            # Simple version sorting - in practice, use semantic versioning
            return sorted(versions)[-1]
        return None
    
    def get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model version."""
        if (model_name in self.registry['models'] and 
            version in self.registry['models'][model_name]['versions']):
            return self.registry['models'][model_name]['versions'][version]
        return None


def create_model_package(
    model_name: str,
    model_params: Dict[str, Any],
    model_config: Dict[str, Any],
    training_metadata: Dict[str, Any],
    output_path: str,
    version: str = "1.0.0"
) -> str:
    """
    Convenience function to create a model package.
    
    Args:
        model_name: Name of the model
        model_params: Trained model parameters
        model_config: Model configuration
        training_metadata: Training metadata
        output_path: Output directory path
        version: Model version
        
    Returns:
        Path to the packaged model
    """
    packager = ModelPackager(model_name, version)
    return packager.package_model(
        model_params, model_config, training_metadata, output_path
    )


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Mock model data
    mock_params = {'dense_1': {'kernel': jnp.ones((3, 64)), 'bias': jnp.zeros(64)}}
    mock_config = {
        'architecture': 'PINN',
        'input_dim': 3,
        'output_dim': 3,
        'hidden_layers': [64, 64, 64],
        'activation': 'tanh'
    }
    mock_metadata = {
        'training_loss': 0.001,
        'validation_loss': 0.002,
        'epochs': 1000,
        'dataset_size': 10000
    }
    
    # Create package
    with tempfile.TemporaryDirectory() as temp_dir:
        package_path = create_model_package(
            "glacier_dynamics_model",
            mock_params,
            mock_config,
            mock_metadata,
            temp_dir
        )
        
        print(f"Model packaged at: {package_path}")
        
        # Validate package
        packager = ModelPackager("glacier_dynamics_model")
        validation_results = packager.validate_package(package_path)
        print("Validation results:", validation_results)