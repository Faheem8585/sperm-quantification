"""Configuration management for sperm quantification pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class Config:
    """
    Configuration manager for loading and merging YAML configurations.
    
    Supports hierarchical configuration with defaults and overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file.
                        If None, loads default configuration.
        """
        self.config_dir = Path(__file__).parent.parent.parent / "configs"
        
        # Load default configuration
        default_path = self.config_dir / "default.yaml"
        with open(default_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override with custom configuration if provided
        if config_path is not None:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        Load and merge configuration from file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        config_path = Path(config_path)
        
        # Check if it's a relative path in configs directory
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Deep merge configurations
        self.config = self._deep_merge(self.config, custom_config)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base configuration dictionary.
            override: Override configuration dictionary.
        
        Returns:
            Merged dictionary.
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value.
                     Example: "preprocessing.denoising.method"
            default: Default value if key not found.
        
        Returns:
            Configuration value or default.
        
        Example:
            >>> config = Config()
            >>> config.get("video.fps")
            30
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value.
            value: Value to set.
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to output YAML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({len(self.config)} sections)"
