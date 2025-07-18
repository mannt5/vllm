"""
Configuration for intermediate tensor logging.

This module defines the configuration data class for intermediate tensor logging,
which controls how intermediate tensors are captured and saved during model execution.
"""

import dataclasses
import re
from pathlib import Path
from typing import Optional, Pattern, List, Set, Union


@dataclasses.dataclass
class IntermediateLoggingConfig:
    """Configuration for intermediate tensor logging."""
    
    # Directory where to save the intermediate tensors
    output_dir: str = "/tmp/vllm_intermediates"
    
    # Regex patterns to filter modules by name (None or empty list means log all modules)
    # Can be a single string or a list of strings
    module_name_regex: Optional[Union[str, List[str]]] = None
    
    # List of step IDs to log (empty list means log all steps)
    log_step_ids: List[int] = dataclasses.field(default_factory=lambda: [0, 1])
    
    # Maximum number of elements in tensors to log (None = no limit)
    max_tensor_size: Optional[int] = None
    
    # Whether logging is enabled
    enabled: bool = True
    
    # Current step counter (incremented after each forward pass)
    current_step: int = 0
    
    # List of device names to log (empty list means log all devices)
    device_names: List[str] = dataclasses.field(default_factory=list)
    
    # Compiled regex patterns for module filtering
    _module_name_patterns: List[Pattern] = dataclasses.field(default_factory=list)
    
    # Set of step IDs for faster lookup
    _step_id_set: Set[int] = dataclasses.field(default_factory=set)
    
    def __post_init__(self):
        """Initialize derived fields after instance creation."""
        self._compile_regex_patterns()
        self._step_id_set = set(self.log_step_ids)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for module name filtering."""
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        
        self._module_name_patterns = []
        
        if self.module_name_regex is None:
            logger.info("No module name regex patterns provided, will log all modules")
            return
            
        # Convert single string to list for uniform handling
        patterns = self.module_name_regex
        if isinstance(patterns, str):
            patterns = [patterns]
            logger.info(f"Converting single regex pattern to list: [{patterns[0]}]")
        else:
            logger.info(f"Using list of regex patterns: {patterns}")
            
        # Compile all patterns
        for pattern in patterns:
            try:
                compiled_pattern = re.compile(pattern)
                self._module_name_patterns.append(compiled_pattern)
                logger.info(f"Successfully compiled regex pattern: '{pattern}'")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        logger.info(f"Compiled {len(self._module_name_patterns)} regex patterns")
    
    def should_log_step(self) -> bool:
        """Check if the current step should be logged based on the step IDs."""
        if not self.enabled:
            return False
        
        # If log_step_ids is empty, log all steps
        if not self.log_step_ids:
            return True
            
        # Otherwise, check if current step is in the set of step IDs to log
        return self.current_step in self._step_id_set
        
    def should_log_device(self, device_name: str) -> bool:
        """Check if a device should be logged based on the device names.
        
        Args:
            device_name: The name of the device to check (e.g., 'cuda:0', 'cpu').
            
        Returns:
            True if the device should be logged, False otherwise.
            If device_names is empty, all devices are logged.
        """
        # If device_names is empty, log all devices
        if not self.device_names:
            return True
            
        # Otherwise, check if device_name is in the list of device names to log
        return device_name in self.device_names
    
    def should_log_module(self, module_name: str) -> bool:
        """Check if a module should be logged based on the name regex patterns.
        
        Args:
            module_name: The name of the module to check.
            
        Returns:
            True if the module should be logged, False otherwise.
            If no patterns are defined, all modules are logged.
            If patterns are defined, the module is logged if it matches ANY pattern.
        """
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        
        # If no patterns are defined, log all modules
        if not self._module_name_patterns:
            logger.debug(f"No patterns defined, will log module: {module_name}")
            return True
        
        # Check if the module name matches any of the patterns
        for i, pattern in enumerate(self._module_name_patterns):
            match = pattern.search(module_name)
            if match:
                logger.info(f"Module {module_name} matches pattern {i}: '{pattern.pattern}'")
                return True
        
        # For debugging, log at a higher level when we're checking layer modules
        if "layer" in module_name or "embed" in module_name:
            logger.info(f"Module {module_name} doesn't match any patterns: {[p.pattern for p in self._module_name_patterns]}")
        else:
            logger.debug(f"Module {module_name} doesn't match any patterns")
        return False
    
    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1
    
    def reset_step(self) -> None:
        """Reset the current step counter to zero."""
        self.current_step = 0
    
    def to_dict(self) -> dict:
        """Convert the config to a dictionary for serialization."""
        return {
            "output_dir": self.output_dir,
            "module_name_regex": self.module_name_regex,
            "log_step_ids": self.log_step_ids,
            "max_tensor_size": self.max_tensor_size,
            "enabled": self.enabled,
            "current_step": self.current_step,
            "device_names": self.device_names
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "IntermediateLoggingConfig":
        """Create a config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            
        Returns:
            An IntermediateLoggingConfig instance.
        """
        # Filter out unknown parameters
        known_params = {"output_dir", "module_name_regex", "log_step_ids", 
                       "max_tensor_size", "enabled", "current_step", "device_names"}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_params}
        
        # Handle backward compatibility with log_step_interval
        if "log_step_interval" in config_dict and "log_step_ids" not in filtered_dict:
            interval = config_dict["log_step_interval"]
            if interval > 0:
                # Convert interval to step IDs (first few steps)
                filtered_dict["log_step_ids"] = list(range(0, 10 * interval, interval))
        
        return cls(**filtered_dict)
