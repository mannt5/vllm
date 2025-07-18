"""
Module for logging intermediate tensors during model execution.

This module provides functionality to capture and save intermediate tensors
(inputs and outputs) from PyTorch modules during forward passes.
"""

import os
import re
import json
import torch
import contextlib
import functools
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional

# Import logger from vllm
from vllm.logger import init_logger
from vllm.v1.worker.il_config import IntermediateLoggingConfig

logger = init_logger(__name__)

# Global configuration that can be accessed by hooks
_config = IntermediateLoggingConfig()
_enabled = True

IL_MODULE_NAME="_il_module_name"


def get_il_module_name(module: torch.nn.Module) -> str:
    return getattr(module, IL_MODULE_NAME, module.__class__.__name__)

def set_il_module_name(module: torch.nn.Module, name: str) -> None:
    setattr(module, IL_MODULE_NAME, name)


def disable_intermediate_logging_decorator(func):
    """Decorator to disable intermediate logging during function execution.
    
    This decorator uses the disable_intermediate_logging context manager internally.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The wrapped function with intermediate logging disabled during execution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Disabling intermediate logging for {func.__name__}")
        with disable_intermediate_logging():
            return func(*args, **kwargs)
    return wrapper


@contextlib.contextmanager
def disable_intermediate_logging():
    """Context manager to temporarily disable intermediate logging."""
    global _enabled
    old_enabled = _enabled
    _enabled = False
    try:
        yield
    finally:
        _enabled = old_enabled
        logger.debug(f"Intermediate logging {'enabled' if _enabled else 'disabled'}")

def dump_intermediates_to_json(intermediates: Any, path: Path) -> Any:
    try:
        # Convert inputs to JSON-serializable format
        intermediates_json = convert_intermediates_to_json(intermediates)
        with open(path, "w") as f:
            json.dump(intermediates_json, f, indent=2)
        logger.debug(f"Saved all intermediates as JSON to {path}")
    except Exception as e:
        logger.warning(f"Failed to save intermediates as JSON: {e}")
        import traceback
        logger.warning(traceback.format_exc())

def convert_intermediates_to_json(tensor: Any) -> Any:
    """Convert a intermediates(including tensor) to a JSON-serializable representation.
    
    Args:
        intermediates: The intermediates to convert.
        
    Returns:
        A JSON-serializable representation of the tensor.
    """
    if isinstance(tensor, torch.Tensor):
        try:
            tensor_cpu = tensor.detach().cpu()
            result = {
                "type": "tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": tensor.numel()
            }
            
            # Include a sample of values if the tensor is not too large
            if tensor.numel() <= 100:
                try:
                    # For small tensors, include all values
                    values = tensor_cpu.tolist()
                    # Convert any non-serializable values to strings
                    if isinstance(values, list):
                        result["values"] = values
                    else:
                        result["values"] = str(values)
                except Exception as e:
                    result["values_error"] = str(e)
            else:
                try:
                    # For larger tensors, include just the first few values
                    flat_tensor = tensor_cpu.flatten()
                    sample_values = flat_tensor[:100].tolist()
                    # Convert any non-serializable values to strings
                    if isinstance(sample_values, list):
                        result["values_sample"] = sample_values
                    else:
                        result["values_sample"] = str(sample_values)
                    
                    # Add statistics
                    try:
                        result["values_min"] = float(tensor_cpu.min().item()) if tensor_cpu.numel() > 0 else None
                    except:
                        result["values_min"] = "error"
                        
                    try:
                        result["values_max"] = float(tensor_cpu.max().item()) if tensor_cpu.numel() > 0 else None
                    except:
                        result["values_max"] = "error"
                        
                    try:
                        result["values_mean"] = float(tensor_cpu.mean().item()) if tensor_cpu.numel() > 0 else None
                    except:
                        result["values_mean"] = "error"
                except Exception as e:
                    result["values_error"] = str(e)
                    
            return result
        except Exception as e:
            # Handle any errors in tensor conversion
            return {
                "type": "tensor_error",
                "error": str(e),
                "tensor_type": str(type(tensor))
            }
    
    elif isinstance(tensor, (list, tuple)):
        # For lists/tuples, recursively convert each element
        container_type = "list" if isinstance(tensor, list) else "tuple"
        
        # If it's a large list, only include a sample
        if len(tensor) > 100:
            return {
                "type": container_type,
                "length": len(tensor),
                "sample": [convert_intermediates_to_json(item) for item in tensor[:100]],
                "note": f"Showing only first 100 of {len(tensor)} items"
            }
        else:
            return {
                "type": container_type,
                "items": [convert_intermediates_to_json(item) for item in tensor]
            }
    
    elif isinstance(tensor, dict):
        # For dictionaries, recursively convert each value
        if len(tensor) > 100:
            # For large dicts, only include keys and a sample of values
            keys = list(tensor.keys())
            sample_keys = keys[:100]
            return {
                "type": "dict",
                "length": len(tensor),
                "keys": keys,
                "sample": {k: convert_intermediates_to_json(tensor[k]) for k in sample_keys},
                "note": f"Showing only first 100 of {len(tensor)} items"
            }
        else:
            return {
                "type": "dict",
                "items": {k: convert_intermediates_to_json(v) for k, v in tensor.items()}
            }
    
    elif tensor is None:
        return None
    
    elif isinstance(tensor, (int, float, bool, str)):
        # Primitive types can be directly serialized
        return tensor
    
    else:
        # For other types, use string representation
        return {
            "type": str(type(tensor).__name__),
            "string_repr": str(tensor)
        }

def save_tensors_metadata_if_too_large(tensor: torch.Tensor, file_path: str) -> bool:
    """Utility function to dump tensor metadata to a file.
    
    Args:
        tensor: The tensor to dump.
        file_path: Base path where to save the tensor (without extension).
    """
    if _config.max_tensor_size is not None and tensor.numel() > _config.max_tensor_size:
        # Save tensor metadata instead of full tensor
        tensor_info = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
            "skipped": f"Tensor size {tensor.numel()} exceeds max_tensor_size {_config.max_tensor_size}"
        }
        os.makedirs(os.path.dirname(f"{file_path}.json"), exist_ok=True)
        with open (f"{file_path}.json", "w") as f:
            json.dump(tensor_info, f, indent=2)
        return True
    return False

def save_tensors(tensor: Any, file_path: str) -> None:
    """Utility function to dump tensor to a file.
    
    Args:
        tensor: The tensor to dump. Can be a torch.Tensor, a list/tuple of tensors,
               or a dictionary containing tensors.
        file_path: Base path where to save the tensor (without extension).
    """
    # Also save the actual tensor data for tensors
    if isinstance(tensor, torch.Tensor):
        # Check if tensor is too large
        if save_tensors_metadata_if_too_large(tensor, file_path):
            return
        # Get device name
        device_name = str(tensor.device)
        # Skip if device filtering is enabled and this device should not be logged
        if not _config.should_log_device(device_name):
            logger.debug(f"Skipping tensor on device {device_name} due to device filter")
            return
        # Append device name to file path
        pt_path = f"{file_path}_{device_name.replace(':', '_')}.pt"
        try:
            # Save tensor directly without detaching or moving to CPU
            torch.save(tensor, pt_path)
            logger.debug(f"Saved tensor of shape {tensor.shape} to {pt_path}")
        except Exception as e:
            logger.warning(f"Failed to save tensor to {pt_path}: {e}")
        
        return
    
    if isinstance(tensor, (list, tuple)):
        # For collections, also save each item individually
        for i, item in enumerate(tensor):
            save_tensors(item, f"{file_path}_{i}")
        return
    if isinstance(tensor, dict):
        # For dictionaries, also save each value individually
        for k, v in tensor.items():
            save_tensors(v, f"{file_path}_{k}")
        return


def log_pre_fwd(module: torch.nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Hook to capture module inputs before forward pass.
    
    Args:
        module: The PyTorch module being executed.
        inputs: The inputs to the module's forward function.
        
    Returns:
        The unchanged inputs.
    """
    # Skip if logging is globally disabled
    if not _enabled:
        return inputs
        
    # Skip if logging is disabled or not on the right step
    if not _config.should_log_step():
        return inputs
    
    # Get the module name from the module._name attribute that was set during registration
    module_name = get_il_module_name(module)
    module_id = id(module)
    
    # Skip if module doesn't match the filter
    if not _config.should_log_module(module_name):
        return inputs
    
    logger.debug(f"Logging module {module_name} with id {module_id} to {_config.output_dir}")
    
    # Create a unique directory for this module and step
    dump_dir = Path(_config.output_dir) / f"step_{_config.current_step}"
    dump_dir.mkdir(exist_ok=True, parents=True)
    
    # Create module directory
    module_dir = dump_dir / f"{module_name}_{module_id}"
    module_dir.mkdir(exist_ok=True)
    
    # Save all inputs as a JSON file
    dump_intermediates_to_json(inputs, module_dir / "inputs.json")
    # Save all tensors as separate PT files
    save_tensors(inputs, str(module_dir / "inputs"))
    
    return inputs


def log_post_fwd(module: torch.nn.Module, 
                 inputs: Tuple[Any, ...], 
                 outputs: Any) -> None:
    """Hook to capture module outputs after forward pass.
    
    Args:
        module: The PyTorch module being executed.
        inputs: The inputs to the module's forward function.
        outputs: The outputs from the module's forward function.
    """
    # Skip if logging is globally disabled or not on the right step
    if not _enabled or not _config.should_log_step():
        return
        
    module_name = get_il_module_name(module)
    module_id = id(module)
    
    # Skip if module doesn't match the filter
    if not _config.should_log_module(module_name):
        return
    
    logger.debug(f"Logging module {module_name} output with id {module_id} to {_config.output_dir}")
    # Create a unique directory for this module and step
    dump_dir = Path(_config.output_dir) / f"step_{_config.current_step}"
    module_dir = dump_dir / f"{module_name}_{module_id}"
    module_dir.mkdir(exist_ok=True, parents=True)
    
    # Save outputs as a JSON file
    outputs_json_path = module_dir / "outputs.json"
    try:
        # Convert outputs to JSON-serializable format
        outputs_data = convert_intermediates_to_json(outputs)
        with open(outputs_json_path, "w") as f:
            json.dump(outputs_data, f, indent=2)
        logger.debug(f"Saved outputs as JSON to {outputs_json_path}")
    except Exception as e:
        logger.warning(f"Failed to save outputs as JSON: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    # Save all outputs as a JSON file
    dump_intermediates_to_json(outputs, module_dir / "outputs.json")

    # Save all tensors as separate PT files
    save_tensors(outputs, str(module_dir / "output"))
    #TODO: also dump inputs as post_fwd_inputs


def increment_step() -> None:
    """Increment the current step counter for intermediate logging."""
    _config.increment_step()
    logger.debug(f"Intermediate logging step incremented to {_config.current_step}")


def reset_step() -> None:
    """Reset the current step counter for intermediate logging."""
    _config.reset_step()
    logger.debug("Intermediate logging step reset to 0")


class IntermediatesLogger:
    """Class to manage logging of intermediate tensors during model execution."""
    
    def __init__(self, config: IntermediateLoggingConfig):
        """Initialize the intermediates logger.
        
        Args:
            output_dir: Directory where to save the intermediate tensors.
            module_name_regex: Optional regex pattern(s) to filter modules by name.
                               Can be a single string or a list of strings.
            log_step_ids: List of step IDs to log (None or empty list means log all steps).
            max_tensor_size: Maximum number of elements in tensors to log (None = no limit).
            enabled: Whether logging is enabled.
        """
        global _config
        _config = config
        self.config = _config
        self.hooks = []
        
        # Log configuration
        config_path = Path(self.config.output_dir) / "logging_config.json"
        with open(config_path, "w") as f:
            json.dump(_config.to_dict(), f, indent=2)
        
        logger.info(f"Initialized intermediate logging with config: {self.config.to_dict()}")
        
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks for the model.
        
        Args:
            model: The PyTorch model to register hooks for.
        """
        for name, module in model.named_modules():
            if name:
                if _config.should_log_module(name):
                    # Store the module name in the module object for the hooks to access
                    set_il_module_name(module, name)
                    
                    pre_hook = module.register_forward_pre_hook(log_pre_fwd)
                    post_hook = module.register_forward_hook(log_post_fwd)
                    self.hooks.append((name, module, pre_hook, post_hook))
        
        logger.info(f"Registered hooks for {len(self.hooks)} modules")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for _, _, pre_hook, post_hook in self.hooks:
            pre_hook.remove()
            post_hook.remove()
        
        logger.info(f"Removed {len(self.hooks)} hooks")
        self.hooks = []
        
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable intermediate logging.
        
        Args:
            enabled: Whether to enable logging.
        """
        _config.enabled = enabled
        logger.info(f"Intermediate logging {'enabled' if enabled else 'disabled'}")
        
    def set_log_step_ids(self, step_ids: List[int]) -> None:
        """Set the step IDs to log.
        
        Args:
            step_ids: List of step IDs to log.
        """
        _config.log_step_ids = step_ids
        _config._step_id_set = set(step_ids)
        logger.info(f"Intermediate logging step IDs set to {step_ids}")
        
    def set_module_name_regex(self, regex: Optional[Union[str, List[str]]]) -> None:
        """Set the module name regex filter.
        
        Args:
            regex: Regex pattern(s) to filter modules by name.
                  Can be a single string or a list of strings.
        """
        _config.module_name_regex = regex
        _config._compile_regex_patterns()
        logger.info(f"Intermediate logging module name regex set to {regex}")
        
    def set_max_tensor_size(self, max_size: Optional[int]) -> None:
        """Set the maximum tensor size to log.
        
        Args:
            max_size: Maximum number of elements in tensors to log (None = no limit).
        """
        _config.max_tensor_size = max_size
        logger.info(f"Intermediate logging max tensor size set to {max_size}")


def register_intermediate_hooks(model: torch.nn.Module, 
                               config: Optional[IntermediateLoggingConfig] = None,
                               **kwargs) -> IntermediatesLogger:
    """Register hooks to log intermediate tensors for a model.
    
    Args:
        model: The PyTorch model to log intermediates for.
        config: Configuration for intermediate logging. If provided, this takes precedence over kwargs.
        **kwargs: Configuration parameters that can include:
            - output_dir: Directory where to save the intermediate tensors.
            - module_name_regex: Optional regex pattern to filter modules by name.
            - log_step_ids: List of step IDs to log.
            - max_tensor_size: Maximum number of elements in tensors to log (None = no limit).
            - enabled: Whether logging is enabled.
        
    Returns:
        An IntermediatesLogger instance that can be used to manage the hooks.
    """
    if config is None:
        # Create config from kwargs
        config = IntermediateLoggingConfig.from_dict(kwargs)
    
    logger_instance = IntermediatesLogger(config)
    logger_instance.register_hooks(model)
    return logger_instance
