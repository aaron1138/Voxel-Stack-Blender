# run_logger.py

"""
Manages logging of Modular-Stacker run configurations and timestamps
to a semi-flat JSON Lines file.
Each line in the log file is a JSON object representing a single run.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional # Import Optional and Any here

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref

    # Ensure the config instance has the run_log_file attribute
    if not hasattr(config_instance, 'run_log_file'):
        raise AttributeError("Config instance must have a 'run_log_file' attribute.")
    
    _config_ref = config_instance

def log_run(run_index: int, config_data: Dict[str, Any]) -> None:
    """
    Logs the details of a processing run to the configured log file.

    Args:
        run_index (int): The serial/index number for this run.
        config_data (Dict[str, Any]): A dictionary containing the configuration
                                       parameters used for this run.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in run_logger. Cannot log run.")
        return

    log_filepath = _config_ref.run_log_file
    
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_filepath) or ".", exist_ok=True)

    log_entry = {
        "run_index": run_index,
        "timestamp": datetime.now().isoformat(),
        "config": config_data
    }

    try:
        with open(log_filepath, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f)
            f.write('\n') # Add a newline to make it JSON Lines
        print(f"RunLogger: Logged run {run_index} to {log_filepath}")
    except Exception as e:
        print(f"Error logging run {run_index} to '{log_filepath}': {e}")

def get_last_run_index() -> int:
    """
    Reads the log file to determine the last used run index.
    Returns 0 if the file does not exist or is empty/invalid.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in run_logger. Cannot get last run index.")
        return 0

    log_filepath = _config_ref.run_log_file
    last_index = 0
    if os.path.exists(log_filepath):
        try:
            with open(log_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "run_index" in entry:
                            last_index = max(last_index, int(entry["run_index"]))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line in log file: {line.strip()}")
        except Exception as e:
            print(f"Error reading log file '{log_filepath}': {e}. Resetting run index.")
            last_index = 0
    return last_index

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Run Logger Module Test ---")
    
    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.run_log_file = "test_modular_stacker_runs.log"
            self.input_dir = "/path/to/test/input"
            self.output_dir = "/path/to/test/output"
            self.blend_mode = "gaussian"
            self.threads = 4
            self.lut_generation_type = "gamma"
            self.gamma_value = 2.2
            self.xy_blend_pipeline = [
                {"type": "gaussian_blur", "params": {"gaussian_ksize_x": 5, "gaussian_sigma_x": 1.0}},
                {"type": "unsharp_mask", "params": {"unsharp_amount": 1.2}}
            ]

        def to_dict(self):
            # Simulate the to_dict method from the actual Config class
            data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            if "xy_blend_pipeline" in data:
                data["xy_blend_pipeline"] = [op if isinstance(op, dict) else op.to_dict() for op in data["xy_blend_pipeline"]]
            return data

    mock_config = MockConfig()
    set_config_reference(mock_config)

    # Clean up previous test log if it exists
    if os.path.exists(mock_config.run_log_file):
        os.remove(mock_config.run_log_file)
        print(f"Cleaned up previous test log: {mock_config.run_log_file}")

    # Test initial run index
    initial_index = get_last_run_index()
    print(f"Initial last run index: {initial_index}")

    # Log a few runs
    print("\nLogging runs...")
    log_run(initial_index + 1, mock_config.to_dict())
    log_run(initial_index + 2, mock_config.to_dict())

    # Test getting last run index after logging
    new_last_index = get_last_run_index()
    print(f"New last run index: {new_last_index}")

    # Log another run with a modified config
    mock_config.blend_mode = "linear"
    log_run(new_last_index + 1, mock_config.to_dict())
    print(f"Final last run index: {get_last_run_index()}")

    # Verify content (optional, manual inspection of test_modular_stacker_runs.log)
    print(f"\nCheck the file '{mock_config.run_log_file}' for logged data.")

    # Clean up test log
    if os.path.exists(mock_config.run_log_file):
        os.remove(mock_config.run_log_file)
        print(f"Cleaned up {mock_config.run_log_file}")
