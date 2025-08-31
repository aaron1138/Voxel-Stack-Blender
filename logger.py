import os
import datetime
import json

class Logger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_filepath = os.path.join(log_dir, f"{self.run_timestamp}.log")
        self.start_time = datetime.datetime.now()

    def _write(self, message: str):
        with open(self.log_filepath, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._write(f"[{timestamp}] {message}")

    def log_config(self, config_obj):
        self.log("Starting run with the following configuration:")
        try:
            # Use a custom encoder to handle non-serializable types if necessary
            config_dict = config_obj.to_dict()
            config_str = json.dumps(config_dict, indent=2)
            self._write(config_str)
        except Exception as e:
            self.log(f"Could not serialize config object: {e}")
        self.log("-" * 20) # Separator

    def log_total_time(self):
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        self.log(f"Total execution time: {duration}")
