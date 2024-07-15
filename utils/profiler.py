import time
from utils.logger import *

class profiler:
    _class_ins = None

    def __new__(cls):
        if cls._class_ins is None:
            cls._class_ins = super().__new__(cls)
            cls._class_ins.__initialized = False
        return cls._class_ins
    
    def __init__(self):
        if not self.__initialized:
            self.__initialized = True
            self.profiles_list = {}
            self.profiles_start = {}
            self.profiles_lock = {}
            self.vars_val_list = {}
            self.monitored_vars = {}

    def add_var_to_monitor(self, var_name, var_key, target_instance):
        def monitor_var_change(new_val):
            if var_key not in self.monitored_vars:
                self.monitored_vars[var_key] = []
            self.monitored_vars[var_key].append((time.time(), None, new_val))

        # Get the current value of the variable
        current_val = getattr(target_instance, var_name)

        # If the variable exists, monitor it
        if current_val is not None:
            monitor_var_change(current_val)

        # Override the variable with a custom setter to monitor changes
        def setter(self, new_val):
            nonlocal current_val
            current_val = new_val
            monitor_var_change(new_val)

        # Set the custom setter for the variable
        setattr(target_instance.__class__, var_name, property(lambda self: current_val, setter))

    
    @staticmethod
    def add_var_monitor_changes(var, obj, var_key):
        profiler().add_var_to_monitor(var, var_key, obj)

    @staticmethod
    def start_measuring(probe_name):
        profiler().start_time_profile(probe_name)

    @staticmethod
    def stop_measuring(probe_name, key):
        profiler().measure_time(probe_name, key)

    @staticmethod
    def save_variable(probe_name, value, key):
        profiler().store_value(probe_name, value, key)

    @staticmethod
    def dump_probes():
        output_dict = {}
        output_dict["time_profiles"] = profiler().dump_time_profile_list()
        output_dict["var_changes"] = profiler().dump_variables_changes_list()
        output_dict["var_values"] = profiler().dump_variables_value_list()
        return output_dict

    def is_probe_available(self, probe_name):
        return ((probe_name in self.vars_val_list.keys()) or (probe_name in self.profiles_start.keys()))

    def store_value(self, probe_name, value, key = 0):
        if probe_name not in self.vars_val_list.keys():
            self.vars_val_list[probe_name] = []
        
        self.vars_val_list[probe_name].append(((time.time(), key, value)))

    def __add_time_profiler(self, name: str):
        self.profiles_start[name] = 0
        self.profiles_lock[name] = False
        self.profiles_list[name] = []
        
    def start_time_profile(self, name: str):
        if name not in self.profiles_start.keys():
            self.__add_time_profiler(name)
    
        if self.profiles_lock[name]:
            logger.log_error(f"Probe '{name}' is busy now!")
            return
        self.profiles_lock[name] = True

        self.profiles_start[name] = common.Common.time_ns() // 1000
        

    def measure_time(self, name: str, key=0):
        elapsed_us = (common.Common.time_ns() // 1000) - self.profiles_start[name]
        self.profiles_list[name].append((time.time(), key, elapsed_us))
        self.profiles_lock[name] = False

    def dump_time_profile_list(self):
        return self.profiles_list
    
    def dump_variables_changes_list(self):
        return self.monitored_vars
    
    def dump_variables_value_list(self):
        return self.vars_val_list