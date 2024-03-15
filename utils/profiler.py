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
            self.profile_list = {}
            self.profiles_start = {}
            self._variables_to_monitor = {}

    def add_variable_to_monitor(self, variable_name, obj, attribute_name):
        if variable_name not in self._variables_to_monitor:
            self._variables_to_monitor[variable_name] = {
                'object': obj,
                'attribute_name': attribute_name,
                'changes': []
            }
            setattr(obj.__class__, attribute_name, self.__create_property(attribute_name))
        else:
            print(f"Variable '{variable_name}' is already being monitored.")

    def __create_property(self, attribute_name):
        def setter(self, value):
            self._variables_to_monitor[attribute_name]['changes'].append((time.time(), attribute_name, value))
            setattr(self, f'_{attribute_name}', value)

        def getter(self):
            return getattr(self, f'_{attribute_name}')

        return property(getter, setter)
    
    @staticmethod
    def start_measuring(probe_name):
        profiler().start_time_profile(probe_name)

    @staticmethod
    def stop_measuring(probe_name, key):
        profiler().measure_time(probe_name, key)


    def __add_time_profiler(self, name: str):
        self.profiles_start[name] = 0
        self.profiles_lock = False
        self.profile_list[name] = []
        
    def start_time_profile(self, name: str):
        if self.profiles_lock:
            logger.log_error(f"Probe [{name}] is busy now!")
            return
        self.profiles_lock = True
        if name not in self.profiles_start.keys():
            self.__add_time_profiler(name)
        self.profiles_start[name] = time.time_ns() // 1000
        

    def measure_time(self, name: str, key=0):
        elapsed_us = (time.time_ns() // 1000) - self.profiles_start[name]
        self.profile_list[name].append((time.time(), key, elapsed_us))
        self.profiles_lock = False

    def dump_time_profile_list(self):
        return self.profile_list
    
    def dump_variables_changes_list(self):
        return self._variables_to_monitor