import time

class profiler:
    _class_ins = None

    def __new__(cls):
        if cls._class_ins is None:
            cls._class_ins = super().__new__(cls)
        return cls._class_ins
    
    def __init__(self):
        self.profile_list = {}
        self.profiles = {}
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
    
    def add_time_profiler(self, name: str):
        self.profiles[name].start = 0
        self.profile_list[name] = []
        
    def start_time_profile(self, name: str):
        self.profiles[name].start = time.time_ns() // 1000
        
    def measure_time(self, name: str):
        elapsed_us = (time.time_ns() // 1000) - self.profiles[name].start
        self.profile_list[name].append((time.time(), name, elapsed_us))

    def dump_time_profile_list(self):
        return self.profile_list
    
    def dump_variables_changes_list(self):
        return self._variables_to_monitor