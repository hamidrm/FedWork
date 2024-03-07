#Architecture Generator
import os
import sys
from torch.utils.data import DataLoader
from torchvision import datasets
from enum import Enum
import re

class BaseArch(Enum):
    FeedForwardNet1 = "FeedForwardNet1"
    FeedForwardNet2 = "FeedForwardNet2"
    ResNet18 = "ResNet18"
    ResNet34 = "ResNet34"

class ActivationFunction(Enum):
    SigmoidFunction = "sigmoid"
    TanhFunction = "tanh"
    ReLUFunction = "relu"
    LeakyReLUFunction = "LeakyReLU"

class FWArch:
    def __init__(self, base_arch: BaseArch):
        self.base_arch = base_arch
        self.arch_class = None
        self.final_code = ""
        base_code_path = os.path.join(os.getcwd(), "arch")
        base_code_path = os.path.join(base_code_path, base_arch.value+".py")
        
        with open(base_code_path, 'r') as file:
            self.base_arch_code = file.read()

        tags = re.findall(r'<<([^>]*)>>', self.base_arch_code)
        self.variables_type = {}
        self.variables_value = {}
        self.error_msg = ""

        for tag in tags:
            name_type = tag.split(':')
            p_name = name_type[0]

            if(len(name_type) == 2):
                p_type = name_type[1]
                if p_type != "":
                    self.variables_type[p_name] = p_type
            self.variables_value[p_name] = "undef"
        
    
    def SetParameter(self, parameter_name, parameter_value):
        if parameter_name in self.variables_type.keys():
            if self.variables_type[parameter_name] == "integer":
                parameter_value = int(parameter_value)
                self.variables_value[parameter_name] = parameter_value
            elif self.variables_type[parameter_name] == "act_fn":
                if not any(parameter_value.value == item.value for item in ActivationFunction):
                    self.error_msg = f"Unknown activation function '{parameter_value}'"
                    return
                self.variables_value[parameter_name] = parameter_value.value

    def Build(self):
        for _, (var_name, var_val) in enumerate(self.variables_value.items()):
            if var_val == "undef":
                self.error_msg = f"Unassigned variable '{var_name}'."
                break

        tags = re.finditer(r'<<([^>]*)>>', self.base_arch_code)

        modifications = []
        for tag in tags:
            start, end = tag.span()
            match_text = tag.group(1)
            p_name = match_text.split(":")[0]

            replacement_text = self.variables_value[p_name]
            modifications.append((start, end, str(replacement_text)))
        
        # Apply modifications to the original text
        final_code = self.base_arch_code
        offset = 0
        for start, end, replacement_text in modifications:
            start += offset
            end += offset
            final_code = final_code[:start] + replacement_text + final_code[end:]
            offset += len(replacement_text) - (end - start)

        import_list = "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torchvision.datasets as datasets\nimport torchvision.transforms as transforms\nimport torch.nn.functional as F\n"
        self.final_code = import_list + final_code
        if self.error_msg == "":
            self.error_msg == "OK"
            exec(self.final_code, globals())
            self.arch_class = globals()[self.base_arch.value]
        
        return self.error_msg
    
    def CreateModel(self):
        if hasattr(self, "arch_class"):
            return self.arch_class()
        else:
            return None