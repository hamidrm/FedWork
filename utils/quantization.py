import torch
from abc import ABC, abstractmethod


class QuantizationClass:
    @abstractmethod
    def quantize(self, tensor):
        pass

    @abstractmethod
    def dequantize(self, quantized_tensor, min_val, scale):
        pass

class NonQuantizer(QuantizationClass):
    def __init__(self):
        pass

    def quantize(self, tensor):
        return tensor, 0, 0

    def dequantize(self, quantized_tensor, min_val, scale):
        return quantized_tensor
    
class RandomizedQuantizer(QuantizationClass):
    def __init__(self, num_levels=256):
        self.num_levels = num_levels

    def quantize(self, tensor):
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (self.num_levels - 1)
        normalized_tensor = (tensor - min_val) / scale
        quantized_tensor = torch.floor(normalized_tensor + torch.rand_like(normalized_tensor))
        return quantized_tensor, min_val, scale

    def dequantize(self, quantized_tensor, min_val, scale):
        return quantized_tensor * scale + min_val

class UniformQuantizer(QuantizationClass):
    def __init__(self, num_levels=256):
        self.num_levels = num_levels

    def quantize(self, tensor):
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (self.num_levels - 1)
        quantized = torch.round((tensor - min_val) / scale)
        return quantized, min_val, scale

    def dequantize(self, quantized_tensor, min_val, scale):
        return quantized_tensor * scale + min_val


class QSGDQuantizer(QuantizationClass):
    def __init__(self, num_levels=256):
        self.num_levels = num_levels
    def quantize(self, tensor):
        norm = tensor.norm(p=2)
        scale = norm / self.num_levels
        sign = tensor.sign()
        abs_tensor = tensor.abs()
        q = (abs_tensor / scale).floor()
        prob = (abs_tensor / scale) - q
        rand_tensor = torch.rand_like(prob)
        q += torch.where(rand_tensor < prob, torch.ones_like(q), torch.zeros_like(q))
        quantized_tensor = sign * q
        return quantized_tensor, 0, scale

    def dequantize(self, quantized_tensor, min_val, scale):
        dequantized_tensor = quantized_tensor * scale
        return dequantized_tensor