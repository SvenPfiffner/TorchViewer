
# Mock Torch for Visualization
import sys

class Tensor:
    def __init__(self, shape, history=None):
        self.shape = shape
        self.history = history or [] # List of (op_name, inputs)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def __add__(self, other):
        return Tensor(self.shape, history=[('add', (self, other))])
    
    def __mul__(self, other):
        return Tensor(self.shape, history=[('mul', (self, other))])
    
    def matmul(self, other):
        # Simplified matmul shape inference
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            new_shape = list(self.shape[:-1]) + [other.shape[-1]]
            return Tensor(tuple(new_shape), history=[('matmul', (self, other))])
        return Tensor(self.shape, history=[('matmul', (self, other))])

    def view(self, *shape):
        return Tensor(shape, history=[('view', (self, shape))])
    
    def reshape(self, *shape):
        return Tensor(shape, history=[('reshape', (self, shape))])

def randn(*shape):
    return Tensor(shape)

def zeros(*shape):
    return Tensor(shape)

def ones(*shape):
    return Tensor(shape)

# Expose nn
from . import nn
