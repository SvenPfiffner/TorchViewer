
from .. import Tensor

class Module:
    def __init__(self):
        self._modules = {}
        self.training = True
        
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if not hasattr(self, '_modules'):
                object.__setattr__(self, '_modules', {})
            self._modules[name] = value
        object.__setattr__(self, name, value)
        
    def __call__(self, *args, **kwargs):
        # Hook for tracing could go here
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self._ordered_modules = []
        for i, module in enumerate(args):
            self.add_module(str(i), module)
            
    def add_module(self, name, module):
        self._modules[name] = module
        self._ordered_modules.append(module)
        
    def forward(self, input):
        for module in self._ordered_modules:
            input = module(input)
        return input

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, input):
        # input shape: (batch, *, in_features)
        # output shape: (batch, *, out_features)
        if hasattr(input, 'shape'):
            new_shape = list(input.shape[:-1]) + [self.out_features]
            return Tensor(tuple(new_shape), history=[('linear', (input, self))])
        return Tensor((1, self.out_features))

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, input):
        # input: (N, C_in, H, W)
        # output: (N, C_out, H_out, W_out)
        if hasattr(input, 'shape') and len(input.shape) == 4:
            N, C, H, W = input.shape
            # Simplified formula
            H_out = int((H + 2 * self.padding - (self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0])) / self.stride + 1)
            W_out = int((W + 2 * self.padding - (self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0])) / self.stride + 1)
            return Tensor((N, self.out_channels, H_out, W_out), history=[('conv2d', (input, self))])
        return Tensor((1, self.out_channels, 1, 1))

class ReLU(Module):
    def forward(self, input):
        return Tensor(input.shape, history=[('relu', (input, self))])

class Flatten(Module):
    def forward(self, input):
        if hasattr(input, 'shape'):
            # Flatten all dims except batch
            flat_dim = 1
            for d in input.shape[1:]:
                flat_dim *= d
            return Tensor((input.shape[0], flat_dim), history=[('flatten', (input, self))])
        return input

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        
    def forward(self, input):
        if hasattr(input, 'shape') and len(input.shape) == 4:
            N, C, H, W = input.shape
            k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
            s = self.stride if isinstance(self.stride, int) else self.stride[0]
            H_out = int((H + 2 * self.padding - k) / s + 1)
            W_out = int((W + 2 * self.padding - k) / s + 1)
            return Tensor((N, C, H_out, W_out), history=[('maxpool2d', (input, self))])
        return input

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, input):
        return input
