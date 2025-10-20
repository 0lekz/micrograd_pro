# core/nn.py
import random
import numpy as np
from engine import Value

"""
micrograd_pro.core.nn
---------------------
Minimal neural network module system built on the micrograd_pro autograd engine.
Includes Linear, LayerNorm, Dropout, and Sequential layers compatible with Value-based computation.
"""


class Layer:
    def parameters(self):
        return []  # By default, no parameters

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.weights = [
            Value(np.random.uniform(-1, 1)) for _ in range(in_features * out_features)
        ]
        self.bias = [Value(0.0) for _ in range(out_features)] if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias  # decided to add this just like in PyTorch since it's a useful option for some layers

    def __call__(self, x):
        out = []
        for i in range(self.out_features):
            # Compute the dot product for the i-th output neuron
            start = i * self.in_features
            end = start + self.in_features
            neuron_weights = self.weights[start:end]
            neuron_output = sum(w * xi for w, xi in zip(neuron_weights, x))
            if self.has_bias:
                neuron_output += self.bias[i]

            # Wrap neuron_output in Value with _prev and _backward
            def make_backward(
                neuron_weights=neuron_weights,
                neuron_output=neuron_output,
                x=x,
                bias=self.bias[i] if self.has_bias else None,
            ):
                # capture local variables in default to bind in closure
                out_val = neuron_output

                def _backward():
                    # propagete gradients to weights, inputs, and bias
                    for w, xi in zip(neuron_weights, x):
                        w.grad += xi.data * out_val.grad  # Gradient w.r.t. weights
                        xi.grad += w.data * out_val.grad  # Gradient w.r.t. inputs
                    if self.has_bias:
                        bias.grad += out_val.grad  # Gradient w.r.t. bias

                out_val._backward = _backward
                return out_val

            out.append(make_backward())

        return out

    def parameters(self):
        return self.weights + (self.bias if self.has_bias else [])


# Let's implement Sequential container for micrograd pro:
class Sequential(Layer):
    def __init__(self, *layers):
        # Store layers in a list
        self.layers = layers

    def __call__(self, x):
        # Pass input through each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            # only call parameters() if the layer has it
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    # Also add repr for better visualization
    def __repr__(self):
        layer_reprs = ", ".join([layer.__class__.__name__ for layer in self.layers])
        return f"Sequential({layer_reprs})"


# Let's implement LayerNorm class with backward method:


class LayerNorm(Layer):
    def __init__(self, normalized_shape, eps=1e-5):
        # normalized_shape can be an int or a tuple of ints
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        n = int(np.prod(normalized_shape))
        self.gamma = [Value(1.0) for _ in range(n)]  # Scale parameter
        self.beta = [Value(0.0) for _ in range(n)]  # Shift parameter

    def __call__(self, x):
        """
        x: list of Value objects (length d)
        returns: list of Value objects y_i = gamma_i * hat_x_i + beta_i
        where hat_x_i = (x_i - mu) / sqrt(sigma^2 + eps)
        """
        d = len(x)
        # 1. Calculate mean and variance as scalars
        mean = sum(xi.data for xi in x) / d
        var = sum((xi.data - mean) ** 2 for xi in x) / d
        s = np.sqrt(var + self.eps)  # s == sqrt(sigma^2 + eps)

        # 2. Compute normalized values hat_x as floats
        hat = [(xi.data - mean) / s for xi in x]

        # 3. Crate output Value objects (data computed from gamma and beta)
        outs = []
        for i, (xi, gi, bi) in enumerate(zip(x, self.gamma, self.beta)):
            out_val = Value(
                gi.data * hat[i] + bi.data, _children=(xi, gi, bi), _op="layernorm"
            )
            outs.append(out_val)

        # 4. Define backward for each output. Each backward uses *current* outs[j].grad
        # since xi.grad depends on sums over all g_j * gamma_j etc.
        def make_backward(i, xi=None, gi=None, bi=None):
            # bind referenced objects in closure
            xi = x[i]
            gi = self.gamma[i]
            bi = self.beta[i]

            def _backward():
                # upstream gradients for y_j (these may have been accumulated already)
                g_list = [oj.grad for oj in outs]  # g_j = dL/dy_j

                # precompute sums used in formuala:
                # sum1 = sum_j g_j * gamma_j
                # sum2 = sum_j g_j * gamma_j * hat_x_j
                sum1, sum2 = 0.0, 0.0
                for j, gj in enumerate(g_list):
                    sum1 += gj * self.gamma[j].data
                    sum2 += gj * self.gamma[j].data * hat[j]

                # local values
                s_val = s  # sqrt(var + eps)
                hat_i = hat[i]  # hat_x_i
                g_i = g_list[i]  # dL/dy_i
                gamma_i = gi.data  # gamma_i

                # d_beta_i = g_i
                bi.grad += g_i

                # d_gamma_i = g_i * hat_i
                gi.grad += g_i * hat_i

                # d_x_i = (1/s) * (g_i * gamma_i - (1/d) * sum1 - (hat_i/d) * sum2)
                dx_i = (1 / s_val) * (
                    g_i * gamma_i - (1 / d) * sum1 - (hat_i / d) * sum2
                )
                xi.grad += dx_i

            return _backward

        for i, out_val in enumerate(outs):
            out_val._backward = make_backward(i)

        return outs

    def parameters(self):
        return super().parameters() + self.gamma + self.beta


class Dropout(Layer):
    def __init__(self, p=0.1, seed=None):
        self.p = p  # dropout probability
        self.training = True  # by default, we are in training mode
        # wanted to implement seed as well
        if seed is not None:
            random.seed(seed)

    def __call__(self, x):
        if self.training:
            # dropout mask is for each element in x, 0 with prob p, 1 with prob (1-p)
            self.mask = [
                0 if random.random() < self.p else 1 for _ in x
            ]  # Store mask for backward pass
            # from wikipedia bernoulli distribution
            out = [xi * m / (1 - self.p) for xi, m in zip(x, self.mask)]
        else:
            out = x  # during evaluation, do nothing

        # we don't need to implement backward since dropout is not learnable and does not have parameters
        # Backward flow is automatically handled via Value.__mul__,
        # no custom backward() needed.
        return out

    def parameters(self):
        return super().parameters()  # no parameters in dropout


# -- activation functions as Layer classes -- #
class ReLU(Layer):
    def __call__(self, x):
        out = [xi.relu() for xi in x]  # use Value.relu()
        return out

    def parameters(self):
        return super().parameters()  # no parameters in ReLU


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        out = [xi.leaky_relu(self.alpha) for xi in x]  # use Value.leaky_relu()
        return out

    def parameters(self):
        return super().parameters()  # no parameters in LeakyReLU


class Tanh(Layer):
    def __call__(self, x):
        out = [xi.tanh() for xi in x]  # use Value.tanh()
        return out

    def parameters(self):
        return super().parameters()  # no parameters in Tanh


class GELU(Layer):
    def __call__(self, x):
        out = [xi.gelu() for xi in x]  # use Value.gelu()
        return out

    def parameters(self):
        return super().parameters()  # no parameters in GELU
