# core/engine.py
import numpy as np

"""
micrograd/core/engine.py
------------------------
Implements a scalar value with automatic differentiation capabilities.
This is the foundational building block for constructing computational graphs
and performing backpropagation in neural networks.
"""


class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    # pro addition - Leaky ReLU
    def leaky_relu(self, alpha=0.01):
        out = Value(
            self.data if self.data > 0 else alpha * self.data, (self,), "LeakyReLU"
        )

        def _backward():
            self.grad += (1 if self.data > 0 else alpha) * out.grad

        out._backward = _backward

        return out

    # pro addition - tanh
    # could and should use np.tanh because of exponents of big numbers but implemented it from scratch for educational purposes
    def tanh(self):
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad  # derivative of tanh is 1 - tanh^2(x)

        out._backward = _backward

        return out

    # pro addition - GELU
    def gelu(self):
        """Gaussian Error Linear Unit (approximation) activation function.
        Reference: Hendrycks & Gimpel, 2016.
        """
        # This is how aproximation looks like from paper. You can read more in the reference section
        x = self.data
        tanh_out = np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))
        )  # calculate tanh part
        out = Value(0.5 * x * (1 + tanh_out), (self,), "gelu")

        def _backward():
            # derivative of GELU is a quite a bit complex, so remember calculus classes!
            # since we have 0.5x * tanh_out that also uses x, so we use the chain rule here
            # don't forget that tanh_out is a complex function of x as well
            # derivative is gonna be:
            # 0.5 * (1 + tanh_out) + 0.5 * x * dtanh/dx
            # where dtanh/dx = (1 - tanh^2(x)) * d(inner)/dx
            # and d(inner)/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
            dtanh = 1 - tanh_out**2  # derivative of tanh
            dgelu_dx = 0.5 * (1 + tanh_out) + 0.5 * x * dtanh * np.sqrt(2 / np.pi) * (
                1 + 3 * 0.044715 * (x**2)
            )
            self.grad += dgelu_dx * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = 0

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __rpow__(self, other):
        return Value(other) ** self

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
