import numpy as np


def unbroadcast(grad, shape):
    extra_axes = tuple(range(grad.ndim - len(shape)))
    if extra_axes:
        grad = grad.sum(axis=extra_axes, keepdims=False)

    broadcast_axes = tuple(
        i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1
    )
    if broadcast_axes:
        grad = grad.sum(axis=broadcast_axes, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data=None, requires_grad=False, _op=None, _parents=()):
        self.data = np.asarray(data, dtype=float)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._op = _op
        self._parents = set(_parents)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """adds two tensor instances"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="+",
            _parents=(self, other),
        )

        def _backward():
            # d(out)/ d(self)
            # d(out)/ d(other)

            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """multiply two tensor instances"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="*",
            _parents=(self, other),
        )

        def _backward():
            # d(out)/ d(self) = other.data
            # d(out)/ d(other) = self.data

            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        """subtracts two tensor instances"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="-",
            _parents=(self, other),
        )

        def _backward():
            # d = a - b
            # f(d) / fa = 1 (b is treated as constant)
            # f(d)/fb = -1 (a is treated as constant)

            self.grad += unbroadcast(1.0 * out.grad, self.data.shape)
            other.grad += unbroadcast(-1.0 * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __truediv__(self, other):
        """divides two tensor instances"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="/",
            _parents=(self, other),
        )

        def _backward():
            # d = a / b
            # f(d) / fa = 1/b (b is treated as constant)
            # f(d)/fb = -a/b ** 2 (a is treated as constant)

            self.grad += unbroadcast(1 / (other.data) * out.grad, self.data.shape)
            other.grad += unbroadcast(
                (-(self.data) / (other.data) ** 2) * out.grad, other.data.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, other):
        """raises each element of self with a constant (other)"""
        assert isinstance(other, (int, float)), "only accepting int or float"
        out = Tensor(
            data=self.data**other,
            _op="**",
            _parents=(self,),
            requires_grad=self.requires_grad,
        )

        def _backward():
            # d(x**n)/dx = n*x**n-1
            self.grad += (other * self.data ** (other - 1)) * (out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        """returns the negative values (signs changed)"""
        return self * -1

    # ---------- utils ----------
    def backward(self, grad: np.ndarray | None = None):
        """DAG generator, traverses the function call on stack"""
        topo = []
        visited = set()

        def build(v):
            stack = [(v, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    topo.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                for child in node._parents:
                    stack.append((child, False))
            return topo

        build(self)
        # seed
        if grad is None:
            self.grad = np.ones_like(self.data, dtype=float)
        else:
            # check shape
            assert self.data.shape == grad.shape, "gradient has to be the same shape"
            self.grad = grad

        for v in topo:
            if v.grad is None:
                v.grad = np.zeros_like(v.data)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """zeros the grad attribute of the class"""
        self.grad = None

    def sum(
        self,
        axis=None,
        keepdims: bool = False,
    ):
        """sums across the mentioned axis, backward respects original shapes"""
        # FP
        data = np.sum(self.data, axis=axis, keepdims=keepdims)

        # BP
        out = Tensor(
            data=data, _op="sum", _parents=(self,), requires_grad=self.requires_grad
        )

        def _backward():
            if keepdims or axis == None:
                self.grad += np.ones_like(self.data) * out.grad

            if not keepdims and axis != None:
                self.grad += np.ones_like(self.data) * np.expand_dims(
                    out.grad, axis=axis
                )

        out._backward = _backward
        return out

    def log(self):
        """computes element-wise log of the operand"""
        out = Tensor(
            data=np.log(self.data),
            _op="log",
            _parents=(self,),
            requires_grad=self.requires_grad,
        )

        def _backward():
            self.grad += ((self.data) ** -1.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """computes element-wise exp of the operand"""
        data = np.exp(self.data)
        out = Tensor(
            data=data,
            _op="exp",
            _parents=(self,),
            requires_grad=self.requires_grad,
        )

        def _backward():
            self.grad += data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """performs matrix multiplication for two tensor instances"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="@",
            _parents=(self, other),
        )

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ---------- Non linearity ----------

    def relu(self):
        """transforms each element as per the relu rule"""
        out = Tensor(
            data=np.maximum(0, self.data),
            _op="relu",
            _parents=(self,),
            requires_grad=self.requires_grad,
        )

        def _backward():
            self.grad += np.where(self.data > 0, 1, 0) * out.grad

        out._backward = _backward
        return out
