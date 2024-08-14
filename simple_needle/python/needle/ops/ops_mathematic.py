"""Operator implementations."""


from numbers import Number
from typing import Optional, List, Tuple, Union

import numpy as np

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        safe_a=array_api.where(a==0,array_api.inf,a)

        return array_api.power(safe_a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        input, =node.inputs
        ### BEGIN YOUR SOLUTION
        return (out_grad * ( self.scalar * PowerScalar( self.scalar-1 )( input ) ),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        #  there exist an error what happened if there exist a zero value in the b vector
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        gradient_a=out_grad * PowerScalar(-1)(b)

        # deal with the condition : if there exist zero in the b vector
        # gradient_a=np.where(gradient_a==np.inf, 0 ,gradient_a)

        gradient_b=out_grad * (a * Negate()(PowerScalar(-2)(b)))

        return gradient_a , gradient_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return  a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * (1/self.scalar),)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes == None:
            return array_api.swapaxes(a,-1,-2)
        return array_api.swapaxes(a,*self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (Transpose(self.axes)(out_grad),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTIO
        input_shape=node.inputs[0].shape
        return (out_grad.reshape(input_shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # my solution
        input=node.inputs[0]
        input_shape=array_api.array(input.shape)
        ouput_shape=array_api.array(out_grad.shape)

        if len(input_shape) ==0:
            input_shape=array_api.array([1])

        broadcast_dim=np.where(input_shape!=ouput_shape)[0]


        broadcast_dim=tuple(broadcast_dim)

        grad=summation(out_grad,broadcast_dim)
        if grad.shape != input.shape:
                grad = reshape(summation(out_grad,broadcast_dim),input.shape)

        return (grad,)

        # better solution

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes==None:
            axes=range(len(node.inputs[0].shape))
        else:
            if isinstance(self.axes,int):
                axes=(self.axes,)
            else:
                axes=self.axes

        input=node.inputs[0]
        keepdim_shape=[]
        for index in range(len(input.shape)):
            if index in list(axes):
                keepdim_shape.append(1)
            else:
                keepdim_shape.append(input.shape[index])

        # print(f"temp_dim = {keepdim_shape}")
        out_grad=reshape(out_grad,keepdim_shape)
        # print(f"out_grad.shape={out_grad.shape}")

        return (broadcast_to(out_grad,input.shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION


    # feel like the gradient acumulation
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a=matmul(out_grad,Transpose(None)(b))
        grad_b=matmul(Transpose(None)(a) ,out_grad)

        if a.shape !=grad_a.shape:
            grad_a = summation(grad_a,tuple(range(len(grad_a.shape)-len(a.shape))))
        if b.shape!= grad_b.shape:
            grad_b = summation(grad_b, tuple(range(len(grad_b.shape) - len(b.shape))))

        return grad_a,grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (Negate()(out_grad),)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(f"in log outgrad shape {out_grad}")
        # print(f"in log outgrad shape {node.inputs}")
        print(f"divide(out_grad,node.inputs[0]) is {divide(out_grad,node.inputs[0])}")
        return (divide(out_grad,node.inputs[0]), )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (multiply(out_grad,exp(node.inputs[0])),)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        input = node.inputs[0]
        grad = divide(relu(input) , input)

        return multiply( out_grad , grad )
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
