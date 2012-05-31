from . import compiler
import numpy as np

class ndarray:
  def __init__(self, dtype):
    self.dtype = dtype

def kernel(*args):
  def inner_kernel(fun):
    return compiler.compile(fun, *args)

  return inner_kernel

