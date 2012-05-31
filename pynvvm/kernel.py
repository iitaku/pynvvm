from . import compiler

def kernel(*args):
  def inner_kernel(fun):
    return compiler.compile(fun, *args)

  return inner_kernel

