from . import compiler

def kernel(*args):
  def kernel_wrapper(fun):
    return compiler.compile(fun, *args)
  return kernel_wrapper

