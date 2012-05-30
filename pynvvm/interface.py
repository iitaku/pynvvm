from . import compiler

def kernel(fun):
  return compiler.compile(fun)

