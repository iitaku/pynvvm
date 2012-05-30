import math
import numpy
#from pynvvm import kernel, blockIdx, blockDim, threadIdx
from pynvvm import kernel

#def main():
#  #m = Mapper(lambda x : math.sqrt(x + 1))
#  m = Mapper(lambda x : math.sqrt(x + 1))
#  r = Reducer(lambda x, y : x if x < y else y)
#  c = Composer([m, r])
#  
#  incoming = numpy.array([1.0, 2.0, 3.0]).astype(numpy.float32)
#  outgoing = c.run(incoming)
#  print(incoming)
#  print(outgoing)
#
#if __name__ == '__main__':
#  main()

@kernel
def vec_add(a, b, c):
  i = blockIdx.x * blockDim.x + threadIdx.x
  c[i] = a[i] + b[i]
  return

a = numpy.array([1.0, 2.0, 3.0]).astype(numpy.float32)
b = numpy.array([1.0, 2.0, 3.0]).astype(numpy.float32)
c = numpy.zeros(3).astype(numpy.float32)

vec_add(a, b, c)

print(c)

#import pycuda.autoinit
#import pycuda.driver as drv
#import numpy
#
#from pycuda.compiler import SourceModule
#
##mod = SourceModule("""
##__global__ void multiply_them(float *dest, float *a, float *b)
##{
##    const int i = threadIdx.x;
##      dest[i] = a[i] * b[i];
##}
##""")
##
##multiply_them = mod.get_function("multiply_them")
##
##a = numpy.random.randn(400).astype(numpy.float32)
##b = numpy.random.randn(400).astype(numpy.float32)
##
##dest = numpy.zeros_like(a)
##multiply_them(
##    drv.Out(dest), drv.In(a), drv.In(b),
##    block=(400,1,1), grid=(1,1))
##
##print(dest-a*b)
#a = numpy.random.randn(400).astype(numpy.float32)
#b = numpy.random.randn(400).astype(numpy.float32)
#
#dest = numpy.zeros_like(a)
#multiply_them(
#    drv.Out(dest), drv.In(a), drv.In(b),
#    block=(400,1,1), grid=(1,1))
#
#print(dest-a*b)
#
