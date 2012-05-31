import math
import numpy as np
#from pynvvm import kernel, blockIdx, blockDim, threadIdx
import pynvvm

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

@pynvvm.kernel.kernel(pynvvm.nvtype.array(pynvvm.nvtype.float32), pynvvm.nvtype.array(pynvvm.nvtype.float32), pynvvm.nvtype.array(pynvvm.nvtype.float32))
def vec_add(a, b, c):
  i = pynvvm_ctaid_x() * pynvvm_ntid_x() + pynvvm_tid_x()
  if (0 == i % 2):
    c[i] = a[i] + b[i]
  else:
    c[i] = a[i] - b[i]
    
  return

a = np.array([1.0, 2.0, 3.0]).astype(np.float32)
b = np.array([1.0, 2.0, 3.0]).astype(np.float32)
c = np.zeros(3).astype(np.float32)

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
