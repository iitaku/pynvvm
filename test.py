import math
import numpy as np
from pynvvm.kernel import kernel
from pynvvm.nvtype import array, float32, int32

@kernel(array(float32), array(float32), array(float32))
def vec_add(a, b, c):
  i = pynvvm_ctaid_x() * pynvvm_ntid_x() + pynvvm_tid_x()
  if (i < 2):
    c[i] = a[i] + b[i]
  else:
    c[i] = a[i] - b[i]
  return

a = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
b = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
c = np.zeros(4).astype(np.float32)

bsz = (4, 1, 1)
gsz  = (1, 1, 1)

vec_add(bsz, gsz)(a, b, c)

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
