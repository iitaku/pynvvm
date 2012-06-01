import math
import numpy as np
from pynvvm.kernel import kernel
from pynvvm.nvtype import array, float32, int32

@kernel(array(float32), array(float32), array(float32), float32(), int32(), int32())
def saxpy(z, x, y, a, w, h):
  xidx = pynvvm_ctaid_x() * pynvvm_ntid_x() + pynvvm_tid_x()
  yidx = pynvvm_ctaid_y() * pynvvm_ntid_y() + pynvvm_tid_y()
  
  if yidx < h and xidx < w:
    i = yidx * w + xidx
    z[i] = a * x[i] + y[i]
  
  return

x = np.ndarray([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
y = np.ndarray([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
z = np.zeros(4).astype(np.float32)

bsz = (4, 1, 1)
gsz  = (1, 1, 1)

saxpy(bsz, gsz)(a, b, c, np.int32(3))

print(c)



#import pycuda.autoinit
#import pycuda.driver as drv
#import numpy
#
#from pycuda.compiler import SourceModule
#
#mod = SourceModule("""
#__global__ void multiply_them(float *dest, float *a, float *b, float v)
#{
#    const int i = threadIdx.x;
#    //dest[i] = a[i] * b[i];
#    dest[i] = v;
#}
#""")
#
#multiply_them = mod.get_function("multiply_them")
#
#a = numpy.random.randn(400).astype(numpy.float32)
#b = numpy.random.randn(400).astype(numpy.float32)
#
#dest = numpy.zeros_like(a)
#multiply_them(
#    drv.Out(dest), drv.In(a), drv.In(b), drv.In()
#    block=(400,1,1), grid=(1,1))
#
#print(dest-a*b)
#
