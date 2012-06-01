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

def gpu(x, y, a, n):
  z = np.zeros_like(x)
  
  bsz = (16, 16, 1)
  gsz  = ((n+16-1)/16, (n+16-1)/16, 1)
  
  saxpy(bsz, gsz)(z, x, y, a, np.int32(n), np.int32(n))

  return z

def cpu(x, y, a, n):
  return a * x + y

if '__main__' == __name__:
  
  n = 1024

  x = np.random.randn(n*n).astype(np.float32)
  y = np.random.randn(n*n).astype(np.float32)
  a = np.float32(2.71828183)

  print cpu(x, y, a, n) - gpu (x, y, a, n)
