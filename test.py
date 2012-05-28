import xenodermus.nvvm as xeno
import xenodermus.gpuarray

#xenodermus.gpuarray.map(lambda x : x + 1, [1.0, 2.0, 3.0])

if __name__ == '__main__':
  result = xeno.Init()
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val
  
  result, cu = xeno.CreateCU()
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val

  code = '''
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define i32 @ave(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  %div = sdiv i32 %add, 2
  ret i32 %div
}

define void @simple(i32* %data) {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %0, %1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %2
  %call = call i32 @ave(i32 %add, i32 %add)
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32* %data, i64 %idxprom
  store i32 %call, i32* %arrayidx, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvm.annotations = !{!1}
!1 = metadata !{void (i32*)* @simple, metadata !"kernel", i32 1}
  '''
  
  result = xeno.CUAddModule(cu, code)
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val
 
  options = ['-g', '-target=ptx']
  result = xeno.CompileCU(cu, options)
  if xeno.nvvmResult.NVVM_SUCCESS != result.val:
    result, msg_buffer = xeno.GetCompilationLog(cu)
    print msg_buffer
    assert xeno.nvvmResult.NVVM_SUCCESS == result.val
  
  result, msg_buffer = xeno.GetCompiledResult(cu)
  print msg_buffer
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val
 
  result = xeno.DestroyCU(cu)
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val

  result = xeno.Fini()
  assert xeno.nvvmResult.NVVM_SUCCESS == result.val
 
#import pycuda.autoinit
#import pycuda.driver as drv
#import numpy
#
#from pycuda.compiler import SourceModule
#mod = SourceModule("""
#__global__ void multiply_them(float *dest, float *a, float *b)
#{
#    const int i = threadIdx.x;
#      dest[i] = a[i] * b[i];
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
#    drv.Out(dest), drv.In(a), drv.In(b),
#    block=(400,1,1), grid=(1,1))
#
#print dest-a*b
#
