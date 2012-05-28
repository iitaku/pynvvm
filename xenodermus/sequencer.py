import ast
import inspect

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from . import nvvm

class ASTTraverser(ast.NodeVisitor):
  def generic_visit(self, node):
    print(' : '.join([type(node).__name__, str(node.__dict__)]))
    ast.NodeVisitor.generic_visit(self, node)
#end class ASTTraverser

class LambdaExtractor(ast.NodeVisitor):
  def generic_visit(self, node):
    if ast.Lambda == type(node):
      self._lambda_node = node
      return
    else:
      ast.NodeVisitor.generic_visit(self, node)
  @property
  def lambda_node(self):
    return self._lambda_node
#end class LambdaExtractor

class Composer:
  def __init__(self, sequences):
    result, self._cu = nvvm.instance.create_cu()
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    
    for s in sequences:
      self._codegen(s)
    return
 
  def __del__(self):
    result = nvvm.instance.destroy_cu(self._cu)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    return

  def _codegen(self, s):
    return

  def run(self, array):
 
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
  
    result = nvvm.instance.cu_add_module(self._cu, code)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
 
    options = ['-target=ptx']
    result = nvvm.instance.compile_cu(self._cu, options)
    if nvvm.nvvmResult.NVVM_SUCCESS != result.val:
      result, msg_buffer = nvvm.instance.get_compilation_log(self._cu)
      print(msg_buffer)
      assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    
    result, ptx_code = nvvm.instance.get_compiled_result(self._cu)
    print(ptx_code)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val

    m = drv.module_from_buffer(ptx_code)
    simple = m.get_function('simple')

    print simple.__dict__
   
    data = numpy.random.randn(16).astype(numpy.int32)
    print data
    simple(drv.InOut(data), block=(16, 1, 1), grid=(1, 1))
    print data

    return array
#end class Composer

class Sequencer:
  def __init__(self, fun):
    t = ast.parse(inspect.getsource(fun).strip())
    lambda_extractor = LambdaExtractor()
    lambda_extractor.visit(t)
    self._lambda_node = lambda_extractor.lambda_node
    return
#end class Sequencer

#class Getter(Sequencer):
#  def __init__(self, fun):
#    Sequencer.__init__(self, fun)
#    return

class Mapper(Sequencer):
  def __init__(self, fun):
    Sequencer.__init__(self, fun)
    #ASTTraverser().visit(self._lambda_node)
    #print('')
    return
#end class Mapper

class Reducer(Sequencer):
  def __init__(self, fun):
    Sequencer.__init__(self, fun)
    #ASTTraverser().visit(self._lambda_node)
    #print('')
    return
#end class Mapper

