import ast
import inspect
import sys
from string import Template
#import pycuda.autoinit
#import pycuda.driver as drv
import numpy

from . import nvvm
from . import llvm

class ASTTraverser(ast.NodeVisitor):
  
  def generic_visit(self, node):
    print('%40s : %s' % (node, str(node.__dict__)))
    ast.NodeVisitor.generic_visit(self, node)

#end class ASTTraverser

def test(a, b, c):
  c[0] = a[0] + b[0]

def compile(fun, *args):
  
  kernel_ast = ast.parse(inspect.getsource(fun))
  
  class ArgCheck(ast.NodeVisitor):
    def generic_visit(self, node):
      ast.NodeVisitor.generic_visit(self, node)
    
    def visit_FunctionDef(self, node):
      self.args = node.args.args
      if not len(args) == (len(node.args.args)):
        raise Exception, 'error : invalid kernel argument number'
  # class ArgCheck

  check = ArgCheck()
  check.visit(kernel_ast)

  class NameTracker(ast.NodeVisitor):
    def __init__(self):
      self.names = set()
    def generic_visit(self, node):
      ast.NodeVisitor.generic_visit(self, node)
    def visit_Name(self, node):
      self.names.add(node.id)
  #end class TrackName

  class IOTracker(ast.NodeVisitor):
    (i, o, io) = (0x1, 0x2, 0x3)
    def __init__(self, args_name):
      self.args_iomap = { arg_name:0x0 for arg_name in args_name }
    def generic_visit(self, node):
      ast.NodeVisitor.generic_visit(self, node)
    def visit_Assign(self, node):
      for target in node.targets:
        track = NameTracker()
        track.visit(target)
        for name in (track.names & set(self.args_iomap.keys())):
          self.args_iomap[name] |= IOTracker.o
      
      track = NameTracker()
      track.visit(node.value)

      for name in (track.names & set(self.args_iomap.keys())):
        self.args_iomap[name] |= IOTracker.i
  
  # end class IOTracker 
  
  args_name = map(lambda x : x.id, check.args)
  io_track = IOTracker(args_name)
  io_track.visit(kernel_ast)
  print(io_track.args_iomap)
  return test

class LambdaExtractor(ast.NodeVisitor):

  def __init__(self):
    self._lambda_node = None
  
  def generic_visit(self, node):
    ast.NodeVisitor.generic_visit(self, node)
  
  def visit_Lambda(self, node):
    if None == self._lambda_node:
      self._lambda_node = node
      ast.NodeVisitor.generic_visit(self, node)
    else:
      raise Exception, 'error : lambda expr contain another lambda expr'
  
  @property
  def lambda_node(self):
    return self._lambda_node

#end class LambdaExtractor


class MapLambdaTraverser(ast.NodeVisitor):
   
  def generic_visit(self, node):
    ast.NodeVisitor.generic_visit(self, node)

   
#end class MapLambdaTraverser


class Composer:
  
  def __init__(self, sequences):
    self._layout = '''
    target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
    '''
    
    self._intrinsics = '''
    declare float  @llvm.sqrt.f32(float)  nounwind readnone
    declare double @llvm.sqrt.f64(double) nounwind readnone
    
    declare float  @llvm.fma.f32(float)  nounwind readnone
    declare double @llvm.fma.f64(double) nounwind readnone
    
    declare i16 @llvm.bswap.i16(i16) nounwind readnone
    declare i32 @llvm.bswap.i32(i32) nounwind readnone
    declare i64 @llvm.bswap.f64(i64) nounwind readnone
    
    declare i8  @llvm.ctpop.i8(i8)   nounwind readnone
    declare i16 @llvm.ctpop.i16(i16) nounwind readnone
    declare i32 @llvm.ctpop.i32(i32) nounwind readnone
    declare i64 @llvm.ctpop.f64(i64) nounwind readnone
      
    declare i8  @llvm.ctlz.i8(i8)   nounwind readnone
    declare i16 @llvm.ctlz.i16(i16) nounwind readnone
    declare i32 @llvm.ctlz.i32(i32) nounwind readnone
    declare i64 @llvm.ctlz.f64(i64) nounwind readnone

    declare i8  @llvm.cttz.i8(i8)   nounwind readnone
    declare i16 @llvm.cttz.i16(i16) nounwind readnone
    declare i32 @llvm.cttz.i32(i32) nounwind readnone
    declare i64 @llvm.cttz.f64(i64) nounwind readnone

    declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
    declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
    declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
    '''
    
    self._kernel ='''
    define void @stub_kernel(${type}* %incoming, ${type}* %outgoing) {
    entry:  
      %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
      %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
      %2 = mul i32 %0, %1
      %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      %id = add i32 %2, %3
      %idx = sext i32 %id to i64
      %incoming_p = getelementptr inbounds ${type}* %incoming, i64 %idx
      %cmp = icmp slt i64 %idx, ${num}
      br i1 %cmp, label %cont, label %exit
    cont:
      ${logic}
      br label %exit
    exit:
      ret void
    }
    '''
    
    self._metadata = '''
    !nvvm.annotations = !{!1}
    !1 = metadata !{void (${type}*, ${type}*)* @stub_kernel, metadata !"kernel", i32 1}
    '''
    
    result, self._cu = nvvm.instance.create_cu()
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    
    self._func = ''
    
    self._sequences = sequences
    
    for s in sequences:
      self._func += s.codegen() + '\n'
    return
 
  def __del__(self):
    result = nvvm.instance.destroy_cu(self._cu)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    return

  def run(self, incoming):
       
    code_template = Template('\n'.join([self._layout, self._func, self._kernel, self._intrinsics, self._metadata]))

    print(incoming.dtype)
    
    test_type = 'float'
    test_logic = '''
    %outgoing_p = getelementptr inbounds float* %outgoing, i64 %idx
    %res = call float @llvm.sqrt.f32(float 2514.0)
    store float %res, float* %outgoing_p, align 4
    '''
    
    code = code_template.safe_substitute({'type':test_type, 'num':len(incoming), 'logic':test_logic})
    
    #for i, line in enumerate(code.strip().split('\n')):
    #  print('%3d : %s' % (i+1, line)) 

    result = nvvm.instance.cu_add_module(self._cu, code)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
 
    options = []
    result = nvvm.instance.compile_cu(self._cu, options)
    if nvvm.nvvmResult.NVVM_SUCCESS != result.val:
      result, msg_buffer = nvvm.instance.get_compilation_log(self._cu)
      print(msg_buffer)
      assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
      sys.exit(-1)
    
    result, ptx_code = nvvm.instance.get_compiled_result(self._cu)
    #print(ptx_code)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    
    #m = drv.module_from_buffer(ptx_code)
    #stub_kernel = m.get_function('stub_kernel')
   
    #block_size = (256, 1, 1)
    #grid_size  = (1024, 1, 1)

    #outgoing = numpy.arange(len(incoming)).astype(numpy.float32)
  
    #stub_kernel(drv.In(incoming), drv.Out(outgoing), block=block_size, grid=grid_size)

    return outgoing
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
    ASTTraverser().visit(self._lambda_node)
    #print('')
    if not 1 == len(self._lambda_node.args.args):
      raise Exception, 'error : lambda expr must accept #1 args'
    return

  def codegen(self):
    class Visitor(ast.NodeVisitor):
      def __init__(self):
        self.fun = ''
        self._id = 0
        return
      
      def visit_Lambda(self, node):
        self.fun += Template('''
        define ${type} @${fun} (${type} %${val}) {
        entry:
        ${local_logic}
        ret ${type} %res
        }
        ''').safe_substitute({'val':node.args.args[0].id})
        
        ast.NodeVisitor.generic_visit(self, node)
            
      def visit_BinOp(self, node):
        pass 

      def visit_Call(self, node):
         print(help(node))
    
    #end class Visitor

    v = Visitor()
    v.visit(self._lambda_node)
    
    return v.fun
#end class Mapper

class Reducer(Sequencer):
  def __init__(self, fun):
    Sequencer.__init__(self, fun)
    #ASTTraverser().visit(self._lambda_node)
    #print('')
    return
  
  def codegen(self):
    return ''
#end class Mapper

