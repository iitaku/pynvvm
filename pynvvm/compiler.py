
# note : type inferitance does not support scope
#        there are a lot of unsupported Op
#        (maybe) leaks memory

import ast
import inspect
import sys
import numpy

import pycuda.autoinit
import pycuda.driver as drv

from . import nvvm
from . import nvtype
from .externals.llvm import pyllvm

_datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

_builtin_types = {
  'pynvvm_sqrt_f32':(nvtype.float32, (nvtype.float32)),
  'pynvvm_sqrt_f64':(nvtype.float64, (nvtype.float64)),
  'pynvvm_fma_f32':(nvtype.float32, (nvtype.float32)),
  'pynvvm_fma_f64':(nvtype.float64, (nvtype.float64)),
  'pynvvm_bswap_i16':(nvtype.int16, (nvtype.int16)),
  'pynvvm_bswap_i32':(nvtype.int32, (nvtype.int32)),
  'pynvvm_bswap_i64':(nvtype.int64, (nvtype.int64)),
  'pynvvm_ctpop_i8' :(nvtype.int8,  (nvtype.int8)),
  'pynvvm_ctpop_i16':(nvtype.int16, (nvtype.int16)),
  'pynvvm_ctpop_i32':(nvtype.int32, (nvtype.int32)),
  'pynvvm_ctpop_i64':(nvtype.int64, (nvtype.int64)),
  'pynvvm_ctlz_i8' :(nvtype.int8,  (nvtype.int8)),
  'pynvvm_ctlz_i16':(nvtype.int16, (nvtype.int16)),
  'pynvvm_ctlz_i32':(nvtype.int32, (nvtype.int32)),
  'pynvvm_ctlz_i64':(nvtype.int64, (nvtype.int64)),
  'pynvvm_cttz_i8' :(nvtype.int8,  (nvtype.int8)),
  'pynvvm_cttz_i16':(nvtype.int16, (nvtype.int16)),
  'pynvvm_cttz_i32':(nvtype.int32, (nvtype.int32)),
  'pynvvm_cttz_i64':(nvtype.int64, (nvtype.int64)),
  'pynvvm_tid_x':(nvtype.int32, ()),
  'pynvvm_tid_y':(nvtype.int32, ()),
  'pynvvm_tid_z':(nvtype.int32, ()),
  'pynvvm_ntid_x':(nvtype.int32, ()),
  'pynvvm_ntid_y':(nvtype.int32, ()),
  'pynvvm_ntid_z':(nvtype.int32, ()),
  'pynvvm_ctaid_x':(nvtype.int32, ()),
  'pynvvm_ctaid_y':(nvtype.int32, ()),
  'pynvvm_ctaid_z':(nvtype.int32, ()),
  'pynvvm_nctaid_x':(nvtype.int32, ()),
  'pynvvm_nctaid_y':(nvtype.int32, ()),
  'pynvvm_nctaid_z':(nvtype.int32, ()),
}

_builtin_code = '''
define linkonce_odr float @pynvvm_sqrt_f32(float %x) {
  %y = call float @llvm.sqrt.rn.f32(float %x)
  ret float %y
}

define linkonce_odr double @pynvvm_sqrt_f64(double %x) {
  %y = call double @llvm.sqrt.rn.f64(double %x)
  ret double %y
}

define linkonce_odr float @pynvvm_fma_f32(float %x) {
  %y = call float @llvm.fma.f32(float %x)
  ret float %y
}

define linkonce_odr double @pynvvm_fma_f64(double %x) {
  %y = call double @llvm.fma.f64(double %x)
  ret double %y
}

define linkonce_odr i16 @pynvvm_bswap_i16(i16 %x) {
  %y = call i16 @llvm.bswap.i16(i16 %x)
  ret i16 %y
}

define linkonce_odr i32 @pynvvm_bswap_i32(i32 %x) {
  %y = call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %y
}

;define linkonce_odr i64 @pynvvm_bswap_i64(i64 %x) {
;  %y = call i64 @llvm.bswap.i64(i64 %x)
;  ret i64 %y
;}

define linkonce_odr i8 @pynvvm_ctpop_i8(i8 %x) {
  %y = call i8 @llvm.ctpop.i8(i8 %x)
  ret i8 %y
}

define linkonce_odr i16 @pynvvm_ctpop_i16(i16 %x) {
  %y = call i16 @llvm.ctpop.i16(i16 %x)
  ret i16 %y
}

define linkonce_odr i32 @pynvvm_ctpop_i32(i32 %x) {
  %y = call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %y
}

;define linkonce_odr i64 @pynvvm_ctpop_i64(i64 %x) {
;  %y = call i64 @llvm.ctpop.i64(i64 %x)
;  ret i64 %y
;}

define linkonce_odr i8 @pynvvm_ctlz_i8(i8 %x) {
  %y = call i8 @llvm.ctlz.i8(i8 %x)
  ret i8 %y
}

define linkonce_odr i16 @pynvvm_ctlz_i16(i16 %x) {
  %y = call i16 @llvm.ctlz.i16(i16 %x)
  ret i16 %y
}

define linkonce_odr i32 @pynvvm_ctlz_i32(i32 %x) {
  %y = call i32 @llvm.ctlz.i32(i32 %x)
  ret i32 %y
}

;define linkonce_odr i64 @pynvvm_ctlz_i64(i64 %x) {
;  %y = call i64 @llvm.ctlz.i64(i64 %x)
;  ret i64 %y
;}

define linkonce_odr i8 @pynvvm_cttz_i8(i8 %x) {
  %y = call i8 @llvm.cttz.i8(i8 %x)
  ret i8 %y
}

define linkonce_odr i16 @pynvvm_cttz_i16(i16 %x) {
  %y = call i16 @llvm.cttz.i16(i16 %x)
  ret i16 %y
}

define linkonce_odr i32 @pynvvm_cttz_i32(i32 %x) {
  %y = call i32 @llvm.cttz.i32(i32 %x)
  ret i32 %y
}

;define linkonce_odr i64 @pynvvm_cttz_i64(i64 %x) {
;  %y = call i64 @llvm.cttz.i64(i64 %x)
;  ret i64 %y
;}

define linkonce_odr i32 @pynvvm_tid_x() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_tid_y() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_tid_z() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ntid_x() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ntid_y() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ntid_z() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ctaid_x() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ctaid_y() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_ctaid_z() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_nctaid_x() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_nctaid_y() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  ret i32 %a
}

define linkonce_odr i32 @pynvvm_nctaid_z() {
  %a = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  ret i32 %a
}

define linkonce_odr void @pynvvm_sync() {
  call void @llvm.cuda.syncthreads()
  ret void
}

declare float  @llvm.sqrt.rn.f32(float)
declare double @llvm.sqrt.rn.f64(double)

declare float  @llvm.fma.f32(float)
declare double @llvm.fma.f64(double)

declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)
;declare i64 @llvm.bswap.f64(i64)

declare i8  @llvm.ctpop.i8(i8)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
;declare i64 @llvm.ctpop.f64(i64)

declare i8  @llvm.ctlz.i8(i8)
declare i16 @llvm.ctlz.i16(i16)
declare i32 @llvm.ctlz.i32(i32)
;declare i64 @llvm.ctlz.f64(i64)

declare i8  @llvm.cttz.i8(i8)
declare i16 @llvm.cttz.i16(i16)
declare i32 @llvm.cttz.i32(i32)
;declare i64 @llvm.cttz.f64(i64)

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

declare void @llvm.cuda.syncthreads()
'''

class Scope:
  def __init__(self, parent=None):
    self._parent = parent
    self.env = dict()
    return
  
  def create_child(self):
    child = Scope()
    child.parent = self
    return child

  def search_env(self, name):
    result = self.env.get(name, (None, None))
    
    if (None, None) == result and None != self.parent:
      return self.parent.search_env(name)
    else:
      return result

  @property
  def parent(self):
    return self._parent

#end class Scope

class ASTTraverser(ast.NodeVisitor):
  
  def generic_visit(self, node):
    print('%12s : %s' % (type(node).__name__, str(node.__dict__)))
    ast.NodeVisitor.generic_visit(self, node)

#end class ASTTraverser

class Checker(ast.NodeVisitor):
  def __init__(self, arg_types):
    self.arg_types = arg_types
    self.is_visited_functiondef = False
  
  def generic_visit(self, node):
    ast.NodeVisitor.generic_visit(self, node)
  
  def visit_FunctionDef(self, node):
    if self.is_visited_functiondef:
      raise Exception, 'error : nested function definition'
    else:
      self.is_visited_functiondef = True
    self.args = node.args.args
    if not len(self.arg_types) == (len(node.args.args)):
      raise Exception, 'error : invalid kernel argument number'
  
  def visit_Assign(self, node):
    if not 1 == len(node.targets):
      raise Exception, 'error : tuppled lvalue'

  def visit_Comparator(self, node):
    if len(node.comparators):
      raise Exception, 'error : multiple comparators expr'
# end class Checker

class NameTracker(ast.NodeVisitor):
  def __init__(self):
    self.names = set()
  def generic_visit(self, node):
    ast.NodeVisitor.generic_visit(self, node)
  def visit_Name(self, node):
    self.names.add(node.id)
# end class TrackName

class IOTracker(ast.NodeVisitor):
  (nio, i, o, io) = (0x0, 0x1, 0x2, 0x3)
  def __init__(self, arg_names):
    self.args_iomap = { arg_name:self.nio for arg_name in arg_names }
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

class TypeInferencer(ast.NodeTransformer):
  def __init__(self, arg_nametypes):
    self.env_nametypes = dict(arg_nametypes)

  def generic_visit(self, node):
    return ast.NodeTransformer.generic_visit(self, node)

  def visit_FunctionDef(self, node):
    for i, body in enumerate(node.body):
      node.body[i] = self.visit(body)
    return node

  def visit_Assign(self, node):
    node.targets[0] = self.visit(node.targets[0])
    node.value = self.visit(node.value)
    
    if None == node.value.type and None == node.targets[0].type:
      raise Exception, 'error : failed type inference'
    
    if None == node.targets[0].type:
     self.env_nametypes[node.targets[0].id] = node.targets[0].type = node.value.type
    
    node.type = node.value.type
    
    return node

  def visit_Compare(self, node):
    node.left = self.visit(node.left)
    node.comparators[0] = self.visit(node.comparators[0])
   
    if not node.left.type.__class__ == node.comparators[0].type.__class__:
      raise Exception, 'error : mismatched type'

    node.type = nvtype.int1()
    
    return node
  
  def visit_Name(self, node):
    node.type = self.env_nametypes.get(node.id, None)
    return self.generic_visit(node)

  def visit_Call(self, node):
    
    for i, arg in enumerate(node.args):
      node.args[i] = self.visit(arg)
    
    if not isinstance(node.func, ast.Name):
      raise Exception, 'error : attribute function call'
    
    ret_type, arg_types = _builtin_types.get(node.func.id, (None, None))
    if (None, None) == (ret_type, arg_types):
      raise Exception, 'error : unknown function call'
    
    if not len(arg_types) == len(node.args):
      raise Exception, 'error : invalid function call'
    
    for i, arg in enumerate(node.args):
      if not isinstance(arg.type, arg_types[i]):
        raise Exception, 'error : invalid function argument'
    
    node.type = ret_type()
    
    return node
  
  def visit_Subscript(self, node):
    node.value = self.visit(node.value)
    node.slice = self.visit(node.slice)
    
    if not isinstance(node.value.type, nvtype.array):
      raise Exception, 'error : invalid subscript'
    
    node.type = node.value.type.dtype
   
    return node

  def visit_BinOp(self, node):
    node.left = self.visit(node.left)
    node.right = self.visit(node.right)

    if not node.left.type.__class__ == node.right.type.__class__:
      raise Exception, 'error : mismatched type'

    if None == node.left.type and None == node.right.type:
      raise Exception, 'error : failed type infererence'

    node.type = node.left.type
    
    return node

  def visit_Num(self, node):
    if float == type(node.n):
      node.type = nvtype.float32()
    elif int == type(node.n):
      node.type = nvtype.int32()
    else:
      raise Exception, 'error : invalid number'
    return node
# end class TypeInferencer

class CodeGenerator(ast.NodeVisitor):
  def __init__(self, arg_nametypes):
    self.context = pyllvm.get_global_context()
    self.builder = pyllvm.builder.create(self.context)
    self.module = pyllvm.module.create('pynvvm_module', self.context)
    self.module.set_data_layout(_datalayout)
    self.arg_nametypes = arg_nametypes
   
    self.llint32_zero = pyllvm.constant_int.get(pyllvm.type.get_int32_ty(self.context), 0)
    self.llint32_one = pyllvm.constant_int.get(pyllvm.type.get_int32_ty(self.context), 1)
  
  def generic_visit(self, node):
    print(type(node))
    raise Exception, 'error : unknown syntax'
  
  def visit_Module(self, node):
    ast.NodeVisitor.generic_visit(self, node)
 
  def visit_Return(self, node):
    ast.NodeVisitor.generic_visit(self, node)

  def visit_FunctionDef(self, node):
    self.scope = Scope()
    self.current_scope = self.scope
   
    llarg_tys = [arg_type.typefun()(self.context) for _, arg_type in self.arg_nametypes]
    llret_ty = pyllvm.type.get_void_ty(self.context)
    fun_ty = pyllvm.function_type.get(llret_ty, llarg_tys, False)
    self.fun = pyllvm.function.create(fun_ty, pyllvm.linkage_type.external_linkage, 'stub', self.module)

    nvvmannotate = self.module.get_or_insert_named_metadata('nvvm.annotations')
    mdstr = pyllvm.md_string.get(self.context, 'kernel')
    md_node = pyllvm.md_node.get(self.context, [self.fun, mdstr, self.llint32_one])
    nvvmannotate.add_operand(md_node)

    bb = pyllvm.basic_block.create(self.context, 'entry', self.fun)
    self.builder.set_insert_point(bb)

    args = self.fun.get_arguments()
    for i, (arg_name, arg_type) in enumerate(self.arg_nametypes):
      args[i].set_name(arg_name)
      llptr = self.builder.create_alloca(arg_type.typefun()(self.context), self.llint32_one, '')
      
      self.builder.create_store(args[i], llptr)
      self.current_scope.env[arg_name] = (arg_type, llptr)
            
    for b in node.body:
      self.visit(b)

    self.builder.create_ret_void()
   
  def visit_Assign(self, node):
    lltarget = self.visit(node.targets[0])
    llvalue = self.visit(node.value)
    self.builder.create_store(llvalue, lltarget)

  def visit_BinOp(self, node):
    lllhs = self.visit(node.left)
    llrhs = self.visit(node.right)
   
    llretval = None
    if isinstance(node.op, ast.Add):
      llretval = node.left.type.addfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.op, ast.Sub):
      llretval = node.left.type.subfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.op, ast.Mult):
      llretval = node.left.type.mulfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.op, ast.Div):
      llretval = node.left.type.divfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.op, ast.BitAnd):
      llretval = self.builder.create_and(lllhs, llrhs, '')
    elif isinstance(node.op, ast.BitOr):
      llretval = self.builder.create_or(lllhs, llrhs, '')
    elif isinstance(node.op, ast.BitXor):
      llretval = self.builder.create_xor(lllhs, llrhs, '')
    else:
      raise Exception, 'error : unsupported binop'
    
    return llretval

  def visit_BoolOp(self, node):
    
    llretval = self.visit(node.values[0])

    for value in node.values[1:]:
      llvalue = self.visit(value)
      
      if isinstance(node.op, ast.And):
        llretval = self.builder.create_and(llretval, llvalue, '')
      elif isinstance(node.op, ast.Or):
        llretval = self.builder.create_or(llretval, llvalue, '')
      elif isinstance(node.op, ast.Xor):
        llretval = self.builder.create_xor(llretval, llvalue, '')
      else:
        raise Exception, 'error : unsupported binop'
    
    return llretval

  def visit_Call(self, node):
    (ret_ty, param_ty) = _builtin_types.get(node.func.id, (None, None))
    llret_ty = ret_ty.typefun()(self.context)
    llparam_ty = [ty.typefun()(self.context) for ty in param_ty]
    llfun_ty = pyllvm.function_type.get(llret_ty, llparam_ty, False)
    llfun = pyllvm.function.create(llfun_ty, pyllvm.linkage_type.external_linkage, node.func.id, self.module)
    
    llarg = [] 
    for arg in node.args:
      llarg.append(self.visit(arg))
    
    return self.builder.create_call(llfun, llarg, '')
 
  def visit_Compare(self, node):
    lllhs = self.visit(node.left)
    llrhs = self.visit(node.comparators[0])
    
    llretval = None
    
    if isinstance(node.ops[0], ast.Eq):
      llretval = node.left.type.eqfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.ops[0], ast.NotEq):
      llretval = node.left.type.nefun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.ops[0], ast.Gt):
      llretval = node.left.type.gtfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.ops[0], ast.GtE):
      llretval = node.left.type.gefun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.ops[0], ast.Lt):
      llretval = node.left.type.ltfun()(self.builder, lllhs, llrhs, '')
    elif isinstance(node.ops[0], ast.LtE):
      llretval = node.left.type.lefun()(self.builder, lllhs, llrhs, '')
    else:
      raise Exception, 'error : unsupported compare'

    return llretval

  def visit_If(self, node):
    ifthenbb = pyllvm.basic_block.create(self.context, 'ifthen', self.fun)
    orelsebb = pyllvm.basic_block.create(self.context, 'orelse', self.fun)
    brexitbb = pyllvm.basic_block.create(self.context, 'brexit', self.fun)
    
    lltest = self.visit(node.test)
    self.builder.create_cond_br(lltest, ifthenbb, orelsebb)

    self.builder.set_insert_point(ifthenbb)
    self.current_scope = self.current_scope.create_child()
    for stm in node.body:
      self.visit(stm)
    self.builder.create_br(brexitbb)
    self.current_scope = self.current_scope.parent

    self.builder.set_insert_point(orelsebb)
    self.current_scope = self.current_scope.create_child()
    for stm in node.orelse:
      self.visit(stm)
    self.builder.create_br(brexitbb)
    self.current_scope = self.current_scope.parent

    self.builder.set_insert_point(brexitbb)

  def visit_Subscript(self, node):
    llbaseptr = self.visit(node.value)
    llidx = self.visit(node.slice.value)
    llptr = self.builder.create_gep(llbaseptr, llidx, '')
    
    if ast.Store == type(node.ctx):
      return llptr
    else:
      return self.builder.create_load(llptr)

  def visit_Name(self, node):
    ty, llptr = self.current_scope.search_env(node.id)
               
    if ast.Store == type(node.ctx):
      if None == llptr:
        assert None != node.type, 'error : failed type inference'
      
        lltype = node.type.typefun()(self.context)
        llptr = self.builder.create_alloca(lltype, self.llint32_one, '')
        self.current_scope.env[node.id] = (node.type, llptr)
      return llptr
    else:
      assert None != llptr, 'error : unknown name'
      
      return self.builder.create_load(llptr)

  def visit_Num(self, node):
    return node.type.valuefun()(node.type.typefun()(self.context), node.n)
# end class CodeGenerator


def compile_ast(tree, *arg_types):

  check = Checker(arg_types)
  check.visit(tree)
  
  arg_names = map(lambda x : x.id, check.args)
  io_track = IOTracker(arg_names)
  io_track.visit(tree)

  arg_nametypes = zip(arg_names, arg_types)
  typeinfer = TypeInferencer(arg_nametypes)
  tree = typeinfer.visit(tree)

  codegen = CodeGenerator(arg_nametypes)
  codegen.visit(tree)

  llcode = pyllvm.print_ir(codegen.module)

  #print(llcode)
  
  return llcode, io_track.args_iomap, arg_nametypes

def compile_nvvm(llcode):
  result, cu = nvvm.instance.create_cu()
  assert nvvm.nvvmResult.NVVM_SUCCESS == result.val

  result = nvvm.instance.cu_add_module_ir(cu, llcode)
  assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
  
  result = nvvm.instance.cu_add_module_ir(cu, _builtin_code)
  assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
  
  options = []
  result = nvvm.instance.compile_cu(cu, options)
  
  if nvvm.nvvmResult.NVVM_SUCCESS != result.val:
    result, msg_buffer = nvvm.instance.get_compilation_log(cu)
    print(msg_buffer)
    assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
    sys.exit(-1)

  result, ptxcode = nvvm.instance.get_compiled_result(cu)
  assert nvvm.nvvmResult.NVVM_SUCCESS == result.val

  result = nvvm.instance.destroy_cu(cu)
  assert nvvm.nvvmResult.NVVM_SUCCESS == result.val
  
  return ptxcode

def create_function(ptxcode, iomap, arg_nametypes):
  m = drv.module_from_buffer(ptxcode)
  #print(ptxcode)
  stub_function = m.get_function('stub')

  iofun = { 
            IOTracker.nio:drv.In, 
            IOTracker.i:drv.In, 
            IOTracker.o:drv.Out, 
            IOTracker.io:drv.InOut
          }

  def param_wrapper(bsz, gsz):
    
    def stub_wrapper(*args):

      assert len(args) == len(arg_nametypes), 'error : invalid number argument'
      
      wrapped_args = []
      for i, arg in enumerate(args):
        arg_name, arg_type = arg_nametypes[i]
        
        if isinstance(arg_type, nvtype.pointer):
          wrapped_args.append(iofun[iomap[arg_name]](arg))
        else:
          wrapped_args.append(arg)

      stub_function(*tuple(wrapped_args), block=bsz, grid=gsz)
      
      return 
    
    return stub_wrapper
  
  return param_wrapper

def compile(fun, *args):
  
  tree = ast.parse(inspect.getsource(fun))
   
  (llcode, iomap, arg_nametypes) = compile_ast(tree, *args)

  ptxcode = compile_nvvm(llcode)
 
  gpu_fun = create_function(ptxcode, iomap, arg_nametypes)
   
  return gpu_fun

