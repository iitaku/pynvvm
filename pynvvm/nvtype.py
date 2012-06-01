
from .externals.llvm import pyllvm

class integer:
  @classmethod
  def addfun(cls):
    return pyllvm.builder.create_add
  
  @classmethod
  def subfun(cls):
    return pyllvm.builder.create_sub
  
  @classmethod
  def mulfun(cls):
    return pyllvm.builder.create_mul
 
  @classmethod
  def divfun(cls):
    return pyllvm.builder.create_sdiv

  @classmethod
  def eqfun(cls):
    return pyllvm.builder.create_icmp_eq

  @classmethod
  def nefun(cls):
    return pyllvm.builder.create_icmp_ne

  @classmethod
  def gtfun(cls):
    return pyllvm.builder.create_icmp_sgt

  @classmethod
  def gefun(cls):
    return pyllvm.builder.create_icmp_sge

  @classmethod
  def ltfun(cls):
    return pyllvm.builder.create_icmp_slt

  @classmethod
  def lefun(cls):
    return pyllvm.builder.create_icmp_sle

#end class integer

class floating:
  @classmethod
  def addfun(cls):
    return pyllvm.builder.create_fadd
  
  @classmethod
  def subfun(cls):
    return pyllvm.builder.create_fsub
  
  @classmethod
  def mulfun(cls):
    return pyllvm.builder.create_fmul
 
  @classmethod
  def divfun(cls):
    return pyllvm.builder.create_fdiv

  @classmethod
  def eqfun(cls):
    return pyllvm.builder.create_fcmp_eq

  @classmethod
  def nefun(cls):
    return pyllvm.builder.create_fcmp_ne

  @classmethod
  def gtfun(cls):
    return pyllvm.builder.create_fcmp_gt

  @classmethod
  def gefun(cls):
    return pyllvm.builder.create_fcmp_ge

  @classmethod
  def ltfun(cls):
    return pyllvm.builder.create_fcmp_lt

  @classmethod
  def lefun(cls):
    return pyllvm.builder.create_fcmp_le

#end class floating

class int1(integer):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int1_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_int.get

class int8(integer):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int8_ty
  
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_int.get

class int16(integer):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int16_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_int.get

class int32(integer):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int32_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_int.get

class int64(integer):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int64_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_int.get

class float32(floating):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_float_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_fp.get

class float64(floating):
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_double_ty
 
  @classmethod
  def valuefun(cls):
    return pyllvm.constant_fp.get

class pointer:
  def __init__(self, dtype):
    self.dtype = dtype()
  def typefun(self):
    def aux_typefun(ctx):
      return pyllvm.pointer_type.get(self.dtype.typefun()(ctx), 0)
    return aux_typefun

# type synonym
array = pointer

