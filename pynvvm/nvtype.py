
from .externals.llvm import pyllvm

class int8:
  @classmethod
  def lltype(cls):
    return pyllvm.type.get_int8_ty

class int16:
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int16_ty

class int32:
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int32_ty

class int64:
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_int64_ty

class float32:
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_float_ty

class float64:
  @classmethod
  def typefun(cls):
    return pyllvm.type.get_double_ty

class array:
  def __init__(self, dtype):
    self.dtype = dtype
  def typefun(self):
    def aux_typefun(ctx):
      return pyllvm.pointer_type.get(self.dtype.typefun()(ctx), 0)
    return aux_typefun

