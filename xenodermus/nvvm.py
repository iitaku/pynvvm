import os
import ctypes

class nvvmResult(ctypes.Structure):
  ( 
   NVVM_SUCCESS,
   NVVM_ERROR_OUT_OF_MEMORY,
   NVVM_ERROR_NOT_INITIALIZED,
   NVVM_ERROR_ALREADY_INITIALIZED,
   NVVM_ERROR_CU_CREATION_FAILURE,
   NVVM_ERROR_IR_VERSION_MISMATCH,
   NVVM_ERROR_INVALID_INPUT,
   NVVM_ERROR_INVALID_CU,
   NVVM_ERROR_INVALID_IR,
   NVVM_ERROR_INVALID_OPTION,
   NVVM_ERROR_NO_MODULE_IN_CU,
   NVVM_ERROR_COMPILATION
  ) = range(12)
  _fields_ = [('val', ctypes.c_int)]
  
  def __str__(self):
    for k, v in nvvmResult.__dict__.items():
      if 'NVVM' in k:
        if v == self.val:
          return k
    return 'NVVM_UNKNOWN'
#end nvvmResult

class nvvmCU(ctypes.Structure):
  _fields_ = [('_nvvmCU', ctypes.c_void_p)]
#end nvvmCU

_this_file = os.path.abspath(__file__)
_dllfile = os.path.join(os.path.dirname(_this_file), 'LibNVVM/libnvvm.so')
_dll = ctypes.CDLL(_dllfile)

def Init():
  _dll.nvvmInit.restype = nvvmResult
  return _dll.nvvmInit()

def Fini():
  _dll.nvvmFini.restype = nvvmResult
  return _dll.nvvmFini()

def Version():
  _dll.nvvmVersion.argtypes = [ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]
  _dll.nvvmVersion.restype = nvvmResult
  major = ctypes.c_int()
  minor = ctypes.c_int()
  result = _dll.nvvmVersion(ctypes.byref(major), ctypes.byref(minor))
  return [result, major.value, minor.value]

def CreateCU():
  _dll.nvvmCreateCU.argtypes = [ctypes.POINTER(nvvmCU)]
  _dll.nvvmCreateCU.restype = nvvmResult
  cu = nvvmCU()
  result = _dll.nvvmCreateCU(ctypes.byref(cu))
  return [result, cu]

def DestroyCU(cu):
  _dll.nvvmDestroyCU.argtypes = [ctypes.POINTER(nvvmCU)]
  _dll.nvvmDestroyCU.restype = nvvmResult
  return _dll.nvvmDestroyCU(ctypes.byref(cu))

def CUAddModule(cu, code):
  _dll.nvvmCUAddModule.argtypes = [nvvmCU, ctypes.c_char_p, ctypes.c_size_t]
  _dll.nvvmCUAddModule.restype = nvvmResult
  return _dll.nvvmCUAddModule(cu, ctypes.c_char_p(code), ctypes.c_size_t(len(code)))

def CompileCU(cu, options):
  numOptions = len(options)
  options_t = ctypes.c_char_p * numOptions
  _dll.nvvmCompileCU.argtypes = [nvvmCU, ctypes.c_int, options_t]
  _dll.nvvmCompileCU.restype = nvvmResult
  options_char_p = options_t()
  for i, s in enumerate(options):
    options_char_p[i] = ctypes.c_char_p(s)
  return _dll.nvvmCompileCU(cu, ctypes.c_int(numOptions), options_char_p)

def GetCompiledResult(cu):
  _dll.nvvmGetCompiledResultSize.argtypes = [nvvmCU, ctypes.POINTER(ctypes.c_size_t)]
  _dll.nvvmGetCompiledResultSize.restype = nvvmResult
  size = ctypes.c_size_t()
  result = _dll.nvvmGetCompiledResultSize(cu, ctypes.byref(size))
  if nvvmResult.NVVM_SUCCESS != result.val:
    return result, ''
  buffer_t = ctypes.c_char * size.value
  _dll.nvvmGetCompiledResult.argtypes = [nvvmCU, buffer_t]
  _dll.nvvmGetCompiledResult.restype = nvvmResult
  msg_buffer = buffer_t()
  result = _dll.nvvmGetCompiledResult(cu, msg_buffer)
  return result, msg_buffer.value

def GetCompilationLog(cu):
  _dll.nvvmGetCompilationLogSize.argtypes = [nvvmCU, ctypes.POINTER(ctypes.c_size_t)]
  _dll.nvvmGetCompilationLogSize.restype = nvvmResult
  size = ctypes.c_size_t()
  result = _dll.nvvmGetCompilationLogSize(cu, ctypes.byref(size))
  if nvvmResult.NVVM_SUCCESS != result.val:
    return result, ''
  buffer_t = ctypes.c_char * size.value
  _dll.nvvmGetCompilationLog.argtypes = [nvvmCU, buffer_t]
  _dll.nvvmGetCompilationLog.restype = nvvmResult
  msg_buffer = buffer_t()
  result = _dll.nvvmGetCompilationLog(cu, msg_buffer)
  return result, msg_buffer.value

