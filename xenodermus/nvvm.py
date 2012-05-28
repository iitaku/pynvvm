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
  ) = map(ctypes.c_int, xrange(12))
  _fields_ = [('val', ctypes.c_int)]
#end nvvmResult

class nvvmCU(ctypes.Structure):
  _fields_ = [('_nvvmCU', ctypes.c_void_p)]
#end nvvmCU

_this_file = os.path.abspath(__file__)
_dllfile = os.path.join(os.path.dirname(_this_file), 'LibNVVM/libnvvm.so')
_dll = ctypes.CDLL(_dllfile)

def init():
  _dll.nvvmInit.restype = nvvmResult
  return _dll.nvvmInit()

def fini():
  _dll.nvvmFini.restype = nvvmResult
  return _dll.nvvmFini()

def version():
  _dll.nvvmVersion.argtypes = [ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]
  _dll.nvvmVersion.restype = nvvmResult
  major = ctypes.c_int()
  minor = ctypes.c_int()
  result = _dll.nvvmVersion(ctypes.byref(major), ctypes.byref(minor))
  return [result, major.value, minor.value]

def createCU():
  _dll.nvvmCreateCU.argtypes = [ctypes.POINTER(nvvmCU)]
  _dll.nvvmCreateCU.restype = nvvmResult
  cu = nvvmCU()
  result = _dll.nvvmCreateCU(ctypes.byref(cu))
  return [result, cu]

def destroyCU(cu):
  _dll.nvvmDestroyCU.argtypes = [ctypes.POINTER(nvvmCU)]
  _dll.nvvmDestroyCU.restype = nvvmResult
  return _dll.nvvmDestroyCU(ctypes.byref(cu))

