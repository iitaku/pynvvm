import ctypes
import os
import sys

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

class _nvvmInstance:
  def __init__(self):
    # initialize
    
    ext = ''
    if 'darwin' == sys.platform:
      ext = 'dylib'
    elif 'windows' == sys.platform:
      ext = 'dll'
    else:
      ext = 'so'
    
    this_file = os.path.abspath(__file__)
    dllfile = os.path.join(os.path.dirname(this_file), 'externals/LibNVVM/libnvvm.'+ext)
    self._dll = ctypes.CDLL(dllfile)
 
    self._dll.nvvmInit.restype = nvvmResult
    self._dll.nvvmInit()

    # method
    self._dll.nvvmVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    self._dll.nvvmVersion.restype = nvvmResult

    self._dll.nvvmCreateCU.argtypes = [ctypes.POINTER(nvvmCU)]
    self._dll.nvvmCreateCU.restype = nvvmResult

    self._dll.nvvmDestroyCU.argtypes = [ctypes.POINTER(nvvmCU)]
    self._dll.nvvmDestroyCU.restype = nvvmResult

    self._dll.nvvmCUAddModule.argtypes = [nvvmCU, ctypes.c_char_p, ctypes.c_size_t]
    self._dll.nvvmCUAddModule.restype = nvvmResult

    self._dll.nvvmCompileCU.restype = nvvmResult

    self._dll.nvvmGetCompiledResultSize.argtypes = [nvvmCU, ctypes.POINTER(ctypes.c_size_t)]
    self._dll.nvvmGetCompiledResultSize.restype = nvvmResult

    self._dll.nvvmGetCompiledResult.restype = nvvmResult

    self._dll.nvvmGetCompilationLogSize.argtypes = [nvvmCU, ctypes.POINTER(ctypes.c_size_t)]
    self._dll.nvvmGetCompilationLogSize.restype = nvvmResult

    self._dll.nvvmGetCompilationLog.restype = nvvmResult
    
    return

  def __del__(self):
    self._dll.nvvmFini.restype = nvvmResult
    return self._dll.nvvmFini()

  def version(self):
    major = ctypes.c_int()
    minor = ctypes.c_int()
    result = self._dll.nvvmVersion(ctypes.byref(major), ctypes.byref(minor))
    return [result, major.value, minor.value]
  
  def create_cu(self):
    cu = nvvmCU()
    result = self._dll.nvvmCreateCU(ctypes.byref(cu))
    return [result, cu]
  
  def destroy_cu(self, cu):
    return self._dll.nvvmDestroyCU(ctypes.byref(cu))
  
  def cu_add_module(self, cu, code):
    return self._dll.nvvmCUAddModule(cu, ctypes.c_char_p(code.encode('ascii')), ctypes.c_size_t(len(code)))
  
  def compile_cu(self, cu, options):
    numOptions = len(options)
    options_t = ctypes.c_char_p * numOptions
    self._dll.nvvmCompileCU.argtypes = [nvvmCU, ctypes.c_int, options_t]
    options_char_p = options_t()
    for i, s in enumerate(options):
      options_char_p[i] = ctypes.c_char_p(s.encode('ascii'))
    return self._dll.nvvmCompileCU(cu, ctypes.c_int(numOptions), options_char_p)
  
  def get_compiled_result(self, cu):
    size = ctypes.c_size_t()
    result = self._dll.nvvmGetCompiledResultSize(cu, ctypes.byref(size))
    if nvvmResult.NVVM_SUCCESS != result.val:
      return result, ''
    buffer_t = ctypes.c_char * size.value
    self._dll.nvvmGetCompiledResult.argtypes = [nvvmCU, buffer_t]
    msg_buffer = buffer_t()
    result = self._dll.nvvmGetCompiledResult(cu, msg_buffer)
    return result, msg_buffer.value.decode('UTF-8')
  
  def get_compilation_log(self, cu):
    size = ctypes.c_size_t()
    result = self._dll.nvvmGetCompilationLogSize(cu, ctypes.byref(size))
    if nvvmResult.NVVM_SUCCESS != result.val:
      return result, ''
    buffer_t = ctypes.c_char * size.value
    self._dll.nvvmGetCompilationLog.argtypes = [nvvmCU, buffer_t]
    msg_buffer = buffer_t()
    result = self._dll.nvvmGetCompilationLog(cu, msg_buffer)
    return result, msg_buffer.value

instance = _nvvmInstance()
