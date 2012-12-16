#include <llvm/Type.h>

#include "pyllvm-type.h"
#include "pyllvm-context.h"

namespace pyllvm {

PyLLVMFunctionType *PyLLVMFunctionType::get(PyLLVMType *result, bool is_var_arg)
{
  return new PyLLVMFunctionType(llvm::FunctionType::get(result->obj_, is_var_arg));
}

PyLLVMFunctionType *PyLLVMFunctionType::get(PyLLVMType *result, std::vector<PyLLVMType *> params, bool is_var_arg)
{
  std::vector<llvm::Type *> params_;
  
  for(size_t i=0; i<params.size(); ++i)
  {
    params_.push_back(params[i]->obj_);
  }
  
  return new PyLLVMFunctionType(llvm::FunctionType::get(result->obj_, params_, is_var_arg));
}

PyLLVMPointerType *PyLLVMPointerType::get(PyLLVMType *type, unsigned address_space)
{
  return new PyLLVMPointerType(llvm::PointerType::get(type->obj_, address_space));
}

} /* namespace pyllvm */

