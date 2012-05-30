#include <llvm/Type.h>

#include "pyllvm-type.h"
#include "pyllvm-context.h"

namespace pyllvm {

PyLLVMType *PyLLVMType::get_int32_ty(PyLLVMContext *context)
{
  return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt32Ty(context->obj_)));
}

PyLLVMType *PyLLVMType::get_void_ty(PyLLVMContext *context)
{
  return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getVoidTy(context->obj_)));
}

PyLLVMFunctionType *PyLLVMFunctionType::get(PyLLVMType *type, bool is_var_arg)
{
  return new PyLLVMFunctionType(llvm::FunctionType::get(type->obj_, is_var_arg));
}

PyLLVMPointerType *PyLLVMPointerType::get(PyLLVMType *type, unsigned address_space)
{
  return new PyLLVMPointerType(llvm::PointerType::get(type->obj_, address_space));
}

} /* namespace pyllvm */

