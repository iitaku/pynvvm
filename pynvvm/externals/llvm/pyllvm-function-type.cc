
#include "pyllvm-type.h"
#include "pyllvm-function-type.h"

namespace pyllvm {

PyLLVMFunctionType *PyLLVMFunctionType::get(PyLLVMType *type, bool is_var_arg)
{
  return new PyLLVMFunctionType(llvm::FunctionType::get(type->obj_, is_var_arg));
}

} /* namespace pyllvm */

