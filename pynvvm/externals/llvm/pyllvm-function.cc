
#include "pyllvm.h"
#include "pyllvm-function.h"
#include "pyllvm-function-type.h"
#include "pyllvm-module.h"

namespace pyllvm {

PyLLVMFunctionType *PyLLVMFunction::get_function_type(void)
{
  return new PyLLVMFunctionType(obj_->getFunctionType());
}

PyLLVMFunction *PyLLVMFunction::create(PyLLVMFunctionType *function_type, PyLLVMLinkageTypes linkage_type, std::string name, PyLLVMModule *module)
{
  return new PyLLVMFunction(llvm::Function::Create(function_type->obj_, linkage_type, name, module->obj_));
}

} /* namespace pyllvm */

