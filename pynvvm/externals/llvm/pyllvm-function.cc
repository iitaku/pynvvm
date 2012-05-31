
#include "pyllvm.h"
#include "pyllvm-argument.h"
#include "pyllvm-function.h"
#include "pyllvm-module.h"
#include "pyllvm-type.h"

namespace pyllvm {

PyLLVMFunction::PyLLVMFunction(llvm::Function* obj)
  : PyLLVMValue(obj), obj_(obj)
{
}

PyLLVMFunctionType *PyLLVMFunction::get_function_type(void)
{
  return new PyLLVMFunctionType(obj_->getFunctionType());
}

std::vector<PyLLVMArgument*> PyLLVMFunction::get_arguments(void)
{
  llvm::iplist<llvm::Argument>::iterator iter;
  std::vector<PyLLVMArgument*> list;
  
  for (iter = obj_->arg_begin(); iter != obj_->arg_end(); ++iter)
  {
    list.push_back(new PyLLVMArgument(static_cast<llvm::Argument*>(iter)));
  }
  
  return list;
}

PyLLVMFunction *PyLLVMFunction::create(PyLLVMFunctionType *function_type, PyLLVMLinkageTypes linkage_type, std::string name, PyLLVMModule *module)
{
  return new PyLLVMFunction(llvm::Function::Create(function_type->obj_, linkage_type, name, module->obj_));
}

} /* namespace pyllvm */

