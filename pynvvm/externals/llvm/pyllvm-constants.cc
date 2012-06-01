#include "pyllvm-constants.h"
#include "pyllvm-type.h"

namespace pyllvm {

PyLLVMConstantInt *PyLLVMConstantInt::get(PyLLVMType *type, int val)
{
  return new PyLLVMConstantInt(llvm::ConstantInt::get(type->obj_, val));
}

PyLLVMConstantFP *PyLLVMConstantFP::get(PyLLVMType *type, double val)
{
  return new PyLLVMConstantFP(llvm::ConstantFP::get(type->obj_, val));
}

} /* namespace pyllvm */
