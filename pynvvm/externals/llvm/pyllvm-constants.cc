#include "pyllvm-constants.h"
#include "pyllvm-type.h"

namespace pyllvm {

PyLLVMConstantInt *PyLLVMConstantInt::get(PyLLVMType *type, int i)
{
  return new PyLLVMConstantInt(llvm::ConstantInt::get(type->obj_, i));
}

} /* namespace pyllvm */
